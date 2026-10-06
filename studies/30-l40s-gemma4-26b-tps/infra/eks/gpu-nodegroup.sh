#!/bin/bash
# Creates (or reconciles) this study's GPU node group: by default llm-serving-l40s-1xl, one
# g6e.xlarge (1x NVIDIA L40S 48 GB, Ada SM 8.9, 4 vCPU / 32 GiB).
#
# History of the instance size (capacity-reservation probes: create a reservation, cancel
# it at once; study 17's infra/eks/gpu-capacity-fallback.sh technique):
#   2026-10-05: only g6e.8xlarge (32 vCPU / 256 GiB) had capacity in us-east-2, so
#     llm-serving-l40s-8xl was created (desiredSize 0, the study was due the next morning).
#   2026-10-06 07:18 UTC: g6e.8xlarge empty in every AZ; only g6e.xlarge had capacity
#     (2a/2b/2c). The user chose to run on it: llm-serving-l40s-1xl. Same GPU; the pod's
#     CPU / memory were lowered to fit (k8s/01-statefulset_template.yaml).
# The instance type of a managed node group is immutable, so each size has its own node
# group and label. llm-serving-l40s-8xl and the older llm-serving-l40s (g6e.4xlarge, study
# 17's fallback) stay at 0. Override with NG=... INSTANCE_TYPE=... to (re)create another.
#
# `aws eks create-nodegroup` with the EKS-managed NVIDIA AMI, no launch template (as
# llm-serving-g7-4500). One taint, nvidia.com/gpu=present:NoSchedule: the cluster-wide
# device plugin (kube-system, static manifest v0.18.0, tolerates only that taint) serves
# this node; there is no GPU sharing here.
#
# Created at desiredSize 0. Idempotent: re-running on an existing node group only re-applies
# the taint.
# Usage: AWS_PROFILE=lab ./gpu-nodegroup.sh [--up] [--always-on]
#   --up         sets desiredSize 1 and waits for the node to be Ready
#   --always-on  tags the node group's ASG AlwaysOn=true (PropagateAtLaunch) and its running
#                instance, so the lab's 17:00 UTC stop Lambda leaves the node running.
#                Remove the ASG tag when the study ends and the node group goes to 0.
set -euo pipefail
CLUSTER=vllm-bench
REGION=us-east-2
NG=${NG:-llm-serving-l40s-1xl}
INSTANCE_TYPE=${INSTANCE_TYPE:-g6e.xlarge}
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

UP=0; ALWAYS_ON=0
for a in "$@"; do
  case $a in
    --up) UP=1 ;;
    --always-on) ALWAYS_ON=1 ;;
    *) echo "usage: $0 [--up] [--always-on]" >&2; exit 2 ;;
  esac
done

if aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG >/dev/null 2>&1; then
  echo "node group $NG exists — re-applying the taint"
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --taints 'addOrUpdateTaints=[{key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE}]' \
    --query 'update.id' --output text || true   # "no changes" is not an error
else
  # Same instance role as the cluster's other node groups.
  ROLE=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name akamas \
    --query 'nodegroup.nodeRole' --output text)
  # Public subnets of 2a/2b/2c (both g6e sizes used here had capacity in all three).
  SUBNETS=$(aws ec2 describe-subnets --region $REGION \
    --filters "Name=tag:Name,Values=eksctl-${CLUSTER}-cluster/SubnetPublic*" \
    --query 'Subnets[].SubnetId' --output text)
  # shellcheck disable=SC2086
  aws eks create-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --node-role "$ROLE" --subnets $SUBNETS \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 \
    --instance-types "$INSTANCE_TYPE" --capacity-type ON_DEMAND \
    --ami-type AL2023_x86_64_NVIDIA --disk-size 200 \
    --labels node-role=$NG \
    --taints key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE \
    --tags Project=vllm-bench,Study=30-l40s-gemma4-26b-tps \
    --query 'nodegroup.[nodegroupName,status,releaseVersion]' --output text
  aws eks wait nodegroup-active --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG
fi

if [ $UP = 1 ]; then
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --scaling-config minSize=0,maxSize=1,desiredSize=1 --query 'update.id' --output text
  echo "waiting for a Ready node with node-role=$NG (an InsufficientInstanceCapacity shows in"
  echo "  aws eks describe-nodegroup ... --query nodegroup.health.issues)"
  for _ in $(seq 1 90); do
    kubectl get nodes -l node-role=$NG -o name 2>/dev/null | grep -q . && break
    sleep 10
  done
  kubectl wait --for=condition=Ready node -l node-role=$NG --timeout=900s
fi

if [ $ALWAYS_ON = 1 ]; then
  ASG=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --query 'nodegroup.resources.autoScalingGroups[0].name' --output text)
  aws autoscaling create-or-update-tags --region $REGION \
    --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
  for i in $(aws ec2 describe-instances --region $REGION --filters Name=tag:eks:nodegroup-name,Values=$NG \
      Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text); do
    aws ec2 create-tags --region $REGION --resources "$i" --tags Key=AlwaysOn,Value=true
  done
  echo "AlwaysOn=true on $ASG and its running instance(s)"
fi
