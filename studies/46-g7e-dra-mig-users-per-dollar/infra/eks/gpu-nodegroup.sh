#!/bin/bash
# Creates (or reconciles) study 46's GPU node group: llm-serving-g7e-4xl, one g7e.4xlarge
# (1x NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB, 16 vCPU / 128 GiB, 4.00 USD/h).
# History: Phase 0 ran on llm-serving-g7e-2xl (g7e.2xlarge, 8 vCPU): with 4 vLLM replicas two
# of them were CPU-throttled ~15 % at the peak minute already at concurrency 64, so on
# 2026-11-03 the user chose g7e.4xlarge (NG=llm-serving-g7e-2xl INSTANCE_TYPE=g7e.2xlarge
# reproduces the old one).
# Copied from study 30's script (self-contained studies) and adapted.
#
# Why g7e.2xlarge: capacity probes on 2026-11-03 (studies/17-.../infra/eks/
# gpu-capacity-fallback.sh probe) found g7e.4xlarge empty in every AZ and g7e.8xlarge gone
# by 13:25 UTC; 2xlarge was the only size seen in both probes. Capacity moved between AZs
# within hours, hence subnets 2a/2b/2c. Probe before --up.
#
# EKS-managed AL2023 NVIDIA AMI for the cluster version (1.36 since 2026-11-03), no launch
# template, no pinned AMI (the old llm-serving-g7e pins a 1.35 AMI and is left alone).
# GPUs on this node are served by the DRA driver (../dra/), NOT by a device plugin:
# ../dra/install.sh excludes this node group from the kube-system nvidia-device-plugin.
# Taint nvidia.com/gpu=present:NoSchedule (the DRA kubelet plugin tolerates it).
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
NG=${NG:-llm-serving-g7e-4xl}
INSTANCE_TYPE=${INSTANCE_TYPE:-g7e.4xlarge}
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
  # The taint update puts the node group in UPDATING; a scaling update right after it is
  # refused ("not in Active State", 2026-10-07). Wait for ACTIVE first.
  aws eks wait nodegroup-active --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG
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
    --tags Project=vllm-bench,Study=46-g7e-dra-mig-users-per-dollar \
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
