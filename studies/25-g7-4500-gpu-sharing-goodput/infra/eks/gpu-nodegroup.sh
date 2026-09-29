#!/bin/bash
# Creates (or reconciles) this study's GPU node group: llm-serving-g7-4500, one
# g7.4xlarge (1x NVIDIA RTX PRO 4500 Blackwell Server Edition, 32 GB, 165 W, SM 12.0).
#
# Created this way on 2026-09-29 and running since: `aws eks create-nodegroup` with the
# EKS-managed NVIDIA AMI type, no launch template. Why not eksctl: see cluster.yaml.
# Why g7 and not g7e (RTX PRO 6000): g7e had no on-demand capacity in any us-east-2 AZ
# for days (studies 2/17), while g7 had capacity in 2a/2b/2c.
#
# Two taints:
#   nvidia.com/gpu=present:NoSchedule        the usual GPU taint
#   akamas.io/gpu-sharing=managed:NoSchedule keeps the cluster-wide NVIDIA device plugin
#       (kube-system, tolerates only nvidia.com/gpu) off this node, so that this study's
#       own plugin (../gpu-sharing/) is the only one registering nvidia.com/gpu here.
#
# Idempotent: re-running on an existing node group only re-applies the taints.
# Usage: AWS_PROFILE=lab ./gpu-nodegroup.sh [--always-on]
#   --always-on  tags the node group's ASG AlwaysOn=true (PropagateAtLaunch) so the lab's
#                17:00 UTC stop Lambda leaves the node running. Remove the tag when the
#                study ends and the node group goes to 0.
set -euo pipefail
CLUSTER=vllm-bench
REGION=us-east-2
NG=llm-serving-g7-4500
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

TAINTS='key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE key=akamas.io/gpu-sharing,value=managed,effect=NO_SCHEDULE'

if aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG >/dev/null 2>&1; then
  echo "node group $NG exists — re-applying taints"
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --taints 'addOrUpdateTaints=[{key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE},{key=akamas.io/gpu-sharing,value=managed,effect=NO_SCHEDULE}]' \
    --query 'update.id' --output text
else
  # Same instance role as the cluster's other node groups.
  ROLE=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name akamas \
    --query 'nodegroup.nodeRole' --output text)
  SUBNETS=$(aws ec2 describe-subnets --region $REGION \
    --filters "Name=tag:Name,Values=eksctl-${CLUSTER}-cluster/SubnetPublic*" \
    --query 'Subnets[].SubnetId' --output text)
  # shellcheck disable=SC2086
  aws eks create-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --node-role "$ROLE" --subnets $SUBNETS \
    --scaling-config minSize=0,maxSize=1,desiredSize=1 \
    --instance-types g7.4xlarge --capacity-type ON_DEMAND \
    --ami-type AL2023_x86_64_NVIDIA --disk-size 200 \
    --labels node-role=$NG \
    --taints $TAINTS \
    --query 'nodegroup.[nodegroupName,status,releaseVersion]' --output text
  aws eks wait nodegroup-active --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG
fi

if [ "${1:-}" = --always-on ]; then
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
