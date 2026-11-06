#!/bin/bash
# Study 47 and 48's GPU node group: llm-serving-g7-kv, one g7.4xlarge (1x NVIDIA RTX PRO 4500
# Blackwell 32 GB, 16 vCPU / 64 GiB, 57 GiB allocatable, 3.04 USD/h). Created on 2026-11-05 for
# the KV offload spike (docs/spikes/2026-11-05-kv-offload/): the older llm-serving-g7-4500
# cannot be used because the StatefulSet gpu-sharing/vllm (an old study) grabs its GPU as soon
# as the node group comes up. GPU served by the kube-system nvidia-device-plugin (exclusive).
# EKS-managed AL2023 NVIDIA AMI for 1.36, taint nvidia.com/gpu=present:NoSchedule.
#
# Usage: AWS_PROFILE=lab ./gpu-nodegroup.sh [--up] [--always-on] | --down
#   (none)       create the node group at desiredSize 0 if missing
#   --up         desiredSize 1, wait for the node to be Ready
#   --always-on  tag the ASG (PropagateAtLaunch) and the running instance AlwaysOn=true, so
#                the lab's 17:00 UTC stop Lambda leaves the node running during the study
#   --down       desiredSize 0 and remove the ASG AlwaysOn tag (end of study)
set -euo pipefail
CLUSTER=vllm-bench
REGION=us-east-2
NG=llm-serving-g7-kv
INSTANCE_TYPE=g7.4xlarge
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

UP=0; ALWAYS_ON=0; DOWN=0
for a in "$@"; do
  case $a in
    --up) UP=1 ;;
    --always-on) ALWAYS_ON=1 ;;
    --down) DOWN=1 ;;
    *) echo "usage: $0 [--up] [--always-on] | --down" >&2; exit 2 ;;
  esac
done

asg() { aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
  --query 'nodegroup.resources.autoScalingGroups[0].name' --output text; }

if [ $DOWN = 1 ]; then
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 --query 'update.id' --output text
  aws autoscaling delete-tags --region $REGION \
    --tags "ResourceId=$(asg),ResourceType=auto-scaling-group,Key=AlwaysOn" || true
  echo "$NG scaled to 0, AlwaysOn removed from its ASG"
  exit 0
fi

if ! aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG >/dev/null 2>&1; then
  ROLE=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name akamas \
    --query 'nodegroup.nodeRole' --output text)
  SUBNETS=$(aws ec2 describe-subnets --region $REGION \
    --filters "Name=tag:Name,Values=eksctl-${CLUSTER}-cluster/SubnetPublic*" \
    --query 'Subnets[].SubnetId' --output text)
  # shellcheck disable=SC2086
  aws eks create-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --node-role "$ROLE" --subnets $SUBNETS \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 \
    --instance-types $INSTANCE_TYPE --capacity-type ON_DEMAND \
    --ami-type AL2023_x86_64_NVIDIA --disk-size 200 \
    --labels node-role=$NG \
    --taints key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE \
    --tags Project=vllm-bench,Study=47-48-kv-offload \
    --query 'nodegroup.[nodegroupName,status]' --output text
  aws eks wait nodegroup-active --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG
fi

if [ $UP = 1 ]; then
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --scaling-config minSize=0,maxSize=1,desiredSize=1 --query 'update.id' --output text
  echo "waiting for a Ready node with node-role=$NG (capacity issues show in"
  echo "  aws eks describe-nodegroup ... --query nodegroup.health.issues)"
  for _ in $(seq 1 90); do
    kubectl get nodes -l node-role=$NG -o name 2>/dev/null | grep -q . && break
    sleep 10
  done
  kubectl wait --for=condition=Ready node -l node-role=$NG --timeout=900s
fi

if [ $ALWAYS_ON = 1 ]; then
  aws autoscaling create-or-update-tags --region $REGION \
    --tags "ResourceId=$(asg),ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
  for i in $(aws ec2 describe-instances --region $REGION --filters Name=tag:eks:nodegroup-name,Values=$NG \
      Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text); do
    aws ec2 create-tags --region $REGION --resources "$i" --tags Key=AlwaysOn,Value=true
  done
  echo "AlwaysOn=true on $(asg) and its running instance(s)"
fi
