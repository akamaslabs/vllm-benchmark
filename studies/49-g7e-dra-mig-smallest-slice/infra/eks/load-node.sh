#!/bin/bash
# Scales the existing CPU node group m8a-2xlarge (m8a.2xlarge, 8 vCPU / 32 GiB, taint
# dedicated=m8a-2xlarge:NoSchedule, subnets 2a/2b/2c) used by study 49's AIPerf clients and slice-roles exporter (copied from study 46).
# Usage: AWS_PROFILE=lab ./load-node.sh up|down
#   up   desiredSize 1, wait Ready, tag ASG + instance AlwaysOn=true (17:00 UTC stop)
#   down desiredSize 0, remove the ASG AlwaysOn tag
set -euo pipefail
CLUSTER=vllm-bench; REGION=us-east-2; NG=m8a-2xlarge
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE
ASG=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
  --query 'nodegroup.resources.autoScalingGroups[0].name' --output text)
case ${1:-} in
  up)
    aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
      --scaling-config minSize=0,maxSize=1,desiredSize=1 --query 'update.id' --output text
    aws autoscaling create-or-update-tags --region $REGION \
      --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
    for _ in $(seq 1 90); do kubectl get nodes -l node-role=$NG -o name | grep -q . && break; sleep 10; done
    kubectl wait --for=condition=Ready node -l node-role=$NG --timeout=900s
    for i in $(aws ec2 describe-instances --region $REGION --filters Name=tag:eks:nodegroup-name,Values=$NG \
        Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text); do
      aws ec2 create-tags --region $REGION --resources "$i" --tags Key=AlwaysOn,Value=true
    done ;;
  down)
    aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
      --scaling-config minSize=0,maxSize=1,desiredSize=0 --query 'update.id' --output text
    aws autoscaling delete-tags --region $REGION \
      --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn" ;;
  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
