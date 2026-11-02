#!/bin/bash
# Study 45 GPU node groups on the vllm-bench cluster (us-east-2), one per instance the optimizer
# can choose (see ../k8s/nodes.sh for the table). All pinned to the public us-east-2b subnet
# (same AZ: the KV stream of a P/D pair would otherwise be billed as inter-AZ traffic), min 0 /
# max 2 (two nodes when node_a and node_b are the same instance), desired 0: the workflow
# scales them per experiment, and its last task scales everything back to 0.
#
# Three groups come from study 44 (llm-serving-l4-pd-xl g6.xlarge, llm-serving-a10g-pd
# g5.2xlarge, llm-serving-l40s-pd g6e.2xlarge, created by ../../44-l4-pd-multinode/infra/
# nodegroups.sh); this script creates the other three. Every group's ASG carries AlwaysOn=true
# with PropagateAtLaunch, so the lab's 17:00 UTC stop Lambda leaves running experiments alone.
#
#   AWS_PROFILE=lab ./nodegroups.sh create     create the missing groups and tag every ASG
#   AWS_PROFILE=lab ./nodegroups.sh down       scale all six to 0
#   AWS_PROFILE=lab ./nodegroups.sh status
#   AWS_PROFILE=lab ./nodegroups.sh delete     delete the three groups this script created
set -euo pipefail
REGION=us-east-2
CLUSTER=vllm-bench
SUBNET=subnet-0cfc3532e8268ea75   # eksctl-vllm-bench-cluster/SubnetPublicUSEAST2B
NODE_ROLE_ARN=arn:aws:iam::916205288457:role/eksctl-vllm-bench-nodegroup-llm-se-NodeInstanceRole-U345Y6VF5KIs
export AWS_PROFILE=${AWS_PROFILE:-lab}
NEW="llm-serving-g5-xl:g5.xlarge llm-serving-g6-2xl:g6.2xlarge llm-serving-g6e-xl:g6e.xlarge"
ALL="llm-serving-l4-pd-xl llm-serving-g6-2xl llm-serving-g5-xl llm-serving-a10g-pd llm-serving-g6e-xl llm-serving-l40s-pd"

exists() { aws eks describe-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$1" >/dev/null 2>&1; }
tag_asg() {
  local asg
  asg=$(aws eks describe-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$1" \
    --query 'nodegroup.resources.autoScalingGroups[0].name' --output text)
  aws autoscaling create-or-update-tags --region $REGION --tags \
    "ResourceId=$asg,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
  echo "$1: ASG $asg tagged AlwaysOn=true (propagated at launch)"
}

case "${1:-status}" in
  create)
    for e in $NEW; do
      g=${e%%:*}; it=${e##*:}
      if exists "$g"; then echo "$g exists"; continue; fi
      aws eks create-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$g" \
        --node-role $NODE_ROLE_ARN --subnets $SUBNET \
        --scaling-config minSize=0,maxSize=2,desiredSize=0 --instance-types "$it" \
        --capacity-type ON_DEMAND --ami-type AL2023_x86_64_NVIDIA --disk-size 200 \
        --labels node-role="$g",study=45 \
        --taints 'key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE' \
        --tags Project=vllm-bench,Study=45,AlwaysOn=true >/dev/null
      echo "$g ($it): creating"
    done
    for e in $NEW; do aws eks wait nodegroup-active --region $REGION --cluster-name $CLUSTER --nodegroup-name "${e%%:*}"; done
    for g in $ALL; do tag_asg "$g"; done
    ;;
  down)
    for g in $ALL; do
      aws eks update-nodegroup-config --region $REGION --cluster-name $CLUSTER --nodegroup-name "$g" \
        --scaling-config minSize=0,maxSize=2,desiredSize=0 >/dev/null && echo "$g -> 0"
    done
    ;;
  delete)
    for e in $NEW; do aws eks delete-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "${e%%:*}" >/dev/null && echo "${e%%:*}: deleting"; done
    ;;
  status)
    for g in $ALL; do
      aws eks describe-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$g" \
        --query 'nodegroup.[nodegroupName,status,instanceTypes[0],scalingConfig.desiredSize]' --output text 2>/dev/null || echo "$g: not created"
    done
    ;;
  *) echo "usage: $0 create|down|status|delete"; exit 2 ;;
esac
