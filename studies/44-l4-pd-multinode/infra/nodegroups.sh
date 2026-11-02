#!/bin/bash
# Study 44 GPU node groups: two single-L4 nodes in ONE availability zone, so prefill and
# decode run on different machines and the KV cache crosses the VPC network (TCP, no EFA).
#
#   llm-serving-l4-pd     g6.4xlarge  (1x L4, 16 vCPU, 64 GiB, network 10 Gbps baseline / 25 burst)
#   llm-serving-l4-pd-xl  g6.xlarge   (1x L4,  4 vCPU, 16 GiB, network 2.5 Gbps baseline / 10 burst)
# Tappa B (different GPU per role), one node each:
#   llm-serving-l40s-pd   g6e.2xlarge (1x L40S, 8 vCPU, 64 GiB, network 5 / 20 Gbps)   prefill
#   llm-serving-a10g-pd   g5.2xlarge  (1x A10G, 8 vCPU, 32 GiB, network 5 / 10 Gbps)   decode
#   (g6e.xlarge had no capacity in any us-east-2 AZ on 2026-10-29; g6e.2xlarge and g5.* had)
#
# Both are pinned to the public us-east-2b subnet: inter-AZ traffic is billed in both
# directions (~$0.02/GB together), and the KV stream (~0.3 GB per 4096-token prompt in fp8)
# would cost more than the GPUs. us-east-2b is also where the system node (Prometheus, AIPerf)
# lives. Capacity for 2 instances of each type was probed free in all three AZs on 2026-10-29.
#
# Usage (AWS_PROFILE=lab, kubectl on the vllm-bench context):
#   ./nodegroups.sh create                  create all node groups at 0 nodes (idempotent)
#   ./nodegroups.sh up   <size> [count]     scale a group (4xl|xl default 2, l40s|a10g default 1),
#                                           tag AlwaysOn, wait Ready
#   ./nodegroups.sh down <size>|all         scale to 0 (stops GPU billing)
#   ./nodegroups.sh delete                  delete all node groups
#   ./nodegroups.sh status
# Only one group should be up at a time: the 4 L4s together fit the 64-vCPU G quota
# (2x16 + 2x4 = 40), but the manifests select the group by label, so there is no reason to.
set -euo pipefail
REGION=us-east-2
CLUSTER=vllm-bench
SUBNET=subnet-0cfc3532e8268ea75   # eksctl-vllm-bench-cluster/SubnetPublicUSEAST2B
NODE_ROLE_ARN=arn:aws:iam::916205288457:role/eksctl-vllm-bench-nodegroup-llm-se-NodeInstanceRole-U345Y6VF5KIs
export AWS_PROFILE=${AWS_PROFILE:-lab}

group_of() {
  case "$1" in
    4xl) echo llm-serving-l4-pd ;;
    xl) echo llm-serving-l4-pd-xl ;;
    l40s) echo llm-serving-l40s-pd ;;
    a10g) echo llm-serving-a10g-pd ;;
    *) echo "unknown size '$1' (4xl|xl|l40s|a10g)" >&2; exit 2 ;;
  esac
}

create_one() {  # name instance-type
  local name=$1 it=$2
  if aws eks describe-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$name" >/dev/null 2>&1; then
    echo "$name exists"; return
  fi
  aws eks create-nodegroup --region $REGION --cluster-name $CLUSTER \
    --nodegroup-name "$name" \
    --node-role $NODE_ROLE_ARN \
    --subnets $SUBNET \
    --scaling-config minSize=0,maxSize=2,desiredSize=0 \
    --instance-types "$it" \
    --capacity-type ON_DEMAND \
    --ami-type AL2023_x86_64_NVIDIA \
    --disk-size 200 \
    --labels node-role="$name",study=44 \
    --taints 'key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE' \
    --tags Project=vllm-bench,Study=44,AlwaysOn=true >/dev/null
  echo "$name: creating"
  aws eks wait nodegroup-active --region $REGION --cluster-name $CLUSTER --nodegroup-name "$name"
  echo "$name: ACTIVE"
}

tag_always_on() {  # name — the nightly Lambda stops untagged instances at 17:00 UTC
  local name=$1 asg ids
  asg=$(aws eks describe-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$name" \
    --query 'nodegroup.resources.autoScalingGroups[0].name' --output text)
  aws autoscaling create-or-update-tags --region $REGION --tags \
    "ResourceId=$asg,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
  ids=$(aws autoscaling describe-auto-scaling-groups --region $REGION --auto-scaling-group-names "$asg" \
    --query 'AutoScalingGroups[0].Instances[].InstanceId' --output text)
  [[ -n $ids ]] && aws ec2 create-tags --region $REGION --resources $ids --tags Key=AlwaysOn,Value=true
  echo "$name: ASG $asg and instances [${ids:-none}] tagged AlwaysOn=true"
}

scale() {  # name desired
  aws eks update-nodegroup-config --region $REGION --cluster-name $CLUSTER --nodegroup-name "$1" \
    --scaling-config minSize=0,maxSize=2,desiredSize="$2" >/dev/null
  echo "$1: desired=$2"
}

case "${1:-status}" in
  create)
    create_one llm-serving-l4-pd g6.4xlarge
    create_one llm-serving-l4-pd-xl g6.xlarge
    create_one llm-serving-l40s-pd g6e.2xlarge
    create_one llm-serving-a10g-pd g5.2xlarge
    ;;
  up)
    g=$(group_of "${2:?size}")
    case $2 in 4xl|xl) want=${3:-2} ;; *) want=${3:-1} ;; esac
    scale "$g" "$want"
    echo "waiting for $want Ready node(s) with node-role=$g ..."
    for _ in $(seq 1 60); do
      n=$(kubectl get nodes -l node-role="$g" --no-headers 2>/dev/null | grep -c ' Ready' || true)
      [[ $n -ge $want ]] && break
      sleep 20
    done
    kubectl get nodes -l node-role="$g" -L topology.kubernetes.io/zone,node.kubernetes.io/instance-type
    tag_always_on "$g"
    ;;
  down)
    if [[ ${2:-} == all ]]; then for g in llm-serving-l4-pd llm-serving-l4-pd-xl llm-serving-l40s-pd llm-serving-a10g-pd; do scale $g 0; done
    else scale "$(group_of "${2:?size}")" 0; fi
    ;;
  delete)
    for g in llm-serving-l4-pd llm-serving-l4-pd-xl llm-serving-l40s-pd llm-serving-a10g-pd; do
      aws eks delete-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$g" >/dev/null && echo "$g: deleting" || true
    done
    ;;
  status)
    for g in llm-serving-l4-pd llm-serving-l4-pd-xl llm-serving-l40s-pd llm-serving-a10g-pd; do
      aws eks describe-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$g" \
        --query 'nodegroup.[nodegroupName,status,instanceTypes[0],scalingConfig.desiredSize]' --output text 2>/dev/null \
        || echo "$g: not created"
    done
    kubectl get nodes -l study=44 -L node-role,topology.kubernetes.io/zone,node.kubernetes.io/instance-type 2>/dev/null || true
    ;;
  *) echo "usage: $0 create|up <size> [count]|down <size>|all|delete|status"; exit 2 ;;
esac
