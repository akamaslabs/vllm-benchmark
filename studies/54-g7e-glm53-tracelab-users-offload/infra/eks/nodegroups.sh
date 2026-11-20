#!/bin/bash
# Study 54 (and 52, 53) GPU node (node group llm-serving-g7e-24xl-spot, see nodegroup-g7e-24xl-spot.yaml): one
# g7e.24xlarge SPOT in us-east-2c (4x RTX PRO 6000 Blackwell 96 GB, 96 vCPU, 1 TiB).
# Usage: AWS_PROFILE=lab ./nodegroups.sh probe | quota | create | up | down | delete | status
#   probe    spot price and placement score of g7e.24xlarge per us-east-2 AZ (spot: no capacity
#            reservation probe). The placement-score API allows few new configurations per 24 h.
#   quota    the "All G and VT Spot Instance Requests" quota in us-east-2 (needs >= 96 vCPU).
#   create   eksctl create nodegroup at desired 0, without touching the cluster's NVIDIA device
#            plugin (--install-nvidia-plugin=false, as in studies 50 and 51).
#   up       checks the quota, tags the ASG AlwaysOn=true (17:00 UTC stop Lambda), scales to 1,
#            waits for the Ready node, tags the instance.
#   down     scales to 0 and removes the ASG AlwaysOn tag.
#   delete   eksctl delete nodegroup.
#   status   node group, instance (lifecycle spot), node with its allocatable GPU, CPU, memory.
# Spot: a reclaimed node comes back as a new instance in the same ASG (empty disk: the vLLM pod
# re-downloads the checkpoint, see k8s/01-vllm_template.yaml).
set -euo pipefail
CLUSTER=vllm-bench; REGION=us-east-2; NG=llm-serving-g7e-24xl-spot; IT=g7e.24xlarge; VCPU=96
HERE=$(cd "$(dirname "$0")" && pwd)
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

asg() { aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
  --query 'nodegroup.resources.autoScalingGroups[0].name' --output text; }
scale() { aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
  --scaling-config "minSize=0,maxSize=1,desiredSize=$1" --query 'update.id' --output text; }
quota() { aws service-quotas get-service-quota --region $REGION --service-code ec2 --quota-code L-3819A6DF \
  --query 'Quota.Value' --output text; }

case ${1:-} in
  probe)
    aws ec2 describe-spot-price-history --region $REGION --instance-types $IT --product-descriptions Linux/UNIX \
      --start-time "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output text
    aws ec2 get-spot-placement-scores --region $REGION --instance-types $IT --target-capacity 1 \
      --single-availability-zone --region-names $REGION --query 'SpotPlacementScores[].[AvailabilityZoneId,Score]' --output text
    echo "(use2-az1 = us-east-2a, use2-az2 = us-east-2b, use2-az3 = us-east-2c)" ;;
  quota)
    echo "spot G vCPU quota in $REGION: $(quota) (need >= $VCPU)" ;;
  create)
    eksctl create nodegroup -f "$HERE/nodegroup-g7e-24xl-spot.yaml" --install-nvidia-plugin=false ;;
  up)
    q=$(quota); awk -v q="$q" -v n=$VCPU 'BEGIN { exit !(q + 0 >= n) }' \
      || { echo "error: spot G quota $q < $VCPU vCPU: the instance would be refused (VcpuLimitExceeded)" >&2; exit 3; }
    ASG=$(asg)
    aws autoscaling create-or-update-tags --region $REGION \
      --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
    scale 1
    echo "waiting for the Ready node with node-role=$NG (spot shortage shows in"
    echo "  aws eks describe-nodegroup ... --query nodegroup.health.issues)"
    for _ in $(seq 1 120); do
      [ -n "$(kubectl get nodes -l node-role=$NG -o name 2>/dev/null)" ] && break; sleep 10
    done
    kubectl wait --for=condition=Ready node -l node-role=$NG --timeout=900s
    for i in $(aws ec2 describe-instances --region $REGION --filters Name=tag:eks:nodegroup-name,Values=$NG \
        Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text); do
      aws ec2 create-tags --region $REGION --resources "$i" --tags Key=AlwaysOn,Value=true
    done
    "$0" status ;;
  down)
    ASG=$(asg)
    scale 0
    aws autoscaling delete-tags --region $REGION \
      --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn" || true
    echo "$NG scaled to 0, AlwaysOn removed from its ASG" ;;
  delete)
    eksctl delete nodegroup -f "$HERE/nodegroup-g7e-24xl-spot.yaml" --approve --wait ;;
  status)
    aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
      --query 'nodegroup.[nodegroupName,status,capacityType,scalingConfig.desiredSize,health.issues[0].code]' --output text 2>&1 | tail -1
    aws ec2 describe-instances --region $REGION --filters Name=tag:eks:nodegroup-name,Values=$NG \
      Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
      --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceLifecycle,Placement.AvailabilityZone,Tags[?Key==`AlwaysOn`]|[0].Value]' --output text
    kubectl get nodes -l node-role=$NG -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory' ;;
  *)
    sed -n '2,16p' "$0"; exit 2 ;;
esac
