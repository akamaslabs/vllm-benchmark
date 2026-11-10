#!/bin/bash
# Study 50 GPU pair (node group llm-serving-g7e-8xl-efa, see nodegroups.yaml).
# Usage: AWS_PROFILE=lab ./nodegroups.sh probe | create <az> | up | down | delete | status
#   probe         asks EC2 for TWO g7e.8xlarge in each AZ with a capacity reservation that is
#                 cancelled at once (study 17's probe): prints YES/no per AZ. Costs nothing.
#   create <az>   eksctl create nodegroup in that AZ, desired 0, without touching the cluster's
#                 NVIDIA device plugin; then applies the study's EFA device plugin (../efa/).
#   up            tags the ASG AlwaysOn=true (17:00 UTC stop Lambda), scales to 2, waits Ready,
#                 tags the instances, labels the nodes study50/role=a and =b (sorted by name).
#   down          scales to 0 and removes the ASG AlwaysOn tag.
#   delete        eksctl delete nodegroup (stack, launch template, placement group).
#   status        node group, instances, nodes with their allocatable GPU / EFA / hugepages.
#   wait-up [h]   re-probes every 5 minutes for up to h hours (default 10); at the first AZ that
#                 can host the pair runs create (or delete + create if the node group sits in
#                 another AZ) and up. If up fails (capacity gone in between) it scales back to 0
#                 and keeps probing. Exit 0 = two labelled Ready nodes, 1 = no capacity in time.
# The G on-demand quota of the lab is 64 vCPU = exactly this pair: no other G instance may run.
set -euo pipefail
CLUSTER=vllm-bench; REGION=us-east-2; NG=llm-serving-g7e-8xl-efa; IT=g7e.8xlarge
HERE=$(cd "$(dirname "$0")" && pwd)
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

asg() { aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
  --query 'nodegroup.resources.autoScalingGroups[0].name' --output text; }
# AZ of the existing node group, empty when there is none.
ngaz() { local sn; sn=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
    --query 'nodegroup.subnets[0]' --output text 2>/dev/null) || { echo ""; return 0; }
  aws ec2 describe-subnets --region $REGION --subnet-ids "$sn" --query 'Subnets[0].AvailabilityZone' --output text; }
scale() { aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
  --scaling-config "minSize=0,maxSize=2,desiredSize=$1" --query 'update.id' --output text; }

case ${1:-} in
  probe)
    end=$(date -u -v+1H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+1 hour' '+%Y-%m-%dT%H:%M:%SZ')
    for az in us-east-2a us-east-2b us-east-2c; do
      out=$(aws ec2 create-capacity-reservation --region $REGION --instance-type $IT \
        --instance-platform Linux/UNIX --availability-zone $az --instance-count 2 \
        --instance-match-criteria targeted --end-date-type limited --end-date "$end" \
        --query 'CapacityReservation.CapacityReservationId' --output text 2>&1) || true
      if [[ "$out" == cr-* ]]; then
        echo "$az YES"
        aws ec2 cancel-capacity-reservation --region $REGION --capacity-reservation-id "$out" >/dev/null \
          || echo "!! reservation $out NOT cancelled, cancel it by hand !!"
      elif echo "$out" | grep -q InsufficientInstanceCapacity; then echo "$az no"
      else echo "$az n/a: $(echo "$out" | tail -1 | cut -c1-200)"; fi
    done
    echo "reservations still active (must be empty):"
    aws ec2 describe-capacity-reservations --region $REGION --filters Name=state,Values=active,pending \
      --query 'CapacityReservations[].[CapacityReservationId,InstanceType,AvailabilityZone,State]' --output text ;;
  create)
    AZ=${2:?usage: $0 create <az>}
    sed "s/@AZ@/$AZ/" "$HERE/nodegroups.yaml" | eksctl create nodegroup -f - --install-nvidia-plugin=false
    kubectl create namespace moe-efa --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f "$HERE/../efa/efa-device-plugin.yaml" ;;
  up)
    ASG=$(asg)
    aws autoscaling create-or-update-tags --region $REGION \
      --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
    scale 2
    echo "waiting for two Ready nodes with node-role=$NG (InsufficientInstanceCapacity shows in"
    echo "  aws eks describe-nodegroup ... --query nodegroup.health.issues)"
    for _ in $(seq 1 120); do
      [ "$(kubectl get nodes -l node-role=$NG -o name 2>/dev/null | wc -l | tr -d ' ')" = 2 ] && break; sleep 10
    done
    kubectl wait --for=condition=Ready node -l node-role=$NG --timeout=900s
    for i in $(aws ec2 describe-instances --region $REGION --filters Name=tag:eks:nodegroup-name,Values=$NG \
        Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text); do
      aws ec2 create-tags --region $REGION --resources "$i" --tags Key=AlwaysOn,Value=true
    done
    r=a
    for n in $(kubectl get nodes -l node-role=$NG -o name | sort); do
      kubectl label --overwrite "$n" study50/role=$r; r=b
    done
    "$0" status ;;
  down)
    ASG=$(asg)
    scale 0
    aws autoscaling delete-tags --region $REGION \
      --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn" || true
    echo "$NG scaled to 0, AlwaysOn removed from its ASG" ;;
  delete)
    kubectl delete -f "$HERE/../efa/efa-device-plugin.yaml" --ignore-not-found
    sed "s/@AZ@/us-east-2a/" "$HERE/nodegroups.yaml" | eksctl delete nodegroup -f - --approve --wait ;;
  status)
    aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name $NG \
      --query 'nodegroup.[nodegroupName,status,scalingConfig.desiredSize,subnets[0],health.issues[0].code]' --output text 2>&1 | tail -1
    aws ec2 describe-instances --region $REGION --filters Name=tag:eks:nodegroup-name,Values=$NG \
      Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down \
      --query 'Reservations[].Instances[].[InstanceId,State.Name,Placement.AvailabilityZone,Tags[?Key==`AlwaysOn`]|[0].Value]' --output text
    kubectl get nodes -l node-role=$NG -o custom-columns='NODE:.metadata.name,ROLE:.metadata.labels.study50/role,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa,HUGE2M:.status.allocatable.hugepages-2Mi,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory' ;;
  wait-up)
    max=${2:-10}; deadline=$(( $(date +%s) + max * 3600 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      az=$("$0" probe | awk '$2 == "YES" {print $1; exit}') || az=""
      if [ -z "$az" ]; then echo "$(date -u +%H:%M) no capacity"; sleep 300; continue; fi
      echo "$(date -u +%H:%M) capacity in $az"
      cur=$(ngaz)
      if [ -n "$cur" ] && [ "$cur" != "$az" ]; then echo "node group is in $cur: recreating"; "$0" delete; cur=""; fi
      [ -n "$cur" ] || "$0" create "$az"
      if "$0" up; then exit 0; fi
      echo "$(date -u +%H:%M) up failed in $az: back to 0"; scale 0 || true; sleep 300
    done
    echo "no capacity after $max h"; exit 1 ;;
  *) echo "usage: $0 probe | create <az> | up | down | delete | status | wait-up [hours]" >&2; exit 2 ;;
esac
