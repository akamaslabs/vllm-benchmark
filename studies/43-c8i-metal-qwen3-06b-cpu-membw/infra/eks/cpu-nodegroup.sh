#!/bin/bash
# shellcheck disable=SC2016  # literal $Latest and in-pod $(cat ...) are intentional
# Creates (or reconciles) this study's CPU node group on the EXISTING vllm-bench cluster:
# llm-serving-c8i, one c8i.12xlarge (Xeon 6 Granite Rapids, AMX, 48 vCPU / 24 cores,
# 96 GiB), AL2023, label node-role=llm-serving-cpu, taint workload=cpu-inference:NoSchedule.
#
# Kubelet: cpuManagerPolicy static (+ full-pcpus-only), topologyManagerPolicy
# single-numa-node, 2 reserved CPUs (EKS kubeReserved 190m + systemReserved 1810m = core 0, eks/nodeconfig.yaml). On an AL2023 managed
# node group the only way to set these is a nodeadm NodeConfig in the user data of a custom
# launch template, so this script owns a launch template `llm-serving-c8i` (no ImageId:
# EKS keeps choosing the AMI from --ami-type and appends its own NodeConfig to ours).
# Settings a launch template forbids on create-nodegroup move into it: the disk
# (BlockDeviceMappings, instead of --disk-size). The instance type stays on the node group.
#
# Capacity: SPOT by default (decided 2026-10-16): c8i.12xlarge spot was 0.68-0.81 $/h in
# us-east-2 against 2.25 $/h on demand; spot placement score 3/10 in 2b and 2c, 1/10 in 2a.
# An interruption kills the running trial (a failed experiment; a failed baseline puts the
# study in ERROR) and the ASG brings a replacement node up, which apply_config.sh waits for
# (UNSCHED_S). CAPACITY_TYPE=ON_DEMAND switches back, but a managed node group's capacity type
# is fixed at creation: delete the node group first (--down, then aws eks delete-nodegroup).
#
# Subnets: the PRIVATE us-east-2b and us-east-2c subnets (egress via the cluster's NAT, no
# public IP); two AZs give spot two capacity pools. The vLLM pod uses hostPath caches only,
# so no EBS volume ties it to an AZ. Override with SUBNET_TAGS="tag1 tag2".
#
# Created at desiredSize 0. Idempotent: re-running re-applies the taint and, if
# nodeconfig.yaml changed, creates a new launch template version and rolls the node group
# to it (instant at desiredSize 0; at 1 it replaces the node).
#
# Usage: AWS_PROFILE=lab ./cpu-nodegroup.sh [--up] [--always-on] [--down]
#   --up         sets desiredSize 1, waits for the node to be Ready, then verifies the live
#                kubelet config (configz) and the SMT sibling / NUMA layout behind
#                the 2 reserved CPUs
#   --always-on  tags the node group's ASG AlwaysOn=true (PropagateAtLaunch) and its running
#                instance, so the lab's 17:00 UTC stop Lambda leaves the node running
#   --down       sets desiredSize 0 and removes the ASG's AlwaysOn tag
#
# Study 43 defaults (2026-10-27): node group llm-serving-c8i-metal, ONE on-demand
# c8i.metal-48xl (Xeon 6975P-C, 96 cores / 192 threads, 1 socket, 3 NUMA nodes, 384 GiB,
# 12 x DDR5-7200) in us-east-2c only: bare metal exposes the uncore IMC counters membw.py
# reads. Spot had no capacity (placement score 1/10) and on demand had none in 2b that day
# (~9 $/h on demand). The verify_node checks below were written for the 12xlarge (cpu0 siblings
# 0,24, 1 NUMA node): on this node they print WARNINGs that are expected (siblings 0,96, 3 NUMA).
set -euo pipefail
CLUSTER=vllm-bench
REGION=us-east-2
NG=${NG:-llm-serving-c8i-metal}
LT=${LT:-$NG}
INSTANCE_TYPE=${INSTANCE_TYPE:-c8i.metal-48xl}
NODE_ROLE=llm-serving-cpu
SUBNET_TAGS=${SUBNET_TAGS:-eksctl-${CLUSTER}-cluster/SubnetPrivateUSEAST2C}
CAPACITY_TYPE=${CAPACITY_TYPE:-ON_DEMAND}
DISK_GB=${DISK_GB:-100}
SIBLING_CHECK_IMAGE=${SIBLING_CHECK_IMAGE:-public.ecr.aws/docker/library/busybox:1.36}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

UP=0; ALWAYS_ON=0; DOWN=0
for a in "$@"; do
  case $a in
    --up) UP=1 ;;
    --always-on) ALWAYS_ON=1 ;;
    --down) DOWN=1 ;;
    *) echo "usage: $0 [--up] [--always-on] [--down]" >&2; exit 2 ;;
  esac
done
if [ $DOWN = 1 ] && { [ $UP = 1 ] || [ $ALWAYS_ON = 1 ]; }; then
  echo "--down cannot be combined with --up / --always-on" >&2; exit 2
fi

asg_name() {
  aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" \
    --query 'nodegroup.resources.autoScalingGroups[0].name' --output text
}

# Pin every kubectl call to this cluster: the README runs --up before provision.sh (the only
# other update-kubeconfig), so the current context may point at another cluster. A private
# kubeconfig, so the user's own kubeconfig and current context are left untouched.
if [ $UP = 1 ]; then
  KC=$(mktemp); trap 'rm -f "$KC"' EXIT
  aws eks update-kubeconfig --name $CLUSTER --region $REGION --kubeconfig "$KC" --alias $CLUSTER >/dev/null
  kubectl() { command kubectl --kubeconfig "$KC" --context $CLUSTER "$@"; }
fi

if [ $DOWN = 1 ]; then
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 --query 'update.id' --output text
  ASG=$(asg_name)
  aws autoscaling delete-tags --region $REGION \
    --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn" || true
  echo "$NG scaled to 0; AlwaysOn removed from $ASG (a forgotten scale-up is stopped at night again)"
  exit 0
fi

# ---------- launch template: user data = MIME multipart with the NodeConfig part ----------
render_user_data() {
  printf 'MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary="BOUNDARY"\n\n'
  printf -- '--BOUNDARY\nContent-Type: application/node.eks.aws\n\n'
  # Comments are valid YAML; strip them anyway to keep the user data small and readable.
  grep -v '^[[:space:]]*#' "$SCRIPT_DIR/nodeconfig.yaml"
  printf -- '\n--BOUNDARY--\n'
}
USER_DATA_B64=$(render_user_data | base64 | tr -d '\n')

LT_DATA=$(cat <<JSON
{
  "BlockDeviceMappings": [
    {"DeviceName": "/dev/xvda",
     "Ebs": {"VolumeSize": $DISK_GB, "VolumeType": "gp3", "Encrypted": true, "DeleteOnTermination": true}}
  ],
  "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 1, "HttpEndpoint": "enabled"},
  "TagSpecifications": [
    {"ResourceType": "instance",
     "Tags": [{"Key": "Project", "Value": "vllm-bench"}, {"Key": "Study", "Value": "43-c8i-metal-qwen3-06b-cpu-membw"},
              {"Key": "Name", "Value": "$NG"}]},
    {"ResourceType": "volume",
     "Tags": [{"Key": "Project", "Value": "vllm-bench"}, {"Key": "Study", "Value": "43-c8i-metal-qwen3-06b-cpu-membw"}]}
  ],
  "UserData": "$USER_DATA_B64"
}
JSON
)
# HttpPutResponseHopLimit 1 = system-2b's disablePodIMDS: pods cannot read the node's IAM
# credentials (a custom launch template otherwise defaults to 2).

if aws ec2 describe-launch-templates --region $REGION --launch-template-names "$LT" >/dev/null 2>&1; then
  LATEST_UD=$(aws ec2 describe-launch-template-versions --region $REGION --launch-template-name "$LT" \
    --versions '$Latest' --query 'LaunchTemplateVersions[0].LaunchTemplateData.UserData' --output text)
  LATEST_BD=$(aws ec2 describe-launch-template-versions --region $REGION --launch-template-name "$LT" \
    --versions '$Latest' --query 'LaunchTemplateVersions[0].LaunchTemplateData.BlockDeviceMappings[0].Ebs.VolumeSize' --output text)
  if [ "$LATEST_UD" = "$USER_DATA_B64" ] && [ "$LATEST_BD" = "$DISK_GB" ]; then
    echo "launch template $LT up to date"
  else
    aws ec2 create-launch-template-version --region $REGION --launch-template-name "$LT" \
      --version-description "nodeconfig $(date -u +%Y-%m-%dT%H:%MZ)" \
      --launch-template-data "$LT_DATA" --query 'LaunchTemplateVersion.VersionNumber' --output text
    echo "launch template $LT: new version (nodeconfig.yaml or DISK_GB changed)"
  fi
else
  aws ec2 create-launch-template --region $REGION --launch-template-name "$LT" \
    --version-description "nodeconfig initial" \
    --tag-specifications 'ResourceType=launch-template,Tags=[{Key=Project,Value=vllm-bench},{Key=Study,Value=43-c8i-metal-qwen3-06b-cpu-membw}]' \
    --launch-template-data "$LT_DATA" --query 'LaunchTemplate.LaunchTemplateId' --output text
fi
LT_VERSION=$(aws ec2 describe-launch-templates --region $REGION --launch-template-names "$LT" \
  --query 'LaunchTemplates[0].LatestVersionNumber' --output text)

# ---------- node group ----------
if aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" >/dev/null 2>&1; then
  echo "node group $NG exists — re-applying the taint"
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" \
    --taints 'addOrUpdateTaints=[{key=workload,value=cpu-inference,effect=NO_SCHEDULE}]' \
    --query 'update.id' --output text || true   # "no changes" is not an error
  # A scaling/version update right after the taint update is refused while UPDATING
  # (study 30, 2026-10-07): wait for ACTIVE first.
  aws eks wait nodegroup-active --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG"
  NG_LT_VERSION=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" \
    --query 'nodegroup.launchTemplate.version' --output text)
  if [ "$NG_LT_VERSION" != "$LT_VERSION" ]; then
    echo "rolling $NG from launch template version $NG_LT_VERSION to $LT_VERSION (replaces a running node)"
    aws eks update-nodegroup-version --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" \
      --launch-template "name=$LT,version=$LT_VERSION" --query 'update.id' --output text
    aws eks wait nodegroup-active --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG"
  fi
else
  # Same instance role as the cluster's other node groups.
  ROLE=$(aws eks describe-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name akamas \
    --query 'nodegroup.nodeRole' --output text)
  SUBNETS=()
  for tag in $SUBNET_TAGS; do
    sn=$(aws ec2 describe-subnets --region $REGION --filters "Name=tag:Name,Values=$tag" \
      --query 'Subnets[0].SubnetId' --output text)
    if [ -z "$sn" ] || [ "$sn" = None ]; then
      echo "no subnet tagged Name=$tag" >&2; exit 1
    fi
    SUBNETS+=("$sn")
  done
  aws eks create-nodegroup --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" \
    --node-role "$ROLE" --subnets "${SUBNETS[@]}" \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 \
    --instance-types "$INSTANCE_TYPE" --capacity-type "$CAPACITY_TYPE" \
    --ami-type AL2023_x86_64_STANDARD \
    --launch-template "name=$LT,version=$LT_VERSION" \
    --labels node-role=$NODE_ROLE \
    --taints key=workload,value=cpu-inference,effect=NO_SCHEDULE \
    --tags Project=vllm-bench,Study=43-c8i-metal-qwen3-06b-cpu-membw \
    --query 'nodegroup.[nodegroupName,status,releaseVersion]' --output text
  aws eks wait nodegroup-active --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG"
fi

# ---------- scale up + verify ----------
verify_node() {
  local node; node=$(kubectl get nodes -l node-role=$NODE_ROLE -o jsonpath='{.items[0].metadata.name}')
  echo "--- kubelet config on $node (configz)"
  local cfg; cfg=$(kubectl get --raw "/api/v1/nodes/$node/proxy/configz" 2>/dev/null) || cfg=""
  if [ -z "$cfg" ]; then
    echo "WARNING: configz not readable; judge by allocatable cpu below (46 = NodeConfig applied, ~47810m = not)" >&2
  elif command -v jq >/dev/null 2>&1; then
    echo "$cfg" | jq '.kubeletconfig | {cpuManagerPolicy, cpuManagerPolicyOptions, topologyManagerPolicy, kubeReserved, systemReserved}'
    if [ "$(echo "$cfg" | jq -r '.kubeletconfig.cpuManagerPolicy')" != static ]; then
      echo "ERROR: cpuManagerPolicy is not static — the NodeConfig part was not applied" >&2; return 1
    fi
  else
    echo "$cfg" | tr ',' '\n' | grep -E 'cpuManagerPolicy|topologyManagerPolicy|reservedSystemCPUs|full-pcpus-only'
  fi
  echo "--- allocatable cpu: $(kubectl get node "$node" -o jsonpath='{.status.allocatable.cpu}') (expected 46)"
  echo "--- SMT siblings / NUMA (the 2 reserved CPUs are core 0 only if cpu0 siblings = 0,24; expected 1 NUMA node)"
  kubectl delete pod -n default cpu-topology-check --ignore-not-found >/dev/null
  kubectl run cpu-topology-check -n default --image="$SIBLING_CHECK_IMAGE" --restart=Never \
    --overrides="{\"spec\":{\"nodeName\":\"$node\",\"tolerations\":[{\"operator\":\"Exists\"}]}}" \
    --command -- sh -c 'echo "cpu0 siblings: $(cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list)"; echo "cpu1 siblings: $(cat /sys/devices/system/cpu/cpu1/topology/thread_siblings_list)"; echo "online cpus: $(cat /sys/devices/system/cpu/online)"; echo "NUMA nodes online: $(cat /sys/devices/system/node/online)"' >/dev/null
  kubectl wait -n default --for=jsonpath='{.status.phase}'=Succeeded pod/cpu-topology-check --timeout=180s >/dev/null || true
  local out; out=$(kubectl logs -n default cpu-topology-check 2>/dev/null) || out="(check pod produced no logs)"
  echo "$out"
  kubectl delete pod -n default cpu-topology-check --ignore-not-found >/dev/null
  if ! echo "$out" | grep -q '^cpu0 siblings: 0,24$'; then
    echo "WARNING: cpu0's sibling set is not 0,24 — the reserved CPUs are not one whole core: revisit systemReserved in eks/nodeconfig.yaml" >&2
  fi
  if ! echo "$out" | grep -q '^NUMA nodes online: 0$'; then
    echo "WARNING: more than one NUMA node — a 44-CPU pod fails single-numa-node admission (infra/README.md)" >&2
  fi
}

if [ $UP = 1 ]; then
  # Tag the ASG BEFORE scaling up, so PropagateAtLaunch tags the instance at launch: a failed
  # wait below (slow join, bad nodeadm user data) must not leave a billing node untagged for
  # the 17:00 UTC stop Lambda. The --always-on block below still tags a running instance.
  if [ $ALWAYS_ON = 1 ]; then
    aws autoscaling create-or-update-tags --region $REGION \
      --tags "ResourceId=$(asg_name),ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
  fi
  aws eks update-nodegroup-config --cluster-name $CLUSTER --region $REGION --nodegroup-name "$NG" \
    --scaling-config minSize=0,maxSize=1,desiredSize=1 --query 'update.id' --output text
  echo "waiting for a Ready node with node-role=$NODE_ROLE (an InsufficientInstanceCapacity shows in"
  echo "  aws eks describe-nodegroup ... --query nodegroup.health.issues)"
  for _ in $(seq 1 90); do
    kubectl get nodes -l node-role=$NODE_ROLE -o name 2>/dev/null | grep -q . && break
    sleep 10
  done
  kubectl wait --for=condition=Ready node -l node-role=$NODE_ROLE --timeout=900s
fi

if [ $ALWAYS_ON = 1 ]; then
  ASG=$(asg_name)
  aws autoscaling create-or-update-tags --region $REGION \
    --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"
  for i in $(aws ec2 describe-instances --region $REGION --filters "Name=tag:eks:nodegroup-name,Values=$NG" \
      Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text); do
    aws ec2 create-tags --region $REGION --resources "$i" --tags Key=AlwaysOn,Value=true
  done
  echo "AlwaysOn=true on $ASG and its running instance(s)"
fi

# Last, after the AlwaysOn tags: a failed check must not leave a running node untagged.
if [ $UP = 1 ]; then
  verify_node
fi
