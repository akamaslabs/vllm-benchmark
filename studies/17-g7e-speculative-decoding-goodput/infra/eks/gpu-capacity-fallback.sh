#!/bin/bash
# GPU capacity fallback for study 17 — added 2026-09-22.
#
# WHY THIS EXISTS
# ---------------
# `llm-serving-g7e` (g7e.4xlarge, 1x RTX PRO 6000 Blackwell 96GB) could not be launched
# for a full day: 56+ consecutive InsufficientInstanceCapacity failures in us-east-2,
# starting 07:27 UTC on 2026-09-22. A second attempt with a different size
# (`llm-serving-g7e-8xl`, g7e.8xlarge) was created at 09:05 and rolled back at 09:38 with
# the same error, so its CloudFormation stack no longer exists. The node group's own
# health message carries AWS's hint verbatim:
#
#   "We currently do not have sufficient g7e.4xlarge capacity in the Availability Zone
#    you requested (us-east-2b). ... You can currently get g7e.4xlarge capacity by not
#    specifying an Availability Zone in your request or choosing us-east-2a."
#
# `llm-serving-g7e` spans BOTH us-east-2a and us-east-2b, so its ASG is free to retry in
# 2a — but the recorded failure is always 2b, and an ASG maintaining AZ balance can stay
# pinned to one zone across retries. Option A below removes that freedom: a node group
# whose subnet list contains ONLY us-east-2a, so every retry lands in the zone AWS says
# has capacity. g7e is offered in 2a and 2b only; 2c does not offer the family at all.
#
# WHY NOT eksctl / cluster.yaml
# ------------------------------
# `cluster.yaml` stays the from-scratch description of the intended cluster. These are
# capacity workarounds against a live cluster, not a change to what the study wants, so
# they live in their own script. If a fallback ever becomes the permanent shape of the
# study, fold it into cluster.yaml then — not before.
#
# WHAT EACH OPTION COSTS THE STUDY
# --------------------------------
#   g7e-2a  ZERO study changes. Same GPU (96GB), same label, same taint — it reuses the
#           existing launch template, whose userdata already writes
#           `node-role=llm-serving-g7e` and the nvidia.com/gpu taint. Both sizes offered
#           carry exactly 1 GPU with 96GB, so the study's single-GPU model is untouched.
#           g7e.2xlarge is deliberately EXCLUDED: 8 vCPU / 32GB cannot even schedule the
#           vLLM pod (it requests cpu: 8 / memory: 32Gi, above an 8-vCPU node's
#           allocatable), and a CPU-starved frontend at concurrency 2048 would cap
#           goodput — contaminating the very metric the study optimizes.
#
#   l40s    MATERIAL study changes, see the header of option B. Different GPU, 48GB
#           instead of 96GB, and a KV-cache budget roughly 5x smaller. Kept at
#           desiredSize 0 so it costs nothing until someone decides to pay that price.
#
# A single L4 is NOT an option and no code path here offers one: Qwen3-32B-FP8 needs
# ~31 GiB of weights and an L4 has 22.35 GiB. The pre-existing `llm-serving-l4` node
# group (g6.12xlarge, 4x L4) would fit the model only across tensor parallelism 4, which
# reintroduces exactly the topology confound the dense-model swap was made to remove —
# and at $4.60/h it is MORE expensive than the g7e node it would replace.
set -euo pipefail

CLUSTER=vllm-bench
REGION=us-east-2
: "${AWS_PROFILE:=lab}"
export AWS_PROFILE

# Everything below is resolved from the live cluster rather than hardcoded, so this
# script keeps working if the node instance role or launch template is ever recreated.
resolve_from_g7e() {
  local q="$1"
  aws eks describe-nodegroup --cluster-name "$CLUSTER" --region "$REGION" \
    --nodegroup-name llm-serving-g7e --query "$q" --output text
}

subnet_in_az() {
  aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=availability-zone,Values=$1" \
              "Name=tag:Name,Values=eksctl-${CLUSTER}-cluster/SubnetPublic*" \
    --query 'Subnets[0].SubnetId' --output text
}

create_g7e_2a() {
  local role lt_id subnet_2a
  role=$(resolve_from_g7e 'nodegroup.nodeRole')
  lt_id=$(resolve_from_g7e 'nodegroup.launchTemplate.id')
  subnet_2a=$(subnet_in_az us-east-2a)
  echo "role=$role launchTemplate=$lt_id subnet(us-east-2a)=$subnet_2a"

  # --launch-template pins the AL2023 NVIDIA AMI (ami-07648720b23706893) AND carries the
  # nodeadm userdata that applies the node-role label and the GPU taint. Because the AMI
  # is custom, EKS cannot inject labels/taints itself — the userdata is what makes them
  # appear, which is precisely why this reuses the existing template verbatim instead of
  # declaring a new one. --labels/--taints below are recorded for console parity only.
  aws eks create-nodegroup \
    --cluster-name "$CLUSTER" --region "$REGION" \
    --nodegroup-name llm-serving-g7e-2a \
    --node-role "$role" \
    --subnets "$subnet_2a" \
    --scaling-config minSize=0,maxSize=1,desiredSize=1 \
    --instance-types g7e.4xlarge g7e.8xlarge \
    --capacity-type ON_DEMAND \
    --launch-template "id=${lt_id},version=1" \
    --labels node-role=llm-serving-g7e \
    --taints 'key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE'

  cat <<'NOTE'

Created. TWO node groups now carry the label node-role=llm-serving-g7e
(llm-serving-g7e and llm-serving-g7e-2a). That is intentional — two independent shots at
scarce capacity — but it is ONLY safe while the study is not running. The moment either
node is Ready, scale the other to 0:

  aws eks update-nodegroup-config --cluster-name vllm-bench --region us-east-2 \
    --nodegroup-name llm-serving-g7e --scaling-config minSize=0,maxSize=1,desiredSize=0

Two Ready nodes sharing one label would put two GPUs behind the study's per-GPU DCGM
queries, and every avg()-based GPU metric would silently average across both.
NOTE
}

create_l40s() {
  local role subnets
  role=$(resolve_from_g7e 'nodegroup.nodeRole')
  # All three AZs: g6e is a mature family offered in 2a/2b/2c, so there is no reason to
  # pin a zone the way option A must.
  subnets=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=tag:Name,Values=eksctl-${CLUSTER}-cluster/SubnetPublic*" \
    --query 'Subnets[].SubnetId' --output text)
  echo "role=$role subnets=$subnets"

  # amiType AL2023_x86_64_NVIDIA (no launch template): L40S is Ada/SM89 and fully
  # supported by the stock EKS NVIDIA AMI, so none of the g7e/Blackwell AMI-pinning
  # gymnastics documented in cluster.yaml applies here. EKS therefore applies the label
  # and taint itself.
  #
  # desiredSize 0 ON PURPOSE. Choosing this node is a study decision, not an infra one:
  #   - 44.7 GiB of VRAM against ~31 GiB of Qwen3-32B-FP8 weights plus ~1.2 GiB of
  #     Qwen3-0.6B drafter leaves roughly 4-8 GiB for KV cache, versus ~54 GiB on the
  #     96GB g7e node. Qwen3-32B's KV is 256 KiB/token at bf16 and 128 KiB/token at fp8,
  #     so the cache holds on the order of 16k-60k tokens instead of several hundred k.
  #   - Consequence: the saturation knee moves down to roughly 64-128 concurrent
  #     requests, the sweep's upper levels become KV-starvation and preemption tests
  #     rather than speculative-decoding tests, and high max_num_seqs x max_model_len
  #     combinations will fail at startup on a KV-fit check instead of running.
  #   - The study's max_num_seqs domain ([16, 1024]) and gpu_memory_utilization domain
  #     ([0.85, 0.95]) were both sized for 96GB and would need re-deriving.
  #   - Config to change before use: the nodeSelector in k8s/01-deployment_template.yaml,
  #     the cluster component's node_role, every telemetry PromQL filtering on the node
  #     label, and the DCGM exporter's target.
  # Scale up only after those edits:
  #   aws eks update-nodegroup-config --cluster-name vllm-bench --region us-east-2 \
  #     --nodegroup-name llm-serving-l40s --scaling-config minSize=0,maxSize=1,desiredSize=1
  aws eks create-nodegroup \
    --cluster-name "$CLUSTER" --region "$REGION" \
    --nodegroup-name llm-serving-l40s \
    --node-role "$role" \
    --subnets $subnets \
    --scaling-config minSize=0,maxSize=1,desiredSize=0 \
    --instance-types g6e.4xlarge \
    --capacity-type ON_DEMAND \
    --ami-type AL2023_x86_64_NVIDIA \
    --disk-size 200 \
    --labels node-role=llm-serving-l40s \
    --taints 'key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE'
}

create_l4_single() {
  local role subnets
  role=$(resolve_from_g7e 'nodegroup.nodeRole')
  subnets=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=tag:Name,Values=eksctl-${CLUSTER}-cluster/SubnetPublic*" \
    --query 'Subnets[].SubnetId' --output text)
  echo "role=$role subnets=$subnets"

  # ONE L4, not the four on the pre-existing llm-serving-l4 (g6.12xlarge). Added
  # 2026-09-22 after a capacity probe (short-lived capacity reservations, created and
  # cancelled immediately) showed the 24GB class is the ONLY GPU capacity us-east-2 had
  # left: g6.4xlarge available in all three AZs, g5.2xlarge in 2a/2b, g5.12xlarge in 2b,
  # and nothing at all for g7e, g6e or g6.12xlarge in any zone.
  #
  # This node CANNOT serve Qwen3-32B-FP8 — 22.35 GiB of VRAM against ~31 GiB of weights.
  # Using it means swapping the study's target model for one that fits, e.g.
  # Qwen/Qwen3-8B-FP8 or Qwen/Qwen3-14B-FP8, both of which keep the Qwen3 151936-token
  # vocabulary the Qwen3-0.6B drafter needs. That is a study decision, not an infra one,
  # so nothing here edits the study. What it does preserve is the study's actual shape:
  # one dense model on one GPU with speculative decoding. An L4's ~300 GB/s of bandwidth
  # is far below the RTX PRO 6000's, which puts decode deeper into the
  # memory-bandwidth-bound regime where speculation has the most to give.
  #
  # Distinct label on purpose: node-role=llm-serving-l4-single never collides with the
  # llm-serving-g7e label the two g7e node groups share, so no avg()-based GPU metric can
  # silently average across a g7e node and this one if g7e capacity ever returns.
  aws eks create-nodegroup \
    --cluster-name "$CLUSTER" --region "$REGION" \
    --nodegroup-name llm-serving-l4-single \
    --node-role "$role" \
    --subnets $subnets \
    --scaling-config minSize=0,maxSize=1,desiredSize=1 \
    --instance-types g6.4xlarge \
    --capacity-type ON_DEMAND \
    --ami-type AL2023_x86_64_NVIDIA \
    --disk-size 200 \
    --labels node-role=llm-serving-l4-single \
    --taints 'key=nvidia.com/gpu,value=present,effect=NO_SCHEDULE'
}

probe_capacity() {
  # Free-in-practice capacity probe: a capacity reservation is refused outright with
  # InsufficientInstanceCapacity when the pool is empty, and when it succeeds it is
  # cancelled within the same second, so billing is negligible. This is the only way to
  # ask "is there capacity right now" without waiting out an ASG's 4-minute retry cycle.
  local end; end=$(date -u -v+1H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+1 hour' '+%Y-%m-%dT%H:%M:%SZ')
  printf '%-16s %-5s %-5s %-5s\n' TYPE 2a 2b 2c
  for it in "${@:-g7e.4xlarge g6e.4xlarge g6.4xlarge g5.2xlarge}"; do
    printf '%-16s ' "$it"
    for az in us-east-2a us-east-2b us-east-2c; do
      out=$(aws ec2 create-capacity-reservation --region "$REGION" \
        --instance-type "$it" --instance-platform Linux/UNIX --availability-zone "$az" \
        --instance-count 1 --instance-match-criteria targeted \
        --end-date-type limited --end-date "$end" \
        --query 'CapacityReservation.CapacityReservationId' --output text 2>&1)
      if [[ "$out" == cr-* ]]; then
        printf '%-5s ' YES
        aws ec2 cancel-capacity-reservation --region "$REGION" --capacity-reservation-id "$out" >/dev/null 2>&1 \
          || printf '\n!! reservation %s NOT cancelled, cancel it by hand !!\n' "$out"
      else
        echo "$out" | grep -q InsufficientInstanceCapacity && printf '%-5s ' no || printf '%-5s ' n/a
      fi
    done
    echo
  done
  echo "still-active reservations (must be empty):"
  aws ec2 describe-capacity-reservations --region "$REGION" --filters Name=state,Values=active \
    --query 'CapacityReservations[].[CapacityReservationId,InstanceType,AvailabilityZone]' --output text
}

status() {
  echo "=== GPU node groups"
  for ng in $(aws eks list-nodegroups --cluster-name "$CLUSTER" --region "$REGION" \
                --query 'nodegroups[?contains(@,`llm-serving`)]' --output text); do
    printf '  %-22s ' "$ng"
    aws eks describe-nodegroup --cluster-name "$CLUSTER" --region "$REGION" \
      --nodegroup-name "$ng" \
      --query 'nodegroup.[instanceTypes[0],status,scalingConfig.desiredSize]' \
      --output text | tr '\t' ' '
  done
  echo "=== GPU instances actually running"
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=instance-state-name,Values=pending,running" \
              "Name=instance-type,Values=g7e.*,g6e.*,g6.*,g5.*" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Placement.AvailabilityZone,State.Name]' \
    --output text
}

case "${1:-}" in
  g7e-2a) create_g7e_2a ;;
  l40s)   create_l40s   ;;
  l4-single) create_l4_single ;;
  probe)  shift; probe_capacity "$@" ;;
  status) status        ;;
  *) echo "usage: $0 {g7e-2a|l40s|l4-single|probe [type...]|status}" >&2; exit 1 ;;
esac
