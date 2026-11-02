#!/bin/bash
# Study 45: the six GPU instances the optimizer can choose, their node groups (EKS cluster
# vllm-bench, all in us-east-2b, min 0 / max 2) and on-demand prices. Sourced by
# apply_config.sh and scale_down.sh; also usable by hand:  bash nodes.sh status
#
# instance      node group (= node-role label)   GPU         vCPU / RAM    $/h us-east-2 (Pricing API, 2026-10-29)
# g6.xlarge     llm-serving-l4-pd-xl             L4 24 GB    4 / 16 GiB    0.8048
# g6.2xlarge    llm-serving-g6-2xl               L4 24 GB    8 / 32 GiB    0.9776
# g5.xlarge     llm-serving-g5-xl                A10G 24 GB  4 / 16 GiB    1.006
# g5.2xlarge    llm-serving-a10g-pd              A10G 24 GB  8 / 32 GiB    1.212
# g6e.xlarge    llm-serving-g6e-xl               L40S 48 GB  4 / 32 GiB    1.861
# g6e.2xlarge   llm-serving-l40s-pd              L40S 48 GB  8 / 64 GiB    2.24208
# Three of the groups were created for study 44 and are reused (same labels).
REGION=us-east-2
CLUSTER=vllm-bench
ALL_GROUPS="llm-serving-l4-pd-xl llm-serving-g6-2xl llm-serving-g5-xl llm-serving-a10g-pd llm-serving-g6e-xl llm-serving-l40s-pd"

group_of() {
  case "$1" in
    g6.xlarge) echo llm-serving-l4-pd-xl ;;   g6.2xlarge) echo llm-serving-g6-2xl ;;
    g5.xlarge) echo llm-serving-g5-xl ;;      g5.2xlarge) echo llm-serving-a10g-pd ;;
    g6e.xlarge) echo llm-serving-g6e-xl ;;    g6e.2xlarge) echo llm-serving-l40s-pd ;;
    *) return 1 ;;
  esac
}
price_of() {
  case "$1" in
    g6.xlarge) echo 0.8048 ;;  g6.2xlarge) echo 0.9776 ;;
    g5.xlarge) echo 1.006 ;;   g5.2xlarge) echo 1.212 ;;
    g6e.xlarge) echo 1.861 ;;  g6e.2xlarge) echo 2.24208 ;;
    *) return 1 ;;
  esac
}
# Pod resources per instance: CPU_REQ MEM_REQ MEM_LIM SHM. With kv_buffer_device=cpu a P/D
# pod keeps a host copy of its KV region (study 44: ~13 GiB on a 24 GB GPU, ~33 GiB on the
# L40S at gmu 0.9), so the 16 GiB / 32 GiB nodes are tight on purpose: an OOM is a result.
resources_of() {
  case "$1" in
    g6.xlarge|g5.xlarge) echo "3 11Gi 13Gi 2Gi" ;;
    g6e.xlarge) echo "3 22Gi 27Gi 4Gi" ;;
    g6.2xlarge|g5.2xlarge) echo "6 20Gi 27Gi 4Gi" ;;
    g6e.2xlarge) echo "6 40Gi 56Gi 8Gi" ;;
    *) return 1 ;;
  esac
}

scale_group() {  # group desired — retries while another update of the same group is in progress
  local g=$1 n=$2 out
  for _ in $(seq 1 20); do
    out=$(aws eks update-nodegroup-config --region $REGION --cluster-name $CLUSTER --nodegroup-name "$g" \
      --scaling-config minSize=0,maxSize=2,desiredSize="$n" 2>&1) && { echo "scale $g -> $n"; return 0; }
    echo "$out" | grep -q ResourceInUseException || { echo "scale $g -> $n FAILED: $out"; return 1; }
    sleep 15
  done
  echo "scale $g -> $n: still busy after 5 min"; return 1
}

desired_of() {
  aws eks describe-nodegroup --region $REGION --cluster-name $CLUSTER --nodegroup-name "$1" \
    --query 'nodegroup.scalingConfig.desiredSize' --output text 2>/dev/null
}

if [[ "${BASH_SOURCE[0]}" == "$0" && "${1:-}" == status ]]; then
  for g in $ALL_GROUPS; do echo "$g desired=$(desired_of $g)"; done
  kubectl get nodes -L node-role,node.kubernetes.io/instance-type --no-headers 2>/dev/null | grep -E "$(echo $ALL_GROUPS | tr ' ' '|')" || true
fi
