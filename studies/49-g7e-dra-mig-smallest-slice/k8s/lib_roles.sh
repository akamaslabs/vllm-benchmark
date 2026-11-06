# shellcheck shell=bash
# Slice roles of study 49, sourced by apply_config.sh (and test_roles.sh).
# Under DRA, dcgm-exporter's per-MIG series have GPU_I_ID but no pod label (study 46). These
# helpers map each vLLM pod to its GPU instance id and render the slice-roles exporter file
# (03-slice-roles.yaml) that the Akamas telemetry joins to DCGM.

# parse_gpu_instance_id: stdin = `nvidia-smi -q` from inside a pod; prints the MIG GPU instance
# id, or nothing for a whole GPU ("N/A" or no such line). Reads all of stdin (no early exit): a
# writer still streaming would get SIGPIPE, and under pipefail apply_config.sh would die with
# rc 141 and no message (final review).
parse_gpu_instance_id() {
  awk -F: '/GPU Instance ID/ && id == "" { v = $2; gsub(/[ \t]/, "", v); if (v ~ /^[0-9]+$/) id = v } END { if (id != "") print id }'
}

# gpu_instance_id NAMESPACE POD: the GPU instance id of the pod's slice (empty for a whole GPU).
# A failed exec gives an empty id (roles_metrics then fails -> apply_config.sh exit 7), never a
# silent abort under set -e.
gpu_instance_id() {
  { kubectl -n "$1" exec "$2" -- nvidia-smi -q 2>/dev/null || true; } | parse_gpu_instance_id
}

# roles_metrics LAYOUT CUSTOMER_ID [NEIGHBOUR_ID...]: the exporter file on stdout.
#   always:   study49_slice_role{role="gpu"} 1   (whole-GPU series, no GPU_I_ID: gpu0)
#   none:     study49_slice_role{role="customer"} 1   (no MIG: the whole GPU is the customer's)
#   MIG:      one line per slice with GPU_I_ID, role customer or neighbour
# rc 1 (reason on stderr) if a MIG layout has an empty or duplicate id, or the wrong number of
# neighbour ids (1g.24gb: 3, 2g.48gb: 1): a wrong mapping would score the wrong slice.
roles_metrics() {
  local layout=$1 cust=$2; shift 2
  local want
  case $layout in
    none) want=0 ;;
    2g.48gb) want=1 ;;
    1g.24gb) want=3 ;;
    *) echo "roles_metrics: unknown layout '$layout'" >&2; return 1 ;;
  esac
  [ "$#" = "$want" ] || { echo "roles_metrics: $layout needs $want neighbour id(s), got $#" >&2; return 1; }
  echo 'study49_slice_role{role="gpu"} 1'
  if [ "$layout" = none ]; then
    echo 'study49_slice_role{role="customer"} 1'
    return 0
  fi
  local id seen=" "
  for id in "$cust" "$@"; do
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "roles_metrics: empty or non-numeric GPU instance id '$id'" >&2; return 1; }
    case $seen in *" $id "*) echo "roles_metrics: duplicate GPU instance id $id" >&2; return 1 ;; esac
    seen="$seen$id "
  done
  echo "study49_slice_role{GPU_I_ID=\"$cust\",role=\"customer\"} 1"
  for id in "$@"; do echo "study49_slice_role{GPU_I_ID=\"$id\",role=\"neighbour\"} 1"; done
}

# parse_prom_value: stdin = a Prometheus /api/v1/query JSON answer; prints the first sample's
# value, or nothing (empty result, error, unparsable).
parse_prom_value() {
  python3 -c '
import json, sys
try:
    r = json.load(sys.stdin)["data"]["result"]
    if r: print(r[0]["value"][1])
except Exception:
    pass'
}

# customer_fb_mib: the customer slice's framebuffer (MiB) through the same DCGM x slice-roles join
# the Akamas goal uses; empty if the join matches nothing. PROM_URL overridable for tests.
customer_fb_mib() {
  local j q
  j='* on(GPU_I_ID) group_left max by(GPU_I_ID)(study49_slice_role{role="customer"})'
  q="sum((DCGM_FI_DEV_FB_USED{modelName=~\".*RTX PRO 6000.*\"} $j) + (DCGM_FI_DEV_FB_FREE{modelName=~\".*RTX PRO 6000.*\"} $j))"
  { curl -s --max-time 10 "${PROM_URL:-http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090}/api/v1/query" \
      --data-urlencode "query=$q" || true; } | parse_prom_value
}
