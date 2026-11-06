# shellcheck shell=bash
# Slice roles of study 49, sourced by apply_config.sh (and test_roles.sh).
# Under DRA, dcgm-exporter's per-MIG series have GPU_I_ID but no pod label (study 46). These
# helpers map each vLLM pod to its GPU instance id and render the slice-roles exporter file
# (03-slice-roles.yaml) that the Akamas telemetry joins to DCGM.

# parse_gpu_instance_id: stdin = `nvidia-smi -q` from inside a pod; prints the MIG GPU instance
# id, or nothing for a whole GPU ("N/A" or no such line).
parse_gpu_instance_id() {
  awk -F: '/GPU Instance ID/ { gsub(/[ \t]/, "", $2); if ($2 ~ /^[0-9]+$/) { print $2; exit } }'
}

# gpu_instance_id NAMESPACE POD: the GPU instance id of the pod's slice (empty for a whole GPU).
gpu_instance_id() {
  kubectl -n "$1" exec "$2" -- nvidia-smi -q 2>/dev/null | parse_gpu_instance_id
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
