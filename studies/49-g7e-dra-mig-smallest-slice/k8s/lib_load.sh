# shellcheck shell=bash
# Load library of study 49 (adapted from study 46's), sourced by run_load.sh.
#
# run_all: every AIPerf client starts at once, closed loop on ShareGPT, for a 20 s warm-up then
# LOAD_SECONDS of measurement:
#   aiperf-client-0 -> vllm-customer-0      at CUSTOMER_USERS (128, the customer's load)
#   aiperf-client-i -> vllm-neighbour-(i-1) at the neighbour load from layout.current
# Prints one line per client:
#   C<i> role=<customer|neighbour> users=<u> ttft_p95=<ms> itl_p95=<ms> requests=<ok> errors=<n>
NS=dra-mig-49
STUDY_DIR=${STUDY_DIR:-/work/study49/49-g7e-dra-mig-smallest-slice}
K8S=$STUDY_DIR/k8s
CUSTOMER_USERS=128
LOAD_SECONDS=${LOAD_SECONDS:-600}
TTFT_MAX=1500
ITL_MAX=300

t() { echo "$(date -u +%T) ${0##*/}: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# shellcheck disable=SC2034  # PROFILE is used by run_load.sh
read -r PROFILE NEIGHBOURS NEIGHBOUR_USERS < "$K8S/layout.current" 2>/dev/null \
  || die "no $K8S/layout.current: apply_config.sh did not run"
CLIENTS=$((1 + NEIGHBOURS))

# Restart counts and uids of every vLLM pod, to detect a crash or a replacement during the load.
restarts() { kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.metadata.name}={.status.containerStatuses[0].restartCount}/{.metadata.uid} {end}'; }

# AIPerf 0.11 CSV parser (study 46): per-request metrics table, then scalar metrics including
# Request Count / Error Request Count. Runs inside the client pod.
PARSER='
import csv, glob, sys
f = sorted(glob.glob(sys.argv[1] + "/**/*aiperf*.csv", recursive=True))
if not f:
    print("ttft_p95=NA itl_p95=NA requests=0 errors=0"); sys.exit(0)
rows = list(csv.reader(open(f[-1])))
hdr, ttft, itl, req, err = None, "NA", "NA", 0, 0
for r in rows:
    if not r: hdr = None; continue
    if r[0] == "Metric": hdr = r; continue
    if hdr and "p95" in hdr:
        if r[0].startswith("Time to First Token"): ttft = r[hdr.index("p95")]
        if r[0].startswith("Inter Token Latency"): itl = r[hdr.index("p95")]
    elif len(r) > 1:
        if r[0] == "Request Count": req = int(float(r[1]))
        if r[0] == "Error Request Count": err = int(float(r[1]))
print(f"ttft_p95={ttft} itl_p95={itl} requests={req} errors={err}")
'

target() { # client index -> "role users url"
  if [ "$1" = 0 ]; then
    echo "customer $CUSTOMER_USERS http://vllm-customer-0.vllm-customer.$NS.svc.cluster.local:8000"
  else
    echo "neighbour $NEIGHBOUR_USERS http://vllm-neighbour-$(($1 - 1)).vllm-neighbour.$NS.svc.cluster.local:8000"
  fi
}

run_all() {
  local i role users url pids=() rc=0 res
  for i in $(seq 0 $((CLIENTS - 1))); do
    read -r role users url < <(target "$i")
    kubectl -n $NS exec aiperf-client-$i -- sh -c "rm -rf /work/out/load; mkdir -p /work/out/load; \
      aiperf profile --model qwen3-8b-dra --tokenizer Qwen/Qwen3-8B-FP8 --url $url \
        --endpoint-type chat --streaming --public-dataset sharegpt \
        --warmup-duration 20 --warmup-concurrency $users \
        --concurrency $users --benchmark-duration $LOAD_SECONDS --benchmark-grace-period 30 \
        --goodput 'time_to_first_token:$TTFT_MAX inter_token_latency:$ITL_MAX' \
        --ui simple --output-artifact-dir /work/out/load > /work/out/load.log 2>&1" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" = 0 ] || t "WARNING: at least one AIPerf client exited non-zero (see its log below)"
  for i in $(seq 0 $((CLIENTS - 1))); do
    read -r role users url < <(target "$i")
    res=$(kubectl -n $NS exec aiperf-client-$i -- python3 -c "$PARSER" /work/out/load 2>/dev/null || echo "ttft_p95=NA itl_p95=NA requests=0 errors=0")
    echo "C$i role=$role users=$users $res"
  done
}
