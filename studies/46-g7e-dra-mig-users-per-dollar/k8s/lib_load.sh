# Shared by run_ramp.sh and run_hold.sh (study 46). Sourced, not executed.
#
# run_level LEVEL DURATION TAG: every replica i is driven by its own client pod
# aiperf-client-i at the same closed-loop concurrency LEVEL, all started together, for
# DURATION seconds of measurement after a 20 s warm-up at the same concurrency (the
# warm-up absorbs the first wave of LEVEL simultaneous prompts; Phase 0 saw TTFT p95 31 s
# from a cold first wave). Prints one line per replica:
#   R<i> L=<level> ttft_p95=<ms> itl_p95=<ms> requests=<n> errors=<n> ok=<0|1>
# ok=1 means TTFT p95 <= 1500 ms, ITL p95 <= 300 ms and >= 95 % of requests completed
# (the study SLA, client-side, per replica).

NS=dra-mig
STUDY_DIR=${STUDY_DIR:-/work/vllm-benchmark/studies/46-g7e-dra-mig-users-per-dollar}
K8S=$STUDY_DIR/k8s
TTFT_MAX=1500
ITL_MAX=300
MIN_COMPLETED=0.95

t() { echo "$(date -u +%T) ${0##*/}: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

REPLICAS=$(cat "$K8S/replicas.current" 2>/dev/null || true)
[ -n "$REPLICAS" ] || die "no $K8S/replicas.current: apply_config.sh did not run"

# Restart counts now, to detect a replica that crashed or was replaced during the load.
restarts() { kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.metadata.name}={.status.containerStatuses[0].restartCount}/{.metadata.uid} {end}'; }

# Parser run inside the client pod (python3 is in python:3.12-slim). AIPerf 0.11 CSV: a
# per-request metrics table (header Metric,avg,min,max,p1,...,p95,...) then, after a
# blank row, scalar metrics (Metric,Value) including Request Count / Error Request Count.
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

run_level() {
  local L=$1 D=$2 TAG=$3 i pids=()
  for i in $(seq 0 $((REPLICAS - 1))); do
    kubectl -n $NS exec aiperf-client-$i -- sh -c "rm -rf /work/out/$TAG; mkdir -p /work/out/$TAG; \
      aiperf profile --model qwen3-8b-dra --tokenizer Qwen/Qwen3-8B-FP8 \
        --url http://vllm-$i.vllm-headless.$NS.svc.cluster.local:8000 \
        --endpoint-type chat --streaming --public-dataset sharegpt \
        --warmup-duration 20 --warmup-concurrency $L \
        --concurrency $L --benchmark-duration $D --benchmark-grace-period 30 \
        --goodput 'time_to_first_token:$TTFT_MAX inter_token_latency:$ITL_MAX' \
        --ui simple --output-artifact-dir /work/out/$TAG > /work/out/$TAG.log 2>&1" &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  for i in $(seq 0 $((REPLICAS - 1))); do
    local res
    res=$(kubectl -n $NS exec aiperf-client-$i -- python3 -c "$PARSER" /work/out/$TAG 2>/dev/null || echo "ttft_p95=NA itl_p95=NA requests=0 errors=0")
    local ok
    ok=$(echo "$res" | awk -v tm=$TTFT_MAX -v im=$ITL_MAX -v mc=$MIN_COMPLETED '{
      for (k = 1; k <= NF; k++) { split($k, a, "="); v[a[1]] = a[2] }
      tot = v["requests"] + v["errors"]
      if (v["ttft_p95"] == "NA" || v["itl_p95"] == "NA" || tot == 0) { print 0; exit }
      print ((v["ttft_p95"] + 0 <= tm && v["itl_p95"] + 0 <= im && v["requests"] / tot >= mc) ? 1 : 0) }')
    echo "R$i L=$L $res ok=$ok"
    if [ "$rc" -ne 0 ] || [ "$ok" != 1 ]; then
      echo "--- aiperf-client-$i, level $L: AIPerf log (tail) ---"
      kubectl -n $NS exec aiperf-client-$i -- tail -40 /work/out/$TAG.log 2>/dev/null || true
    fi
  done
}
