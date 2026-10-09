#!/bin/bash
# Manual smoke run of study 32, outside Akamas, before the study is created (study 31's, on the
# synthetic multi-turn load). One vLLM start with SMOKE_CONFIG's values, then up to two AIPerf
# Jobs:
#   1. Length run  closed loop, SMOKE_LEN_CONC concurrent conversations, SMOKE_LEN_COUNT requests
#                  (default 32 / 320): the "LENGTHS" lines of the Job log give the input per
#                  request (input_sequence_length: does the profile hit the customer's median
#                  ~3000 / p90 ~6000 tokens with this checkpoint's answers?), the reasoning and
#                  answer tokens, TTFT and time to the first answer token, and the requests near
#                  max_model_len 96000.
#   2. Smoke ramp  ../k8s/run_test.sh with a steep ramp, 0 -> SMOKE_RATE req/s over SMOKE_RAMP_S s
#                  (1800 s) and wide first-trial guards: the watchdog ends it past the knee.
#                  smoke_analyze.py then gives the knee and the score. SMOKE_RATE, unless given,
#                  comes from the length run and the KV cache vLLM reports (Little's law): the
#                  requests that fit, min(max_num_seqs, KV tokens / (mean input + mean output)),
#                  over a request's life, mean output x 80 ms (study 31's ITL at its knee), times
#                  2 so the knee falls mid-ramp.
# SMOKE_CONFIG: "compose" (default; the customer's compose values = the study baseline) or
# "large" (the "kv fp8 large batch" preset). The study's R must let both reach their knee
# (README "Runbook"): run the compose config (length + ramp), then the large one (ramp only:
# SMOKE_CONFIG=large SMOKE_PHASES=ramp, its rate from the compose length run via SMOKE_LEN_LOG).
# SMOKE_PHASES="length ramp" (default) runs both; "length" or "ramp" runs one, and
# SMOKE_SKIP_APPLY=1 reuses the running vLLM (no restart) for a second invocation.
# The watchdog reads Prometheus through a self-restarting port-forward.
# Usage: SMOKE_OUT=<dir> bash smoke/smoke_manual.sh   (workstation: kubectl, python3)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); STUDY=$(dirname "$HERE")
OUT=${SMOKE_OUT:-$HERE/results}; mkdir -p "$OUT"
PORT=${SMOKE_PROM_PORT:-19090}
NS=llm-l40s JOB=aiperf-l40s
LEN_CONC=${SMOKE_LEN_CONC:-32} LEN_COUNT=${SMOKE_LEN_COUNT:-320} LEN_TIMEOUT_S=${SMOKE_LEN_TIMEOUT_S:-3600}
RATE=${SMOKE_RATE:-} RAMP_S=${SMOKE_RAMP_S:-1800}
PHASES=${SMOKE_PHASES:-length ramp} SKIP_APPLY=${SMOKE_SKIP_APPLY:-0}
CONFIG=${SMOKE_CONFIG:-compose} LEN_LOG=${SMOKE_LEN_LOG:-$OUT/length_run.log}
# compose = the study baseline: empty lines = not rendered = vLLM's defaults (its
# doNotRenderParameters); large = the "kv fp8 large batch" preset, every value written.
case $CONFIG in
  compose) GMU=0.90 SEQS=64 MNBT="" KV="" MODE="" OPT="" POL="" ASYNC="" CG="" BS="" LB="" SM="" ST="" ;;
  large)   GMU=0.94 SEQS=512 MNBT=8192 KV=fp8 MODE=throughput OPT=2 POL=fcfs ASYNC=true CG=512 BS=16 LB=auto SM=none ST=0 ;;
  *) echo "SMOKE_CONFIG '$CONFIG' is not compose or large" >&2; exit 2 ;;
esac
cat > "$OUT/params.env" <<P
GPU_MEMORY_UTILIZATION=$GMU
MAX_NUM_SEQS=$SEQS
MAX_NUM_BATCHED_TOKENS=$MNBT
KV_CACHE_DTYPE=$KV
PERFORMANCE_MODE=$MODE
OPTIMIZATION_LEVEL=$OPT
ENFORCE_EAGER=false
SCHEDULING_POLICY=$POL
ASYNC_SCHEDULING=$ASYNC
MAX_CUDAGRAPH_CAPTURE_SIZE=$CG
BLOCK_SIZE=$BS
LINEAR_BACKEND=$LB
SPEC_METHOD=$SM
SPEC_TOKENS=$ST
P
( while true; do kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus "$PORT:9090" >/dev/null 2>&1; sleep 1; done ) &
PF=$!
trap 'kill $PF 2>/dev/null; pkill -f "port-forward svc/kube-prometheus-stack-prometheus $PORT:9090" 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) SMOKE START (config: $CONFIG, phases: $PHASES)" | tee -a "$OUT/times.txt"
if [ "$SKIP_APPLY" != 1 ]; then
  STUDY_DIR=$STUDY PARAMS=$OUT/params.env RENDERED=$OUT/sts.yaml bash "$STUDY/k8s/apply_config.sh" > "$OUT/apply_config.log" 2>&1
  RC=$?; echo "$(date -u +%FT%TZ) APPLY END rc=$RC" | tee -a "$OUT/times.txt"
  [ $RC -eq 0 ] || { tail -40 "$OUT/apply_config.log"; exit $RC; }
fi

# --- 1. Length run ---------------------------------------------------------------------
if [[ " $PHASES " == *" length "* ]]; then
echo "$(date -u +%FT%TZ) LENGTH RUN START ($LEN_CONC concurrent, $LEN_COUNT requests)" | tee -a "$OUT/times.txt"
kubectl -n $NS delete job -l app=$JOB --ignore-not-found --wait=true
bash "$STUDY/k8s/render_job.sh" --closed "$LEN_CONC" "$LEN_COUNT" "$OUT/length_job.yaml" || exit 2
kubectl apply -f "$OUT/length_job.yaml" || exit 3
T0=$SECONDS; LRC=1
while [ $((SECONDS - T0)) -lt "$LEN_TIMEOUT_S" ]; do
  S=$(kubectl -n $NS get job $JOB -o jsonpath='{.status.succeeded}' 2>/dev/null)
  F=$(kubectl -n $NS get job $JOB -o jsonpath='{.status.failed}' 2>/dev/null)
  [ "${S:-0}" -ge 1 ] && { LRC=0; break; }
  [ "${F:-0}" -ge 1 ] && break
  sleep 20
done
kubectl -n $NS logs "job/$JOB" -c aiperf --tail=-1 > "$OUT/length_run.log" 2>&1
echo "$(date -u +%FT%TZ) LENGTH RUN END rc=$LRC after $((SECONDS - T0)) s" | tee -a "$OUT/times.txt"
grep -E '^LENGTHS' "$OUT/length_run.log"
kubectl -n $NS delete job -l app=$JOB --ignore-not-found --wait=true
[ $LRC -eq 0 ] || { echo "length run failed or timed out: see $OUT/length_run.log"; tail -30 "$OUT/length_run.log"; exit 1; }
fi

# --- 2. Smoke ramp -----------------------------------------------------------------------
[[ " $PHASES " == *" ramp "* ]] || exit 0
if [ -z "$RATE" ]; then
  mean_of() { awk -v k="$1" '$2 == k && match($0, /mean= *[0-9.]+/) { s = substr($0, RSTART, RLENGTH); sub(/mean= */, "", s); print s; exit }' "$LEN_LOG" 2>/dev/null; }
  ISL=$(mean_of input_sequence_length); OSL=$(mean_of output_sequence_length)
  KVTOK=$(grep -o -m1 'GPU KV cache size: [0-9,]* tokens' "$OUT/apply_config.log" 2>/dev/null | tr -dc '0-9')
  if [ -n "$ISL" ] && [ -n "$OSL" ] && [ -n "$KVTOK" ]; then
    RATE=$(awk -v i="$ISL" -v o="$OSL" -v kv="$KVTOK" -v n="$SEQS" 'BEGIN {
      fit = kv / (i + o); if (fit > n) fit = n; k = fit / (o * 0.08); r = 2 * k; if (r < 0.2) r = 0.2; printf "%.2f", r }')
    echo "ramp rate: KV $KVTOK tokens, mean input $ISL + output $OSL, $SEQS seqs -> 0 -> $RATE req/s" | tee -a "$OUT/times.txt"
  else
    echo "no length run ($LEN_LOG) or KV size ($OUT/apply_config.log): set SMOKE_RATE" >&2; exit 2
  fi
fi
echo "$(date -u +%FT%TZ) RUNTEST START (0 -> $RATE req/s over $RAMP_S s)" | tee -a "$OUT/times.txt"
K8S=$STUDY/k8s RT_PROM=http://127.0.0.1:$PORT RT_RATE=$RATE RT_RAMP_S=$RAMP_S RT_FIRST_OK_S=2400 RT_STALL_S=1500 \
  RT_DEADLINE_S=$((RAMP_S + 1500)) bash "$STUDY/k8s/run_test.sh" > "$OUT/run_test.log" 2>&1
RC=$?; echo "$(date -u +%FT%TZ) RUNTEST END rc=$RC" | tee -a "$OUT/times.txt"
grep -E 'watchdog|measured run|Watchdog|error|completed after' "$OUT/run_test.log" | head -20
exit $RC
