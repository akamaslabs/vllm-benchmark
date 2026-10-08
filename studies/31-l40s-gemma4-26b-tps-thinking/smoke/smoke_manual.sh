#!/bin/bash
# Manual smoke run of study 31 (thinking mode), outside Akamas, before the study is created.
# One vLLM start with the study's baseline values, then two AIPerf Jobs:
#   1. Length run  closed loop, SMOKE_LEN_CONC concurrent requests, SMOKE_LEN_COUNT requests
#                  (default 32 / 320, ~5-15 min): how long Gemma 4 reasons and answers on ShareGPT
#                  prompts once max_tokens is gone. The Job log ends with the "LENGTHS" lines
#                  (reasoning / answer tokens, TTFT and time to the first answer token, outputs
#                  that may have hit max_model_len): they confirm or change --max-model-len and give
#                  the expected knee. 32 requests stay inside the bf16 KV cache (~64k tokens) if
#                  a request holds up to ~2000 tokens. The first run also builds the ShareGPT cache
#                  and its copy without max_tokens.
#   2. Smoke ramp  ../k8s/run_test.sh with a steep ramp, 0 -> SMOKE_RATE req/s over SMOKE_RAMP_S s
#                  (1800 s) and wide first-trial guards: the watchdog ends it past the knee.
#                  smoke_analyze.py then gives the knee and the score; R and D of the study
#                  follow from them (README "Load"). SMOKE_RATE, unless given, comes from the
#                  length run: 2.5 x 2900 / mean output tokens per request (2900 = generated
#                  tokens/s of study 30's baseline at its knee), so that the knee falls in the
#                  middle of the ramp even if thinking halves the tokens/s at the knee; 6 when
#                  the length run is skipped.
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
cat > "$OUT/params.env" <<'P'
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_DTYPE=auto
PERFORMANCE_MODE=balanced
OPTIMIZATION_LEVEL=2
ENFORCE_EAGER=false
SCHEDULING_POLICY=fcfs
ASYNC_SCHEDULING=true
MAX_CUDAGRAPH_CAPTURE_SIZE=512
BLOCK_SIZE=16
LINEAR_BACKEND=auto
SPEC_METHOD=none
SPEC_TOKENS=0
P
( while true; do kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus "$PORT:9090" >/dev/null 2>&1; sleep 1; done ) &
PF=$!
trap 'kill $PF 2>/dev/null; pkill -f "port-forward svc/kube-prometheus-stack-prometheus $PORT:9090" 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) SMOKE START (phases: $PHASES)" | tee -a "$OUT/times.txt"
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
grep -E '^nomax:|^LENGTHS' "$OUT/length_run.log"
kubectl -n $NS delete job -l app=$JOB --ignore-not-found --wait=true
[ $LRC -eq 0 ] || { echo "length run failed or timed out: see $OUT/length_run.log"; tail -30 "$OUT/length_run.log"; exit 1; }
fi

# --- 2. Smoke ramp -----------------------------------------------------------------------
[[ " $PHASES " == *" ramp "* ]] || exit 0
if [ -z "$RATE" ]; then
  MEAN=$(awk '$2 == "output_sequence_length" && match($0, /mean= *[0-9.]+/) { s = substr($0, RSTART, RLENGTH); sub(/mean= */, "", s); print s }' \
    "$OUT/length_run.log" 2>/dev/null | head -1)
  if [ -n "$MEAN" ]; then
    RATE=$(awk -v m="$MEAN" 'BEGIN { r = 2.5 * 2900 / m; if (r < 0.5) r = 0.5; printf "%.1f", r }')
    echo "ramp rate from the length run: mean output $MEAN tokens -> 0 -> $RATE req/s" | tee -a "$OUT/times.txt"
  else
    RATE=6; echo "no length run output in $OUT: ramp rate $RATE req/s" | tee -a "$OUT/times.txt"
  fi
fi
echo "$(date -u +%FT%TZ) RUNTEST START (0 -> $RATE req/s over $RAMP_S s)" | tee -a "$OUT/times.txt"
K8S=$STUDY/k8s RT_PROM=http://127.0.0.1:$PORT RT_RATE=$RATE RT_RAMP_S=$RAMP_S RT_FIRST_OK_S=2400 RT_STALL_S=1500 \
  RT_DEADLINE_S=$((RAMP_S + 1500)) bash "$STUDY/k8s/run_test.sh" > "$OUT/run_test.log" 2>&1
RC=$?; echo "$(date -u +%FT%TZ) RUNTEST END rc=$RC" | tee -a "$OUT/times.txt"
grep -E 'watchdog|measured run|Watchdog|error|completed after' "$OUT/run_test.log" | head -20
exit $RC
