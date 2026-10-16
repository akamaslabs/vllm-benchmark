#!/bin/bash
# Manual smoke run of study 39, outside Akamas: the workflow's two Executor scripts exactly,
# from this checkout (the same work as the Akamas Smoke study, for when Akamas is not at hand):
#   Apply config  ../k8s/apply_config.sh with the study's baseline values (params below = what
#                 the FileConfigurator renders for the baseline step: vLLM 0.31.0 CPU defaults
#                 except gpu_memory_utilization 0.90)
#   RunTest       ../k8s/run_test.sh with the smoke ramp (default 0 -> 40 req/s over 900 s,
#                 as the Smoke workflow; SMOKE_RATE / SMOKE_RAMP_S override, e.g. with the
#                 probe's proposal)
# The watchdog reads Prometheus through a self-restarting port-forward. Afterwards
# smoke_analyze.py recomputes the Akamas score (best valid 3-minute window) from Prometheus.
# Usage: SMOKE_OUT=<dir> bash smoke/smoke_manual.sh   (workstation: kubectl, python3)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); STUDY=$(dirname "$HERE")
OUT=${SMOKE_OUT:-$HERE/results}; mkdir -p "$OUT"
PORT=${SMOKE_PROM_PORT:-19090}
RATE=${SMOKE_RATE:-40}; RAMP_S=${SMOKE_RAMP_S:-900}
cat > "$OUT/params.env" <<'P'
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=128
KV_CACHE_DTYPE=auto
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
LONG_PREFILL_TOKEN_THRESHOLD=0
STREAM_INTERVAL=1
P
( while true; do kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus "$PORT:9090" >/dev/null 2>&1; sleep 1; done ) &
PF=$!
trap 'kill $PF 2>/dev/null; pkill -f "port-forward svc/kube-prometheus-stack-prometheus $PORT:9090" 2>/dev/null' EXIT
echo "$(date -u +%FT%TZ) SMOKE START" | tee "$OUT/times.txt"
STUDY_DIR=$STUDY PARAMS=$OUT/params.env RENDERED=$OUT/sts.yaml bash "$STUDY/k8s/apply_config.sh" > "$OUT/apply_config.log" 2>&1
RC=$?; echo "$(date -u +%FT%TZ) APPLY END rc=$RC" | tee -a "$OUT/times.txt"
[ $RC -eq 0 ] || { tail -40 "$OUT/apply_config.log"; exit $RC; }
echo "$(date -u +%FT%TZ) RUNTEST START" | tee -a "$OUT/times.txt"
K8S=$STUDY/k8s RT_PROM=http://127.0.0.1:$PORT RT_RATE=$RATE RT_RAMP_S=$RAMP_S RT_FIRST_OK_S=2400 RT_STALL_S=1500 \
  RT_DEADLINE_S=$((RAMP_S + 2400)) bash "$STUDY/k8s/run_test.sh" > "$OUT/run_test.log" 2>&1
RC=$?; echo "$(date -u +%FT%TZ) RUNTEST END rc=$RC" | tee -a "$OUT/times.txt"
grep -E 'watchdog|measured run|Watchdog|error|completed after' "$OUT/run_test.log" | head -20
START=$(sed -n 's/ RUNTEST START$//p' "$OUT/times.txt"); END=$(sed -n 's/ RUNTEST END.*//p' "$OUT/times.txt")
python3 "$HERE/smoke_analyze.py" "http://127.0.0.1:$PORT" "$START" "$END" "$OUT/timeseries.csv" | tee "$OUT/summary.txt"
exit $RC
