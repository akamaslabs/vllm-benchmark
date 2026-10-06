#!/bin/bash
# Manual smoke run of study 30 (2026-10-06), outside Akamas, while the Akamas 4.1 instance is
# being installed. Runs the workflow's two Executor scripts exactly, from this checkout:
#   Apply config  ../k8s/apply_config.sh with the study's baseline values (params below = what
#                 the FileConfigurator renders for the baseline step)
#   RunTest       ../k8s/run_test.sh with the smoke workflow's ramp and guards
# The watchdog reads Prometheus through a self-restarting port-forward. Afterwards
# smoke_analyze.py recomputes the Akamas score (best valid 3-minute window) from Prometheus.
# Usage: SMOKE_OUT=<dir> bash smoke/smoke_manual.sh   (workstation: kubectl, python3)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); STUDY=$(dirname "$HERE")
OUT=${SMOKE_OUT:-$HERE/results}; mkdir -p "$OUT"
PORT=${SMOKE_PROM_PORT:-19090}
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
echo "$(date -u +%FT%TZ) SMOKE START" | tee "$OUT/times.txt"
STUDY_DIR=$STUDY PARAMS=$OUT/params.env RENDERED=$OUT/sts.yaml bash "$STUDY/k8s/apply_config.sh" > "$OUT/apply_config.log" 2>&1
RC=$?; echo "$(date -u +%FT%TZ) APPLY END rc=$RC" | tee -a "$OUT/times.txt"
[ $RC -eq 0 ] || { tail -40 "$OUT/apply_config.log"; exit $RC; }
echo "$(date -u +%FT%TZ) RUNTEST START" | tee -a "$OUT/times.txt"
K8S=$STUDY/k8s RT_PROM=http://127.0.0.1:$PORT RT_RATE=40 RT_RAMP_S=900 RT_FIRST_OK_S=2400 RT_STALL_S=1500 RT_DEADLINE_S=3300 \
  bash "$STUDY/k8s/run_test.sh" > "$OUT/run_test.log" 2>&1
RC=$?; echo "$(date -u +%FT%TZ) RUNTEST END rc=$RC" | tee -a "$OUT/times.txt"
grep -E 'watchdog|measured run|Watchdog|error|completed after' "$OUT/run_test.log" | head -20
exit $RC
