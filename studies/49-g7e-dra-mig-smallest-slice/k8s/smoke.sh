#!/bin/bash
# Run in the toolbox before the study (README "smoke checks" 1-4): for each layout, apply with
# vLLM defaults, print the slice-roles mapping, then a 120 s load. Stops at the first non-zero
# exit. 2026-11-07: all three layouts passed (see README).
# Study 49 smoke checks 1-3: apply + short load for each layout.
K=${STUDY_DIR:-/work/study49/49-g7e-dra-mig-smallest-slice}/k8s
for L in ${LAYOUTS:-1g.24gb 2g.48gb none}; do
  cat > "$K"/params.env <<P
MIG_PROFILE=$L
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_DTYPE=auto
PERFORMANCE_MODE=
OPTIMIZATION_LEVEL=
ASYNC_SCHEDULING=
MAX_CUDAGRAPH_CAPTURE_SIZE=
BLOCK_SIZE=
WATERMARK=
SCHEDULER_RESERVE_FULL_ISL=
STREAM_INTERVAL=
P
  echo "===== SMOKE layout=$L apply $(date -u +%T)"
  bash "$K"/apply_config.sh; rc=$?
  echo "===== SMOKE layout=$L apply rc=$rc $(date -u +%T)"
  [ $rc = 0 ] || exit $rc
  cat "$K"/layout.current 2>/dev/null
  kubectl -n dra-mig-49 get cm slice-roles -o jsonpath='{.data.metrics}'; echo
  echo "===== SMOKE layout=$L load $(date -u +%T)"
  LOAD_SECONDS=120 bash "$K"/run_load.sh; rc=$?
  echo "===== SMOKE layout=$L load rc=$rc $(date -u +%T)"
  [ $rc = 0 ] || exit $rc
done
echo "===== SMOKE all done"
