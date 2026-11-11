#!/bin/bash
# Smoke run in the toolbox before the study (README "Smoke checks"): for each layout, apply it at
# the presets' vLLM flags (apply_config.sh proves pod args, cache_config_info, ranks, expert
# parallelism and the NCCL transport), then one short ramp level. Stops at the first non-zero exit.
#   bash smoke.sh [layout ...]      default: all seven (single replicas dp2 dp2ep tp2 tp2ep pp2)
# Environment passes through to apply_config.sh (NCCL_ENV for NCCL socket tuning,
# DP_MODE=internal) and to run_ramp.sh (LEVELS, LEVEL_SECONDS, STEADY_FROM; defaults below).
K=${STUDY_DIR:-/work/study51/51-g7e-tcp-moe-parallelism-users-ramp}/k8s
HERE=$(cd "$(dirname "$0")" && pwd)
for L in ${@:-single replicas dp2 dp2ep tp2 tp2ep pp2}; do
  case "$L" in
    single)   P="1 1 1 1 false" ;;
    replicas) P="2 1 1 1 false" ;;
    dp2)      P="1 1 2 1 false" ;;
    dp2ep)    P="1 1 2 1 true" ;;
    tp2)      P="1 2 1 1 false" ;;
    tp2ep)    P="1 2 1 1 true" ;;
    pp2)      P="1 1 1 2 false" ;;
    *) echo "unknown layout $L" >&2; exit 2 ;;
  esac
  set -- $P
  cat > "$K"/params.env <<P
REPLICAS=$1
TENSOR_PARALLEL_SIZE=$2
DATA_PARALLEL_SIZE=$3
PIPELINE_PARALLEL_SIZE=$4
ENABLE_EXPERT_PARALLEL=$5
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=512
MAX_NUM_BATCHED_TOKENS=8192
KV_CACHE_DTYPE=auto
MOE_BACKEND=auto
P
  echo "===== SMOKE layout=$L apply $(date -u +%T)"
  bash "$HERE"/apply_config.sh; rc=$?
  echo "===== SMOKE layout=$L apply rc=$rc $(date -u +%T)"
  [ $rc = 0 ] || exit $rc
  echo "===== SMOKE layout=$L load $(date -u +%T)"
  LEVELS=${LEVELS:-32} LEVEL_SECONDS=${LEVEL_SECONDS:-180} STEADY_FROM=${STEADY_FROM:-60} bash "$HERE"/run_ramp.sh; rc=$?
  echo "===== SMOKE layout=$L load rc=$rc $(date -u +%T)"
  [ $rc = 0 ] || exit $rc
done
echo "===== SMOKE all done"
