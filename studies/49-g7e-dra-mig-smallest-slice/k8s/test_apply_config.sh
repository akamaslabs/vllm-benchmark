#!/bin/bash
# Local tests of apply_config.sh's parameter handling (RENDER_ONLY=1: no cluster).
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/k8s"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
render() { # params.env lines as arguments; output in $TMP/out, rc in $RC
  printf '%s\n' "$@" > "$TMP/k8s/params.env"
  STUDY_DIR="$TMP" RENDER_ONLY=1 bash "$HERE/apply_config.sh" > "$TMP/out" 2>&1; RC=$?
}
FULL=(GPU_MEMORY_UTILIZATION=0.92 MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=4096 KV_CACHE_DTYPE=fp8
      PERFORMANCE_MODE=interactivity OPTIMIZATION_LEVEL=2 ASYNC_SCHEDULING=false
      MAX_CUDAGRAPH_CAPTURE_SIZE=256 BLOCK_SIZE=32 WATERMARK=0.01 SCHEDULER_RESERVE_FULL_ISL=true
      STREAM_INTERVAL=4)
DEFAULTS=(GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS= KV_CACHE_DTYPE=
      PERFORMANCE_MODE= OPTIMIZATION_LEVEL= ASYNC_SCHEDULING= MAX_CUDAGRAPH_CAPTURE_SIZE= BLOCK_SIZE=
      WATERMARK= SCHEDULER_RESERVE_FULL_ISL= STREAM_INTERVAL=)

render MIG_PROFILE=1g.24gb "${FULL[@]}"
[ $RC = 0 ] && grep -q 'layout=1g.24gb claim=mig-1g24gb neighbours=3 neighbour_users=64' "$TMP/out" \
  && grep -qx -- '--kv-cache-dtype=fp8' "$TMP/out" && grep -qx -- '--no-async-scheduling' "$TMP/out" \
  && grep -qx -- '--scheduler-reserve-full-isl' "$TMP/out" && grep -qx -- '--block-size=32' "$TMP/out" \
  && ok || ko "1g layout and flags"

render MIG_PROFILE=2g.48gb "${FULL[@]}"
[ $RC = 0 ] && grep -q 'layout=2g.48gb claim=mig-2g48gb neighbours=1 neighbour_users=128' "$TMP/out" && ok || ko "2g layout"

render MIG_PROFILE=none "${DEFAULTS[@]}"
[ $RC = 0 ] && grep -q 'layout=none claim=gpu-full neighbours=0 neighbour_users=0' "$TMP/out" \
  && [ "$(grep -c -- '^--' "$TMP/out")" = 2 ] && ok || ko "none layout, defaults = 2 flags"

render MIG_PROFILE=4g.96gb "${FULL[@]}"
[ $RC = 2 ] && ok || ko "bad profile -> rc 2 (rc $RC)"

render MIG_PROFILE=1g.24gb "${FULL[@]/WATERMARK=0.01/WATERMARK=2.19}"
[ $RC = 0 ] && grep -qx -- '--watermark=0' "$TMP/out" && grep -q WARNING "$TMP/out" && ok || ko "watermark without exponent -> 0"

render MIG_PROFILE=1g.24gb "${FULL[@]/GPU_MEMORY_UTILIZATION=0.92/GPU_MEMORY_UTILIZATION=9.2e-01}"
[ $RC = 0 ] && grep -qx -- '--gpu-memory-utilization=0.920000' "$TMP/out" && ok || ko "gmu scientific notation"

render MIG_PROFILE=1g.24gb "${FULL[@]/MAX_NUM_BATCHED_TOKENS=4096/MAX_NUM_BATCHED_TOKENS=128}"
[ $RC = 2 ] && ok || ko "mnbt < mns -> rc 2"

render MIG_PROFILE=1g.24gb "${FULL[@]/STREAM_INTERVAL=4/}"
[ $RC = 2 ] && ok || ko "missing line -> rc 2"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
