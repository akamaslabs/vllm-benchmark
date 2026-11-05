#!/bin/bash
# Local tests of apply_config.sh's parameter section (RENDER_ONLY=1: no cluster needed).
# Usage: bash k8s/test_apply_config.sh   → "N passed, M failed", exit 1 on any failure.
HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
VARS="KV_OFFLOADING_SIZE KV_CACHE_DTYPE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE"

# render NAME "VAR=value ..." → sets OUT and RC
render() {
  local tmp; tmp=$(mktemp -d); mkdir -p "$tmp/k8s"
  for v in $VARS; do echo "$v=" >> "$tmp/k8s/params.env"; done
  for kv in $2; do sed -i.bak "s|^${kv%%=*}=.*|$kv|" "$tmp/k8s/params.env"; done
  OUT=$(RENDER_ONLY=1 STUDY_DIR="$tmp" bash "$HERE/apply_config.sh" 2>&1); RC=$?
  rm -rf "$tmp"
}
ok()   { PASS=$((PASS + 1)); }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL: $1"; echo "$OUT" | sed 's/^/    /'; }
has()  { echo "$OUT" | grep -qx -- "$2" && ok || bad "$1: expected line '$2'"; }
hasnt(){ echo "$OUT" | grep -q -- "$2" && bad "$1: unexpected '$2'" || ok; }
rc()   { [ "$RC" = "$2" ] && ok || bad "$1: exit $RC, expected $2"; }

render "all empty" ""
rc "all empty" 0; hasnt "all empty" "--kv-offloading"; hasnt "all empty" "--gpu-memory-utilization"; has "all empty" "shm=2 mem=20"

render "offload 0" "KV_OFFLOADING_SIZE=0"
rc "offload 0" 0; hasnt "offload 0" "--kv-offloading"; has "offload 0" "shm=2 mem=20"

render "offload 24" "KV_OFFLOADING_SIZE=24"
rc "offload 24" 0; has "offload 24" "--kv-offloading-size=24"; has "offload 24" "--kv-offloading-backend=native"; has "offload 24" "shm=25 mem=44"

render "gmu sci" "GPU_MEMORY_UTILIZATION=9.2E-1"
rc "gmu sci" 0; has "gmu sci" "--gpu-memory-utilization=0.920000"

render "gmu no exponent" "GPU_MEMORY_UTILIZATION=2.19"
rc "gmu no exponent" 2

render "mnbt < mns" "MAX_NUM_SEQS=128 MAX_NUM_BATCHED_TOKENS=64"
rc "mnbt < mns" 2

render "offload too big" "KV_OFFLOADING_SIZE=40"
rc "offload too big" 2

render "block 24" "BLOCK_SIZE=24"
rc "block 24" 2

render "full set" "KV_OFFLOADING_SIZE=12 KV_CACHE_DTYPE=fp8 GPU_MEMORY_UTILIZATION=0.85 MAX_NUM_SEQS=96 MAX_NUM_BATCHED_TOKENS=4096 BLOCK_SIZE=64"
rc "full set" 0; has "full set" "--kv-cache-dtype=fp8"; has "full set" "--max-num-seqs=96"; has "full set" "--max-num-batched-tokens=4096"; has "full set" "--block-size=64"; has "full set" "--gpu-memory-utilization=0.85"; has "full set" "shm=13 mem=32"

render "offload float" "KV_OFFLOADING_SIZE=12.0"
rc "offload float" 0; has "offload float" "--kv-offloading-size=12"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
