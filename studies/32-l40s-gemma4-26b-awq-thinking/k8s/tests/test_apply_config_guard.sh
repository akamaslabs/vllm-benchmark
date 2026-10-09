#!/bin/bash
# apply_config.sh must refuse an invalid params.env BEFORE any kubectl call, and run the
# whole sequence on a valid one (stub kubectl).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); STUDY=$(dirname "$K8S")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
GOOD='GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_DTYPE=auto
PERFORMANCE_MODE=balanced
OPTIMIZATION_LEVEL=2
ENFORCE_EAGER=false
SCHEDULING_POLICY=fcfs
ASYNC_SCHEDULING=true
MAX_CUDAGRAPH_CAPTURE_SIZE=512
BLOCK_SIZE=16'
run_with() {  # $1 params.env content; sets RC
  printf '%s\n' "$1" > "$TMP/params.env"; : > "$TMP/kubectl.log"
  PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUDY_DIR="$STUDY" PARAMS="$TMP/params.env" \
    RENDERED="$TMP/sts.yaml" bash "$K8S/apply_config.sh" > "$TMP/out.txt" 2>&1; RC=$?
}
reject() {
  run_with "$2"
  { [ $RC = 2 ] && [ ! -s "$TMP/kubectl.log" ]; } && ok "$1: exit 2, no kubectl call" \
    || ko "$1: rc=$RC, kubectl calls: $(wc -l < "$TMP/kubectl.log")"
}
with() { printf '%s\n' "$GOOD" | sed "$1"; }
reject "empty always-rendered value" "$(with 's|^GPU_MEMORY_UTILIZATION=.*|GPU_MEMORY_UTILIZATION=|')"
# shellcheck disable=SC2016
reject "leftover token" "$(with 's|^MAX_NUM_SEQS=.*|MAX_NUM_SEQS=${vllm.max_num_seqs}|')"
reject "batched tokens below max_num_seqs" "$(with 's|^MAX_NUM_BATCHED_TOKENS=.*|MAX_NUM_BATCHED_TOKENS=128|')"
run_with "$GOOD"
[ $RC = 0 ] && ok "valid params: exit 0" || ko "valid params: exit $RC ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied vllm$' "$TMP/kubectl.log" && ok "valid params: StatefulSet applied" || ko "valid params: StatefulSet applied"
grep -q 'scale sts vllm --replicas=0' "$TMP/kubectl.log" && ok "valid params: GPU freed first" || ko "valid params: GPU freed first"
# The baseline steps' doNotRenderParameters: empty values, applied with vLLM's defaults.
run_with "$(printf '%s\n' "$GOOD" | sed -E 's/^(MAX_NUM_BATCHED_TOKENS|KV_CACHE_DTYPE|PERFORMANCE_MODE|OPTIMIZATION_LEVEL|SCHEDULING_POLICY|ASYNC_SCHEDULING|MAX_CUDAGRAPH_CAPTURE_SIZE|BLOCK_SIZE)=.*/\1=/')"
grep -q '^BLOCK_SIZE=$' "$TMP/params.env" && ok "not-rendered values: params.env has empty values" || ko "not-rendered values: params.env has empty values"
[ $RC = 0 ] && grep -q '^applied vllm$' "$TMP/kubectl.log" && ok "not-rendered values: applied" || ko "not-rendered values: rc=$RC"
grep -q -- '--block-size\|--kv-cache-dtype\|--max-num-batched-tokens' "$TMP/sts.yaml" && ko "not-rendered values: no flag for them" || ok "not-rendered values: no flag for them"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
