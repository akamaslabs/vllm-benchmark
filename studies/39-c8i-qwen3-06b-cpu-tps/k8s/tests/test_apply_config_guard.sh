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
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=128
KV_CACHE_DTYPE=auto
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
LONG_PREFILL_TOKEN_THRESHOLD=0
STREAM_INTERVAL=1'
run_with() {  # $1 params.env content; sets RC
  printf '%s\n' "$1" > "$TMP/params.env"; : > "$TMP/kubectl.log"
  PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUDY_DIR="$STUDY" PARAMS="$TMP/params.env" \
    RENDERED="$TMP/sts.yaml" AC_POLL_S=0 bash "$K8S/apply_config.sh" > "$TMP/out.txt" 2>&1; RC=$?
}
reject() {
  run_with "$2"
  { [ $RC = 2 ] && [ ! -s "$TMP/kubectl.log" ]; } && ok "$1: exit 2, no kubectl call" \
    || ko "$1: rc=$RC, kubectl calls: $(wc -l < "$TMP/kubectl.log")"
}
with() { printf '%s\n' "$GOOD" | sed "$1"; }
reject "empty value" "$(with 's|^BLOCK_SIZE=.*|BLOCK_SIZE=|')"
# shellcheck disable=SC2016
reject "leftover token" "$(with 's|^MAX_NUM_SEQS=.*|MAX_NUM_SEQS=${vllm.max_num_seqs}|')"
reject "batched tokens below max_num_seqs" "$(with 's|^MAX_NUM_BATCHED_TOKENS=.*|MAX_NUM_BATCHED_TOKENS=64|')"
reject "chunked prefill off below max_model_len" "$(with 's|^ENABLE_CHUNKED_PREFILL=.*|ENABLE_CHUNKED_PREFILL=false|')"
reject "block size 48" "$(with 's|^BLOCK_SIZE=.*|BLOCK_SIZE=48|')"
run_with "$GOOD"
[ $RC = 0 ] && ok "valid params: exit 0" || ko "valid params: exit $RC ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied vllm$' "$TMP/kubectl.log" && ok "valid params: StatefulSet applied" || ko "valid params: StatefulSet applied"
grep -q 'scale sts vllm --replicas=0' "$TMP/kubectl.log" && ok "valid params: old replica stopped first" || ko "valid params: old replica stopped first"
grep -q 'node-role=llm-serving-cpu' "$TMP/kubectl.log" && ok "valid params: looks for the CPU node" || ko "valid params: looks for the CPU node"
grep -q -- '-n llm-cpu' "$TMP/kubectl.log" && ok "valid params: namespace llm-cpu" || ko "valid params: namespace llm-cpu"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
