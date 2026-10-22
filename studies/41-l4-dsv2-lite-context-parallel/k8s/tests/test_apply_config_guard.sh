#!/bin/bash
# apply_config.sh must refuse an invalid params.env BEFORE any kubectl call, refuse a node in
# the wrong AZ or GPUs held by another pod BEFORE touching the StatefulSet, and run the whole
# sequence on a valid one (stub kubectl).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); STUDY=$(dirname "$K8S")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
GOOD='TENSOR_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=1
DECODE_CONTEXT_PARALLEL_SIZE=4
PREFILL_CONTEXT_PARALLEL_SIZE=1
ENABLE_EXPERT_PARALLEL=false
KV_CACHE_DTYPE=auto
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.90
ENABLE_PREFIX_CACHING='
run_with() {  # $1 params.env content, then extra VAR=value; sets RC
  local p=$1; shift
  printf '%s\n' "$p" > "$TMP/params.env"; : > "$TMP/kubectl.log"
  env PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUDY_DIR="$STUDY" PARAMS="$TMP/params.env" \
    RENDERED="$TMP/sts.yaml" AC_POLL_S=0 "$@" bash "$K8S/apply_config.sh" > "$TMP/out.txt" 2>&1; RC=$?
}
reject() {
  run_with "$2"
  { [ $RC = 2 ] && [ ! -s "$TMP/kubectl.log" ]; } && ok "$1: exit 2, no kubectl call" \
    || ko "$1: rc=$RC, kubectl calls: $(wc -l < "$TMP/kubectl.log")"
}
with() { printf '%s\n' "$GOOD" | sed "$1"; }
reject "missing key" "$(with '/^ENABLE_PREFIX_CACHING=/d')"
# shellcheck disable=SC2016
reject "leftover token" "$(with 's|^MAX_NUM_SEQS=.*|MAX_NUM_SEQS=${vllm.max_num_seqs}|')"
reject "batched tokens below max_num_seqs" "$(with 's|^MAX_NUM_BATCHED_TOKENS=.*|MAX_NUM_BATCHED_TOKENS=64|')"
reject "not 4 GPUs" "$(with 's|^DATA_PARALLEL_SIZE=.*|DATA_PARALLEL_SIZE=2|')"
reject "PCP without EP" "$(with 's|^TENSOR_PARALLEL_SIZE=.*|TENSOR_PARALLEL_SIZE=2|; s|^PREFILL_CONTEXT_PARALLEL_SIZE=.*|PREFILL_CONTEXT_PARALLEL_SIZE=2|; s|^DECODE_CONTEXT_PARALLEL_SIZE=.*|DECODE_CONTEXT_PARALLEL_SIZE=1|')"
run_with "$GOOD"
[ $RC = 0 ] && ok "valid params: exit 0" || ko "valid params: exit $RC ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied vllm-dsv2$' "$TMP/kubectl.log" && ok "valid params: StatefulSet applied" || ko "valid params: StatefulSet applied"
grep -q 'scale sts vllm-dsv2 --replicas=0' "$TMP/kubectl.log" && ok "valid params: old replica stopped first" || ko "valid params: old replica stopped first"
grep -q 'node-role=llm-serving-l4' "$TMP/kubectl.log" && ok "valid params: looks for the L4 node" || ko "valid params: looks for the L4 node"
grep -q -- '-n llm-serving' "$TMP/kubectl.log" && ok "valid params: namespace llm-serving" || ko "valid params: namespace llm-serving"
grep -q 'get pods -A --field-selector spec.nodeName=node-l4' "$TMP/kubectl.log" && ok "valid params: GPU-busy guard ran" || ko "valid params: GPU-busy guard ran"
grep -qE 'app=vllm( |$)' "$TMP/kubectl.log" && ko "never touches the other study's app=vllm" || ok "never touches the other study's app=vllm"
grep -q -- '--decode-context-parallel-size=4' "$TMP/sts.yaml" && ok "valid params: DCP4 rendered" || ko "valid params: DCP4 rendered"
# Wrong AZ: exit 3 before the StatefulSet is touched.
run_with "$GOOD" STUB_ZONE=us-east-2b
{ [ $RC = 3 ] && ! grep -q 'scale sts\|apply -f' "$TMP/kubectl.log"; } && ok "node in us-east-2b: exit 3, nothing touched" || ko "node in us-east-2b: rc=$RC"
grep -q 'model PVC is in us-east-2a' "$TMP/out.txt" && ok "wrong AZ: says why" || ko "wrong AZ: says why"
# GPU-busy guard: another pod holds the L4s.
run_with "$GOOD" STUB_GPU_PODS=$'llm-serving/dsv2-smoke:4\nkube-system/aws-node-x:'
{ [ $RC = 5 ] && ! grep -q 'scale sts\|apply -f' "$TMP/kubectl.log"; } && ok "GPUs busy: exit 5, nothing touched" || ko "GPUs busy: rc=$RC"
grep -q 'llm-serving/dsv2-smoke' "$TMP/out.txt" && ok "GPUs busy: names the pod" || ko "GPUs busy: names the pod"
# Our own previous replica on the node does not trip the guard.
run_with "$GOOD" STUB_GPU_PODS='llm-serving/vllm-dsv2-0:4'
[ $RC = 0 ] && ok "own replica on the node: exit 0" || ko "own replica on the node: rc=$RC"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
