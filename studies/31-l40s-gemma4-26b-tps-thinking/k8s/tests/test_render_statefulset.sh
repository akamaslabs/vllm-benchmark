#!/bin/bash
# Tests for ../render_statefulset.sh. Run: bash k8s/tests/test_render_statefulset.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
R=$K8S/render_statefulset.sh; T=$K8S/01-statefulset_template.yaml
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
params() {  # the baseline (vLLM 0.29.0 defaults on a 48 GB GPU), then KEY=VALUE overrides; KEY=- deletes
  cat > "$TMP/params.env" <<'P'
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
P
  local kv
  for kv in "$@"; do
    if [ "${kv#*=}" = - ]; then sed -i.bak "/^${kv%%=*}=/d" "$TMP/params.env"
    elif grep -q "^${kv%%=*}=" "$TMP/params.env"; then sed -i.bak "s|^${kv%%=*}=.*|$kv|" "$TMP/params.env"
    else echo "$kv" >> "$TMP/params.env"; fi
  done
}
C='.spec.template.spec.containers[0]'
args() { yq "$C.args[]" "$TMP/out.yaml"; }
render() { rm -f "$TMP/out.yaml"; bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" >/dev/null 2>&1; }

# 1. The baseline renders every flag, booleans as --x / --no-x, no backend flag.
params
if render; then
  [ "$(yq '.kind' "$TMP/out.yaml")" = StatefulSet ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.spec.replicas' "$TMP/out.yaml")" = 1 ] && ok "one replica" || ko "one replica"
  for a in --gpu-memory-utilization=0.92 --max-num-seqs=256 --max-num-batched-tokens=2048 --kv-cache-dtype=auto \
           --performance-mode=balanced --optimization-level=2 --no-enforce-eager --scheduling-policy=fcfs \
           --async-scheduling --max-cudagraph-capture-size=512 --block-size=16 \
           --model=RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic --served-model-name=gemma4-26b-l40s-think \
           --language-model-only --no-enable-prefix-caching --max-model-len=16384 --reasoning-parser=gemma4; do
    args | grep -qx -- "$a" && ok "arg $a" || ko "arg $a"
  done
  # Thinking mode: the chat-template kwargs flag is followed by enable_thinking true.
  args | grep -A1 -x -- --default-chat-template-kwargs | tail -1 | grep -qx '{"enable_thinking": true}' \
    && ok "enable_thinking true" || ko "enable_thinking true"
  args | grep -q -- '--attention-backend\|--linear-backend' && ko "no backend flag by default" || ok "no backend flag by default"
  args | grep -q -- '=true\|=false' && ko "no =true/=false" || ok "no =true/=false"
  [ "$(yq "$C.resources.requests.\"nvidia.com/gpu\"" "$TMP/out.yaml")" = 1 ] && ok "one GPU" || ko "one GPU"
  [ "$(yq '.spec.template.spec.nodeSelector."node-role"' "$TMP/out.yaml")" = llm-serving-l40s-1xl ] && ok "node selector" || ko "node selector"
  grep -q '@[A-Z_]*@' "$TMP/out.yaml" && ko "no token left" || ok "no token left"
else
  ko "baseline renders"
fi

# 2. Booleans flipped and the optional backend lines.
params ENFORCE_EAGER=true ASYNC_SCHEDULING=false LINEAR_BACKEND=triton ATTENTION_BACKEND=TRITON_ATTN
if render; then
  args | grep -qx -- --enforce-eager && ok "--enforce-eager" || ko "--enforce-eager"
  args | grep -qx -- --no-async-scheduling && ok "--no-async-scheduling" || ko "--no-async-scheduling"
  args | grep -qx -- --linear-backend=triton && ok "linear backend rendered" || ko "linear backend rendered"
  args | grep -qx -- --attention-backend=TRITON_ATTN && ok "attention backend rendered" || ko "attention backend rendered"
else
  ko "flipped booleans render"
fi
params LINEAR_BACKEND=auto ATTENTION_BACKEND=auto
render && ! args | grep -q -- '--attention-backend\|--linear-backend' && ok "auto backends: no flag" || ko "auto backends: no flag"

# 2b. Speculative decoding: mtp/K renders the three flags; none/0 renders none.
params SPEC_METHOD=mtp SPEC_TOKENS=3
if render; then
  for a in --spec-method=mtp --spec-model=google/gemma-4-26B-A4B-it-assistant --spec-tokens=3; do
    args | grep -qx -- "$a" && ok "arg $a" || ko "arg $a"
  done
else
  ko "mtp renders"
fi
params SPEC_METHOD=none SPEC_TOKENS=0
render && ! args | grep -q -- '--spec-' && ok "none/0: no --spec- flag" || ko "none/0: no --spec- flag"

# 3. Invalid inputs exit 2 and write nothing.
expect_reject() {
  local name=$1; shift
  params "$@"; render; local rc=$?
  { [ $rc != 0 ] && [ ! -f "$TMP/out.yaml" ]; } && ok "reject: $name" || ko "reject: $name"
}
# shellcheck disable=SC2016
expect_reject "unsubstituted token" 'MAX_NUM_SEQS=${vllm.max_num_seqs}'
expect_reject "empty value" 'BLOCK_SIZE='
expect_reject "missing value" 'PERFORMANCE_MODE=-'
expect_reject "optional line present but empty" 'LINEAR_BACKEND='
expect_reject "boolean as yes" 'ENFORCE_EAGER=yes'
expect_reject "gpu_memory_utilization not a fraction" 'GPU_MEMORY_UTILIZATION=92'
expect_reject "block size not a multiple of 16" 'BLOCK_SIZE=40'
expect_reject "optimization level 4" 'OPTIMIZATION_LEVEL=4'
expect_reject "batched tokens below max_num_seqs" 'MAX_NUM_SEQS=512' 'MAX_NUM_BATCHED_TOKENS=256'
expect_reject "unknown kv dtype" 'KV_CACHE_DTYPE=int8'
expect_reject "unknown performance mode" 'PERFORMANCE_MODE=fast'
expect_reject "unknown attention backend" 'ATTENTION_BACKEND=XFORMERS'
expect_reject "spec none with tokens" 'SPEC_METHOD=none' 'SPEC_TOKENS=2'
expect_reject "spec mtp with 0 tokens" 'SPEC_METHOD=mtp' 'SPEC_TOKENS=0'
expect_reject "spec method without tokens" 'SPEC_METHOD=mtp'
expect_reject "unknown spec method" 'SPEC_METHOD=ngram' 'SPEC_TOKENS=2'
expect_reject "FLASH_ATTN with fp8 KV" 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
params 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
RENDER_ALLOW_FA_FP8=1 bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" >/dev/null 2>&1 \
  && ok "FLASH_ATTN with fp8 KV when the probe allows it" || ko "FLASH_ATTN with fp8 KV when the probe allows it"
# Every token of params.env.template is a parameter the renderer reads.
for k in $(grep -o '^[A-Z_]*=\${' "$K8S/params.env.template" | cut -d= -f1); do
  grep -q "\b$k\b" "$R" && ok "template key $k is read" || ko "template key $k is read"
done
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
