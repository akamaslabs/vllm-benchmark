#!/bin/bash
# Tests for ../render_statefulset.sh. Run: bash k8s/tests/test_render_statefulset.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
R=$K8S/render_statefulset.sh; T=$K8S/01-statefulset_template.yaml
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
params() {  # the baseline (the customer's compose values, vLLM 0.29.0 defaults otherwise), then KEY=VALUE overrides; KEY=- deletes
  cat > "$TMP/params.env" <<'P'
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_DTYPE=auto
PERFORMANCE_MODE=balanced
OPTIMIZATION_LEVEL=2
ENFORCE_EAGER=false
SCHEDULING_POLICY=fcfs
ASYNC_SCHEDULING=true
MAX_CUDAGRAPH_CAPTURE_SIZE=512
BLOCK_SIZE=16
LINEAR_BACKEND=
MOE_BACKEND=
SPEC_METHOD=
SPEC_TOKENS=
LONG_PREFILL_TOKEN_THRESHOLD=
WATERMARK=
SCHEDULER_RESERVE_FULL_ISL=
STREAM_INTERVAL=
WNA16_ACTIVATION_DTYPE=
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
  for a in --gpu-memory-utilization=0.90 --max-num-seqs=64 --max-num-batched-tokens=2048 --kv-cache-dtype=auto \
           --performance-mode=balanced --optimization-level=2 --no-enforce-eager --scheduling-policy=fcfs \
           --async-scheduling --max-cudagraph-capture-size=512 --block-size=16 \
           --model=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --revision=0ef577a --served-model-name=aiservices_gemma \
           --max-model-len=96000 --enable-auto-tool-choice --tool-call-parser=gemma4 --reasoning-parser=gemma4; do
    args | grep -qx -- "$a" && ok "arg $a" || ko "arg $a"
  done
  # The customer's compose passes neither: vision tower loaded, prefix caching at vLLM's default.
  for a in --language-model-only --enable-prefix-caching --no-enable-prefix-caching; do
    args | grep -qx -- "$a" && ko "no $a" || ok "no $a"
  done
  env_of() { yq "$C.env[] | select(.name==\"$1\") | .value" "$TMP/out.yaml"; }
  [ "$(env_of PYTORCH_CUDA_ALLOC_CONF)" = expandable_segments:True ] && ok "compose env PYTORCH_CUDA_ALLOC_CONF" || ko "compose env PYTORCH_CUDA_ALLOC_CONF"
  [ "$(env_of RAY_DISABLE_NODE_DISCOVERY)" = 1 ] && ok "compose env RAY_DISABLE_NODE_DISCOVERY" || ko "compose env RAY_DISABLE_NODE_DISCOVERY"
  # Study 35, the compose: no server-wide thinking default (a client asks for it per request).
  args | grep -q -- '--default-chat-template-kwargs' && ko "no server-wide chat-template kwargs" || ok "no server-wide chat-template kwargs"
  args | grep -qx -- '--served-model-name=aiservices_gemma' && ok "served as aiservices_gemma" || ko "served as aiservices_gemma"
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

# 2a. doNotRenderParameters (the study's baseline steps): Akamas renders an empty value, and an
# empty value means "no flag", so vLLM picks its own default, as the customer's compose does.
DNR="MAX_NUM_BATCHED_TOKENS= KV_CACHE_DTYPE= PERFORMANCE_MODE= OPTIMIZATION_LEVEL= SCHEDULING_POLICY= ASYNC_SCHEDULING= MAX_CUDAGRAPH_CAPTURE_SIZE= BLOCK_SIZE= LINEAR_BACKEND= SPEC_METHOD= SPEC_TOKENS="
# shellcheck disable=SC2086
params $DNR
if render; then
  for a in --gpu-memory-utilization=0.90 --max-num-seqs=64 --no-enforce-eager; do
    args | grep -qx -- "$a" && ok "not-rendered baseline: arg $a" || ko "not-rendered baseline: arg $a"
  done
  # vLLM 0.29.0 raises an unset max_num_batched_tokens to Gemma 4's image size (2496); an
  # explicit 2048 crashes the engine (study 32, experiment 1).
  for f in --max-num-batched-tokens --kv-cache-dtype --performance-mode --optimization-level --scheduling-policy --async-scheduling \
           --no-async-scheduling --max-cudagraph-capture-size --block-size --linear-backend --spec-; do
    args | grep -q -- "^$f" && ko "not-rendered baseline: no $f" || ok "not-rendered baseline: no $f"
  done
else
  ko "not-rendered baseline renders"
fi
params LINEAR_BACKEND=
render && ! args | grep -q -- '--linear-backend' && ok "empty linear backend: no flag" || ko "empty linear backend: no flag"

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
expect_reject "empty gpu_memory_utilization (always rendered)" 'GPU_MEMORY_UTILIZATION='
expect_reject "empty max_num_seqs (always rendered)" 'MAX_NUM_SEQS='
expect_reject "missing line (the template writes every line)" 'PERFORMANCE_MODE=-'
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
expect_reject "spec method with empty tokens" 'SPEC_METHOD=mtp' 'SPEC_TOKENS='
expect_reject "spec tokens with empty method" 'SPEC_METHOD=' 'SPEC_TOKENS=2'
expect_reject "unknown spec method" 'SPEC_METHOD=ngram' 'SPEC_TOKENS=2'
expect_reject "FLASH_ATTN with fp8 KV" 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
params 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
RENDER_ALLOW_FA_FP8=1 bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" >/dev/null 2>&1 \
  && ok "FLASH_ATTN with fp8 KV when the probe allows it" || ko "FLASH_ATTN with fp8 KV when the probe allows it"
# Every token of params.env.template is a parameter the renderer reads.
for k in $(grep -o '^[A-Z_]*=\${' "$K8S/params.env.template" | cut -d= -f1); do
  grep -q "\b$k\b" "$R" && ok "template key $k is read" || ko "template key $k is read"
done
# Study 35's scheduler and MoE parameters.
params MOE_BACKEND=humming LONG_PREFILL_TOKEN_THRESHOLD=2048 WATERMARK=0.05 SCHEDULER_RESERVE_FULL_ISL=false STREAM_INTERVAL=4
if render; then
  for a in --moe-backend=humming --long-prefill-token-threshold=2048 --watermark=0.05 --no-scheduler-reserve-full-isl --stream-interval=4; do
    args | grep -qx -- "$a" && ok "renders $a" || ko "renders $a"
  done
else ko "study 35 parameters render"; fi
params MOE_BACKEND=auto LONG_PREFILL_TOKEN_THRESHOLD=0 SCHEDULER_RESERVE_FULL_ISL=true
if render; then
  args | grep -q -- '--moe-backend\|--long-prefill-token-threshold' && ko "auto / 0: no flag" || ok "auto / 0: no flag"
  args | grep -qx -- --scheduler-reserve-full-isl && ok "reserve full isl true: --scheduler-reserve-full-isl" || ko "reserve full isl true"
else ko "auto / 0 renders"; fi
for bad in WATERMARK=1.5 STREAM_INTERVAL=0 LONG_PREFILL_TOKEN_THRESHOLD=100000 SCHEDULER_RESERVE_FULL_ISL=yes; do
  params "$bad"; render && ko "rejects $bad" || ok "rejects $bad"
done
# Study 36: wna16_activation_dtype -> the environment of both W4A8 paths.
params WNA16_ACTIVATION_DTYPE=fp8
if render; then
  [ "$(yq "$C.env[] | select(.name == \"VLLM_MARLIN_INPUT_DTYPE\") | .value" "$TMP/out.yaml")" = fp8 ] && ok "fp8: VLLM_MARLIN_INPUT_DTYPE=fp8" || ko "fp8: VLLM_MARLIN_INPUT_DTYPE"
  [ "$(yq "$C.env[] | select(.name == \"VLLM_HUMMING_INPUT_QUANT_CONFIG\") | .value" "$TMP/out.yaml")" = '{"dtype":"float8e4m3"}' ] && ok "fp8: Humming input quant fp8" || ko "fp8: Humming input quant"
else ko "fp8 renders"; fi
for v in auto ""; do
  params "WNA16_ACTIVATION_DTYPE=$v"
  if render; then yq "$C.env[].name" "$TMP/out.yaml" | grep -q 'VLLM_MARLIN_INPUT_DTYPE\|VLLM_HUMMING_INPUT' && ko "'$v': no activation variable" || ok "'$v': no activation variable"
  else ko "'$v' renders"; fi
done
params WNA16_ACTIVATION_DTYPE=fp16; render && ko "rejects fp16" || ok "rejects fp16"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
