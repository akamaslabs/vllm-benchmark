#!/bin/bash
# Tests for ../render_statefulset.sh. Run: bash k8s/tests/test_render_statefulset.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
R=$K8S/render_statefulset.sh; T=$K8S/01-statefulset_template.yaml
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
params() {  # the baseline (TP4, vLLM 0.32.0 L4 defaults, gmu 0.90), then KEY=VALUE overrides; KEY=- deletes the line
  cat > "$TMP/params.env" <<'P'
TENSOR_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=1
DECODE_CONTEXT_PARALLEL_SIZE=1
PREFILL_CONTEXT_PARALLEL_SIZE=1
ENABLE_EXPERT_PARALLEL=false
KV_CACHE_DTYPE=auto
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.90
ENABLE_PREFIX_CACHING=
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
envv() { yq "$C.env[] | select(.name==\"$1\") | .value" "$TMP/out.yaml"; }
render() { rm -f "$TMP/out.yaml"; bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" >/dev/null 2>&1; }
accept() {  # $1 name, then overrides: must render
  local name=$1; shift; params "$@"
  render && ok "accept: $name" || ko "accept: $name"
}
expect_reject() {  # $1 name, then overrides: exit 2, nothing written
  local name=$1; shift
  params "$@"; render; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/out.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}

# 1. The baseline renders every tuned flag, the fixed flags, the GPU pod shape.
params
if render; then
  [ "$(yq '.kind' "$TMP/out.yaml")" = StatefulSet ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/out.yaml")" = vllm-dsv2 ] && ok "name vllm-dsv2 (not the namespace's other 'vllm')" || ko "name vllm-dsv2"
  [ "$(yq '.metadata.namespace' "$TMP/out.yaml")" = llm-serving ] && ok "namespace llm-serving" || ko "namespace llm-serving"
  [ "$(yq '.spec.template.metadata.labels.app' "$TMP/out.yaml")" = vllm-dsv2 ] && ok "label app=vllm-dsv2" || ko "label app=vllm-dsv2"
  [ "$(yq '.spec.serviceName' "$TMP/out.yaml")" = vllm-dsv2-headless ] && ok "governing Service" || ko "governing Service"
  [ "$(yq '.spec.replicas' "$TMP/out.yaml")" = 1 ] && ok "one replica" || ko "one replica"
  for a in --tensor-parallel-size=4 --data-parallel-size=1 --decode-context-parallel-size=1 \
           --prefill-context-parallel-size=1 --no-enable-expert-parallel --kv-cache-dtype=auto \
           --max-num-seqs=256 --max-num-batched-tokens=2048 --gpu-memory-utilization=0.90 \
           deepseek-ai/DeepSeek-V2-Lite-Chat --served-model-name=dsv2-lite --trust-remote-code \
           --max-model-len=32768 --enable-prompt-tokens-details; do
    args | grep -qxF -- "$a" && ok "arg $a" || ko "arg $a"
  done
  args | grep -q -- 'prefix-caching' && ko "prefix caching empty = no flag (vLLM default)" || ok "prefix caching empty = no flag (vLLM default)"
  args | grep -q -- '=true\|=false' && ko "no =true/=false" || ok "no =true/=false"
  [ "$(yq "$C.image" "$TMP/out.yaml")" = vllm/vllm-openai:v0.32.0 ] && ok "image v0.32.0" || ko "image v0.32.0"
  [ "$(yq "$C.resources.limits.\"nvidia.com/gpu\"" "$TMP/out.yaml")" = 4 ] && ok "4 GPUs" || ko "4 GPUs"
  [ "$(yq '.spec.template.spec.nodeSelector."node-role"' "$TMP/out.yaml")" = llm-serving-l4 ] && ok "node selector" || ko "node selector"
  [ "$(yq '.spec.template.spec.tolerations[0].key' "$TMP/out.yaml")" = nvidia.com/gpu ] && ok "GPU toleration" || ko "GPU toleration"
  [ "$(yq '.spec.template.spec.volumes[] | select(.name=="model-cache") | .persistentVolumeClaim.claimName' "$TMP/out.yaml")" = vllm-model-cache-dsv2lite ] \
    && ok "model PVC vllm-model-cache-dsv2lite" || ko "model PVC vllm-model-cache-dsv2lite"
  [ "$(yq "$C.volumeMounts[] | select(.name==\"model-cache\") | .mountPath" "$TMP/out.yaml")" = "$(envv HF_HOME)" ] \
    && ok "HF_HOME on the model PVC" || ko "HF_HOME on the model PVC"
  [ "$(yq '.spec.template.spec.volumes[] | select(.name=="shm") | .emptyDir.medium' "$TMP/out.yaml")" = Memory ] && ok "/dev/shm in memory" || ko "/dev/shm in memory"
  [ "$(envv VLLM_ENGINE_READY_TIMEOUT_S)" = 1800 ] && ok "engine ready timeout 1800" || ko "engine ready timeout 1800"
  [ $(( $(yq "$C.startupProbe.periodSeconds" "$TMP/out.yaml") * $(yq "$C.startupProbe.failureThreshold" "$TMP/out.yaml") )) -gt 1800 ] \
    && ok "startupProbe above the engine ready timeout" || ko "startupProbe above the engine ready timeout"
  grep -q 'hf-token\|HF_TOKEN' "$TMP/out.yaml" && ko "no HF token" || ok "no HF token"
  grep -q '@[A-Z_]*@' "$TMP/out.yaml" && ko "no token left" || ok "no token left"
else
  ko "baseline renders"
fi

# 2. The study's presets render (the layouts measured in the smoke run).
accept "TP4+DCP2" DECODE_CONTEXT_PARALLEL_SIZE=2
accept "TP4+DCP4" DECODE_CONTEXT_PARALLEL_SIZE=4
expect_reject "TP4+DCP4+fp8 (crashes TRITON_MLA on the L4)" DECODE_CONTEXT_PARALLEL_SIZE=4 KV_CACHE_DTYPE=fp8
accept "TP1xPCP4+EP" TENSOR_PARALLEL_SIZE=1 PREFILL_CONTEXT_PARALLEL_SIZE=4 ENABLE_EXPERT_PARALLEL=true
accept "TP2xPCP2+EP" TENSOR_PARALLEL_SIZE=2 PREFILL_CONTEXT_PARALLEL_SIZE=2 ENABLE_EXPERT_PARALLEL=true
accept "DP4+EP" TENSOR_PARALLEL_SIZE=1 DATA_PARALLEL_SIZE=4 ENABLE_EXPERT_PARALLEL=true
accept "TP2xDP2" TENSOR_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE=2
accept "TP2xDP2+DCP2+EP" TENSOR_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE=2 DECODE_CONTEXT_PARALLEL_SIZE=2 ENABLE_EXPERT_PARALLEL=true
params TENSOR_PARALLEL_SIZE=1 PREFILL_CONTEXT_PARALLEL_SIZE=4 ENABLE_EXPERT_PARALLEL=true ENABLE_PREFIX_CACHING=false
if render; then
  for a in --tensor-parallel-size=1 --prefill-context-parallel-size=4 --enable-expert-parallel --no-enable-prefix-caching; do
    args | grep -qx -- "$a" && ok "PCP arg $a" || ko "PCP arg $a"
  done
else
  ko "PCP layout renders"
fi
params MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=32768 GPU_MEMORY_UTILIZATION=0.94 ENABLE_PREFIX_CACHING=true
render && args | grep -qx -- --enable-prefix-caching && ok "domain corner + prefix caching on" || ko "domain corner + prefix caching on"
# Scientific notation from Akamas becomes a plain decimal (commit 3d2a36c).
params GPU_MEMORY_UTILIZATION=8.5e-01
render && args | grep -qx -- --gpu-memory-utilization=0.850000 && ok "8.5e-01 -> 0.850000" || ko "8.5e-01 -> 0.850000"
# Empty value = flag omitted (vLLM default); the defaults are what the cross checks use.
params MAX_NUM_BATCHED_TOKENS= KV_CACHE_DTYPE=
if render; then
  args | grep -q -- '--max-num-batched-tokens\|--kv-cache-dtype' && ko "empty values omit their flags" || ok "empty values omit their flags"
else
  ko "empty values render"
fi

# 3. Invalid inputs exit 2 and write nothing.
# shellcheck disable=SC2016
expect_reject "unsubstituted token" 'MAX_NUM_SEQS=${vllm.max_num_seqs}'
expect_reject "missing line" 'ENABLE_PREFIX_CACHING=-'
expect_reject "missing tuned line" 'DATA_PARALLEL_SIZE=-'
expect_reject "boolean as yes" 'ENABLE_EXPERT_PARALLEL=yes'
expect_reject "boolean as True" 'ENABLE_PREFIX_CACHING=True'
expect_reject "gpu_memory_utilization not a fraction" 'GPU_MEMORY_UTILIZATION=92'
expect_reject "gpu_memory_utilization 1" 'GPU_MEMORY_UTILIZATION=1'
expect_reject "tp 3" 'TENSOR_PARALLEL_SIZE=3'
expect_reject "dcp 8" 'DECODE_CONTEXT_PARALLEL_SIZE=8'
expect_reject "fewer than 4 GPUs (TP2)" 'TENSOR_PARALLEL_SIZE=2'
expect_reject "more than 4 GPUs (TP4xDP2)" 'DATA_PARALLEL_SIZE=2'
expect_reject "dcp above tp (TP2xDP2+DCP4)" 'TENSOR_PARALLEL_SIZE=2' 'DATA_PARALLEL_SIZE=2' 'DECODE_CONTEXT_PARALLEL_SIZE=4'
expect_reject "PCP + DCP (dense MLA)" 'TENSOR_PARALLEL_SIZE=2' 'PREFILL_CONTEXT_PARALLEL_SIZE=2' 'ENABLE_EXPERT_PARALLEL=true' 'DECODE_CONTEXT_PARALLEL_SIZE=2'
expect_reject "PCP without EP (smoke crash)" 'TENSOR_PARALLEL_SIZE=2' 'PREFILL_CONTEXT_PARALLEL_SIZE=2'
expect_reject "PCP + DP" 'TENSOR_PARALLEL_SIZE=1' 'PREFILL_CONTEXT_PARALLEL_SIZE=2' 'DATA_PARALLEL_SIZE=2' 'ENABLE_EXPERT_PARALLEL=true'
accept "DP4 without EP (experts TP-sharded across all ranks)" TENSOR_PARALLEL_SIZE=1 DATA_PARALLEL_SIZE=4
expect_reject "batched tokens below max_num_seqs" 'MAX_NUM_SEQS=512' 'MAX_NUM_BATCHED_TOKENS=256'
expect_reject "batched tokens empty (2048) below max_num_seqs 4096" 'MAX_NUM_SEQS=4096' 'MAX_NUM_BATCHED_TOKENS='
expect_reject "unknown kv dtype" 'KV_CACHE_DTYPE=int8'
expect_reject "fp8_e5m2 (not in TRITON_MLA's list)" 'KV_CACHE_DTYPE=fp8_e5m2'
expect_reject "max_num_seqs 0" 'MAX_NUM_SEQS=0'
# Every token of params.env.template is a key the renderer reads, every tuned key is an
# Akamas token, and prefix caching is a fixed key (empty = vLLM default).
for k in $(grep -o '^[A-Z_]*=\${' "$K8S/params.env.template" | cut -d= -f1); do
  grep -q "\b$k\b" "$R" && ok "template key $k is read" || ko "template key $k is read"
done
for k in TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE DECODE_CONTEXT_PARALLEL_SIZE PREFILL_CONTEXT_PARALLEL_SIZE \
         ENABLE_EXPERT_PARALLEL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION; do
  grep -q "^$k=\${vllm\.[a-z_]*}$" "$K8S/params.env.template" && ok "contract key $k in the template" || ko "contract key $k in the template"
done
grep -qx "KV_CACHE_DTYPE=" "$K8S/params.env.template" && ok "fixed KV_CACHE_DTYPE= (default auto) in the template" || ko "fixed KV_CACHE_DTYPE= in the template"
grep -qx "ENABLE_PREFIX_CACHING=" "$K8S/params.env.template" && ok "fixed ENABLE_PREFIX_CACHING= (default) in the template" || ko "fixed ENABLE_PREFIX_CACHING= in the template"

# 6. The real template as the FileConfigurator leaves it (every token replaced, comments kept):
#    its header mentions the ${...} syntax, which must not be read as a leftover token
#    (first Smoke trial, 2026-10-22: the baseline was refused for exactly this).
sed -e 's/${vllm.tensor_parallel_size}/4/' -e 's/${vllm.data_parallel_size}/1/' \
    -e 's/${vllm.decode_context_parallel_size}/1/' -e 's/${vllm.prefill_context_parallel_size}/1/' \
    -e 's/${vllm.enable_expert_parallel}/false/' -e 's/${vllm.kv_cache_dtype}/auto/' \
    -e 's/${vllm.max_num_seqs}/256/' -e 's/${vllm.max_num_batched_tokens}/2048/' \
    -e 's/${vllm.gpu_memory_utilization}/0.9/' "$K8S/params.env.template" > "$TMP/params.env"
grep -q '^#.*\${' "$TMP/params.env" && ok "substituted template keeps a commented \${...}" || ko "substituted template keeps a commented \${...}"
render && ok "real template, all tokens substituted: renders" || ko "real template, all tokens substituted: renders"
sed -i.bak 's/^MAX_NUM_SEQS=256$/MAX_NUM_SEQS=${vllm.max_num_seqs}/' "$TMP/params.env"
render; [ $? = 2 ] && ok "real template, one token left: refused" || ko "real template, one token left: refused"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
