#!/bin/bash
# Tests for ../render_statefulset.sh. Run: bash k8s/tests/test_render_statefulset.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
R=$K8S/render_statefulset.sh; T=$K8S/01-statefulset_template.yaml
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
params() {  # the baseline (vLLM 0.31.0 CPU online-serving defaults, gmu 0.90), then KEY=VALUE overrides; KEY=- deletes
  cat > "$TMP/params.env" <<'P'
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=128
KV_CACHE_DTYPE=auto
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
LONG_PREFILL_TOKEN_THRESHOLD=0
STREAM_INTERVAL=1
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

# 1. The baseline renders every flag, booleans as --x / --no-x, CPU pod shape.
params
if render; then
  [ "$(yq '.kind' "$TMP/out.yaml")" = StatefulSet ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.namespace' "$TMP/out.yaml")" = llm-cpu ] && ok "namespace llm-cpu" || ko "namespace llm-cpu"
  [ "$(yq '.spec.replicas' "$TMP/out.yaml")" = 1 ] && ok "one replica" || ko "one replica"
  for a in --gpu-memory-utilization=0.90 --max-num-seqs=128 --max-num-batched-tokens=2048 --block-size=128 \
           --kv-cache-dtype=auto --enable-prefix-caching --enable-chunked-prefill \
           --long-prefill-token-threshold=0 --stream-interval=1 \
           --model=Qwen/Qwen3-0.6B --served-model-name=qwen3-06b --dtype=bfloat16 --max-model-len=16384 \
           --default-chat-template-kwargs '{"enable_thinking": false}'; do
    args | grep -qxF -- "$a" && ok "arg $a" || ko "arg $a"
  done
  args | grep -q -- '=true\|=false' && ko "no =true/=false" || ok "no =true/=false"
  [ "$(yq "$C.image" "$TMP/out.yaml")" = vllm/vllm-openai-cpu:v0.31.0 ] && ok "CPU image v0.31.0" || ko "CPU image v0.31.0"
  # Guaranteed QoS: integer CPUs, requests == limits, no GPU.
  for r in cpu memory; do
    [ "$(yq "$C.resources.requests.$r" "$TMP/out.yaml")" = "$(yq "$C.resources.limits.$r" "$TMP/out.yaml")" ] \
      && ok "requests.$r == limits.$r" || ko "requests.$r == limits.$r"
  done
  [ "$(yq "$C.resources.limits.cpu" "$TMP/out.yaml")" = 44 ] && ok "44 CPUs" || ko "44 CPUs"
  [ "$(yq "$C.resources.limits.memory" "$TMP/out.yaml")" = 48Gi ] && ok "48Gi" || ko "48Gi"
  grep -q 'nvidia' "$TMP/out.yaml" && ko "no GPU anywhere" || ok "no GPU anywhere"
  [ "$(yq "$C.securityContext.seccompProfile.type" "$TMP/out.yaml")" = Unconfined ] && ok "seccomp Unconfined" || ko "seccomp Unconfined"
  [ "$(yq "$C.securityContext.capabilities.add[0]" "$TMP/out.yaml")" = SYS_NICE ] && ok "SYS_NICE" || ko "SYS_NICE"
  [ "$(yq '.spec.template.spec.volumes[] | select(.name=="shm") | .emptyDir.sizeLimit' "$TMP/out.yaml")" = 4Gi ] && ok "/dev/shm 4Gi" || ko "/dev/shm 4Gi"
  [ "$(yq '.spec.template.spec.nodeSelector."node-role"' "$TMP/out.yaml")" = llm-serving-cpu ] && ok "node selector" || ko "node selector"
  [ "$(yq '.spec.template.spec.tolerations[0].key + "=" + .spec.template.spec.tolerations[0].value + ":" + .spec.template.spec.tolerations[0].effect' "$TMP/out.yaml")" = "workload=cpu-inference:NoSchedule" ] \
    && ok "toleration" || ko "toleration"
  [ "$(envv VLLM_CPU_OMP_THREADS_BIND)" = auto ] && ok "OMP bind auto" || ko "OMP bind auto"
  [ "$(envv VLLM_CPU_NUM_OF_RESERVED_CPU)" = 2 ] && ok "2 reserved CPUs" || ko "2 reserved CPUs"
  [ "$(envv VLLM_ENGINE_READY_TIMEOUT_S)" = 1800 ] && ok "engine ready timeout 1800" || ko "engine ready timeout 1800"
  yq "$C.env[].name" "$TMP/out.yaml" | grep -qx VLLM_CPU_KVCACHE_SPACE && ko "VLLM_CPU_KVCACHE_SPACE never set" || ok "VLLM_CPU_KVCACHE_SPACE never set"
  grep -q 'hf-token\|HF_TOKEN' "$TMP/out.yaml" && ko "no HF token" || ok "no HF token"
  [ $(( $(yq "$C.startupProbe.periodSeconds" "$TMP/out.yaml") * $(yq "$C.startupProbe.failureThreshold" "$TMP/out.yaml") )) -gt 1800 ] \
    && ok "startupProbe above the engine ready timeout" || ko "startupProbe above the engine ready timeout"
  grep -q '@[A-Z_]*@' "$TMP/out.yaml" && ko "no token left" || ok "no token left"
else
  ko "baseline renders"
fi

# 2. Booleans flipped (chunked prefill off needs batched tokens >= max_model_len).
params ENABLE_PREFIX_CACHING=false ENABLE_CHUNKED_PREFILL=false MAX_NUM_BATCHED_TOKENS=16384
if render; then
  args | grep -qx -- --no-enable-prefix-caching && ok "--no-enable-prefix-caching" || ko "--no-enable-prefix-caching"
  args | grep -qx -- --no-enable-chunked-prefill && ok "--no-enable-chunked-prefill" || ko "--no-enable-chunked-prefill"
else
  ko "flipped booleans render"
fi
# 2b. The study's other corners render.
params GPU_MEMORY_UTILIZATION=0.15 MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 BLOCK_SIZE=32 KV_CACHE_DTYPE=fp8 \
  LONG_PREFILL_TOKEN_THRESHOLD=4096 STREAM_INTERVAL=16
if render; then
  for a in --gpu-memory-utilization=0.15 --block-size=32 --kv-cache-dtype=fp8 --long-prefill-token-threshold=4096 --stream-interval=16; do
    args | grep -qx -- "$a" && ok "corner arg $a" || ko "corner arg $a"
  done
else
  ko "corner renders"
fi
for b in 64 96; do params BLOCK_SIZE=$b; render && ok "block size $b" || ko "block size $b"; done
# 2c. Scientific notation from Akamas becomes a plain decimal (commit 3d2a36c).
params GPU_MEMORY_UTILIZATION=1.5e-01
render && args | grep -qx -- --gpu-memory-utilization=0.150000 && ok "1.5e-01 -> 0.150000" || ko "1.5e-01 -> 0.150000"

# 3. Invalid inputs exit 2 and write nothing.
expect_reject() {
  local name=$1; shift
  params "$@"; render; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/out.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
# shellcheck disable=SC2016
expect_reject "unsubstituted token" 'MAX_NUM_SEQS=${vllm.max_num_seqs}'
expect_reject "empty value" 'BLOCK_SIZE='
expect_reject "missing line" 'STREAM_INTERVAL=-'
expect_reject "boolean as yes" 'ENABLE_PREFIX_CACHING=yes'
expect_reject "boolean as True" 'ENABLE_CHUNKED_PREFILL=True'
expect_reject "gpu_memory_utilization not a fraction" 'GPU_MEMORY_UTILIZATION=92'
expect_reject "gpu_memory_utilization 0" 'GPU_MEMORY_UTILIZATION=0.0'
expect_reject "gpu_memory_utilization 1" 'GPU_MEMORY_UTILIZATION=1'
expect_reject "block size 16 (not a multiple of 32)" 'BLOCK_SIZE=16'
expect_reject "block size 48" 'BLOCK_SIZE=48'
expect_reject "block size 256 (outside the domain)" 'BLOCK_SIZE=256'
expect_reject "batched tokens below max_num_seqs" 'MAX_NUM_SEQS=256' 'MAX_NUM_BATCHED_TOKENS=128'
expect_reject "chunked prefill off, batched tokens 8192 < max_model_len" 'ENABLE_CHUNKED_PREFILL=false' 'MAX_NUM_BATCHED_TOKENS=8192'
expect_reject "chunked prefill off, baseline batched tokens" 'ENABLE_CHUNKED_PREFILL=false'
expect_reject "long prefill threshold above max_model_len" 'LONG_PREFILL_TOKEN_THRESHOLD=20000'
expect_reject "stream interval 0" 'STREAM_INTERVAL=0'
expect_reject "stream interval not an integer" 'STREAM_INTERVAL=1.5'
expect_reject "unknown kv dtype" 'KV_CACHE_DTYPE=int8'
expect_reject "max_num_seqs 0" 'MAX_NUM_SEQS=0'
# Every token of params.env.template is a parameter the renderer reads, every tuned key of
# the contract is an Akamas token in the template, and the two keys fixed on 2026-10-16
# (chunked prefill on, stream_interval 1) are hard-coded there.
for k in $(grep -o '^[A-Z_]*=\${' "$K8S/params.env.template" | cut -d= -f1); do
  grep -q "\b$k\b" "$R" && ok "template key $k is read" || ko "template key $k is read"
done
for k in GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE KV_CACHE_DTYPE ENABLE_PREFIX_CACHING \
         LONG_PREFILL_TOKEN_THRESHOLD; do
  grep -q "^$k=\${vllm\.[a-z_]*}$" "$K8S/params.env.template" && ok "contract key $k in the template" || ko "contract key $k in the template"
done
for kv in ENABLE_CHUNKED_PREFILL=true STREAM_INTERVAL=1; do
  grep -qx "$kv" "$K8S/params.env.template" && ok "fixed $kv in the template" || ko "fixed $kv in the template"
done
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
