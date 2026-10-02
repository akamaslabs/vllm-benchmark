#!/bin/bash
# Tests for ../render_statefulset.sh. Run: bash k8s/tests/test_render_statefulset.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
R=$K8S/render_statefulset.sh; T=$K8S/01-statefulset_template.yaml
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
params() {  # a valid params.env, then KEY=VALUE overrides
  cat > "$TMP/params.env" <<'EOF'
MIG_PROFILE=1g.16gb
CPU_LIMIT=2500
MEMORY_LIMIT=12000
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=4096
LINEAR_BACKEND=cutlass
ATTENTION_BACKEND=FLASHINFER
EOF
  local kv
  for kv in "$@"; do sed -i.bak "s|^${kv%%=*}=.*|$kv|" "$TMP/params.env"; done
}
C='.spec.template.spec.containers[0]'

# 1. A valid params.env renders every value; requests == limits with units.
params
if bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 2 >/dev/null 2>&1; then
  Y=$TMP/out.yaml
  [ "$(yq '.spec.replicas' "$Y")" = 2 ] && ok "replicas" || ko "replicas"
  [ "$(yq "$C.resources.requests.cpu" "$Y")" = 2500m ] && [ "$(yq "$C.resources.limits.cpu" "$Y")" = 2500m ] \
    && ok "cpu request = limit = 2500m" || ko "cpu request = limit = 2500m"
  [ "$(yq "$C.resources.requests.memory" "$Y")" = 12000M ] && [ "$(yq "$C.resources.limits.memory" "$Y")" = 12000M ] \
    && ok "memory request = limit = 12000M" || ko "memory request = limit = 12000M"
  [ "$(yq "$C.resources.requests.\"nvidia.com/gpu\"" "$Y")" = 1 ] && [ "$(yq "$C.resources.limits.\"nvidia.com/gpu\"" "$Y")" = 1 ] \
    && ok "gpu request = limit = 1" || ko "gpu request = limit = 1"
  for a in --gpu-memory-utilization=0.92 --kv-cache-dtype=fp8 --max-num-seqs=64 --max-num-batched-tokens=4096 \
           --linear-backend=cutlass --attention-backend=FLASHINFER --served-model-name=qwen3-8b-mig --model=Qwen/Qwen3-8B-FP8; do
    yq "$C.args[]" "$Y" | grep -qx -- "$a" && ok "arg $a" || ko "arg $a"
  done
  grep -q '@[A-Z_]*@' "$Y" && ko "no token left" || ok "no token left"
else
  ko "valid params render"
fi

# 1b. What the FileConfigurator really writes: the Kubernetes pack renders cpu_limit and
#     memory_limit through confTemplate "${value}m" / "${value}M" (component type
#     Kubernetes Container). The unit must end up in the manifest exactly once.
params 'CPU_LIMIT=7000m' 'MEMORY_LIMIT=28000M'
if bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1; then
  [ "$(yq "$C.resources.limits.cpu" "$TMP/out.yaml")" = 7000m ] && [ "$(yq "$C.resources.limits.memory" "$TMP/out.yaml")" = 28000M ] \
    && [ "$(yq "$C.resources.requests.cpu" "$TMP/out.yaml")" = 7000m ] && [ "$(yq "$C.resources.requests.memory" "$TMP/out.yaml")" = 28000M ] \
    && ok "confTemplate suffixes render once" || ko "confTemplate suffixes render once"
else
  ko "confTemplate suffixes accepted"
fi

# 2. Invalid inputs exit 2 and write nothing.
expect_reject() {
  local name=$1; shift
  params "$@"; rm -f "$TMP/out.yaml"
  bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/out.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
expect_reject "unsubstituted token" 'CPU_LIMIT=${container.cpu_limit}'
expect_reject "empty value" 'LINEAR_BACKEND='
expect_reject "MIG profile outside the study" 'MIG_PROFILE=2g.32gb'
expect_reject "non-integer cpu" 'CPU_LIMIT=2.5'
expect_reject "non-integer memory" 'MEMORY_LIMIT=12G'
expect_reject "doubled cpu suffix" 'CPU_LIMIT=7000mm'
expect_reject "memory unit on cpu" 'CPU_LIMIT=7000M'
expect_reject "cpu unit on memory" 'MEMORY_LIMIT=28000m'
expect_reject "gpu_memory_utilization not a fraction" 'GPU_MEMORY_UTILIZATION=92'
expect_reject "unknown kv dtype" 'KV_CACHE_DTYPE=fp8_e5m2'
expect_reject "unknown attention backend" 'ATTENTION_BACKEND=XFORMERS'
expect_reject "FLASH_ATTN with fp8 KV" 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
params; rm -f "$TMP/out.yaml"
bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 3 >/dev/null 2>&1; rc=$?
{ [ $rc = 2 ] && [ ! -f "$TMP/out.yaml" ]; } && ok "reject: 3 replicas" || ko "reject: 3 replicas (rc=$rc)"

# 3. Accepted edge cases.
params 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=auto'
bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1 && ok "FLASH_ATTN with auto KV" || ko "FLASH_ATTN with auto KV"
params 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
RENDER_ALLOW_FA_FP8=1 bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1 \
  && ok "FLASH_ATTN with fp8 KV when the probe allows it" || ko "FLASH_ATTN with fp8 KV when the probe allows it"
params 'MIG_PROFILE=none' 'LINEAR_BACKEND=auto' 'ATTENTION_BACKEND=auto' 'KV_CACHE_DTYPE=auto'
bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1 && ok "none / auto everywhere" || ko "none / auto everywhere"

echo "$FAILS failure(s)"; [ $FAILS = 0 ]
