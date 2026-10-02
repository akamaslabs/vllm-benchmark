#!/bin/bash
# Validates params.env and renders the StatefulSet template (28-g7-4500-mig-min-cost-fixed-load).
# Usage: render_statefulset.sh <params.env> <template> <output> <replicas 1|2>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
# RENDER_ALLOW_FA_FP8=1 lifts the FLASH_ATTN + fp8 guard: the kernel probe uses it to check
# that guard on this GPU; the study never sets it.
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 4 ] || die "usage: render_statefulset.sh <params.env> <template> <output> <replicas>"
PARAMS=$1 TEMPLATE=$2 OUT=$3 REPLICAS=$4
[ -f "$PARAMS" ] || die "no params file $PARAMS"
[ -f "$TEMPLATE" ] || die "no template $TEMPLATE"
# shellcheck disable=SC2016  # a literal ${ is what an unsubstituted Akamas token looks like
if grep -q '\${' "$PARAMS"; then
  die "params.env still has unsubstituted tokens (a parameter is missing from parametersSelection): $(grep '\${' "$PARAMS" | tr '\n' ' ')"
fi
# shellcheck disable=SC1090
source "$PARAMS"
for v in MIG_PROFILE CPU_LIMIT MEMORY_LIMIT GPU_MEMORY_UTILIZATION KV_CACHE_DTYPE MAX_NUM_SEQS \
         MAX_NUM_BATCHED_TOKENS LINEAR_BACKEND ATTENTION_BACKEND; do
  [ -n "${!v:-}" ] || die "$v is empty in params.env (doNotRenderParameters renders an empty string, not the token)"
done
# The Kubernetes pack's FileConfigurator confTemplates write cpu_limit as "<n>m" and
# memory_limit as "<n>M" (component type Kubernetes Container); the kernel probe writes bare
# integers. Strip one unit here; the template adds it back exactly once.
CPU_LIMIT=${CPU_LIMIT%m}; MEMORY_LIMIT=${MEMORY_LIMIT%M}
[[ "$REPLICAS" =~ ^[12]$ ]] || die "replicas '$REPLICAS' is not 1 or 2"
[[ "$MIG_PROFILE" =~ ^(none|1g\.16gb)$ ]] || die "mig_profile '$MIG_PROFILE' is not none or 1g.16gb"
for v in CPU_LIMIT MEMORY_LIMIT MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not a fraction"
[[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8)$ ]] || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto or fp8"
[[ "$LINEAR_BACKEND" =~ ^[a-z0-9_]+$ ]] || die "linear_backend '$LINEAR_BACKEND' is not a backend name"
[[ "$ATTENTION_BACKEND" =~ ^(auto|FLASHINFER|FLASH_ATTN|TRITON_ATTN)$ ]] || die "attention_backend '$ATTENTION_BACKEND' is not auto, FLASHINFER, FLASH_ATTN or TRITON_ATTN"
# FlashAttention 2 rejects an fp8 KV cache (vLLM 0.29.0 fa_utils.py: fp8 needs FA3 on SM 9.x
# or FA4 on SM 10.x). The kernel probe checks it on SM 12.0; until then, refuse the pair
# instead of losing ~8 min to a vLLM that does not start.
if [ "$ATTENTION_BACKEND" = FLASH_ATTN ] && [ "$KV_CACHE_DTYPE" = fp8 ] && [ "${RENDER_ALLOW_FA_FP8:-0}" != 1 ]; then
  die "FLASH_ATTN with kv_cache_dtype fp8 (FlashAttention 2 rejects an fp8 KV cache)"
fi
sed -e "s|@REPLICAS@|$REPLICAS|g" \
    -e "s|@CPU_LIMIT@|$CPU_LIMIT|g" \
    -e "s|@MEMORY_LIMIT@|$MEMORY_LIMIT|g" \
    -e "s|@GMU@|$GPU_MEMORY_UTILIZATION|g" \
    -e "s|@KV_CACHE_DTYPE@|$KV_CACHE_DTYPE|g" \
    -e "s|@MAX_NUM_SEQS@|$MAX_NUM_SEQS|g" \
    -e "s|@MAX_NUM_BATCHED_TOKENS@|$MAX_NUM_BATCHED_TOKENS|g" \
    -e "s|@LINEAR_BACKEND@|$LINEAR_BACKEND|g" \
    -e "s|@ATTENTION_BACKEND@|$ATTENTION_BACKEND|g" \
    "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then
  LEFT=$(grep -o '@[A-Z_]*@' "$OUT.tmp" | sort -u | tr '\n' ' '); rm -f "$OUT.tmp"
  die "rendered StatefulSet still has tokens: $LEFT"
fi
mv "$OUT.tmp" "$OUT"
