#!/bin/bash
# Validates a combination's params.env and renders the StatefulSet template (W4A16 kernel probe).
# Usage: render_statefulset.sh <params.env> <template> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
#
# params.env keys (shell syntax, sourced):
#   GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN, MAX_NUM_SEQS   required (customer: 0.90 / 96000 / 64)
#   KV_CACHE_DTYPE                                        required (auto | fp8)
#   VLLM_ARGS   optional, space-separated extra flags: --x, --no-x or --x=value (value without
#               spaces or quotes), e.g. "--moe-backend=marlin --language-model-only"
#   KERNEL_ENV  optional, space-separated NAME=value pairs for the container environment, e.g.
#               "VLLM_MARLIN_INPUT_DTYPE=fp8". A value may hold JSON without spaces
#               (VLLM_HUMMING_INPUT_QUANT_CONFIG={"dtype":"float8e4m3"}).
# vLLM 0.29.0 parses booleans with BooleanOptionalAction: --x / --no-x, never --x=false.
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_statefulset.sh <params.env> <template> <output>"
PARAMS=$1 TEMPLATE=$2 OUT=$3
[ -f "$PARAMS" ] || die "no params file $PARAMS"
[ -f "$TEMPLATE" ] || die "no template $TEMPLATE"
VLLM_ARGS="" KERNEL_ENV=""
# shellcheck disable=SC1090
source "$PARAMS"
for v in GPU_MEMORY_UTILIZATION MAX_MODEL_LEN MAX_NUM_SEQS KV_CACHE_DTYPE; do
  [ -n "${!v:-}" ] || die "$v is empty in $PARAMS"
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not a fraction"
for v in MAX_MODEL_LEN MAX_NUM_SEQS; do [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"; done
[[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8|fp8_e4m3|fp8_e5m2)$ ]] || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto or fp8*"

ARGS=(
  "--gpu-memory-utilization=$GPU_MEMORY_UTILIZATION"
  "--max-model-len=$MAX_MODEL_LEN"
  "--max-num-seqs=$MAX_NUM_SEQS"
  "--kv-cache-dtype=$KV_CACHE_DTYPE"
)
for a in $VLLM_ARGS; do
  [[ "$a" =~ ^--[a-z0-9-]+(=[A-Za-z0-9_./:,+-]+)?$ ]] || die "VLLM_ARGS item '$a' is not a plain --flag[=value]"
  ARGS+=("$a")
done

INDENT=$(grep -m1 '@VARIANT_ARGS@' "$TEMPLATE" | sed 's/@VARIANT_ARGS@.*//')
ENV_INDENT=$(grep -m1 '@VARIANT_ENV@' "$TEMPLATE" | sed 's/@VARIANT_ENV@.*//')
[ -n "$INDENT" ] && [ -n "$ENV_INDENT" ] || die "template lacks @VARIANT_ARGS@ or @VARIANT_ENV@"
: > "$OUT.args"
for a in "${ARGS[@]}"; do printf '%s- "%s"\n' "$INDENT" "$a" >> "$OUT.args"; done
: > "$OUT.env"
for kv in $KERNEL_ENV; do
  k=${kv%%=*} val=${kv#*=}
  [[ "$k" =~ ^[A-Z][A-Z0-9_]*$ ]] && [ "$k" != "$kv" ] || die "KERNEL_ENV item '$kv' is not NAME=value"
  [[ "$val" != *"'"* ]] || die "KERNEL_ENV value of $k holds a single quote"
  printf "%s- name: %s\n%s  value: '%s'\n" "$ENV_INDENT" "$k" "$ENV_INDENT" "$val" >> "$OUT.env"
done
sed -e "/@VARIANT_ARGS@/r $OUT.args" -e '/@VARIANT_ARGS@/d' -e "/@VARIANT_ENV@/r $OUT.env" -e '/@VARIANT_ENV@/d' "$TEMPLATE" > "$OUT.tmp"
rm -f "$OUT.args" "$OUT.env"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "rendered StatefulSet still has tokens"; fi
mv "$OUT.tmp" "$OUT"
