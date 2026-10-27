#!/bin/bash
# Validates params.env and renders the StatefulSet template (43-c8i-metal-qwen3-06b-cpu-membw).
# Usage: render_statefulset.sh <params.env> <template> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
#
# The tuned flags are written here, not in the template: vLLM 0.31.0 parses every boolean
# with argparse.BooleanOptionalAction (arg_utils.py get_kwargs), so true/false must become
# --x / --no-x (--x=false is an argparse error). Every check below mirrors a vLLM 0.31.0
# startup error, so an invalid configuration fails here in a second, before any kubectl
# call, instead of after a CPU engine start (minutes).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_statefulset.sh <params.env> <template> <output>"
PARAMS=$1 TEMPLATE=$2 OUT=$3
[ -f "$PARAMS" ] || die "no params file $PARAMS"
[ -f "$TEMPLATE" ] || die "no template $TEMPLATE"
# --max-model-len is fixed in the template; two checks below depend on it.
MAX_MODEL_LEN=$(grep -o -- '--max-model-len=[0-9]*' "$TEMPLATE" | head -1 | cut -d= -f2)
[[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || die "template has no --max-model-len=<int>"
# shellcheck disable=SC2016  # a literal ${ is what an unsubstituted Akamas token looks like
if grep -q '\${' "$PARAMS"; then
  die "params.env still has unsubstituted tokens (a parameter is missing from parametersSelection): $(grep '\${' "$PARAMS" | tr '\n' ' ')"
fi
REQUIRED="GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE KV_CACHE_DTYPE
ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL LONG_PREFILL_TOKEN_THRESHOLD STREAM_INTERVAL"
for v in $REQUIRED; do unset "$v"; done
# shellcheck disable=SC1090
source "$PARAMS"
for v in $REQUIRED; do
  [ -n "${!v:-}" ] || die "$v is empty or missing in params.env (doNotRenderParameters renders an empty string, not the token)"
done

# Akamas writes a tiny real value in scientific notation (study 35 exp. 23); make it a
# plain decimal before the checks (commit 3d2a36c, studies 35/36).
if [[ "$GPU_MEMORY_UTILIZATION" =~ ^[0-9]+(\.[0-9]+)?[eE][-+]?[0-9]+$ ]]; then
  LC_ALL=C printf -v GPU_MEMORY_UTILIZATION '%.6f' "$GPU_MEMORY_UTILIZATION"
fi
# vllm/config/cache.py: gt=0, le=1. A fraction strictly between 0 and 1 here.
{ [[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] && [[ "$GPU_MEMORY_UTILIZATION" =~ [1-9] ]]; } \
  || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not a fraction in (0, 1)"
for v in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE LONG_PREFILL_TOKEN_THRESHOLD STREAM_INTERVAL; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"
done
[ "$MAX_NUM_SEQS" -ge 1 ] || die "max_num_seqs must be >= 1"
# The CPU attention backend supports block sizes MultipleOf(32) only (cpu_attn.py); the
# study's categorical domain.
[[ "$BLOCK_SIZE" =~ ^(32|64|96|128)$ ]] || die "block_size '$BLOCK_SIZE' is not 32, 64, 96 or 128"
# cpu_attn.py supported_kv_cache_dtypes (fp8 needs AVX-512 or AMX: c8i has both). The
# study domain is {auto, fp8}; the explicit fp8 variants are accepted for the probe.
[[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8|fp8_e4m3|fp8_e5m2)$ ]] || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto, fp8, fp8_e4m3 or fp8_e5m2"
for v in ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL; do
  [[ "${!v}" =~ ^(true|false)$ ]] || die "$v='${!v}' is not true or false"
done
# vllm/config/scheduler.py verify_max_model_len, in the same order.
if [ "$ENABLE_CHUNKED_PREFILL" = false ] && [ "$MAX_NUM_BATCHED_TOKENS" -lt "$MAX_MODEL_LEN" ]; then
  die "enable_chunked_prefill false needs max_num_batched_tokens >= max_model_len $MAX_MODEL_LEN (got $MAX_NUM_BATCHED_TOKENS): vLLM refuses to start"
fi
[ "$MAX_NUM_BATCHED_TOKENS" -ge "$MAX_NUM_SEQS" ] || die "max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS < max_num_seqs $MAX_NUM_SEQS"
[ "$LONG_PREFILL_TOKEN_THRESHOLD" -le "$MAX_MODEL_LEN" ] || die "long_prefill_token_threshold $LONG_PREFILL_TOKEN_THRESHOLD > max_model_len $MAX_MODEL_LEN"
[ "$STREAM_INTERVAL" -ge 1 ] || die "stream_interval must be >= 1 (got $STREAM_INTERVAL)"

flag() { if [ "$2" = true ]; then echo "--$1"; else echo "--no-$1"; fi; }
ARGS=(
  "--gpu-memory-utilization=$GPU_MEMORY_UTILIZATION"
  "--max-num-seqs=$MAX_NUM_SEQS"
  "--max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS"
  "--block-size=$BLOCK_SIZE"
  "--kv-cache-dtype=$KV_CACHE_DTYPE"
  "$(flag enable-prefix-caching "$ENABLE_PREFIX_CACHING")"
  "$(flag enable-chunked-prefill "$ENABLE_CHUNKED_PREFILL")"
  "--long-prefill-token-threshold=$LONG_PREFILL_TOKEN_THRESHOLD"
  "--stream-interval=$STREAM_INTERVAL"
)

INDENT=$(grep -m1 '@TUNED_ARGS@' "$TEMPLATE" | sed 's/@TUNED_ARGS@.*//')
[ -n "$INDENT" ] || die "template has no TUNED_ARGS line"
: > "$OUT.args"
for a in "${ARGS[@]}"; do printf '%s- "%s"\n' "$INDENT" "$a" >> "$OUT.args"; done
sed -e "/@TUNED_ARGS@/r $OUT.args" -e '/@TUNED_ARGS@/d' "$TEMPLATE" > "$OUT.tmp"
rm -f "$OUT.args"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then
  LEFT=$(grep -o '@[A-Z_]*@' "$OUT.tmp" | sort -u | tr '\n' ' '); rm -f "$OUT.tmp"
  die "rendered StatefulSet still has tokens: $LEFT"
fi
mv "$OUT.tmp" "$OUT"
