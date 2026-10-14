#!/bin/bash
# Validates params.env and renders the StatefulSet template (36-l40s-gemma4-26b-awq-replay-users-v2).
# Usage: render_statefulset.sh <params.env> <template> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
#
# The tuned flags are written here, not in the template, for two reasons:
#   - vLLM 0.29.0 parses every boolean with argparse.BooleanOptionalAction, so true/false
#     must become --x / --no-x (--x=false is an error, study 27's note);
#   - LINEAR_BACKEND / ATTENTION_BACKEND are optional (see params.env.template): absent or
#     "auto" means no flag, so vLLM keeps its own choice (for Gemma 4 on SM 8.9: TRITON_ATTN,
#     forced by Gemma4Config only when no backend is given);
#   - SPEC_METHOD / SPEC_TOKENS are optional too (vLLM pack spec_method / spec_tokens): absent
#     or none/0 means no --spec-* flag at all (vLLM builds a SpeculativeConfig as soon as any
#     of them is passed, and spec_tokens 0 is the pack's off sentinel, never valid for vLLM);
#     mtp/K renders Gemma 4's MTP drafter (google/gemma-4-26B-A4B-it-assistant, 0.78 GiB,
#     shares the target's KV cache).
# Study 32: an EMPTY value means "no flag", so vLLM picks its own default. Akamas renders a
# parameter listed in a step's doNotRenderParameters as an empty string (not the token), and the
# study's baseline steps list every parameter the customer's compose does not pass, so the
# baseline starts vLLM with the compose's flags only. GPU_MEMORY_UTILIZATION, MAX_NUM_SEQS and
# ENFORCE_EAGER are always rendered and must not be empty; the other template keys must have
# their line (the template writes them all), empty or not. MAX_NUM_BATCHED_TOKENS empty matters:
# with the vision tower loaded, vLLM 0.29.0 raises an unset value to Gemma 4's largest image
# (2496 tokens, arg_utils.py _get_min_mm_batched_tokens), while an explicit 2048 stops the engine
# ("Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than
# max_num_batched_tokens (2048)", study 32 experiment 1).
# RENDER_ALLOW_FA_FP8=1 lifts the FLASH_ATTN + fp8 guard (the startup probe uses it to check
# that guard on this GPU; the study never sets it).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_statefulset.sh <params.env> <template> <output>"
PARAMS=$1 TEMPLATE=$2 OUT=$3
[ -f "$PARAMS" ] || die "no params file $PARAMS"
[ -f "$TEMPLATE" ] || die "no template $TEMPLATE"
# shellcheck disable=SC2016  # a literal ${ is what an unsubstituted Akamas token looks like
if grep -q '\${' "$PARAMS"; then
  die "params.env still has unsubstituted tokens (a parameter is missing from parametersSelection): $(grep '\${' "$PARAMS" | tr '\n' ' ')"
fi
LINEAR_BACKEND="" ATTENTION_BACKEND="" MOE_BACKEND="" SPEC_METHOD="" SPEC_TOKENS=""
LONG_PREFILL_TOKEN_THRESHOLD="" WATERMARK="" SCHEDULER_RESERVE_FULL_ISL="" STREAM_INTERVAL="" WNA16_ACTIVATION_DTYPE=""
MAX_NUM_BATCHED_TOKENS="" KV_CACHE_DTYPE="" PERFORMANCE_MODE="" OPTIMIZATION_LEVEL="" SCHEDULING_POLICY="" ASYNC_SCHEDULING=""
MAX_CUDAGRAPH_CAPTURE_SIZE="" BLOCK_SIZE=""
# shellcheck disable=SC1090
source "$PARAMS"
REQUIRED="GPU_MEMORY_UTILIZATION MAX_NUM_SEQS ENFORCE_EAGER"
for v in $REQUIRED; do
  [ -n "${!v:-}" ] || die "$v is empty in params.env (it is always rendered: never in doNotRenderParameters)"
done
DEFAULTABLE="MAX_NUM_BATCHED_TOKENS KV_CACHE_DTYPE PERFORMANCE_MODE OPTIMIZATION_LEVEL SCHEDULING_POLICY ASYNC_SCHEDULING
MAX_CUDAGRAPH_CAPTURE_SIZE BLOCK_SIZE LINEAR_BACKEND MOE_BACKEND SPEC_METHOD SPEC_TOKENS LONG_PREFILL_TOKEN_THRESHOLD WATERMARK
SCHEDULER_RESERVE_FULL_ISL STREAM_INTERVAL WNA16_ACTIVATION_DTYPE"
for v in $DEFAULTABLE; do
  grep -q "^$v=" "$PARAMS" || die "$v has no line in params.env (empty means vLLM's default, absent means a broken template)"
done

[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not a fraction"
for v in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE BLOCK_SIZE OPTIMIZATION_LEVEL LONG_PREFILL_TOKEN_THRESHOLD STREAM_INTERVAL; do
  [ -z "${!v}" ] || [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"
done
[ -z "$OPTIMIZATION_LEVEL" ] || [ "$OPTIMIZATION_LEVEL" -le 3 ] || die "optimization_level '$OPTIMIZATION_LEVEL' is not 0-3"
[ -z "$BLOCK_SIZE" ] || { [ "$BLOCK_SIZE" -ge 16 ] && [ $((BLOCK_SIZE % 16)) = 0 ]; } \
  || die "block_size '$BLOCK_SIZE' is not a multiple of 16"
# vLLM 0.29.0 vllm/config/scheduler.py raises ValueError when max_num_batched_tokens < max_num_seqs.
[ -z "$MAX_NUM_BATCHED_TOKENS" ] || [ "$MAX_NUM_BATCHED_TOKENS" -ge "$MAX_NUM_SEQS" ] || die "max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS < max_num_seqs $MAX_NUM_SEQS"
[ -z "$KV_CACHE_DTYPE" ] || [[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8|fp8_e4m3|fp8_e5m2)$ ]] || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto, fp8, fp8_e4m3 or fp8_e5m2"
[ -z "$PERFORMANCE_MODE" ] || [[ "$PERFORMANCE_MODE" =~ ^(balanced|interactivity|throughput)$ ]] || die "performance_mode '$PERFORMANCE_MODE' is not balanced, interactivity or throughput"
[ -z "$SCHEDULING_POLICY" ] || [[ "$SCHEDULING_POLICY" =~ ^(fcfs|priority)$ ]] || die "scheduling_policy '$SCHEDULING_POLICY' is not fcfs or priority"
[[ "$ENFORCE_EAGER" =~ ^(true|false)$ ]] || die "ENFORCE_EAGER='$ENFORCE_EAGER' is not true or false"
[ -z "$ASYNC_SCHEDULING" ] || [[ "$ASYNC_SCHEDULING" =~ ^(true|false)$ ]] || die "ASYNC_SCHEDULING='$ASYNC_SCHEDULING' is not true or false"
[ -z "$LINEAR_BACKEND" ] || [[ "$LINEAR_BACKEND" =~ ^[a-z0-9_]+$ ]] || die "linear_backend '$LINEAR_BACKEND' is not a backend name"
[ -z "$MOE_BACKEND" ] || [[ "$MOE_BACKEND" =~ ^[a-z0-9_]+$ ]] || die "moe_backend '$MOE_BACKEND' is not a backend name"
[ -z "$WNA16_ACTIVATION_DTYPE" ] || [[ "$WNA16_ACTIVATION_DTYPE" =~ ^(auto|fp8|int8)$ ]] || die "wna16_activation_dtype '$WNA16_ACTIVATION_DTYPE' is not auto, fp8 or int8"
[ -z "$WATERMARK" ] || [[ "$WATERMARK" =~ ^0(\.[0-9]+)?$ ]] || die "watermark '$WATERMARK' is not a fraction in [0, 1)"
[ -z "$SCHEDULER_RESERVE_FULL_ISL" ] || [[ "$SCHEDULER_RESERVE_FULL_ISL" =~ ^(true|false)$ ]] || die "SCHEDULER_RESERVE_FULL_ISL='$SCHEDULER_RESERVE_FULL_ISL' is not true or false"
[ -z "$STREAM_INTERVAL" ] || [ "$STREAM_INTERVAL" -ge 1 ] || die "stream_interval '$STREAM_INTERVAL' is below 1"
# vLLM 0.29.0 config/scheduler.py: the threshold cannot exceed max_model_len (96000, fixed here).
[ -z "$LONG_PREFILL_TOKEN_THRESHOLD" ] || [ "$LONG_PREFILL_TOKEN_THRESHOLD" -le 96000 ] || die "long_prefill_token_threshold $LONG_PREFILL_TOKEN_THRESHOLD > max_model_len 96000"
[ -z "$ATTENTION_BACKEND" ] || [[ "$ATTENTION_BACKEND" =~ ^(auto|FLASHINFER|FLASH_ATTN|TRITON_ATTN)$ ]] \
  || die "attention_backend '$ATTENTION_BACKEND' is not auto, FLASHINFER, FLASH_ATTN or TRITON_ATTN"
# Speculative decoding: method and token count come together (both empty = not rendered, no
# flag), and none <-> 0 (the pack's sentinel; the study pairs them with a parameterConstraint).
if [ -n "$SPEC_METHOD$SPEC_TOKENS" ]; then
  [[ "$SPEC_METHOD" =~ ^(none|mtp)$ ]] || die "spec_method '$SPEC_METHOD' is not none or mtp"
  [[ "$SPEC_TOKENS" =~ ^[0-9]+$ ]] || die "spec_tokens '$SPEC_TOKENS' is not an integer"
  if [ "$SPEC_METHOD" = none ] && [ "$SPEC_TOKENS" != 0 ]; then die "spec_method none needs spec_tokens 0 (got $SPEC_TOKENS)"; fi
  if [ "$SPEC_METHOD" = mtp ] && { [ "$SPEC_TOKENS" -lt 1 ] || [ "$SPEC_TOKENS" -gt 16 ]; }; then
    die "spec_method mtp needs spec_tokens 1-16 (got $SPEC_TOKENS)"
  fi
fi
# FlashAttention 2 rejects an fp8 KV cache (vLLM 0.29.0 fa_utils.py: fp8 needs FA3 on SM 9.x or
# FA4 on SM 10.x); confirmed on SM 8.9 (study 24) and SM 12.0 (study 28).
if [ "$ATTENTION_BACKEND" = FLASH_ATTN ] && [ "$KV_CACHE_DTYPE" != auto ] && [ "${RENDER_ALLOW_FA_FP8:-0}" != 1 ]; then
  die "FLASH_ATTN with kv_cache_dtype $KV_CACHE_DTYPE (FlashAttention 2 rejects an fp8 KV cache)"
fi

flag() { [ "$2" = true ] && echo "--$1" || echo "--no-$1"; }
ARGS=(
  "--gpu-memory-utilization=$GPU_MEMORY_UTILIZATION"
  "--max-num-seqs=$MAX_NUM_SEQS"
  "$(flag enforce-eager "$ENFORCE_EAGER")"
)
# Empty = not rendered = no flag (vLLM's default).
[ -z "$MAX_NUM_BATCHED_TOKENS" ] || ARGS+=("--max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS")
[ -z "$KV_CACHE_DTYPE" ] || ARGS+=("--kv-cache-dtype=$KV_CACHE_DTYPE")
[ -z "$PERFORMANCE_MODE" ] || ARGS+=("--performance-mode=$PERFORMANCE_MODE")
[ -z "$OPTIMIZATION_LEVEL" ] || ARGS+=("--optimization-level=$OPTIMIZATION_LEVEL")
[ -z "$SCHEDULING_POLICY" ] || ARGS+=("--scheduling-policy=$SCHEDULING_POLICY")
[ -z "$ASYNC_SCHEDULING" ] || ARGS+=("$(flag async-scheduling "$ASYNC_SCHEDULING")")
[ -z "$MAX_CUDAGRAPH_CAPTURE_SIZE" ] || ARGS+=("--max-cudagraph-capture-size=$MAX_CUDAGRAPH_CAPTURE_SIZE")
[ -z "$BLOCK_SIZE" ] || ARGS+=("--block-size=$BLOCK_SIZE")
[ -n "$LINEAR_BACKEND" ] && [ "$LINEAR_BACKEND" != auto ] && ARGS+=("--linear-backend=$LINEAR_BACKEND")
[ -n "$ATTENTION_BACKEND" ] && [ "$ATTENTION_BACKEND" != auto ] && ARGS+=("--attention-backend=$ATTENTION_BACKEND")
[ -n "$MOE_BACKEND" ] && [ "$MOE_BACKEND" != auto ] && ARGS+=("--moe-backend=$MOE_BACKEND")
# 0 is vLLM's own "off": no flag, as the compose.
[ -z "$LONG_PREFILL_TOKEN_THRESHOLD" ] || [ "$LONG_PREFILL_TOKEN_THRESHOLD" = 0 ] || ARGS+=("--long-prefill-token-threshold=$LONG_PREFILL_TOKEN_THRESHOLD")
[ -z "$WATERMARK" ] || ARGS+=("--watermark=$WATERMARK")
[ -z "$SCHEDULER_RESERVE_FULL_ISL" ] || ARGS+=("$(flag scheduler-reserve-full-isl "$SCHEDULER_RESERVE_FULL_ISL")")
[ -z "$STREAM_INTERVAL" ] || ARGS+=("--stream-interval=$STREAM_INTERVAL")
if [ "$SPEC_METHOD" = mtp ]; then
  ARGS+=("--spec-method=mtp" "--spec-model=google/gemma-4-26B-A4B-it-assistant" "--spec-tokens=$SPEC_TOKENS")
fi

INDENT=$(grep -m1 '@TUNED_ARGS@' "$TEMPLATE" | sed 's/@TUNED_ARGS@.*//')
[ -n "$INDENT" ] || die "template has no TUNED_ARGS line"
: > "$OUT.args"
for a in "${ARGS[@]}"; do printf '%s- "%s"\n' "$INDENT" "$a" >> "$OUT.args"; done
# wna16_activation_dtype (vLLM pack 1.14.0, a deployment-side switch): the environment of both W4A8
# paths, Marlin's and Humming's; auto or empty = nothing set (W4A16, vLLM's default).
ENV_INDENT=$(grep -m1 '@TUNED_ENV@' "$TEMPLATE" | sed 's/@TUNED_ENV@.*//')
[ -n "$ENV_INDENT" ] || die "template has no TUNED_ENV line"
: > "$OUT.env"
case "$WNA16_ACTIVATION_DTYPE" in
  fp8) printf '%s- name: VLLM_MARLIN_INPUT_DTYPE\n%s  value: "fp8"\n%s- name: VLLM_HUMMING_INPUT_QUANT_CONFIG\n%s  value: '"'"'{"dtype":"float8e4m3"}'"'"'\n' \
         "$ENV_INDENT" "$ENV_INDENT" "$ENV_INDENT" "$ENV_INDENT" >> "$OUT.env" ;;
  int8) printf '%s- name: VLLM_MARLIN_INPUT_DTYPE\n%s  value: "int8"\n%s- name: VLLM_HUMMING_INPUT_QUANT_CONFIG\n%s  value: '"'"'{"dtype":"int8"}'"'"'\n' \
         "$ENV_INDENT" "$ENV_INDENT" "$ENV_INDENT" "$ENV_INDENT" >> "$OUT.env" ;;
esac
sed -e "/@TUNED_ARGS@/r $OUT.args" -e '/@TUNED_ARGS@/d' -e "/@TUNED_ENV@/r $OUT.env" -e '/@TUNED_ENV@/d' "$TEMPLATE" > "$OUT.tmp"
rm -f "$OUT.args" "$OUT.env"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then
  LEFT=$(grep -o '@[A-Z_]*@' "$OUT.tmp" | sort -u | tr '\n' ' '); rm -f "$OUT.tmp"
  die "rendered StatefulSet still has tokens: $LEFT"
fi
mv "$OUT.tmp" "$OUT"
