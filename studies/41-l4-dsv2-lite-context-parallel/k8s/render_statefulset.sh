#!/bin/bash
# Validates params.env and renders the StatefulSet template (41-l4-dsv2-lite-context-parallel).
# Usage: render_statefulset.sh <params.env> <template> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
#
# The tuned flags are written here, not in the template: vLLM 0.32.0 parses every boolean
# with argparse.BooleanOptionalAction (arg_utils.py get_kwargs), so true/false must become
# --x / --no-x (--x=false is an argparse error). An EMPTY value writes no flag (vLLM's own
# default); every key must still be present. Every check below mirrors a vLLM 0.32.0 startup
# error or a crash measured in the smoke run (../smoke/), so an invalid configuration fails
# here in a second, before any kubectl call, instead of after a 6-9 minute engine start.
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_statefulset.sh <params.env> <template> <output>"
PARAMS=$1 TEMPLATE=$2 OUT=$3
[ -f "$PARAMS" ] || die "no params file $PARAMS"
[ -f "$TEMPLATE" ] || die "no template $TEMPLATE"
# shellcheck disable=SC2016  # a literal ${ is what an unsubstituted Akamas token looks like
# Comment lines are skipped: the template's own header mentions the token syntax.
if grep -v '^[[:space:]]*#' "$PARAMS" | grep -q '\${'; then
  die "params.env still has unsubstituted tokens (a parameter is missing from parametersSelection): $(grep -v '^[[:space:]]*#' "$PARAMS" | grep '\${' | tr '\n' ' ')"
fi
REQUIRED="TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE DECODE_CONTEXT_PARALLEL_SIZE PREFILL_CONTEXT_PARALLEL_SIZE
ENABLE_EXPERT_PARALLEL KV_CACHE_DTYPE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION
ENABLE_PREFIX_CACHING"
for v in $REQUIRED; do
  grep -q "^$v=" "$PARAMS" || die "$v is missing from params.env (every key must be present; an empty value means vLLM's default)"
done
for v in $REQUIRED; do unset "$v"; done
# shellcheck disable=SC1090
source "$PARAMS"

# Akamas writes a tiny real value in scientific notation (study 35 exp. 23); make it a
# plain decimal before the checks (commit 3d2a36c, studies 35/36).
if [[ "${GPU_MEMORY_UTILIZATION:-}" =~ ^[0-9]+(\.[0-9]+)?[eE][-+]?[0-9]+$ ]]; then
  LC_ALL=C printf -v GPU_MEMORY_UTILIZATION '%.6f' "$GPU_MEMORY_UTILIZATION"
fi
if [ -n "${GPU_MEMORY_UTILIZATION:-}" ]; then
  # vllm/config/cache.py: gt=0, le=1. A fraction strictly between 0 and 1 here.
  { [[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] && [[ "$GPU_MEMORY_UTILIZATION" =~ [1-9] ]]; } \
    || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not a fraction in (0, 1)"
fi
for v in TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE DECODE_CONTEXT_PARALLEL_SIZE PREFILL_CONTEXT_PARALLEL_SIZE; do
  [ -z "${!v}" ] || [[ "${!v}" =~ ^(1|2|4)$ ]] || die "$v='${!v}' is not 1, 2 or 4"
done
for v in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
  [ -z "${!v}" ] || [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v='${!v}' is not a positive integer"
done
for v in ENABLE_EXPERT_PARALLEL ENABLE_PREFIX_CACHING; do
  [ -z "${!v}" ] || [[ "${!v}" =~ ^(true|false)$ ]] || die "$v='${!v}' is not true or false"
done
# TRITON_MLA supported_kv_cache_dtypes on SM 8.9 (the study's domain is {auto, fp8}).
[ -z "$KV_CACHE_DTYPE" ] || [[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8|fp8_e4m3)$ ]] \
  || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto, fp8 or fp8_e4m3"

# Effective values (empty = vLLM 0.32.0's default) for the cross-parameter checks.
TP=${TENSOR_PARALLEL_SIZE:-1} DP=${DATA_PARALLEL_SIZE:-1}
DCP=${DECODE_CONTEXT_PARALLEL_SIZE:-1} PCP=${PREFILL_CONTEXT_PARALLEL_SIZE:-1}
EP=${ENABLE_EXPERT_PARALLEL:-false}
SEQS=${MAX_NUM_SEQS:-256} BATCHED=${MAX_NUM_BATCHED_TOKENS:-2048}
# Parity: the pod requests all four L4s and vLLM must use all of them.
[ $((TP * PCP * DP)) -eq 4 ] || die "tp $TP x pcp $PCP x dp $DP = $((TP * PCP * DP)), not the 4 GPUs of the pod"
# vllm/config/parallel.py (pcp == 1): tp_size must be divisible by dcp_size. MLA has one
# KV head, so dcp <= tp is the useful range.
[ $((TP % DCP)) -eq 0 ] || die "tp $TP is not divisible by dcp $DCP"
if [ "$PCP" -gt 1 ]; then
  # v1/worker/gpu/pcp_manager.py: "MRV2 PCP + DCP supports sparse MLA models only".
  [ "$DCP" -eq 1 ] || die "pcp $PCP with dcp $DCP: vLLM 0.32.0 supports PCP + DCP on sparse MLA models only"
  # Smoke run 1: TP2 x PCP2 without EP crashed (topk_softmax is_padding size mismatch).
  [ "$EP" = true ] || die "pcp $PCP without expert parallelism crashes (topk_softmax is_padding size mismatch)"
  # PCP + DP: unverified (open upstream work), excluded by the study.
  [ "$DP" -eq 1 ] || die "pcp $PCP with dp $DP is excluded (unverified)"
fi
# Excluded by the study (unmeasured; see README "Deviations from the spec").
# vllm/config/scheduler.py
[ "$BATCHED" -ge "$SEQS" ] || die "max_num_batched_tokens $BATCHED < max_num_seqs $SEQS"

flag() { if [ "$2" = true ]; then echo "--$1"; else echo "--no-$1"; fi; }
ARGS=()
add() { [ -z "$2" ] || ARGS+=("--$1=$2"); }
add tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
add data-parallel-size "$DATA_PARALLEL_SIZE"
add decode-context-parallel-size "$DECODE_CONTEXT_PARALLEL_SIZE"
add prefill-context-parallel-size "$PREFILL_CONTEXT_PARALLEL_SIZE"
[ -z "$ENABLE_EXPERT_PARALLEL" ] || ARGS+=("$(flag enable-expert-parallel "$ENABLE_EXPERT_PARALLEL")")
add kv-cache-dtype "$KV_CACHE_DTYPE"
add max-num-seqs "$MAX_NUM_SEQS"
add max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
add gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
[ -z "$ENABLE_PREFIX_CACHING" ] || ARGS+=("$(flag enable-prefix-caching "$ENABLE_PREFIX_CACHING")")

INDENT=$(grep -m1 '@TUNED_ARGS@' "$TEMPLATE" | sed 's/@TUNED_ARGS@.*//')
[ -n "$INDENT" ] || die "template has no TUNED_ARGS line"
: > "$OUT.args"
for a in ${ARGS[@]+"${ARGS[@]}"}; do printf '%s- "%s"\n' "$INDENT" "$a" >> "$OUT.args"; done
sed -e "/@TUNED_ARGS@/r $OUT.args" -e '/@TUNED_ARGS@/d' "$TEMPLATE" > "$OUT.tmp"
rm -f "$OUT.args"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then
  LEFT=$(grep -o '@[A-Z_]*@' "$OUT.tmp" | sort -u | tr '\n' ' '); rm -f "$OUT.tmp"
  die "rendered StatefulSet still has tokens: $LEFT"
fi
mv "$OUT.tmp" "$OUT"
