#!/bin/bash
# Accuracy check for study 30 (eval/README.md). Per run: params.env = a base file in configs/
# plus the run's overrides, ../k8s/apply_config.sh (vLLM restarted with it), the lm-eval Job
# (render_eval_job.sh), then /benchmarks/eval/<run>/ copied to results/<run>/lmeval/.
# Ends with vLLM at 0 replicas. Never while a study 30 experiment is in flight.
#
# Usage: run_eval.sh [--limit N] [run ...]   (no run: the whole plan, in order)
# Exit codes: 0 done; 2 bad input or missing config; 3 a study experiment is running;
# 4 apply_config failed; 5 the lm-eval Job failed or timed out.
set -euo pipefail
EVAL_DIR=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=${STUDY_DIR:-$(dirname "$EVAL_DIR")}
NS=llm-l40s
OUT=${EVAL_OUT:-$EVAL_DIR/results}
APPLY=${APPLY:-$STUDY_DIR/k8s/apply_config.sh}
WAIT_S=${WAIT_S:-5400}
POLL_S=${POLL_S:-20}
t() { echo "$(date -u +%T) run_eval: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# run | base params.env in configs/ | overrides | protocols
PLAN=(
  "baseline-a|baseline.params.env||greedy card"
  "best|best.params.env||greedy"
  "best-kv-auto|best.params.env|KV_CACHE_DTYPE=auto|greedy"
  "best-no-mtp|best.params.env|SPEC_METHOD=none SPEC_TOKENS=0|greedy"
  "baseline-b|baseline.params.env||greedy"
)

LIMIT=""
if [ "${1:-}" = --limit ]; then
  [ $# -ge 2 ] || die "--limit needs a value"
  LIMIT=$2; shift 2
fi
[ -z "$LIMIT" ] || [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "limit '$LIMIT' is not a positive integer"

val() { sed -n "s/^$1=//p" "$2" | tail -1; }   # KEY file -> value

# An ablation only when best uses the lever it removes.
needed() {
  case $1 in
    best-kv-auto) [ "$(val KV_CACHE_DTYPE "$EVAL_DIR/configs/best.params.env")" != auto ] ;;
    best-no-mtp)  [ "$(val SPEC_METHOD "$EVAL_DIR/configs/best.params.env")" != none ] ;;
    *) return 0 ;;
  esac
}

# --- 0. Select and validate every run before touching the cluster ---------------------
for a in "$@"; do
  printf '%s\n' "${PLAN[@]}" | grep -q "^$a|" || die "unknown run '$a'"
done
SELECTED=()
for line in "${PLAN[@]}"; do
  name=${line%%|*}
  if [ $# -gt 0 ]; then
    want=0; for a in "$@"; do [ "$a" = "$name" ] && want=1; done
    [ $want = 1 ] || continue
  fi
  SELECTED+=("$line")
done
for line in "${SELECTED[@]}"; do
  IFS='|' read -r name base _ _ <<<"$line"
  [ -f "$EVAL_DIR/configs/$base" ] || die "$name: configs/$base missing"
done

# --- 1. Never under a live study experiment (fails closed if kubectl fails) ----------
live=$(kubectl -n $NS get job -l app=aiperf-l40s -o name)
[ -z "$live" ] || die "an aiperf-l40s Job exists: a study 30 experiment is running" 3

# --- 2. One vLLM start and one lm-eval Job per run ------------------------------------
for line in "${SELECTED[@]}"; do
  IFS='|' read -r name base overrides protocols <<<"$line"
  if ! needed "$name"; then t "$name: skipped (best does not use that lever)"; continue; fi
  D=$OUT/$name; rm -rf "$D"; mkdir -p "$D"
  cp "$EVAL_DIR/configs/$base" "$D/params.env"
  for kv in $overrides; do
    k=${kv%%=*}
    grep -q "^$k=" "$D/params.env" || die "$name: $k not in configs/$base"
    sed -i.bak "s|^$k=.*|$kv|" "$D/params.env" && rm -f "$D/params.env.bak"
  done
  t "$name: apply_config ($(grep -v '^#' "$D/params.env" | tr '\n' ' '))"
  STUDY_DIR=$STUDY_DIR PARAMS=$D/params.env RENDERED=$D/sts.yaml bash "$APPLY" > "$D/apply_config.log" 2>&1 \
    || die "$name: apply_config failed, see $D/apply_config.log" 4
  bash "$EVAL_DIR/render_eval_job.sh" "$name" "$protocols" "$LIMIT" "$D/job.yaml"
  kubectl -n $NS delete job -l app=lmeval-l40s --ignore-not-found --wait=true >/dev/null
  kubectl apply -f "$D/job.yaml" >/dev/null
  t "$name: lm-eval Job started ($protocols${LIMIT:+, limit $LIMIT})"
  waited=0; pod=""
  while :; do
    # A transient kubectl error is retried until WAIT_S, not fatal mid-run.
    failed=$(kubectl -n $NS get job lmeval-l40s -o jsonpath='{.status.failed}' 2>/dev/null || true)
    if [ -n "$failed" ] && [ "$failed" != 0 ]; then
      kubectl -n $NS logs job/lmeval-l40s -c lmeval > "$D/job.log" 2>&1 || true
      die "$name: the lm-eval Job failed, see $D/job.log" 5
    fi
    pod=$(kubectl -n $NS get pods -l app=lmeval-l40s -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "$pod" ] && kubectl -n $NS exec "$pod" -c lmeval -- test -f "/benchmarks/eval/$name/DONE" >/dev/null 2>&1; then
      break
    fi
    [ "$waited" -lt "$WAIT_S" ] || die "$name: no DONE after ${WAIT_S}s (Job left in place)" 5
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  kubectl -n $NS logs "$pod" -c lmeval > "$D/job.log"
  kubectl -n $NS cp -c lmeval "$pod:/benchmarks/eval/$name" "$D/lmeval" >/dev/null
  find "$D/lmeval" -name 'samples_*.jsonl' -exec gzip -f {} +
  kubectl -n $NS delete job lmeval-l40s --wait=false >/dev/null
  t "$name: done -> $D"
done

kubectl -n $NS scale sts vllm --replicas=0 >/dev/null
t "vLLM scaled to 0"
