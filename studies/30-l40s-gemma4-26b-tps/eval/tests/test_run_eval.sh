#!/bin/bash
# Tests for ../run_eval.sh with a kubectl stub and a stub apply_config.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); E=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
# A private copy of eval/ per case, with a best.params.env using the given KV dtype and
# speculative method (MAX_NUM_SEQS=448 marks it as best's).
setup() {  # kv_dtype spec_method | "none-file" for no best.params.env
  rm -rf "$TMP/e" "$TMP/out" "$TMP/log"; mkdir -p "$TMP/e/configs"; : > "$TMP/log"
  cp "$E/run_eval.sh" "$E/render_eval_job.sh" "$E/eval_job_template.yaml" "$TMP/e/"
  cp "$E/configs/baseline.params.env" "$TMP/e/configs/"
  [ "$1" = none-file ] && return
  local k=0; [ "$2" = none ] || k=3
  sed -e "s/^KV_CACHE_DTYPE=.*/KV_CACHE_DTYPE=$1/" -e "s/^SPEC_METHOD=.*/SPEC_METHOD=$2/" \
      -e "s/^SPEC_TOKENS=.*/SPEC_TOKENS=$k/" -e "s/^MAX_NUM_SEQS=.*/MAX_NUM_SEQS=448/" \
      "$E/configs/baseline.params.env" > "$TMP/e/configs/best.params.env"
}
run() {
  PATH="$HERE/stub:$PATH" STUB_LOG=$TMP/log APPLY=$HERE/stub/apply_config.sh EVAL_OUT=$TMP/out \
    POLL_S=0 bash "$TMP/e/run_eval.sh" "$@" > "$TMP/stdout" 2>&1
}
order() { grep '^apply ' "$TMP/log" | awk '{print $2}' | tr '\n' ' '; }
val() { sed -n "s/^$1=//p" "$2"; }

setup fp8 mtp; run; rc=$?
[ $rc = 0 ] && ok "fp8+mtp: exit 0" || ko "fp8+mtp: exit 0 (rc=$rc, $(tail -1 "$TMP/stdout"))"
[ "$(order)" = "baseline-a best best-kv-auto best-no-mtp baseline-b " ] && ok "fp8+mtp: all five runs in order" || ko "fp8+mtp: order '$(order)'"
P=$TMP/out/best-kv-auto/params.env
{ [ "$(val KV_CACHE_DTYPE "$P")" = auto ] && [ "$(val SPEC_METHOD "$P")" = mtp ] && [ "$(val MAX_NUM_SEQS "$P")" = 448 ]; } \
  && ok "kv-auto ablation = best + KV auto" || ko "kv-auto ablation = best + KV auto"
P=$TMP/out/best-no-mtp/params.env
{ [ "$(val SPEC_METHOD "$P")" = none ] && [ "$(val SPEC_TOKENS "$P")" = 0 ] && [ "$(val KV_CACHE_DTYPE "$P")" = fp8 ] && [ "$(val MAX_NUM_SEQS "$P")" = 448 ]; } \
  && ok "no-mtp ablation = best - MTP" || ko "no-mtp ablation = best - MTP"
grep -qF 'applied-job baseline-a "greedy card" ""' "$TMP/log" && ok "baseline-a: greedy + card" || ko "baseline-a: greedy + card"
grep -qF 'applied-job best "greedy" ""' "$TMP/log" && ok "best: greedy only" || ko "best: greedy only"
[ -f "$TMP/out/baseline-a/lmeval/greedy/ifeval/gemma4-26b-l40s/samples_ifeval_x.jsonl.gz" ] && ok "samples gzipped" || ko "samples gzipped"
[ -f "$TMP/out/baseline-a/job.log" ] && ok "job log saved" || ko "job log saved"
grep -q 'scale sts vllm --replicas=0' "$TMP/log" && ok "vLLM to 0 at the end" || ko "vLLM to 0 at the end"
[ "$(grep -c 'delete job lmeval-l40s' "$TMP/log")" = 5 ] && ok "each Job deleted" || ko "each Job deleted"

setup auto none; run; rc=$?
{ [ $rc = 0 ] && [ "$(order)" = "baseline-a best baseline-b " ]; } && ok "auto+none: ablations skipped" || ko "auto+none: order '$(order)' rc=$rc"

setup fp8 none; run
[ "$(order)" = "baseline-a best best-kv-auto baseline-b " ] && ok "fp8 only: kv ablation only" || ko "fp8 only: order '$(order)'"

setup fp8 mtp; STUB_AIPERF=1 run; rc=$?
{ [ $rc = 3 ] && ! grep -q '^apply ' "$TMP/log"; } && ok "refuses under a live experiment" || ko "refuses under a live experiment (rc=$rc)"

setup fp8 mtp; STUB_GET_FAILS=1 run; rc=$?
{ [ $rc != 0 ] && ! grep -q '^apply ' "$TMP/log"; } && ok "kubectl failing: fails closed" || ko "kubectl failing: fails closed (rc=$rc)"

setup fp8 mtp; run --limit 20 baseline-a; rc=$?
{ [ $rc = 0 ] && [ "$(order)" = "baseline-a " ] && grep -qF 'applied-job baseline-a "greedy card" "20"' "$TMP/log"; } \
  && ok "--limit 20 baseline-a" || ko "--limit 20 baseline-a (rc=$rc, order '$(order)')"

setup fp8 mtp; run nope; rc=$?
{ [ $rc = 2 ] && [ ! -s "$TMP/log" ]; } && ok "unknown run: exit 2, cluster untouched" || ko "unknown run (rc=$rc)"

setup fp8 mtp; run --limit 0 baseline-a; rc=$?
{ [ $rc = 2 ] && [ ! -s "$TMP/log" ]; } && ok "limit 0 rejected" || ko "limit 0 rejected (rc=$rc)"

setup none-file; run; rc=$?
{ [ $rc = 2 ] && [ ! -s "$TMP/log" ]; } && ok "no best.params.env: exit 2, cluster untouched" || ko "no best.params.env (rc=$rc)"

setup none-file; run baseline-a baseline-b; rc=$?
{ [ $rc = 0 ] && [ "$(order)" = "baseline-a baseline-b " ]; } && ok "baselines only without best" || ko "baselines only without best (rc=$rc)"

setup fp8 mtp; STUB_APPLY_RC=4 run; rc=$?
[ $rc = 4 ] && ok "apply_config failure: exit 4" || ko "apply_config failure: exit 4 (rc=$rc)"

setup fp8 mtp; STUB_FAILED=1 run; rc=$?
{ [ $rc = 5 ] && [ -f "$TMP/out/baseline-a/job.log" ]; } && ok "Job failed: exit 5, log kept" || ko "Job failed (rc=$rc)"

setup fp8 mtp; STUB_NOT_DONE=1 WAIT_S=0 run; rc=$?
[ $rc = 5 ] && ok "no DONE: timeout exit 5" || ko "no DONE: timeout exit 5 (rc=$rc)"

echo "$FAILS failure(s)"; [ $FAILS = 0 ]
