#!/bin/bash
# probe.sh end to end with the stub kubectl of ../../k8s/tests: phase A applies the
# baseline and records it, C-nochunk is refused by the renderer without any cluster call,
# phase B renders and applies the sweep Job, and the summary is written.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); P=$(dirname "$HERE"); STUB=$(dirname "$P")/k8s/tests/stub
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
: > "$TMP/kubectl.log"
PATH="$STUB:$PATH" STUB_LOG="$TMP/kubectl.log" AC_POLL_S=0 KP_OUT="$TMP/out" KP_PROM=http://127.0.0.1:9 \
  KP_RATES="1 2" KP_STEP_S=60 KP_COMBOS=$'B-default -\nC-nochunk ENABLE_CHUNKED_PREFILL=false' \
  bash "$P/probe.sh" > "$TMP/probe.out" 2>&1; rc=$?
[ $rc = 0 ] && ok "exit 0" || ko "exit $rc ($(tail -3 "$TMP/probe.out"))"
grep -q '"name": "B-default", .*"started": true' "$TMP/out/B-default.json" && ok "B-default recorded as started" || ko "B-default recorded as started"
grep -q '"rejected_by_renderer": true' "$TMP/out/C-nochunk.json" && ok "C-nochunk refused by the renderer" || ko "C-nochunk refused by the renderer"
[ -s "$TMP/out/B-default.facts.txt" ] && ok "node facts collected" || ko "node facts collected"
grep -q 'exec -i vllm-0 -c vllm -- bash -s' "$TMP/kubectl.log" && ok "node_facts.sh run in vllm-0" || ko "node_facts.sh run in vllm-0"
grep -q '^applied aiperf-cpu-sweep$' "$TMP/kubectl.log" && ok "sweep Job applied" || ko "sweep Job applied"
grep -q -- '--request-rate "$R"' "$TMP/out/sweep-job.yaml" && grep -q 'for R in 1 2; do' "$TMP/out/sweep-job.yaml" \
  && ok "sweep rates rendered" || ko "sweep rates rendered"
grep -q -- '--random-seed $((39 + 100 \* I))' "$TMP/out/sweep-job.yaml" && ! grep -q -- '--random-seed 39 ' "$TMP/out/sweep-job.yaml" \
  && ok "a new seed per sweep step" || ko "a new seed per sweep step"
grep -q 'scale sts vllm --replicas=0' "$TMP/kubectl.log" && ok "vLLM scaled to 0 at the end" || ko "vLLM scaled to 0 at the end"
grep -q '^Phase B' "$TMP/out/summary.txt" && ok "summary written" || ko "summary written"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
