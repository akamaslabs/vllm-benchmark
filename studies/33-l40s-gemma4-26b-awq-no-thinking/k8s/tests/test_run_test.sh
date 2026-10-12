#!/bin/bash
# run_test.sh: one Job when vllm-0 is ready, none otherwise (stub kubectl).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); export K8S
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
run() {  # $1 ready replicas, then env assignments
  : > "$TMP/kubectl.log"
  local ready=$1; shift
  env PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUB_READY=$ready \
    RT_POLL_S=0 RT_PROM=http://127.0.0.1:9 "$@" bash "$K8S/run_test.sh" > "$TMP/out.txt" 2>&1
}
deletes() { grep -c 'delete job -l app=aiperf-l40s' "$TMP/kubectl.log"; }
run 1; rc=$?
[ $rc = 0 ] && ok "ready: exit 0" || ko "ready: exit $rc ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied aiperf-l40s$' "$TMP/kubectl.log" && ok "ready: Job applied" || ko "ready: Job applied"
grep -q 'delete job -l app=aiperf-l40s' "$TMP/kubectl.log" && ok "leftover Job deleted first" || ko "leftover Job deleted first"
grep -q 'ramp 0 -> 6 req/s over 6000 s' "$TMP/out.txt" && ok "default ramp 6 req/s / 6000 s (provisional)" || ko "default ramp"
run 0; rc=$?
[ $rc = 2 ] && ok "not ready: exit 2" || ko "not ready: exit $rc"
grep -q '^applied ' "$TMP/kubectl.log" && ko "not ready: no Job" || ok "not ready: no Job"
# The Job fails: deleted here by default (the leftover delete at the start, then this one); left
# running for the workflow's Stop AIPerf task with RT_STOP_AIPERF=0.
run 1 STUB_FAILED=1; rc=$?
[ $rc = 1 ] && [ "$(deletes)" = 2 ] && ok "job failed: exit 1, Job deleted" || ko "job failed: exit $rc, $(deletes) deletes"
run 1 STUB_FAILED=1 RT_STOP_AIPERF=0; rc=$?
[ $rc = 1 ] && [ "$(deletes)" = 1 ] && grep -q 'left running for the Stop AIPerf task' "$TMP/out.txt" \
  && ok "RT_STOP_AIPERF=0: exit 1, Job left to Stop AIPerf" || ko "RT_STOP_AIPERF=0: exit $rc, $(deletes) deletes"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
