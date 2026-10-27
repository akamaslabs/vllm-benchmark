#!/bin/bash
# run_test.sh: one Job when vllm-0 is ready, none otherwise (stub kubectl).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); export K8S
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
run() {  # $1 ready replicas, then extra VAR=value
  local ready=$1; shift
  : > "$TMP/kubectl.log"
  env PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUB_READY="$ready" \
    RT_POLL_S=0 RT_PROM=http://127.0.0.1:9 "$@" bash "$K8S/run_test.sh" > "$TMP/out.txt" 2>&1
}
run 1; rc=$?
[ $rc = 0 ] && ok "ready: exit 0" || ko "ready: exit $rc ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied aiperf-cpu$' "$TMP/kubectl.log" && ok "ready: Job applied" || ko "ready: Job applied"
grep -q 'delete job -l app=aiperf-cpu' "$TMP/kubectl.log" && ok "leftover Job deleted first" || ko "leftover Job deleted first"
grep -q 'ramp 0 -> 20 req/s over 3600 s' "$TMP/out.txt" && ok "default ramp 20 req/s / 3600 s" || ko "default ramp"
grep -q 'TTFT p95 > 3000 ms or TPOT p95 > 100 ms for 120 s' "$TMP/out.txt" && ok "watchdog at the SLO" || ko "watchdog at the SLO"
grep -q 'deadline 5100 s' "$TMP/out.txt" && ok "deadline ramp + 1500 s" || ko "deadline"
run 1 RT_RATE=8 RT_RAMP_S=1200; rc=$?
{ [ $rc = 0 ] && grep -q 'ramp 0 -> 8 req/s over 1200 s' "$TMP/out.txt"; } && ok "RT_RATE / RT_RAMP_S honoured" || ko "RT_RATE / RT_RAMP_S honoured"
run 0; rc=$?
[ $rc = 2 ] && ok "not ready: exit 2" || ko "not ready: exit $rc"
grep -q '^applied ' "$TMP/kubectl.log" && ko "not ready: no Job" || ok "not ready: no Job"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
