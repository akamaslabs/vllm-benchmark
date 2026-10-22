#!/bin/bash
# run_test.sh: one Job when vllm-dsv2-0 is ready, none otherwise (stub kubectl).
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
grep -q '^applied aiperf-dsv2$' "$TMP/kubectl.log" && ok "ready: Job applied" || ko "ready: Job applied"
grep -q 'delete job -l app=aiperf-dsv2' "$TMP/kubectl.log" && ok "leftover Job deleted first" || ko "leftover Job deleted first"
grep -q -- '-n llm-serving' "$TMP/kubectl.log" && ok "namespace llm-serving" || ko "namespace llm-serving"
grep -qE 'app=vllm( |$)' "$TMP/kubectl.log" && ko "never selects the other study's app=vllm" || ok "never selects the other study's app=vllm"
grep -q 'users ramp 1 -> 96 over 3600 s' "$TMP/out.txt" && ok "default ramp 96 users / 3600 s" || ko "default ramp"
grep -q 'TTFT p95 > 20000 ms or TPOT p95 > 100 ms for 120 s' "$TMP/out.txt" && ok "watchdog at 2 x the SLO" || ko "watchdog at 2 x the SLO"
grep -q 'deadline 6000 s' "$TMP/out.txt" && ok "deadline ramp + 2400 s" || ko "deadline"
run 1 RT_USERS=48 RT_RAMP_S=1200; rc=$?
{ [ $rc = 0 ] && grep -q 'users ramp 1 -> 48 over 1200 s' "$TMP/out.txt"; } && ok "RT_USERS / RT_RAMP_S honoured" || ko "RT_USERS / RT_RAMP_S honoured"
run 0; rc=$?
[ $rc = 2 ] && ok "not ready: exit 2" || ko "not ready: exit $rc"
grep -q '^applied ' "$TMP/kubectl.log" && ko "not ready: no Job" || ok "not ready: no Job"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
