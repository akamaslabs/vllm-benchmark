#!/bin/bash
# run_test.sh: one Job when vllm-0 is ready, none otherwise (stub kubectl).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); export K8S
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
run() {  # $1 ready replicas
  : > "$TMP/kubectl.log"
  PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUB_READY=$1 \
    RT_POLL_S=0 RT_PROM=http://127.0.0.1:9 bash "$K8S/run_test.sh" > "$TMP/out.txt" 2>&1
}
run 1; rc=$?
[ $rc = 0 ] && ok "ready: exit 0" || ko "ready: exit $rc ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied aiperf-l40s$' "$TMP/kubectl.log" && ok "ready: Job applied" || ko "ready: Job applied"
grep -q 'delete job -l app=aiperf-l40s' "$TMP/kubectl.log" && ok "leftover Job deleted first" || ko "leftover Job deleted first"
grep -q 'ramp 0 -> 6 req/s over 6000 s' "$TMP/out.txt" && ok "default ramp 6 req/s / 6000 s" || ko "default ramp"
run 0; rc=$?
[ $rc = 2 ] && ok "not ready: exit 2" || ko "not ready: exit $rc"
grep -q '^applied ' "$TMP/kubectl.log" && ko "not ready: no Job" || ok "not ready: no Job"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
