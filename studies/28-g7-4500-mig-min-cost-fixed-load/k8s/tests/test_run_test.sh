#!/bin/bash
# run_test.sh must start one AIPerf Job per ready replica, and only those (Review Focus 2).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); export K8S
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
run() {  # $1 ready replicas, $2 replicas asked for (default: $1)
  : > "$TMP/kubectl.log"
  PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUB_REPLICAS=$1 STUB_SPEC=${2:-$1} \
    RT_POLL_S=0 RT_PROM=http://127.0.0.1:9 bash "$K8S/run_test.sh" > "$TMP/out.txt" 2>&1
}
run 1; rc=$?
[ $rc = 0 ] && ok "one replica: exit 0" || ko "one replica: exit $rc ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied aiperf-mig-r0$' "$TMP/kubectl.log" && ok "one replica: r0 Job" || ko "one replica: r0 Job"
grep -q '^applied aiperf-mig-r1$' "$TMP/kubectl.log" && ko "one replica: no r1 Job" || ok "one replica: no r1 Job"
run 2; rc=$?
[ $rc = 0 ] && ok "two replicas: exit 0" || ko "two replicas: exit $rc"
[ "$(grep -c '^applied aiperf-mig-r[01]$' "$TMP/kubectl.log")" = 2 ] && ok "two replicas: r0 and r1 Jobs" || ko "two replicas: r0 and r1 Jobs"
# A neighbour that is not Ready (e.g. OOMKilled by the warm-up) must not let the tenant run
# alone on a half GPU: that would measure it with an idle neighbour (~13 % faster, study 25).
run 1 2; rc=$?
[ $rc = 2 ] && ok "neighbour not ready: exit 2" || ko "neighbour not ready: exit $rc"
grep -q '^applied ' "$TMP/kubectl.log" && ko "neighbour not ready: no Job" || ok "neighbour not ready: no Job"
run 0; rc=$?
[ $rc = 2 ] && ok "no replica: exit 2" || ko "no replica: exit $rc"
grep -q '^applied ' "$TMP/kubectl.log" && ko "no replica: no Job" || ok "no replica: no Job"
RT_MODE=burst PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" bash "$K8S/run_test.sh" >/dev/null 2>&1; rc=$?
[ $rc = 2 ] && ok "unknown mode: exit 2" || ko "unknown mode: exit $rc"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
