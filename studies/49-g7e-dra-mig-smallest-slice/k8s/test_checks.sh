#!/bin/bash
# Local tests of lib_checks.sh check_clients (study 49 Load task).
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$HERE/lib_checks.sh"
PASS=0; FAIL=0
expect() { # name expected_rc summary
  check_clients "$3" >/dev/null 2>&1; local rc=$?
  if [ "$rc" = "$2" ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); echo "FAIL: $1 (rc $rc, expected $2)"; fi
}
C='C0 role=customer users=128 ttft_p95=900 itl_p95=40 requests=5000 errors=0'
N1='C1 role=neighbour users=64 ttft_p95=700 itl_p95=35 requests=2500 errors=0'
N2='C2 role=neighbour users=64 ttft_p95=710 itl_p95=36 requests=2480 errors=3'
expect "all healthy"              0 "$C"$'\n'"$N1"$'\n'"$N2"
expect "whole GPU, no neighbour"  0 "$C"
expect "customer slow"            0 "${C/ttft_p95=900/ttft_p95=9000}"$'\n'"$N1"
expect "neighbour errors 2 %"     1 "$C"$'\n'"${N1/errors=0/errors=52}"
expect "neighbour NA"             1 "$C"$'\n'"C1 role=neighbour users=64 ttft_p95=NA itl_p95=NA requests=0 errors=0"
expect "neighbour no requests"    1 "$C"$'\n'"${N1/requests=2500/requests=0}"
expect "customer completed 90 %"  1 "${C/errors=0/errors=556}"$'\n'"$N1"
expect "customer no result"       1 "C0 role=customer users=128 ttft_p95=NA itl_p95=NA requests=0 errors=0"
expect "no customer line"         1 "$N1"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
