#!/bin/bash
# Local tests of lib_checks.sh (client-side trial checks used by run_ramp.sh and run_hold.sh).
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$HERE/lib_checks.sh"
PASS=0; FAIL=0
expect() { # name expected_rc summary expected_sent
  check_result "$3" "$4" >/dev/null 2>&1; local rc=$?
  if [ "$rc" = "$2" ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); echo "FAIL: $1 (rc $rc, expected $2)"; fi
}
expect "healthy"            0 "RESULT sent=839 completed=839 errors=0 ttft_p50_ms=408 ttft_p95_ms=884 itl_p95_ms=33.1" 840
expect "no steady requests" 1 "RESULT sent=0 completed=0 errors=0 ttft_p50_ms=nan ttft_p95_ms=nan itl_p95_ms=nan" 840
expect "errors 1 %"         1 "RESULT sent=839 completed=830 errors=9 ttft_p50_ms=1 ttft_p95_ms=1 itl_p95_ms=1" 840
expect "completed 94 %"     1 "RESULT sent=839 completed=788 errors=0 ttft_p50_ms=1 ttft_p95_ms=1 itl_p95_ms=1" 840
expect "fell behind"        1 "RESULT sent=700 completed=700 errors=0 ttft_p50_ms=1 ttft_p95_ms=1 itl_p95_ms=1" 840
expect "sent 95 %"          0 "RESULT sent=798 completed=798 errors=0 ttft_p50_ms=1 ttft_p95_ms=1 itl_p95_ms=1" 840
expect "no RESULT line"     1 "garbage" 840
expect_level() { # name expected_rc summary expected_sent slo_ms
  check_level "$3" "$4" "$5" >/dev/null 2>&1; local rc=$?
  if [ "$rc" = "$2" ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); echo "FAIL: $1 (rc $rc, expected $2)"; fi
}
expect_level "level ok"          0 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=190 ttft_p95_ms=480 itl_p95_ms=15" 360 500
expect_level "level at SLO"      0 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=190 ttft_p95_ms=500 itl_p95_ms=15" 360 500
expect_level "level over SLO"    1 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=190 ttft_p95_ms=501 itl_p95_ms=15" 360 500
expect_level "level ttft nan"    1 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=nan ttft_p95_ms=nan itl_p95_ms=nan" 360 500
expect_level "level ttft NA"     1 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA" 360 500
expect_level "level fell behind" 1 "RESULT sent=300 completed=300 errors=0 ttft_p50_ms=190 ttft_p95_ms=480 itl_p95_ms=15" 360 500
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
