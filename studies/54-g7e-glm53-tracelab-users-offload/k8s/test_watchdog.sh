#!/bin/bash
# Local tests of lib_watchdog.sh (pure functions). Usage: bash k8s/test_watchdog.sh
HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); }
bad() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
# shellcheck disable=SC1091
source "$HERE/lib_watchdog.sh"
WD_TTFT_MS=10000 WD_HOLD_S=120 WD_ARM_DELAY_S=300
wd_over 10001 && ok || bad "10001 ms is over 10000"
wd_over 10000 && bad "10000 ms is not over 10000" || ok
wd_over "" && bad "no data is not over" || ok
wd_over "n/a" && bad "a non-number is not over" || ok
[ "$(wd_next_since 500 "" 0)" = 500 ] && ok || bad "over now, under before: since = now"
[ "$(wd_next_since 600 500 0)" = 500 ] && ok || bad "still over: since unchanged"
[ -z "$(wd_next_since 600 500 1)" ] && ok || bad "back under: since cleared"
wd_fired 619 500 && bad "119 s over is not 120" || ok
wd_fired 620 500 && ok || bad "120 s over fires"
wd_fired 620 "" && bad "never over: no fire" || ok
wd_ready 399 100 && bad "armed 299 s ago: not ready" || ok
wd_ready 400 100 && ok || bad "armed 300 s ago: ready"
wd_ready 400 "" && bad "not armed: not ready" || ok
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
