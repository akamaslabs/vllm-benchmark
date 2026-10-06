#!/bin/bash
# Tests for ../lib_watchdog.sh (Review Focus 3). Run: bash k8s/tests/test_lib_watchdog.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../lib_watchdog.sh
source "$(dirname "$HERE")/lib_watchdog.sh"
export WD_TTFT_MS=3000 WD_ITL_MS=600 WD_HOLD_S=120 WD_ARM_DELAY_S=150   # read by the sourced library
FAILS=0
t() { local name=$1; shift; if "$@"; then echo "ok   $name"; else echo "FAIL $name"; FAILS=$((FAILS + 1)); fi; }
n() { local name=$1; shift; if "$@"; then echo "FAIL $name"; FAILS=$((FAILS + 1)); else echo "ok   $name"; fi; }
eq() { [ "$1" = "$2" ]; }
n "no data is not over"             wd_over "" ""
n "under both thresholds"           wd_over 3000 600
t "TTFT over"                       wd_over 3001 100
t "ITL over"                        wd_over 100 601
n "non-numeric is no data"          wd_over NaN +Inf
t "starts counting when over"       eq "$(wd_next_since 1000 "" 0)" 1000
t "keeps the first over time"       eq "$(wd_next_since 1100 1000 0)" 1000
t "resets when back under"          eq "$(wd_next_since 1200 1000 1)" ""
n "not fired before the hold"       wd_fired 1119 1000
t "fired at the hold"               wd_fired 1120 1000
n "never fired when under"          wd_fired 5000 ""
n "not armed before the marker"     wd_armed "warm-up: 60 s at concurrency 4"
t "armed by the marker"             wd_armed $'line\n12:00:00 MEASURED RUN START: --request-rate 3.3'
n "not ready without the marker"    wd_ready 5000 ""
n "not ready inside the delay"      wd_ready 1149 1000
t "ready after the delay"           wd_ready 1150 1000
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
