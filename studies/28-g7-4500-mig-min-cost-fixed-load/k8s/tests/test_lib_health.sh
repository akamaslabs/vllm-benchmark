#!/bin/bash
# Tests for ../lib_health.sh: apply_config.sh's decision after the warm-up requests.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../lib_health.sh
source "$(dirname "$HERE")/lib_health.sh"
FAILS=0
t() { local name=$1; shift; if "$@" >/dev/null; then echo "ok   $name"; else echo "FAIL $name"; FAILS=$((FAILS + 1)); fi; }
n() { local name=$1; shift; if "$@" >/dev/null; then echo "FAIL $name"; FAILS=$((FAILS + 1)); else echo "ok   $name"; fi; }
t "two replicas ready, no restart, warm-up ok"   replicas_healthy 2 2 0 0
t "one replica ready, no restart, warm-up ok"    replicas_healthy 1 1 0 0
n "neighbour not ready"                          replicas_healthy 2 1 0 0
n "nothing ready (empty jsonpath)"               replicas_healthy 1 "" 0 0
n "a replica restarted once during startup"      replicas_healthy 2 2 1 0
n "a replica failed the warm-up"                 replicas_healthy 2 2 0 1
REASON=$(replicas_healthy 2 1 0 0); [[ "$REASON" == *"1 of 2"* ]] && echo "ok   reason names the counts" || { echo "FAIL reason names the counts: $REASON"; FAILS=$((FAILS + 1)); }
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
