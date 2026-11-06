#!/bin/bash
# Local tests of run_ramp.sh with a stub lib_load.sh (no cluster). The stub's run_level prints a
# RESULT line whose client TTFT p95 comes from STUB_TTFT_<level> ("rcfail" = AIPerf exits 1 with
# no RESULT line); its restarts changes once a level >= STUB_RESTART_AT has run.
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/stub_lib.sh" <<'EOF'
t() { echo "$(date -u +%T) run_ramp: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }
expected_sent() { awk -v u="$1" -v d="$LEVEL_SECONDS" -v s="$STEADY_FROM" -v p="$SECONDS_PER_TURN" 'BEGIN { printf "%d", (d - s) * u / p }'; }
restarts() {
  local last; last=$(cat "$STUB_DIR/last" 2>/dev/null || echo 0)
  if [ -n "${STUB_RESTART_AT:-}" ] && [ "$last" -ge "$STUB_RESTART_AT" ]; then echo r2; else echo r1; fi
}
run_level() {
  echo "$1" > "$STUB_DIR/last"
  local v="STUB_TTFT_$1"; local x="${!v:-100}"
  [ "$x" = rcfail ] && { echo "aiperf crashed"; return 1; }
  [ "$x" = killed ] && { echo "Killed"; return 137; }
  local n rc=0; n=$(expected_sent "$1")
  [ "$x" = rc1result ] && { x=100; rc=1; }
  echo "RESULT sent=$n completed=$n errors=0 ttft_p50_ms=50 ttft_p95_ms=$x itl_p95_ms=15"
  return $rc
}
EOF
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
ramp() { # runs run_ramp.sh with the stub; extra env as arguments
  rm -f "$TMP/level" "$TMP/last"
  env LIB_LOAD="$TMP/stub_lib.sh" STUB_DIR="$TMP" LEVEL_FILE="$TMP/level" \
    LEVELS="16 24 32 48 64 80 96 112 128" "$@" bash "$HERE/run_ramp.sh" > "$TMP/out" 2>&1
  echo $? > "$TMP/rc"
}

ramp STUB_TTFT_48=700
[ "$(cat "$TMP/level")" = 32 ] && grep -q 'RAMP level=48 fail' "$TMP/out" && [ "$(cat "$TMP/last")" = 48 ] \
  && [ "$(cat "$TMP/rc")" = 0 ] && ok || ko "ramp stops at first fail"

ramp STUB_TTFT_16=700
[ "$(head -1 "$TMP/level")" = 16 ] && grep -qx 'best=none' "$TMP/level" && grep -q 'RESULT ramp best=none hold=16' "$TMP/out" && ok || ko "ramp none passes"

ramp
[ "$(cat "$TMP/level")" = 128 ] && grep -q WARNING "$TMP/out" && grep -q 'RESULT ramp best=128 hold=128' "$TMP/out" \
  && ok || ko "ramp top passes"

ramp STUB_TTFT_24=rcfail
[ "$(cat "$TMP/rc")" = 6 ] && ok || ko "ramp aiperf crash without RESULT -> exit 6 (rc $(cat "$TMP/rc"))"

ramp STUB_TTFT_24=killed
[ "$(cat "$TMP/rc")" = 6 ] && ok || ko "ramp aiperf killed (rc 137) -> exit 6 (rc $(cat "$TMP/rc"))"

ramp STUB_TTFT_24=rc1result
[ "$(cat "$TMP/rc")" = 0 ] && [ "$(cat "$TMP/level")" = 16 ] && grep -q 'RAMP level=24 fail' "$TMP/out" \
  && ok || ko "ramp aiperf rc 1 with RESULT -> level fails"

# Hold: a ramp where no level passed must not score.
hold() { env LIB_LOAD="$TMP/stub_lib.sh" STUB_DIR="$TMP" LEVEL_FILE="$TMP/level" bash "$HERE/run_hold.sh" > "$TMP/out" 2>&1; echo $? > "$TMP/rc"; }
ramp STUB_TTFT_16=700; hold
[ "$(cat "$TMP/rc")" = 7 ] && ok || ko "hold after no passing level -> exit 7 (rc $(cat "$TMP/rc"))"
ramp STUB_TTFT_48=700; hold
[ "$(cat "$TMP/rc")" = 0 ] && grep -q 'RESULT hold users=32' "$TMP/out" && ok || ko "hold at the passing level"

ramp STUB_RESTART_AT=24
[ "$(cat "$TMP/rc")" = 5 ] && ok || ko "ramp restart -> exit 5 (rc $(cat "$TMP/rc"))"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
