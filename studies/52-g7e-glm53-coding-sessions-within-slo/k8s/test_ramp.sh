#!/bin/bash
# Local tests of run_ramp.sh and run_hold.sh with a stub lib_load.sh (no cluster). The stub's
# run_level prints a RESULT line whose client TTFT p95 comes from STUB_TTFT_<level> and ITL p95 from
# STUB_ITL_<level> ("rcfail" = AIPerf exits 1 with no RESULT line, "killed" = exit 137,
# "rc1result" = exit 1 with a RESULT line); its restarts changes once a level >= STUB_RESTART_AT ran.
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/stub_lib.sh" <<'EOF'
t() { echo "$(date -u +%T) stub: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }
expected_sent() { awk -v u="$1" -v d="$LEVEL_SECONDS" -v s="$STEADY_FROM" -v p="$SECONDS_PER_TURN" 'BEGIN { printf "%d", (d - s) * u / p }'; }
restarts() {
  local last; last=$(cat "$STUB_DIR/last" 2>/dev/null || echo 0)
  if [ -n "${STUB_RESTART_AT:-}" ] && [ "$last" -ge "$STUB_RESTART_AT" ]; then echo r2; else echo r1; fi
}
run_level() {
  echo "$1" > "$STUB_DIR/last"; echo "$2" > "$STUB_DIR/secs"; printf '%s ' "$1" >> "$STUB_DIR/probes"
  local v="STUB_TTFT_$1" w="STUB_ITL_$1"; local x="${!v:-1000}" y="${!w:-30}"
  [ "$x" = rcfail ] && { echo "aiperf crashed"; return 1; }
  [ "$x" = killed ] && { echo "Killed"; return 137; }
  local n rc=0; n=$(expected_sent "$1")
  [ "$x" = rc1result ] && { x=1000; rc=1; }
  echo "RESULT sent=$n completed=$n errors=0 ttft_p50_ms=500 ttft_p95_ms=$x itl_p95_ms=$y worst_session_ttft_p95_ms=$x sessions_over_slo=0"
  return $rc
}
EOF
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
ramp() { # runs run_ramp.sh with the stub; extra env as arguments
  rm -f "$TMP/level" "$TMP/last" "$TMP/probes"
  env LIB_LOAD="$TMP/stub_lib.sh" STUB_DIR="$TMP" LEVEL_FILE="$TMP/level" "$@" bash "$HERE/run_ramp.sh" > "$TMP/out" 2>&1
  echo $? > "$TMP/rc"
}

probes() { cat "$TMP/probes" 2>/dev/null | sed 's/ $//'; }
# Bisection over LEVELS="8 12 16 24 32 48 64" (2026-11-18, the 90-minute cap): 24 first, then 48
# or 12, then the level between; never more than 3 levels.
ramp STUB_TTFT_48=5200 STUB_ITL_32=55
[ "$(cat "$TMP/level")" = 24 ] && [ "$(probes)" = "24 48 32" ] && [ "$(cat "$TMP/rc")" = 0 ] \
  && grep -q 'RAMP level=48 fail client TTFT p95 5200 ms > 5000 ms' "$TMP/out" \
  && grep -q 'RAMP level=32 fail client ITL p95 55 ms > 50 ms' "$TMP/out" \
  && grep -q 'RESULT ramp best=24 hold=24' "$TMP/out" && ok || ko "bisection 24 pass, 48 and 32 fail -> 24 (probes $(probes))"

ramp STUB_ITL_64=60
[ "$(cat "$TMP/level")" = 48 ] && [ "$(probes)" = "24 48 64" ] && ! grep -q WARNING "$TMP/out" \
  && ok || ko "bisection 48 pass, 64 fail -> 48 (probes $(probes))"

ramp STUB_ITL_48=60
[ "$(cat "$TMP/level")" = 32 ] && [ "$(probes)" = "24 48 32" ] && ok || ko "bisection 32 pass, 48 fail -> 32 (probes $(probes))"

ramp STUB_TTFT_24=6000 STUB_TTFT_16=6000
[ "$(cat "$TMP/level")" = 12 ] && [ "$(probes)" = "24 12 16" ] && ok || ko "bisection 24 fail, 12 pass, 16 fail -> 12 (probes $(probes))"

ramp STUB_TTFT_24=6000 STUB_TTFT_12=6000 STUB_TTFT_8=6000
[ "$(head -1 "$TMP/level")" = 8 ] && grep -qx 'best=none' "$TMP/level" && [ "$(probes)" = "24 12 8" ] \
  && grep -q 'RESULT ramp best=none hold=8' "$TMP/out" && ok || ko "bisection none passes (probes $(probes))"

ramp
[ "$(cat "$TMP/level")" = 64 ] && [ "$(probes)" = "24 48 64" ] && grep -q WARNING "$TMP/out" \
  && grep -q 'RESULT ramp best=64 hold=64' "$TMP/out" && ok || ko "bisection top passes (probes $(probes))"
[ "$(cat "$TMP/secs")" = 960 ] && ok || ko "ramp levels run LEVEL_SECONDS 960 ($(cat "$TMP/secs"))"

# A one-level ladder (smoke.sh: LEVELS=16) runs that level only.
ramp LEVELS=16
[ "$(cat "$TMP/level")" = 16 ] && [ "$(probes)" = 16 ] && grep -q 'RESULT ramp best=16 hold=16' "$TMP/out" \
  && ok || ko "one-level ladder runs that level (probes $(probes))"

ramp STUB_TTFT_24=rcfail
[ "$(cat "$TMP/rc")" = 6 ] && ok || ko "ramp aiperf crash without RESULT -> exit 6 (rc $(cat "$TMP/rc"))"

ramp STUB_TTFT_48=killed
[ "$(cat "$TMP/rc")" = 6 ] && ok || ko "ramp aiperf killed (rc 137) -> exit 6 (rc $(cat "$TMP/rc"))"

ramp STUB_TTFT_48=rc1result
[ "$(cat "$TMP/rc")" = 0 ] && [ "$(cat "$TMP/level")" = 32 ] && grep -q 'RAMP level=48 fail' "$TMP/out" \
  && ok || ko "ramp aiperf rc 1 with RESULT -> level fails (level $(cat "$TMP/level"))"

ramp STUB_RESTART_AT=48
[ "$(cat "$TMP/rc")" = 5 ] && ok || ko "ramp restart (e.g. spot node reclaimed) -> exit 5 (rc $(cat "$TMP/rc"))"

# Hold: a ramp where no level passed must not score; the hold runs HOLD_SECONDS.
hold() { rm -f "$TMP/last"; env LIB_LOAD="$TMP/stub_lib.sh" STUB_DIR="$TMP" LEVEL_FILE="$TMP/level" "$@" bash "$HERE/run_hold.sh" > "$TMP/out" 2>&1; echo $? > "$TMP/rc"; }
ramp STUB_TTFT_24=6000 STUB_TTFT_12=6000 STUB_TTFT_8=6000; hold
[ "$(cat "$TMP/rc")" = 7 ] && ok || ko "hold after no passing level -> exit 7 (rc $(cat "$TMP/rc"))"
ramp STUB_ITL_32=60 STUB_ITL_48=60; hold
[ "$(cat "$TMP/rc")" = 0 ] && grep -q 'RESULT hold sessions=24' "$TMP/out" && [ "$(cat "$TMP/secs")" = 1080 ] \
  && ok || ko "hold at the passing level for 1080 s (rc $(cat "$TMP/rc"), secs $(cat "$TMP/secs"))"
ramp STUB_ITL_32=60 STUB_ITL_48=60; hold STUB_RESTART_AT=24
[ "$(cat "$TMP/rc")" = 5 ] && ok || ko "hold restart -> exit 5 (rc $(cat "$TMP/rc"))"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
