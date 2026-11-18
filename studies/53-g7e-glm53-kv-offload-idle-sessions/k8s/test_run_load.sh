#!/bin/bash
# Local tests of run_load.sh with a stub lib_load.sh (no cluster). The stub's run_level prints both
# clients' RESULT and TURNS lines and the merged RESULT; STUB_MODE picks the outcome: ok (resumed turns
# fast), slow (every turn recomputed: TTFT 11-22 s), behind (completed < 95 % of sent), none (nothing
# completed), rcfail (AIPerf exit 1 with no RESULT), killed (exit 137), restart (vLLM replaced).
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/stub_lib.sh" <<'EOF'
t() { echo "$(date -u +%T) stub: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }
restarts() { if [ "${STUB_MODE:-}" = restart ] && [ -f "$STUB_DIR/ran" ]; then echo r2; else echo r1; fi; }
merge_turns() { awk '/^client [0-9]+: TURNS / { n++; l = $0 } END { if (n == 2) { sub(/^client [0-9]+: /, "", l); print l } }'; }
run_level() {
  touch "$STUB_DIR/ran"; echo "$1 $2 $3" > "$STUB_DIR/args"
  local sent=26 comp=26 p50=1700 p95=1750
  case "${STUB_MODE:-ok}" in
    rcfail) echo "aiperf crashed"; return 1 ;;
    killed) echo "Killed"; return 137 ;;
    slow) p50=21600 p95=22500 ;;
    behind) comp=20 p50=60000 p95=90000 ;;
    none) comp=0 ;;
  esac
  for c in 0 1; do
    echo "client $c: RESULT sent=$sent completed=$comp errors=0 ttft_p50_ms=$p50 ttft_p95_ms=$p95 itl_p95_ms=30 worst_session_ttft_p95_ms=$p95 sessions_over_slo=0"
    echo "client $c: TURNS first_n=3 first_ttft_p50_ms=11000 first_ttft_p95_ms=21000 resumed_n=23 resumed_ttft_p50_ms=$p50 resumed_ttft_p95_ms=$p95"
  done
  echo "RESULT sent=$((2 * sent)) completed=$((2 * comp)) errors=0 ttft_p50_ms=$p50 ttft_p95_ms=$p95 itl_p95_ms=30 worst_session_ttft_p95_ms=$p95 sessions_over_slo=0"
}
EOF
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
load() { rm -f "$TMP/ran" "$TMP/args"; env LIB_LOAD="$TMP/stub_lib.sh" STUB_DIR="$TMP" "$@" bash "$HERE/run_load.sh" > "$TMP/out" 2>&1; echo $? > "$TMP/rc"; }

load STUB_MODE=ok
[ "$(cat "$TMP/rc")" = 0 ] && grep -q '^RESULT load sessions=40 sent=52 completed=52 ' "$TMP/out" \
  && grep -q '^TURNS first_n=3 .* resumed_n=23 resumed_ttft_p50_ms=1700 resumed_ttft_p95_ms=1750$' "$TMP/out" \
  && ok || ko "fast load -> exit 0 with RESULT load and TURNS (rc $(cat "$TMP/rc"))"
[ "$(cat "$TMP/args")" = "40 1500 load-40" ] && ok || ko "one run of SESSIONS for LOAD_SECONDS, label load-40 ($(cat "$TMP/args"))"

# Slow or behind is a result, not a failure: the Akamas goal and constraints score it (design: the
# no-offload baseline must score badly, not fail).
load STUB_MODE=slow
[ "$(cat "$TMP/rc")" = 0 ] && grep -q 'NOTE: resumed turns TTFT p95 22500 ms above the 5000 ms SLO' "$TMP/out" && ok || ko "slow resumed turns -> exit 0 with a NOTE (rc $(cat "$TMP/rc"))"
load STUB_MODE=behind
[ "$(cat "$TMP/rc")" = 0 ] && grep -q 'NOTE: completed 40 of 52' "$TMP/out" && ok || ko "load behind -> exit 0 with a NOTE (rc $(cat "$TMP/rc"))"

load STUB_MODE=none
[ "$(cat "$TMP/rc")" = 6 ] && ok || ko "nothing completed -> exit 6 (rc $(cat "$TMP/rc"))"
load STUB_MODE=rcfail
[ "$(cat "$TMP/rc")" = 6 ] && ok || ko "AIPerf crash without RESULT -> exit 6 (rc $(cat "$TMP/rc"))"
load STUB_MODE=killed
[ "$(cat "$TMP/rc")" = 6 ] && ok || ko "AIPerf killed -> exit 6 (rc $(cat "$TMP/rc"))"
load STUB_MODE=restart
[ "$(cat "$TMP/rc")" = 5 ] && ok || ko "vLLM restarted or replaced -> exit 5 (rc $(cat "$TMP/rc"))"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
