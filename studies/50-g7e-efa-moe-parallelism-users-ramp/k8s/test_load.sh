#!/bin/bash
# Local tests of lib_load.sh with a stub kubectl (no cluster): where the two clients point, the
# merge of the two clients' RESULT lines, and the per-client parser on a synthetic AIPerf export.
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/run"
cat > "$TMP/bin/kubectl" <<'EOF'
#!/bin/bash
case "$*" in
  *"get pod vllm-a-0"*) printf '%s' "${STUB_PHASE_A:-Running}" ;;
  *"get pod vllm-b-0"*) [ "${STUB_SERVING_B:-}" = absent ] && exit 1; printf '%s' "${STUB_SERVING_B:-true}" ;;
esac
EOF
chmod +x "$TMP/bin/kubectl"
PATH="$TMP/bin:$PATH"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
# shellcheck disable=SC1091
source "$HERE/workload.env"

# --- where the clients point -----------------------------------------------------------
urls() { ( export STUB_SERVING_B=$1; source "$HERE/lib_load.sh"; echo "${URLS[*]}" ); }
[ "$(urls true)" = "http://vllm-a.moe-efa.svc:8000 http://vllm-b.moe-efa.svc:8000" ] && ok || ko "serving b: one client per node ($(urls true))"
[ "$(urls false)" = "http://vllm-a.moe-efa.svc:8000 http://vllm-a.moe-efa.svc:8000" ] && ok || ko "headless b: both clients on node a"
[ "$(urls absent)" = "http://vllm-a.moe-efa.svc:8000 http://vllm-a.moe-efa.svc:8000" ] && ok || ko "no pod b: both clients on node a"
( export STUB_PHASE_A=Pending; source "$HERE/lib_load.sh" ) >/dev/null 2>&1; [ $? = 3 ] && ok || ko "vllm-a-0 not Running -> exit 3"

# shellcheck disable=SC1091
source "$HERE/lib_load.sh"
LEVEL_SECONDS=960 STEADY_FROM=600 SECONDS_PER_TURN=64
[ "$(expected_sent 128)" = 720 ] && ok || ko "expected_sent 128 = $(expected_sent 128)"

# --- merge of the two clients ------------------------------------------------------------
M=$(printf '%s\n' 'client 0: first request 9 s after the level started' \
  'client 0: RESULT sent=360 completed=358 errors=1 ttft_p50_ms=120 ttft_p95_ms=310 itl_p95_ms=14.2' \
  'client 1: RESULT sent=361 completed=361 errors=0 ttft_p50_ms=131 ttft_p95_ms=295 itl_p95_ms=15.0' \
  'client 1: minute  requests  ttft_p50_ms  ttft_p95_ms' | merge_results)
[ "$M" = "RESULT sent=721 completed=719 errors=1 ttft_p50_ms=131 ttft_p95_ms=310 itl_p95_ms=15.0" ] && ok || ko "merge: sums and worst latency ($M)"
M=$(printf '%s\n' 'client 0: RESULT sent=360 completed=360 errors=0 ttft_p50_ms=120 ttft_p95_ms=1310 itl_p95_ms=14.2' \
  'client 1: RESULT sent=360 completed=360 errors=0 ttft_p50_ms=90 ttft_p95_ms=295 itl_p95_ms=9' | merge_results)
[ "$M" = "RESULT sent=720 completed=720 errors=0 ttft_p50_ms=120 ttft_p95_ms=1310 itl_p95_ms=14.2" ] && ok || ko "merge: numeric max, not string max ($M)"
M=$(printf '%s\n' 'client 0: RESULT sent=360 completed=360 errors=0 ttft_p50_ms=120 ttft_p95_ms=310 itl_p95_ms=14.2' \
  'client 1: RESULT sent=0 completed=0 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA' | merge_results)
[ "$M" = "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA" ] && ok || ko "merge: a client without latencies gives NA ($M)"
M=$(printf '%s\n' 'client 0: RESULT sent=360 completed=360 errors=0 ttft_p50_ms=120 ttft_p95_ms=nan itl_p95_ms=nan' \
  'client 1: RESULT sent=360 completed=360 errors=0 ttft_p50_ms=90 ttft_p95_ms=295 itl_p95_ms=9' | merge_results)
[ "$M" = "RESULT sent=720 completed=720 errors=0 ttft_p50_ms=120 ttft_p95_ms=NA itl_p95_ms=NA" ] && ok || ko "merge: nan gives NA ($M)"
M=$(printf '%s\n' 'client 0: RESULT sent=360 completed=360 errors=0 ttft_p50_ms=120 ttft_p95_ms=310 itl_p95_ms=14.2' \
  'client 1: Traceback (most recent call last):' | merge_results)
[ -z "$M" ] && ok || ko "merge: one client without RESULT gives no merged line ($M)"

# --- the per-client parser on a synthetic export -----------------------------------------
# 300 profiling requests, one per second from t0; TTFT 100 ms before 60 s and 200..439 ms after;
# 2 errors and 1 cancelled request in the steady window; 5 warm-up rows to ignore.
python3 - "$TMP/run/profile_export.jsonl" <<'PY'
import json, sys
t0 = 1_800_000_000
with open(sys.argv[1], "w") as f:
    for i in range(5):
        f.write(json.dumps({"metadata": {"benchmark_phase": "warmup", "request_start_ns": int((t0 - 30 + i) * 1e9)},
                            "metrics": {"time_to_first_token": {"value": 9999}}}) + "\n")
    for i in range(300):
        row = {"metadata": {"benchmark_phase": "profiling", "request_start_ns": int((t0 + i) * 1e9)},
               "metrics": {"time_to_first_token": {"value": 100 if i < 60 else 200 + (i - 60)},
                           "inter_token_latency": {"value": 10 + (i % 10)},
                           "output_token_count": {"value": 200}, "input_sequence_length": {"value": 600 + 800 * (i % 10)}}}
        if i in (100, 101): row = {"metadata": row["metadata"], "metrics": {}, "error": {"code": 500, "message": "boom"}}
        if i == 102: row = {"metadata": dict(row["metadata"], was_cancelled=True), "metrics": {}}
        f.write(json.dumps(row) + "\n")
PY
printf '%s\n' 'usage_usec 1000000' 'nr_throttled 3' 'throttled_usec 500000' > "$TMP/cpu0"
printf '%s\n' 'usage_usec 601000000' 'nr_throttled 13' 'throttled_usec 2500000' > "$TMP/cpu1"
O=$(python3 -c "${PARSER//\/work\/out/$TMP}" 60 1799999990 300 2>&1)
echo "$O" | grep -qx 'first request 10 s after the level started' && ok || ko "parser: first request line"
# Steady window: requests 60..299 = 240 sent, 237 completed, 3 errors (2 errors + 1 cancelled).
echo "$O" | grep -q '^RESULT sent=240 completed=237 errors=3 ttft_p50_ms=3[12][0-9] ttft_p95_ms=42[0-9] itl_p95_ms=19.0$' && ok || ko "parser: RESULT line"
echo "$O" | grep -q '^first error: .*boom' && ok || ko "parser: first error"
echo "$O" | grep -q '^per answer output_token_count: mean 200.0 min 200.0 max 200.0 (n=237)$' && ok || ko "parser: output tokens per answer"
echo "$O" | grep -q '^per answer input_sequence_length: ' && ok || ko "parser: input length per answer"
echo "$O" | grep -q '^cpu used 600 s (2.00 cores on average), throttled 2.0 s in 10 periods$' && ok || ko "parser: cpu line"
echo "$O" | grep -q '^     0        60          100          100$' && ok || ko "parser: minute 0 row"
[ "$(echo "$O" | grep -c '^ *[0-9]* *[0-9]* *[0-9]* *[0-9]*$')" = 5 ] && ok || ko "parser: five minute rows"
: > "$TMP/run/profile_export.jsonl"
O=$(python3 -c "${PARSER//\/work\/out/$TMP}" 60 1799999990 300 2>&1)
[ "$O" = "RESULT sent=0 completed=0 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA" ] && ok || ko "parser: empty export"
[ $FAIL = 0 ] || echo "$O"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
