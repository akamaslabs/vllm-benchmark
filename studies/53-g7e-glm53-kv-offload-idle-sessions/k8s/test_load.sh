#!/bin/bash
# Local tests of lib_load.sh with a stub kubectl (no cluster): the Running-pod check, the expected
# requests, the merge of the two clients' RESULT lines (with the per-session report), the AIPerf
# command (DRY_RUN=1) and the per-client parser on a synthetic AIPerf export, with study 53's split
# of first turns from resumed turns (TURNS line, merge_turns).
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/run"
cat > "$TMP/bin/kubectl" <<'EOF'
#!/bin/bash
case "$*" in
  *"get pods -l app=vllm --field-selector=status.phase=Running"*) printf '%s' "${STUB_POD-vllm-7d9f-abcde}" ;;
esac
EOF
chmod +x "$TMP/bin/kubectl"
PATH="$TMP/bin:$PATH"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
# shellcheck disable=SC1091
source "$HERE/workload.env"

# --- Running vLLM pod -------------------------------------------------------------------
( export STUB_POD=""; source "$HERE/lib_load.sh" ) >/dev/null 2>&1; [ $? = 3 ] && ok || ko "no Running vLLM pod -> exit 3"
# shellcheck disable=SC1091
source "$HERE/lib_load.sh"
[ "$POD" = vllm-7d9f-abcde ] && ok || ko "POD is the Running pod ($POD)"

# --- expected requests: (1500 - 700) s x 40 sessions / 600 s ----------------------------
LEVEL_SECONDS=$LOAD_SECONDS
[ "$(expected_sent 40)" = 53 ] && ok || ko "expected_sent 40 = $(expected_sent 40)"

# --- merge of the two clients -------------------------------------------------------------
M=$(printf '%s\n' 'client 0: first request 9 s after the level started' \
  'client 0: RESULT sent=92 completed=92 errors=0 ttft_p50_ms=900 ttft_p95_ms=3100 itl_p95_ms=31.2 worst_session_ttft_p95_ms=4200 sessions_over_slo=0' \
  'client 1: RESULT sent=92 completed=91 errors=1 ttft_p50_ms=950 ttft_p95_ms=2950 itl_p95_ms=33.0 worst_session_ttft_p95_ms=5600 sessions_over_slo=1' \
  'client 1: minute  requests  ttft_p50_ms  ttft_p95_ms' | merge_results)
[ "$M" = "RESULT sent=184 completed=183 errors=1 ttft_p50_ms=950 ttft_p95_ms=3100 itl_p95_ms=33.0 worst_session_ttft_p95_ms=5600 sessions_over_slo=1" ] && ok || ko "merge: sums, worst latency, per-session report ($M)"
M=$(printf '%s\n' 'client 0: RESULT sent=92 completed=92 errors=0 ttft_p50_ms=900 ttft_p95_ms=10100 itl_p95_ms=31.2 worst_session_ttft_p95_ms=12000 sessions_over_slo=3' \
  'client 1: RESULT sent=92 completed=92 errors=0 ttft_p50_ms=950 ttft_p95_ms=2950 itl_p95_ms=9 worst_session_ttft_p95_ms=4000 sessions_over_slo=0' | merge_results)
[ "$M" = "RESULT sent=184 completed=184 errors=0 ttft_p50_ms=950 ttft_p95_ms=10100 itl_p95_ms=31.2 worst_session_ttft_p95_ms=12000 sessions_over_slo=3" ] && ok || ko "merge: numeric max, not string max ($M)"
M=$(printf '%s\n' 'client 0: RESULT sent=92 completed=92 errors=0 ttft_p50_ms=900 ttft_p95_ms=3100 itl_p95_ms=31.2 worst_session_ttft_p95_ms=4200 sessions_over_slo=0' \
  'client 1: RESULT sent=0 completed=0 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA worst_session_ttft_p95_ms=NA sessions_over_slo=0' | merge_results)
[ "$M" = "RESULT sent=92 completed=92 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA worst_session_ttft_p95_ms=NA sessions_over_slo=0" ] && ok || ko "merge: a client without latencies gives NA ($M)"
M=$(printf '%s\n' 'client 0: RESULT sent=92 completed=92 errors=0 ttft_p50_ms=900 ttft_p95_ms=3100 itl_p95_ms=31.2 worst_session_ttft_p95_ms=4200 sessions_over_slo=0' \
  'client 1: Traceback (most recent call last):' | merge_results)
[ -z "$M" ] && ok || ko "merge: one client without RESULT gives no merged line ($M)"

# --- the AIPerf command (DRY_RUN=1: printed, not run) ------------------------------------
O=$(DRY_RUN=1 run_level 40 1500 load-40 2>&1)
for f in "--model zai-org/GLM-5.3-Flash" "--url http://vllm.glm53-coding.svc:8000" \
         "--shared-system-prompt-length 20000" "--user-context-prompt-length 100000" "--conversation-turn-mean 10" \
         "--isl 4000" "--osl 400" "--extra-inputs ignore_eos:true" "--num-users 20 --user-centric-rate 0.0333" \
         "--benchmark-duration 1500" "--num-dataset-entries 40 " "--conversation-num 40 "; do
  [ "$(echo "$O" | grep -c -- "$f")" = 2 ] && ok || ko "aiperf command: '$f' in both clients"
done
# I3: one seed per client AND per level, another for the Hold: no level reuses another's conversations.
[ "$(seed_for ramp-12 0) $(seed_for ramp-12 1)" = "52120 52121" ] && ok || ko "seed ramp-12: $(seed_for ramp-12 0) $(seed_for ramp-12 1)"
[ "$(seed_for ramp-16 0)" = 52160 ] && ok || ko "seed ramp-16: $(seed_for ramp-16 0)"
[ "$(seed_for hold-12 0) $(seed_for hold-12 1)" = "57120 57121" ] && ok || ko "seed hold-12: $(seed_for hold-12 0) $(seed_for hold-12 1)"
# Study 53: one fixed load, the same seed in every experiment (vLLM restarts between experiments).
[ "$(seed_for load-40 0) $(seed_for load-40 1)" = "61400 61401" ] && ok || ko "seed load-40: $(seed_for load-40 0) $(seed_for load-40 1)"
echo "$O" | grep -q -- "--random-seed 61400" && echo "$O" | grep -q -- "--random-seed 61401" && ok || ko "aiperf command: the load's seeds"
echo "$O" | grep -q -- "--tokenizer /work/tokenizer" && ok || ko "aiperf command: pinned local tokenizer"
( DRY_RUN=1 run_level 5 960 odd ) >/dev/null 2>&1; [ $? != 0 ] && ok || ko "odd level refused"

# --- the per-client parser on a synthetic export -----------------------------------------
# 3 sessions x 100 requests, one request per second per session from t0, 5 warm-up rows to
# ignore. Session a: TTFT 1000 ms; session b: 2000 ms; session c: 3000 ms, but 6000 ms for its
# last 10 requests (its p95 over the steady window goes above 5000 ms). 1 error in the window.
python3 - "$TMP/run/profile_export.jsonl" <<'PY'
import json, sys
t0 = 1_800_000_000
with open(sys.argv[1], "w") as f:
    for i in range(5):
        f.write(json.dumps({"metadata": {"benchmark_phase": "warmup", "request_start_ns": int((t0 - 30 + i) * 1e9), "x_correlation_id": "w"},
                            "metrics": {"time_to_first_token": {"value": 9999}}}) + "\n")
    for s, base in (("a", 1000), ("b", 2000), ("c", 3000)):
        for i in range(100):
            ttft = 6000 if (s == "c" and i >= 90) else base
            row = {"metadata": {"benchmark_phase": "profiling", "request_start_ns": int((t0 + i) * 1e9), "x_correlation_id": s, "turn_index": i % 16},
                   "metrics": {"time_to_first_token": {"value": ttft}, "inter_token_latency": {"value": 30 + (i % 10)},
                               "output_token_count": {"value": 400}}}
            if s == "a" and i == 70: row = {"metadata": row["metadata"], "metrics": {}, "error": {"code": 500, "message": "boom"}}
            f.write(json.dumps(row) + "\n")
PY
O=$(python3 -c "${PARSER//\/work\/out/$TMP}" 60 1799999990 300 5000 2>&1)
echo "$O" | grep -qx 'first request 10 s after the level started' && ok || ko "parser: first request line"
# Steady window: requests 60..99 of each session = 120 sent, 119 completed, 1 error.
echo "$O" | grep -q '^RESULT sent=120 completed=119 errors=1 ttft_p50_ms=2000 ttft_p95_ms=6000 itl_p95_ms=39.0 worst_session_ttft_p95_ms=6000 sessions_over_slo=1$' && ok || ko "parser: RESULT line"
echo "$O" | grep -q '^first error: .*boom' && ok || ko "parser: first error"
echo "$O" | grep -q '^sessions in the steady window: 3$' && ok || ko "parser: session count"
# Study 53: first turns (turn_index 0, a cold prefill of the session's context) apart from resumed
# turns (>= 1, history from the GPU, the CPU offload, or recomputed), over the steady window.
# Window rows i = 60..99, turn_index = i % 16: first turns at i = 64, 80, 96 in each session (9:
# a, b x 3 at 1000 / 2000 ms, c at 3000, 3000, 6000 ms); resumed: the other 110 completed.
echo "$O" | grep -q '^TURNS first_n=9 first_ttft_p50_ms=2000 first_ttft_p95_ms=6000 resumed_n=110 resumed_ttft_p50_ms=2000 resumed_ttft_p95_ms=6000$' && ok || ko "parser: TURNS line"
echo "$O" | grep -q '^resumed 0-300 s: n=' && ok || ko "parser: resumed TTFT per 5 minutes"
T=$(printf '%s\n' 'client 0: TURNS first_n=3 first_ttft_p50_ms=11000 first_ttft_p95_ms=21000 resumed_n=23 resumed_ttft_p50_ms=1700 resumed_ttft_p95_ms=1750' \
  'client 1: TURNS first_n=3 first_ttft_p50_ms=11500 first_ttft_p95_ms=12000 resumed_n=23 resumed_ttft_p50_ms=950 resumed_ttft_p95_ms=11000' | merge_turns)
[ "$T" = "TURNS first_n=6 first_ttft_p50_ms=11500 first_ttft_p95_ms=21000 resumed_n=46 resumed_ttft_p50_ms=1700 resumed_ttft_p95_ms=11000" ] && ok || ko "merge_turns: sums, worse client ($T)"
T=$(printf '%s\n' 'client 0: TURNS first_n=0 first_ttft_p50_ms=NA first_ttft_p95_ms=NA resumed_n=23 resumed_ttft_p50_ms=1700 resumed_ttft_p95_ms=1750' \
  'client 1: TURNS first_n=1 first_ttft_p50_ms=11500 first_ttft_p95_ms=11500 resumed_n=23 resumed_ttft_p50_ms=950 resumed_ttft_p95_ms=1100' | merge_turns)
[ "$T" = "TURNS first_n=1 first_ttft_p50_ms=11500 first_ttft_p95_ms=11500 resumed_n=46 resumed_ttft_p50_ms=1700 resumed_ttft_p95_ms=1750" ] && ok || ko "merge_turns: a client without first turns ($T)"
[ -z "$(printf '%s\n' 'client 0: TURNS first_n=0 first_ttft_p50_ms=NA first_ttft_p95_ms=NA resumed_n=1 resumed_ttft_p50_ms=1 resumed_ttft_p95_ms=1' | merge_turns)" ] && ok || ko "merge_turns: one client only gives nothing"
: > "$TMP/run/profile_export.jsonl"
O=$(python3 -c "${PARSER//\/work\/out/$TMP}" 60 1799999990 300 5000 2>&1)
[ "$O" = "RESULT sent=0 completed=0 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA worst_session_ttft_p95_ms=NA sessions_over_slo=0
TURNS first_n=0 first_ttft_p50_ms=NA first_ttft_p95_ms=NA resumed_n=0 resumed_ttft_p50_ms=NA resumed_ttft_p95_ms=NA" ] && ok || ko "parser: empty export"
[ $FAIL = 0 ] || echo "$O"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
