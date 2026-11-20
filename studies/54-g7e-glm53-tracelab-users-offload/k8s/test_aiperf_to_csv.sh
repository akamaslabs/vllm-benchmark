#!/bin/bash
# Local tests of aiperf_to_csv.py and score_offline.py on synthetic AIPerf 0.13 records (no cluster).
# Usage: bash k8s/test_aiperf_to_csv.sh   → "N passed, M failed", exit 1 on any failure.
HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); }
bad() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# gen.py MODE OUTDIR: two clients' exports and pools.
#   ramp:   one request every 3 s per client, one new session per client per minute (never finished),
#           TTFT 1000 ms, ITL 20 ms, except ITL 80 ms for requests ending in minutes 10-11; 30 minutes.
#   sparse: the same, but the first 6 minutes send one request per 30 s per client with ITL 80 ms
#           (under 20 requests in any 150 s window), then the normal rate with ITL 20 ms.
#   users:  client 0 only: A (2 pool rows) at +31 s and +100 s (a pause), B (3 rows) at +40 and +45,
#           C (3 rows) at +140, +170 and +185 — users at the bucket ends +90/+120/+150/+180 = 2/1/2/2.
#   twice:  ramp, plus client 0's first session started a second time (turn_index 0 again).
cat > "$TMP/gen.py" <<'PY'
import gzip, json, os, sys
mode, out = sys.argv[1], sys.argv[2]
T0 = 1_800_000_000  # a multiple of 30 s
def rec(sid, ti, start, dur=2.0, ttft=1000.0, itl=20.0, outer=None, err=None):
    r = {"metadata": {"conversation_id": sid, "turn_index": ti, "benchmark_phase": "profiling",
                      "request_start_ns": int(start * 1e9), "request_end_ns": int((start + dur) * 1e9),
                      "source_outer_idx": ti if outer is None else outer, "agent_depth": 0, "was_cancelled": False},
         "metrics": {"time_to_first_token": {"value": ttft, "unit": "ms"},
                     "inter_token_latency": {"value": itl, "unit": "ms"},
                     "request_latency": {"value": dur * 1000, "unit": "ms"},
                     "input_sequence_length": {"value": 20000, "unit": "tokens"},
                     "output_sequence_length": {"value": 300, "unit": "tokens"}}}
    if err:
        r["error"] = {"code": 500, "message": err}
    return r
exports = {0: [], 1: []}; pools = {0: {}, 1: {}}
if mode in ("ramp", "sparse", "twice"):
    for c in (0, 1):
        turns = {}
        t = 5.0
        while t < 30 * 60:
            minute = int(t // 60)
            sid = "c%d_s%02d" % (c, minute)
            ti = turns.get(sid, 0); turns[sid] = ti + 1
            end_min = (t + 2) / 60
            itl = 80.0 if 10 <= end_min < 12 else 20.0
            step = 3.0
            if mode == "sparse":
                itl = 80.0 if t < 360 else 20.0
                step = 30.0 if t < 360 else 3.0
            exports[c].append(rec(sid, ti, T0 + t + c * 0.5, itl=itl))
            pools[c][sid] = 1000
            t += step
    if mode == "twice":
        exports[0].append(rec("c0_s00", 0, T0 + 20 * 60 + 1.0))
elif mode == "users":
    exports[0] = [rec("A", 0, T0 + 31), rec("A", 1, T0 + 100), rec("B", 0, T0 + 40), rec("B", 1, T0 + 45),
                  rec("C", 0, T0 + 140), rec("C", 1, T0 + 170), rec("C", 2, T0 + 185)]
    pools[0] = {"A": 2, "B": 3, "C": 3}
os.makedirs(out, exist_ok=True)
for c in (0, 1):
    if not exports[c] and c == 1:
        continue
    with open(os.path.join(out, "export_c%d.jsonl" % c), "w") as f:
        for r in exports[c]:
            f.write(json.dumps(r) + "\n")
    with gzip.open(os.path.join(out, "pool_c%d.jsonl.gz" % c), "wt") as f:
        for sid, n in pools[c].items():
            for i in range(n):
                f.write(json.dumps({"session_id": sid, "round_index": i}) + "\n")
PY
CSV="$HERE/aiperf_to_csv.py"; SCORE="$HERE/score_offline.py"
run2() {  # run2 DIR [args...] -> OUT, RC (both clients)
  local d=$1; shift
  OUT=$(python3 -I "$CSV" --pair "$d/export_c0.jsonl" "$d/pool_c0.jsonl.gz" --pair "$d/export_c1.jsonl" "$d/pool_c1.jsonl.gz" \
        --out "$d/cut.csv" --full-out "$d/full.csv" "$@" 2>&1); RC=$?
}
col() {  # col CSV COLUMN -> the column's values, one per line
  python3 -I -c 'import csv,sys; [print(r[sys.argv[2]]) for r in csv.DictReader(open(sys.argv[1]))]' "$1" "$2"
}
ts_of_minute() { python3 -I -c 'import datetime,sys; print(datetime.datetime.fromtimestamp(1_800_000_000 + int(float(sys.argv[1]) * 60), datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))' "$1"; }

# 1. The ITL breach in minutes 10-11 (8 bad buckets with the 150 s window) cuts the CSV before it;
#    the full CSV keeps every bucket; the cut scores lower than the full one (later windows recover).
python3 -I "$TMP/gen.py" ramp "$TMP/ramp"
run2 "$TMP/ramp"
[ "$RC" = 0 ] && ok || bad "ramp: exit $RC: $OUT"
echo "$OUT" | grep -q '^knee: bucket .* (ITL)' && ok || bad "ramp: knee line missing or wrong metric: $OUT"
# The first request with ITL 80 ms ends at minute 10:01, so the first bad bucket is (10:00, 10:30]:
# the knee names it and the cut CSV ends with the bucket ending at 10:00.
echo "$OUT" | grep -q "^knee: bucket $(ts_of_minute 10.5) " && ok || bad "ramp: knee not at $(ts_of_minute 10.5): $OUT"
LAST=$(col "$TMP/ramp/cut.csv" TS | tail -1)
[ "$LAST" = "$(ts_of_minute 10)" ] && ok || bad "ramp: cut CSV ends at $LAST, expected $(ts_of_minute 10)"
[ "$(col "$TMP/ramp/full.csv" TS | wc -l)" -gt "$(col "$TMP/ramp/cut.csv" TS | wc -l)" ] && ok || bad "ramp: full CSV not longer than the cut one"
[ "$(col "$TMP/ramp/full.csv" TS | tail -1)" \> "$(ts_of_minute 25)" ] && ok || bad "ramp: full CSV does not reach minute 25"
head -1 "$TMP/ramp/cut.csv" | tr -d '\r' | grep -q '^TS,COMPONENT,aiperf_request_success_rate,.*,aiperf_active_sessions$' && ok || bad "ramp: header: $(head -1 "$TMP/ramp/cut.csv")"
S_CUT=$(python3 -I "$SCORE" "$TMP/ramp/cut.csv" | awk '/^score/ {print $2}')
S_FULL=$(python3 -I "$SCORE" "$TMP/ramp/full.csv" | awk '/^score/ {print $2}')
awk -v a="$S_CUT" -v b="$S_FULL" 'BEGIN { exit !(a > 0 && b > a) }' && ok || bad "score: cut $S_CUT should be > 0 and below full $S_FULL"
# The same breach with --breach-buckets 9 (longer than the 8 bad buckets): no cut.
run2 "$TMP/ramp" --breach-buckets 9
echo "$OUT" | grep -q '^knee: none' && ok || bad "ramp, 9 buckets: expected 'knee: none': $OUT"
[ "$(col "$TMP/ramp/cut.csv" TS | wc -l)" = "$(col "$TMP/ramp/full.csv" TS | wc -l)" ] && ok || bad "ramp, 9 buckets: the CSV was cut"
# A tighter TTFT SLO (500 ms < every TTFT) breaks from the first counted bucket: TTFT named.
run2 "$TMP/ramp" --slo-ttft-ms 500
echo "$OUT" | grep -q '^knee: bucket .* (TTFT' && ok || bad "ramp, TTFT 500: knee not on TTFT: $OUT"

# 2. Bad ITL while windows hold under 20 requests never counts.
python3 -I "$TMP/gen.py" sparse "$TMP/sparse"
run2 "$TMP/sparse"
echo "$OUT" | grep -q '^knee: none' && ok || bad "sparse: expected 'knee: none': $OUT"

# 3. A session started twice by one client (the pool ran out): exit 3, session named.
python3 -I "$TMP/gen.py" twice "$TMP/twice"
run2 "$TMP/twice"
[ "$RC" = 3 ] && ok || bad "twice: exit $RC, expected 3: $OUT"
echo "$OUT" | grep -q 'c0_s00' && ok || bad "twice: the session is not named: $OUT"

# 4. Users: started minus finished (all pool rows sent and answered), a pause still counts.
python3 -I "$TMP/gen.py" users "$TMP/users"
OUT=$(python3 -I "$CSV" --pair "$TMP/users/export_c0.jsonl" "$TMP/users/pool_c0.jsonl.gz" --out "$TMP/users/cut.csv" \
      --full-out "$TMP/users/full.csv" --min-window-reqs 1000 2>&1); RC=$?
[ "$RC" = 0 ] && ok || bad "users: exit $RC: $OUT"
U=$(col "$TMP/users/full.csv" aiperf_active_sessions | tr '\n' ' ')
[ "$U" = "2 1 2 2 " ] && ok || bad "users: active sessions '$U', expected '2 1 2 2 '"

# 5. No usable record: exit 2.
: > "$TMP/empty.jsonl"
python3 -I "$CSV" --pair "$TMP/empty.jsonl" "$TMP/users/pool_c0.jsonl.gz" --out "$TMP/e.csv" --full-out "$TMP/e2.csv" >/dev/null 2>&1
[ $? = 2 ] && ok || bad "empty export: exit not 2"

# 6. score_offline.py: a CSV with no valid window scores none.
printf 'TS,COMPONENT,aiperf_time_to_first_token_p95_150s,aiperf_inter_token_latency_p95_150s,aiperf_active_sessions\n' > "$TMP/s.csv"
for i in 1 2 3 4 5 6 7; do printf '2027-01-15 08:0%d:00,aiperf,9000,20,%d\n' "$i" "$i" >> "$TMP/s.csv"; done
python3 -I "$SCORE" "$TMP/s.csv" | grep -qx 'score none' && ok || bad "score: TTFT 9 s everywhere must score none"
sed -i.bak 's/,9000,/,1000,/' "$TMP/s.csv"
OUT=$(python3 -I "$SCORE" "$TMP/s.csv")
[[ "$OUT" == "score 4.5 window 2027-01-15 08:02:00..2027-01-15 08:07:00" ]] && ok || bad "score: '$OUT', expected 4.5 over rows 2-7"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
