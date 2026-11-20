#!/bin/bash
# Local test of prepare_pool.py on a synthetic TraceLab fixture (no network, no cluster).
# Fixture: s1-s3 three normal rounds; s4 one round (not eligible: < 2 rounds); s5 a round of 1,100
# tokens (dropped: under K + 200 with K = 1,000); s6 whose third round (250,000 in + 12,000 out)
# exceeds the 262,144 max model length (cut before it). Kept: s1, s2, s3, s6 -> two pools of two.
# Usage: bash load/test_prepare_pool.sh   → "N passed, M failed", exit 1 on any failure.
HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); }
bad() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

python3 -I - "$TMP/trace.jsonl.gz" <<'PY'
import gzip, json, sys
def rnd(sid, i, t, inp, out, prefix=0):
    ts = "2026-01-01T00:%02d:%02dZ" % (t // 60, t % 60)
    te = "2026-01-01T00:%02d:%02dZ" % ((t + 5) // 60, (t + 5) % 60)
    return {"session_id": sid, "round_index": i, "input_tokens_total": inp, "output_tokens": out,
            "prefix_tokens": prefix, "newly_append_tokens": inp - prefix,
            "timing_events": [{"event_type": "user_message" if i == 0 else "tool_result", "timestamp": ts},
                              {"event_type": "api_response", "timestamp": te}]}
rows = []
for s in ("s1", "s2", "s3"):
    rows += [rnd(s, 0, 0, 5000, 100), rnd(s, 1, 10, 6000, 100, 4000), rnd(s, 2, 20, 7000, 100, 6000)]
rows += [rnd("s4", 0, 0, 5000, 100)]
rows += [rnd("s5", 0, 0, 1100, 100), rnd("s5", 1, 10, 3000, 100, 1100)]
rows += [rnd("s6", 0, 0, 200000, 100), rnd("s6", 1, 10, 220000, 100, 200000),
         rnd("s6", 2, 20, 250000, 12000, 220000)]
with gzip.open(sys.argv[1], "wt") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
PY

OUT=$(python3 -I "$HERE/prepare_pool.py" "$TMP/trace.jsonl.gz" "$TMP/a" --sessions 2 --pools 2 --sys-tokens 1000 2>&1); RC=$?
[ "$RC" = 0 ] && ok || bad "prepare_pool.py exit $RC: $OUT"
echo "$OUT" | grep -q "kept 4" && ok || bad "expected 'kept 4' in: $OUT"
python3 -I "$HERE/prepare_pool.py" "$TMP/trace.jsonl.gz" "$TMP/b" --sessions 2 --pools 2 --sys-tokens 1000 >/dev/null 2>&1

CHECK=$(python3 -I - "$TMP" <<'PY'
import gzip, json, os, sys
d = sys.argv[1]
rec = {}
for l in gzip.open(os.path.join(d, "trace.jsonl.gz"), "rt"):
    r = json.loads(l); rec[(r["session_id"], r["round_index"])] = r["input_tokens_total"]
pools = []
for k in (0, 1):
    p = os.path.join(d, "a_%d.jsonl.gz" % k)
    rows = [json.loads(l) for l in gzip.open(p, "rt")] if os.path.exists(p) else []
    pools.append(rows)
sids = [sorted({r["session_id"] for r in rows}) for rows in pools]
print("sessions", len(sids[0]), len(sids[1]))
print("disjoint", not set(sids[0]) & set(sids[1]))
allrows = pools[0] + pools[1]
print("k_subtracted", all(r["input_tokens_total"] == rec[(r["session_id"], r["round_index"])] - 1000 for r in allrows))
print("prefix_floor", all(r["prefix_tokens"] >= 0 for r in allrows))
print("s5_dropped", "s5" not in sids[0] + sids[1])
print("s6_rounds", sum(1 for r in allrows if r["session_id"] == "s6"))
PY
)
for want in "sessions 2 2" "disjoint True" "k_subtracted True" "prefix_floor True" "s5_dropped True" "s6_rounds 2"; do
  echo "$CHECK" | grep -qx "$want" && ok || bad "expected '$want' in: $(echo "$CHECK" | tr '\n' ';')"
done
for k in 0 1; do
  A=$(gzip -dc "$TMP/a_$k.jsonl.gz" 2>/dev/null | shasum -a 256); B=$(gzip -dc "$TMP/b_$k.jsonl.gz" 2>/dev/null | shasum -a 256)
  [ -n "$A" ] && [ "$A" = "$B" ] && ok || bad "pool $k not reproducible across runs"
done

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
