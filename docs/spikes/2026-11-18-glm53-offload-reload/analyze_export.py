"""Per-request analysis of one AIPerf client's export kept by spike.sh (/work/keep-<tag>-<off>.jsonl).

Usage (inside an AIPerf client pod): python3 - FILE STEADY_FROM PERIOD < analyze_export.py
Prints, for first turns (turn_index 0: a session's context is prefilled cold) and revisits
(turn_index >= 1: the history is cached on GPU, in RAM, or recomputed), the count and TTFT p50/p95
over the whole run and over requests started >= STEADY_FROM s after the first request; the first
turns started after the creation phase (> PERIOD s: sessions AIPerf restarted, each a cold prefill);
and revisit TTFT per 5-minute bin.
"""
import json
import sys

path, steady, period = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = []
for line in open(path):
    r = json.loads(line)
    m, x = r["metadata"], r["metrics"]
    ttft = x.get("time_to_first_token", {}).get("value")
    rows.append((m["request_start_ns"], m["turn_index"], m["conversation_id"], ttft,
                 x.get("input_sequence_length", {}).get("value"), r.get("error")))
t0 = min(r[0] for r in rows)
rows = [((s - t0) / 1e9, ti, c, ttft, isl, err) for s, ti, c, ttft, isl, err in rows]


def q(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] / 1000, 1) if v else None


def show(name, sel):
    t = [r[3] for r in sel if r[3] is not None]
    print(f"{name}: n={len(sel)} ttft_s p50={q(t, .5)} p95={q(t, .95)} max={q(t, 1)} errors={sum(1 for r in sel if r[5])}")


print(f"requests {len(rows)}, conversations {len({r[2] for r in rows})}, last start {max(r[0] for r in rows):.0f} s")
show("first turns, all", [r for r in rows if r[1] == 0])
show("revisits, all", [r for r in rows if r[1] >= 1])
show(f"first turns >= {steady:.0f} s", [r for r in rows if r[1] == 0 and r[0] >= steady])
show(f"revisits >= {steady:.0f} s", [r for r in rows if r[1] >= 1 and r[0] >= steady])
late = [r for r in rows if r[1] == 0 and r[0] > period]
print(f"first turns started after {period:.0f} s (restarted sessions): {len(late)}")
for b in range(0, int(max(r[0] for r in rows)) + 1, 300):
    sel = [r for r in rows if r[1] >= 1 and b <= r[0] < b + 300]
    if sel:
        show(f"  revisits {b:4d}-{b + 300:4d} s", sel)
