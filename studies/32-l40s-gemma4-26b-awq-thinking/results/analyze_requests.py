"""Per-request analysis of an AIPerf run: where the e2e time goes, with thinking on.

Usage: python3 analyze_requests.py <profile_export.jsonl> [--ramp RATE SECONDS] [--json out.json]

Reads AIPerf's per-request records (profile_export.jsonl, written while the run goes on) and
prints, as plain tables:
  1. the load's shape: ISL, OSL split into reasoning and answer tokens, per turn of the
     conversation (the history makes the ISL grow turn after turn);
  2. the e2e time and its three phases, per request:
        TTFT                 queue + prefill, until the first token (reasoning included)
        reasoning phase      TTFT -> first answer token (time_to_first_output_token - TTFT)
        answer phase         first answer token -> last token (request_latency - TTFOT)
     and how much of the e2e variance the OSL explains (r^2) against the ISL;
  3. the same percentiles per bucket of offered load (from the open-loop ramp: the offered
     rate at time t is RATE * t / SECONDS), to see what queueing adds on top of the lengths;
  4. how noisy a p95 is over a 150 s window (what the study's constraint reads): the p95 of
     every 150 s window at low load, and how many requests each one rests on.

Percentiles are exact here (one value per request), unlike Prometheus' histogram buckets.
Every metric name is AIPerf 0.11.0's (records' "metrics" / "metadata" keys).
"""
import argparse
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("records")
ap.add_argument("--ramp", nargs=2, type=float, default=[2.0, 6000.0], metavar=("RATE", "SECONDS"))
ap.add_argument("--json", help="write the per-request table (for charts) to this file")
a = ap.parse_args()
RATE, RAMP_S = a.ramp

rows = []
for line in open(a.records):
    r = json.loads(line)
    md, m = r["metadata"], r.get("metrics", {})
    if r.get("error") or md.get("was_cancelled") or md.get("benchmark_phase") != "profiling":
        continue

    def g(k):
        return (m.get(k) or {}).get("value")
    if g("request_latency") is None or g("output_sequence_length") is None:
        continue
    rows.append({
        "start_s": md["request_start_ns"] / 1e9, "turn": md["turn_index"],
        "isl": g("input_sequence_length"), "osl": g("output_sequence_length"),
        "reasoning": g("reasoning_token_count") or 0, "answer": g("output_token_count") or 0,
        "e2e_s": g("request_latency") / 1000, "ttft_s": g("time_to_first_token") / 1000,
        "ttfot_s": (g("time_to_first_output_token") or g("time_to_first_token")) / 1000,
        "itl_ms": g("inter_token_latency"),
    })
t0 = min(r["start_s"] for r in rows)
for r in rows:
    r["t_s"] = r["start_s"] - t0
    r["offered"] = RATE * r["t_s"] / RAMP_S
    r["reasoning_s"] = max(r["ttfot_s"] - r["ttft_s"], 0.0)
    r["answer_s"] = max(r["e2e_s"] - r["ttfot_s"], 0.0)


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def line(name, v, fmt="%8.1f"):
    v = [x for x in v if x is not None]
    cells = [statistics.mean(v)] + [pct(v, q) for q in (.5, .9, .95, .99)] + [max(v)]
    print("  %-26s n=%5d " % (name, len(v)) + " ".join(fmt % c for c in cells))


def corr(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** .5


HDR = "  %-26s %7s " % ("", "") + " ".join("%8s" % h for h in ("mean", "p50", "p90", "p95", "p99", "max"))
low = [r for r in rows if r["offered"] < 0.2]
span = max(r["t_s"] for r in rows) / 60
print("%d requests over %.0f min of ramp (offered 0 -> %.2f req/s); low load = offered < 0.2 req/s"
      " (%d requests)" % (len(rows), span, max(r["offered"] for r in rows), len(low)))

print("\n1. Load shape at low load (tokens)")
print(HDR)
for k in ("isl", "osl", "reasoning", "answer"):
    line(k, [r[k] for r in low])
print("  reasoning share of the output tokens: %.0f %%; ISL:OSL (means) = %.1f : 1"
      % (100 * sum(r["reasoning"] for r in low) / sum(r["osl"] for r in low),
         statistics.mean(r["isl"] for r in low) / statistics.mean(r["osl"] for r in low)))
print("  per turn (history included in the ISL):")
for t in sorted({r["turn"] for r in low}):
    s = [r for r in low if r["turn"] == t]
    print("    turn %d: n=%4d  ISL p50 %6d p90 %6d   OSL p50 %5d p90 %5d" % (
        t + 1, len(s), pct([r["isl"] for r in s], .5), pct([r["isl"] for r in s], .9),
        pct([r["osl"] for r in s], .5), pct([r["osl"] for r in s], .9)))

print("\n2. e2e and its phases at low load (seconds; ITL in ms)")
print(HDR)
for k in ("e2e_s", "ttft_s", "reasoning_s", "answer_s", "ttfot_s"):
    line(k, [r[k] for r in low])
line("itl_ms", [r["itl_ms"] for r in low])
print("  share of the e2e (sum over requests): TTFT %.0f %%, reasoning %.0f %%, answer %.0f %%" % tuple(
    100 * sum(r[k] for r in low) / sum(r["e2e_s"] for r in low) for k in ("ttft_s", "reasoning_s", "answer_s")))
print("  e2e explained (r^2): by the OSL %.2f, by the ISL %.2f" % (
    corr([r["osl"] for r in low], [r["e2e_s"] for r in low]) ** 2,
    corr([r["isl"] for r in low], [r["e2e_s"] for r in low]) ** 2))
slow = [r for r in low if r["e2e_s"] > pct([x["e2e_s"] for x in low], .9)]
print("  the slowest 10 %% (e2e > p90): OSL mean %d (all: %d), reasoning mean %d (all: %d), ISL mean %d (all: %d)" % (
    statistics.mean(r["osl"] for r in slow), statistics.mean(r["osl"] for r in low),
    statistics.mean(r["reasoning"] for r in slow), statistics.mean(r["reasoning"] for r in low),
    statistics.mean(r["isl"] for r in slow), statistics.mean(r["isl"] for r in low)))

print("\n3. Per offered-load bucket (seconds; ITL ms)")
print("  %-11s %5s %7s %7s %7s %8s %8s %7s %7s %7s" % ("offered", "n", "e2e p50", "e2e p95", "TTFT p95",
                                                         "reas p50", "ans p50", "ITL p50", "OSL p50", "OSL p95"))
edges = [0, .1, .2, .3, .4, .5, .6, .8, 1.0, 1.2, 1.5, 2.0]
for lo, hi in zip(edges, edges[1:]):
    s = [r for r in rows if lo <= r["offered"] < hi]
    if len(s) < 10:
        continue
    print("  %4.1f-%-6.1f %5d %7.1f %7.1f %8.2f %8.1f %8.1f %7.1f %7d %7d" % (
        lo, hi, len(s), pct([r["e2e_s"] for r in s], .5), pct([r["e2e_s"] for r in s], .95),
        pct([r["ttft_s"] for r in s], .95), pct([r["reasoning_s"] for r in s], .5),
        pct([r["answer_s"] for r in s], .5), pct([r["itl_ms"] for r in s], .5),
        pct([r["osl"] for r in s], .5), pct([r["osl"] for r in s], .95)))

print("\n4. p95 of e2e over 150 s windows (by completion time), offered < 0.4 req/s")
done = sorted((r["t_s"] + r["e2e_s"], r) for r in rows if r["offered"] < 0.4)
w0 = done[0][0]
while w0 + 150 <= done[-1][0]:
    s = [r for t, r in done if w0 <= t < w0 + 150]
    if len(s) >= 5:
        print("  window at %5.1f min: %3d requests, e2e p95 %5.1f s, max OSL %5d" % (
            w0 / 60, len(s), pct([r["e2e_s"] for r in s], .95), max(r["osl"] for r in s)))
    w0 += 150

if a.json:
    json.dump(rows, open(a.json, "w"))
