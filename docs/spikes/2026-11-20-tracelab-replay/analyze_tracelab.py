"""Shape of the TraceLab coding trace for a vLLM long-context replay.

Usage: python3 -I analyze_tracelab.py TRACE.jsonl.gz
Per round: input_tokens_total, output_tokens, the gap since the previous round's end (send time =
latest input event, end = latest output event), user- vs tool-initiated. Per session: rounds,
max context, first-round size.
"""
import gzip
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime

INPUT_EV = {"user_message", "tool_result"}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def pct(xs, ps=(50, 90, 99)):
    if not xs:
        return "n=0"
    xs = sorted(xs)
    out = [f"n={len(xs)}"]
    for p in ps:
        out.append(f"p{p}={xs[min(len(xs) - 1, int(len(xs) * p / 100))]:,.0f}")
    out.append(f"max={xs[-1]:,.0f} mean={statistics.fmean(xs):,.0f}")
    return " ".join(out)


def main():
    sessions = defaultdict(list)
    for line in gzip.open(sys.argv[1], "rt"):
        r = json.loads(line)
        ev = r.get("timing_events") or []
        ins = [ts(e["timestamp"]) for e in ev if e.get("event_type") in INPUT_EV and e.get("timestamp")]
        outs = [ts(e["timestamp"]) for e in ev if e.get("event_type") not in INPUT_EV and e.get("timestamp")]
        sessions[r["session_id"]].append({
            "i": r.get("round_index", 0), "prov": r.get("provider"),
            "inp": r.get("input_tokens_total") or 0, "out": r.get("output_tokens") or 0,
            "app": r.get("newly_append_tokens") or 0,
            "first": r.get("first_input_event_type"),
            "send": max(ins) if ins else None, "end": max(outs) if outs else None,
        })
    prov = defaultdict(int)
    rounds_per, max_ctx, first_inp, inp, out, gap_user, gap_tool, drops = [], [], [], [], [], [], [], 0
    over262 = 0
    for sid, rs in sessions.items():
        rs.sort(key=lambda x: x["i"])
        prov[rs[0]["prov"]] += 1
        rounds_per.append(len(rs))
        max_ctx.append(max(x["inp"] for x in rs))
        first_inp.append(rs[0]["inp"])
        for a, b in zip(rs, rs[1:]):
            if b["inp"] < a["inp"] * 0.7:
                drops += 1
            if a["end"] and b["send"] and b["send"] >= a["end"]:
                (gap_user if b["first"] == "user_message" else gap_tool).append(b["send"] - a["end"])
        for x in rs:
            inp.append(x["inp"]); out.append(x["out"])
            over262 += x["inp"] > 262144
    print(f"sessions={len(sessions)} by provider={dict(prov)} rounds={len(inp)}")
    print("rounds per session      ", pct(rounds_per, (50, 75, 90, 99)))
    print("max context per session ", pct(max_ctx, (50, 75, 90, 99)))
    print("first-round input       ", pct(first_inp, (50, 75, 90, 99)))
    print("input tokens per round  ", pct(inp, (50, 75, 90, 99)))
    print(f"rounds over 262144 input tokens: {over262} ({100 * over262 / len(inp):.1f}%)")
    print("output tokens per round ", pct(out, (50, 75, 90, 99)))
    print("gap before tool round, s", pct(gap_tool, (50, 75, 90, 99)))
    print("gap before user round, s", pct(gap_user, (50, 75, 90, 99)))
    print(f"context drops > 30% (compaction / reset): {drops}")
    multi = [m for m, n in zip(max_ctx, rounds_per) if n >= 20]
    print("max context, sessions >= 20 rounds", pct(multi, (50, 75, 90, 99)))
    fit = sum(1 for m, n in zip(max_ctx, rounds_per) if n >= 20 and m <= 262144)
    print(f"sessions >= 20 rounds whose max context <= 262144: {fit}")


if __name__ == "__main__":
    main()
