"""How much of the previous round's context TraceLab's recorded prefix keeps, by the gap before
the round. continuity = min(input_i, input_{i-1} + output_{i-1}) (the history the agent re-sent,
unless it compacted); recorded = prefix_tokens (what the provider's cache served).

Usage: python3 -I prefix_vs_gap.py TRACE.jsonl.gz
"""
import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime

INPUT_EV = {"user_message", "tool_result"}
BUCKETS = [(0, 10), (10, 60), (60, 300), (300, 600), (600, 1800), (1800, 3600), (3600, 1e12)]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main():
    sessions = defaultdict(list)
    for line in gzip.open(sys.argv[1], "rt"):
        r = json.loads(line)
        ev = r.get("timing_events") or []
        ins = [ts(e["timestamp"]) for e in ev if e.get("event_type") in INPUT_EV and e.get("timestamp")]
        outs = [ts(e["timestamp"]) for e in ev if e.get("event_type") not in INPUT_EV and e.get("timestamp")]
        sessions[r["session_id"]].append((r.get("round_index", 0), r.get("provider"), r.get("input_tokens_total") or 0,
                                          r.get("output_tokens") or 0, r.get("prefix_tokens") or 0,
                                          max(ins) if ins else None, max(outs) if outs else None))
    agg = defaultdict(lambda: [0, 0, 0])  # (provider, bucket) -> rounds, continuity tokens, recorded tokens
    for rs in sessions.values():
        rs.sort()
        for a, b in zip(rs, rs[1:]):
            if not (a[6] and b[5]) or b[5] < a[6]:
                continue
            gap = b[5] - a[6]
            cont = min(b[2], a[2] + a[3])
            if b[2] < 0.7 * a[2]:
                continue  # compaction / reset: the history really changed
            for lo, hi in BUCKETS:
                if lo <= gap < hi:
                    k = (b[1], f"{lo:>5.0f}-{hi if hi < 1e11 else 'inf'}s")
                    agg[k][0] += 1; agg[k][1] += cont; agg[k][2] += min(b[4], cont)
    print("provider gap           rounds  recorded prefix / continuity prefix")
    for k in sorted(agg, key=lambda k: (k[0], float(k[1].split('-')[0]))):
        n, c, rec = agg[k]
        print(f"{k[0]:8s} {k[1]:14s} {n:7d}  {100 * rec / c if c else 0:5.1f}%")


if __name__ == "__main__":
    main()
