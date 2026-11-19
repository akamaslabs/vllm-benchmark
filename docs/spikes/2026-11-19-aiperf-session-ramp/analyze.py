"""Read the stub's request log: sessions opened over time and the gap between a session's turns.

Usage: python3 -I analyze.py LOGFILE [BUCKET_S]
"""
import json
import statistics
import sys
from collections import defaultdict


def main():
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    bucket = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    rows.sort(key=lambda r: r["t"])
    t0 = rows[0]["t"]
    first_seen, turns = {}, defaultdict(list)
    for r in rows:
        first_seen.setdefault(r["sid"], r["t"] - t0)
        turns[r["sid"]].append((r["t"] - t0, r["turn"]))
    end = rows[-1]["t"] - t0
    print(f"requests={len(rows)} sessions={len(first_seen)} span={end:.0f}s")
    b = 0.0
    while b <= end:
        opened = sum(1 for v in first_seen.values() if v < b + bucket)
        new = sum(1 for v in first_seen.values() if b <= v < b + bucket)
        reqs = sum(1 for r in rows if b <= r["t"] - t0 < b + bucket)
        print(f"  {b:6.0f}-{b + bucket:<6.0f}s open={opened:4d} new={new:3d} requests={reqs:4d}")
        b += bucket
    gaps = [b2[0] - a[0] for v in turns.values() for a, b2 in zip(v, v[1:])]
    if gaps:
        print(f"turn gaps: n={len(gaps)} min={min(gaps):.1f} median={statistics.median(gaps):.1f} max={max(gaps):.1f} s")
    first_turns = [v[0][1] for v in turns.values()]
    print(f"turn index of each session's first request: {sorted(set(first_turns))}")


if __name__ == "__main__":
    main()
