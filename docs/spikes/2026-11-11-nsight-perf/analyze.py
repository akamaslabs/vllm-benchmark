#!/usr/bin/env python3
"""Client-side numbers per time window from AIPerf's profile_export.jsonl files.

usage: analyze.py <window-seconds> <HH:MM:SS UTC start> <windows> <jsonl> [<jsonl> ...]
Requests are assigned to the window they started in. Prints one row per window:
requests, requests/s, TTFT p50 and p95 (ms), request latency p50 and p95 (ms), output tokens/s.
"""
import datetime, json, sys

def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]

win, start, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
today = datetime.datetime.now(datetime.timezone.utc).date()
h, m, s = (int(x) for x in start.split(":"))
t0 = datetime.datetime(today.year, today.month, today.day, h, m, s, tzinfo=datetime.timezone.utc).timestamp()
rows = [dict(ttft=[], lat=[], out=0, n=0, err=0) for _ in range(n)]
for path in sys.argv[4:]:
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        md, mt = r.get("metadata", {}), r.get("metrics", {})
        i = int((md.get("request_start_ns", 0) / 1e9 - t0) // win)
        if not 0 <= i < n:
            continue
        if "time_to_first_token" not in mt:
            rows[i]["err"] += 1
            continue
        rows[i]["n"] += 1
        rows[i]["ttft"].append(mt["time_to_first_token"]["value"])
        rows[i]["lat"].append(mt["request_latency"]["value"])
        rows[i]["out"] += mt.get("output_sequence_length", {}).get("value", 0)
print("window_start  reqs  err  req/s  ttft_p50  ttft_p95  lat_p50  lat_p95  out_tok/s")
for i, r in enumerate(rows):
    ts = datetime.datetime.fromtimestamp(t0 + i * win, datetime.timezone.utc).strftime("%H:%M:%S")
    print(f"{ts}  {r['n']:5d} {r['err']:4d} {r['n'] / win:6.2f} {pct(r['ttft'], 50):9.0f} {pct(r['ttft'], 95):9.0f}"
          f" {pct(r['lat'], 50):8.0f} {pct(r['lat'], 95):8.0f} {r['out'] / win:10.0f}")
