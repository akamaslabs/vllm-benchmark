"""TraceLab pools for study 54 (the calibration's pools, 2026-11-20, are reproduced bit for bit).

prepare_tracelab.py's rewrite (history continuity, cut at --max-input, >= --min-rounds rounds,
fixed-seed shuffle), plus:
- horizon cut: a session ends once the rounds it could reach within --horizon-min minutes of
  replay are written (capped think time + a FAST service bound: new tokens at 20k tok/s, output
  at 200 tok/s). Lossless for an experiment that ends by then; AIPerf stores every round's full
  message list, so the dataset grows with the square of the rounds and the unreachable tail of
  multi-day sessions would dominate it.
- disjoint pools: --pools K writes K files OUT_0..OUT_{K-1} of --sessions each from one shuffle.
- shared system prompt (--sys-tokens K): AIPerf's --system-prompt-file adds one identical system
  message of K tokens to every round, so K is subtracted from every round's input_tokens_total and
  prefix_tokens (floored at 0) AT WRITE TIME, after every cut ran on the recorded totals; sessions
  with a kept round under K + 200 tokens are dropped before the shuffle (the loader would clamp it).
Usage: python3 -I prepare_pool.py IN.jsonl.gz OUT_PREFIX [--sessions N] [--pools K]
       [--seed S] [--horizon-min H] [--cap-s C] [--sys-tokens K]
"""
import argparse
import gzip
import json
import random
import os
import sys
from collections import defaultdict
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--sessions", type=int, default=600)
ap.add_argument("--pools", type=int, default=2)
ap.add_argument("--seed", type=int, default=54)
ap.add_argument("--max-input", type=int, default=250000)
ap.add_argument("--min-rounds", type=int, default=2)
ap.add_argument("--horizon-min", type=float, default=80)
ap.add_argument("--cap-s", type=float, default=1200)
ap.add_argument("--sys-tokens", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_tracelab import rewrite, submitted  # noqa: E402


def api_end(row):
    t = [e["timestamp"] for e in row.get("timing_events") or []
         if e.get("event_type") not in ("user_message", "tool_result") and e.get("timestamp")]
    return datetime.fromisoformat(max(t).replace("Z", "+00:00")).timestamp() if t else None


def horizon(rs):
    clock, prev_end, out = 0.0, None, []
    for i, r in enumerate(rs):
        s = submitted(r)
        if prev_end is not None:
            clock += min(max(s - prev_end, 0.0), a.cap_s)
        if clock > a.horizon_min * 60:
            break
        new = int(r.get("input_tokens_total") or 0) if i == 0 else int(r.get("newly_append_tokens") or 0)
        o = int(r.get("output_tokens") or 0) + int(r.get("reasoning_output_tokens") or 0)
        # vLLM rejects prompt + max_tokens > max_model_len (262,144); the server counts the trace's
        # input within +-0.2 % plus ~30-80 tokens of chat template (calibration 2026-11-20).
        if int(r.get("input_tokens_total") or 0) * 1.003 + 100 + o > 262144:
            break
        clock += new / 20000 + o / 200
        e = api_end(r)
        prev_end = e if e is not None else s
        out.append(r)
    return out


by = defaultdict(list)
for line in gzip.open(a.inp, "rt"):
    r = json.loads(line)
    by[r["session_id"]].append(r)
kept, n_elig, n_empty, n_small = {}, 0, 0, 0
for sid, rows in by.items():
    rs = rewrite(rows, a.max_input)
    if len(rs) < a.min_rounds:
        continue
    n_elig += 1
    h = horizon(rs)
    if not h:
        n_empty += 1
        continue
    if a.sys_tokens and min(int(r.get("input_tokens_total") or 0) for r in h) < a.sys_tokens + 200:
        n_small += 1
        continue
    kept[sid] = h
print(f"eligible {n_elig}, empty after the cuts {n_empty}, a round under K+200 {n_small}, kept {len(kept)}")
sids = sorted(kept)
random.Random(a.seed).shuffle(sids)
assert len(sids) >= a.pools * a.sessions, "not enough sessions for the pools"
K = a.sys_tokens
for k in range(a.pools):
    part = sids[k * a.sessions:(k + 1) * a.sessions]
    n_rows = n_tok = 0
    lo = None
    path = f"{a.out}_{k}.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for sid in part:
            for r in kept[sid]:
                total = int(r.get("input_tokens_total") or 0)
                if K:
                    r = dict(r)
                    r["input_tokens_total"] = total - K
                    r["prefix_tokens"] = max(int(r.get("prefix_tokens") or 0) - K, 0)
                lo = r["input_tokens_total"] if lo is None else min(lo, r["input_tokens_total"])
                f.write(json.dumps(r) + "\n")
                n_rows += 1; n_tok += total
    print(f"{path}: sessions {len(part)}, rounds {n_rows:,}, recorded input tokens summed {n_tok / 1e9:.2f} B "
          f"(~{n_tok * 3.7 / 2**30:.1f} GiB stored), K {K}, smallest written input {lo}")
