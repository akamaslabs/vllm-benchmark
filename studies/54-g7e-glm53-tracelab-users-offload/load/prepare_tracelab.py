"""Prepare a TraceLab sample for AIPerf's `tracelab` loader (AIPerf >= 0.13.0).

Usage: python3 -I prepare_tracelab.py IN.jsonl.gz OUT.jsonl.gz [--sessions N] [--seed S]
       [--max-input T] [--min-rounds R]

1. History continuity. TraceLab's prefix_tokens is what the ORIGINAL provider's cache served
   (Anthropic's 5-minute TTL: ~47 % of the history after a 5-10 min pause, ~5 % after an hour).
   AIPerf mints fresh block ids for everything past prefix_tokens, so those rounds would reach
   the server with new content and miss whatever it kept. Here prefix_tokens becomes the history
   the agent re-sent, min(input_i, input_{i-1} + output_{i-1}), unless the context shrank below
   70 % of the previous round (compaction / reset), where the recorded value stays.
   newly_append_tokens = input_tokens_total - prefix_tokens.
2. Context limit. A session ends before its first round with input_tokens_total > max-input.
3. Sample. N sessions with >= min-rounds rounds (after the cut), drawn with a fixed seed; rows of
   other sessions are dropped (subagent sessions are top-level session_ids in TraceLab, so a
   sampled parent loses its subagents unless they are sampled too).
Rows keep the TraceLab schema; rounds are ordered by submission time (latest input event), the
loader's own order.
"""
import argparse
import gzip
import json
import random
from collections import defaultdict
from datetime import datetime

INPUT_EV = {"user_message", "tool_result"}


def submitted(row):
    t = [e["timestamp"] for e in row.get("timing_events") or []
         if e.get("event_type") in INPUT_EV and e.get("timestamp")]
    return datetime.fromisoformat(max(t).replace("Z", "+00:00")).timestamp() if t else None


def rewrite(rows, max_input):
    rows = sorted((r for r in rows if submitted(r) is not None),
                  key=lambda r: (submitted(r), r.get("round_index") or 0))
    out, prev = [], None
    for r in rows:
        total = int(r.get("input_tokens_total") or 0)
        if total > max_input:
            break
        if prev is not None:
            p_in = int(prev.get("input_tokens_total") or 0)
            if total >= 0.7 * p_in:
                prefix = min(total, p_in + int(prev.get("output_tokens") or 0))
                r = dict(r, prefix_tokens=prefix, newly_append_tokens=total - prefix)
        out.append(r)
        prev = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--sessions", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=54)
    ap.add_argument("--max-input", type=int, default=250000)
    ap.add_argument("--min-rounds", type=int, default=2)
    a = ap.parse_args()
    by = defaultdict(list)
    for line in gzip.open(a.inp, "rt"):
        r = json.loads(line)
        by[r["session_id"]].append(r)
    kept = {}
    for sid, rows in by.items():
        rs = rewrite(rows, a.max_input)
        if len(rs) >= a.min_rounds:
            kept[sid] = rs
    sids = sorted(kept)
    random.Random(a.seed).shuffle(sids)
    sids = sids[:a.sessions]
    n_rows = n_tok = 0
    with gzip.open(a.out, "wt") as f:
        for sid in sids:
            for r in kept[sid]:
                f.write(json.dumps(r) + "\n")
                n_rows += 1; n_tok += int(r.get("input_tokens_total") or 0)
    print(f"sessions in: {len(by)}, eligible: {len(kept)}, written: {len(sids)}; "
          f"rounds: {n_rows}; input tokens summed over rounds: {n_tok:,}")


if __name__ == "__main__":
    main()
