"""Turns study 54's AIPerf 0.13 per-request records (both clients) into the CSV Akamas reads.

Usage: python3 -I aiperf_to_csv.py --pair EXPORT POOL [--pair EXPORT POOL] --out CSV --full-out CSV
         [--slo-ttft-ms 5000] [--slo-itl-ms 50] [--min-window-reqs 20] [--breach-buckets 4]
         [--component aiperf] [--bucket-s 30] [--window-s 150]

Study 38's converter (AIPerf optimization pack 1.0.0's columns: one per AIPerf component-type metric,
named as the metric) adapted to AIPerf 0.13's TraceLab replay:

- Input: one profile_export.jsonl per client, each followed by the pool it replayed. Records:
  metadata.conversation_id (a TraceLab session id; AIPerf's flattened-agent split adds
  "<session>::sa:..." and "<session>::aux:..." conversations), turn_index, source_outer_idx (the
  round's index in the session), request_start_ns / request_end_ns (wall clock), benchmark_phase,
  was_cancelled; a failed request carries a top-level `error`; metrics.<tag>.value in ms / tokens.
- Rows: one per 30 s bucket (epoch-aligned), TS = the bucket's end in UTC, a request belongs to the
  bucket in which it ended; latency percentiles exact, linearly interpolated; the _150s columns take
  the percentile over the requests completed in the trailing 150 s. First and last (partial) buckets
  dropped. See study 38's converter for every column.
- aiperf_active_sessions = the USERS: TraceLab sessions started minus sessions finished at the
  bucket's end, both clients together. A session starts with its first request and finishes when
  every round of its pool rows has been sent (its own and its agents' conversations together) and
  the last one answered. A user in a developer pause still counts (study 38's span count would drop
  exactly those users). With AIPerf's --concurrency a lane starts its next session at once, so this is
  the number of lanes in use.
- The knee: the first of --breach-buckets consecutive buckets whose client TTFT p95 (150 s) is above
  --slo-ttft-ms or ITL p95 (150 s) above --slo-itl-ms, counting only buckets with at least
  --min-window-reqs completed requests in their 150 s window (a p95 over a handful of early requests
  is noise). --out (what Akamas reads) ends before it: a window after the knee that dips back under
  the SLO can never be scored. --full-out keeps every bucket. Prints "knee: bucket <TS> users <n>
  (<metric>)", n = the users of the last bucket before the knee, or "knee: none".
- Exit 2 if there is no usable record; 3 if a client started a session twice (its pool ran out:
  two users would replay one session and share its cache, design "Load").
"""
import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

NS = 1_000_000_000
PREFIX = 'aiperf_'

LATENCIES = [  # (column prefix, AIPerf metric tag, percentiles)
    ('e2e_request_latency', 'request_latency', (50, 95, 99)),
    ('time_to_first_token', 'time_to_first_token', (50, 95)),
    ('time_to_first_output_token', 'time_to_first_output_token', (50, 95)),
    ('inter_token_latency', 'inter_token_latency', (50, 95)),
]
LENGTHS = [  # (column prefix, AIPerf metric tag, p95 too)
    ('input_sequence_length', 'input_sequence_length', True),
    ('output_sequence_length', 'output_sequence_length', True),
    ('reasoning_token_count', 'reasoning_token_count', False),
]


def columns():
    cols = ['request_success_rate', 'request_error_rate', 'output_token_throughput']
    for name, _, ps in LATENCIES:
        cols += ['%s_avg' % name] + ['%s_p%d' % (name, p) for p in ps]
        cols += ['%s_p%d_150s' % (name, p) for p in ps]
    for name, _, p95 in LENGTHS:
        cols += ['%s_avg' % name] + (['%s_p95' % name] if p95 else [])
    return cols + ['requests_in_flight', 'active_sessions']


def percentile(sorted_vals, p):
    """numpy.percentile(method='linear') on an already sorted list."""
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def avg(vals):
    return sum(vals) / len(vals) if vals else None


def value(rec, tag):
    v = (rec.get('metrics') or {}).get(tag)
    v = v.get('value') if isinstance(v, dict) else None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def pool_rounds(path):
    rounds = Counter()
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.strip():
                rounds[json.loads(line)['session_id'].replace(':', '_')] += 1
    return rounds


def load(pairs):
    """Requests (start_ns, end_ns, ok, metrics) of every client; session (start_ns, finish_ns or None)
    per client-qualified session; sessions started twice; skipped record count."""
    tags = {t for _, t, _ in LATENCIES} | {t for _, t, _ in LENGTHS}
    reqs, sessions, twice, skipped = [], {}, [], 0
    for client, (export, pool) in enumerate(pairs):
        rounds = pool_rounds(pool)
        first, last_end, outer, starts0 = {}, {}, defaultdict(set), Counter()
        with open(export) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    md = rec['metadata']
                    start, end = int(md['request_start_ns']), int(md['request_end_ns'])
                    conv = md['conversation_id']
                except (ValueError, KeyError, TypeError):
                    skipped += 1
                    continue
                if md.get('benchmark_phase', 'profiling') != 'profiling' or end < start:
                    skipped += 1
                    continue
                ok = not rec.get('error') and not md.get('was_cancelled')
                reqs.append((start, end, ok, {t: value(rec, t) for t in tags}))
                root = conv.split('::')[0]
                if '::' not in conv and md.get('turn_index') == 0:
                    starts0[root] += 1
                first[root] = min(first.get(root, start), start)
                last_end[root] = max(last_end.get(root, end), end)
                if md.get('source_outer_idx') is not None:
                    outer[root].add(md['source_outer_idx'])
        twice += ['c%d:%s' % (client, s) for s, n in starts0.items() if n > 1]
        for root, s in first.items():
            n = rounds.get(root)
            done = n is not None and len(outer[root]) >= n
            sessions['c%d:%s' % (client, root)] = (s, last_end[root] if done else None)
    return reqs, sessions, sorted(twice), skipped


def aggregate(reqs, sessions, bucket_s, window_s):
    b_ns, w_ns = bucket_s * NS, window_s * NS
    first = min(r[0] for r in reqs)
    last = max(r[1] for r in reqs)
    b0 = -(-first // b_ns)
    b1 = -(-last // b_ns) - 1
    by_bucket = defaultdict(list)
    for r in reqs:
        by_bucket[-(-r[1] // b_ns) - 1].append(r)
    n_win = max(1, round(w_ns / b_ns))
    rows = []
    for k in range(b0, b1):
        lo, hi = k * b_ns, (k + 1) * b_ns
        done = by_bucket.get(k, [])
        ok = [r for r in done if r[2]]
        win_ok = [r for j in range(k - n_win + 1, k + 1) for r in by_bucket.get(j, []) if r[2]]
        row = {
            'request_success_rate': len(ok) / bucket_s,
            'request_error_rate': (len(done) - len(ok)) / bucket_s,
            'output_token_throughput': sum(r[3]['output_sequence_length'] or 0 for r in ok) / bucket_s,
        }
        for name, tag, ps in LATENCIES:
            vals = sorted(r[3][tag] for r in ok if r[3][tag] is not None)
            wvals = sorted(r[3][tag] for r in win_ok if r[3][tag] is not None)
            row['%s_avg' % name] = avg(vals)
            for p in ps:
                row['%s_p%d' % (name, p)] = percentile(vals, p)
                row['%s_p%d_150s' % (name, p)] = percentile(wvals, p)
        for name, tag, p95 in LENGTHS:
            vals = sorted(r[3][tag] for r in ok if r[3][tag] is not None)
            row['%s_avg' % name] = avg(vals)
            if p95:
                row['%s_p95' % name] = percentile(vals, 95)
        row['requests_in_flight'] = sum(max(0, min(e, hi) - max(s, lo)) for s, e, _, _ in reqs) / b_ns
        row['active_sessions'] = sum(1 for s, f in sessions.values() if s <= hi and (f is None or f > hi))
        rows.append((hi, row, len(win_ok)))
    return rows


def knee(rows, slo_ttft, slo_itl, min_reqs, n_breach):
    """Index of the first row of the first run of n_breach bad rows, and what broke; (None, '')."""
    def bad(row, n):
        t, i = row['time_to_first_token_p95_150s'], row['inter_token_latency_p95_150s']
        what = [m for m, v, lim in (('TTFT', t, slo_ttft), ('ITL', i, slo_itl)) if v is not None and v > lim]
        return what if n >= min_reqs else []
    flags = [bad(row, n) for _, row, n in rows]
    for j in range(len(rows) - n_breach + 1):
        if all(flags[j:j + n_breach]):
            return j, ' and '.join(flags[j])
    return None, ''


def fmt(v):
    if v is None:
        return ''
    return '%d' % v if isinstance(v, int) else '%.6g' % v


def ts(end_ns):
    return datetime.fromtimestamp(end_ns // NS, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def write(path, rows, component):
    cols = columns()
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['TS', 'COMPONENT'] + [PREFIX + c for c in cols])
        for end_ns, row, _ in rows:
            w.writerow([ts(end_ns), component] + [fmt(row[c]) for c in cols])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pair', nargs=2, action='append', required=True, metavar=('EXPORT', 'POOL'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--full-out', required=True)
    ap.add_argument('--slo-ttft-ms', type=float, default=5000)
    ap.add_argument('--slo-itl-ms', type=float, default=50)
    ap.add_argument('--min-window-reqs', type=int, default=20)
    ap.add_argument('--breach-buckets', type=int, default=4)
    ap.add_argument('--component', default='aiperf')
    ap.add_argument('--bucket-s', type=int, default=30)
    ap.add_argument('--window-s', type=int, default=150)
    a = ap.parse_args(argv)
    reqs, sessions, twice, skipped = load(a.pair)
    if not reqs:
        print('error: no usable record (%d skipped)' % skipped, file=sys.stderr)
        return 2
    if twice:
        print('error: sessions started twice (the pool ran out, a replay would share its cache): %s'
              % ' '.join(twice[:10]), file=sys.stderr)
        return 3
    rows = aggregate(reqs, sessions, a.bucket_s, a.window_s)
    write(a.full_out, rows, a.component)
    j, what = knee(rows, a.slo_ttft_ms, a.slo_itl_ms, a.min_window_reqs, a.breach_buckets)
    write(a.out, rows if j is None else rows[:j], a.component)
    n_ok = sum(1 for r in reqs if r[2])
    print('%d requests (%d failed, %d records skipped), %d sessions (%d finished) -> %d buckets of %d s'
          % (len(reqs), len(reqs) - n_ok, skipped, len(sessions),
             sum(1 for _, f in sessions.values() if f is not None), len(rows), a.bucket_s))
    if j is None:
        print('knee: none')
    else:
        users = rows[j - 1][1]['active_sessions'] if j > 0 else 0
        print('knee: bucket %s users %d (%s)' % (ts(rows[j][0]), users, what))
    print('written: %s (%d rows, read by Akamas), %s (%d rows)' % (a.out, len(rows) if j is None else j,
                                                                   a.full_out, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
