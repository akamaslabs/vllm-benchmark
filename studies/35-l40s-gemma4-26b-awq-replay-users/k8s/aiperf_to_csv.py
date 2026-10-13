"""Turns AIPerf's per-request records into a CSV for Akamas's CSV File telemetry provider.

Usage: python3 aiperf_to_csv.py <profile_export.jsonl> <out.csv> [--component aiperf]
                                [--bucket-s 30] [--window-s 150]

Part of the AIPerf optimization pack: its columns are the AIPerf component type's metrics
(tools/tests/test_aiperf_to_csv.sh checks it). A study copies this file at the pack's tag.

Input: the profile_export.jsonl that AIPerf 0.11.0 writes with --export-level records (its
default). One line per request, a MetricRecordInfo (aiperf common/models/record_models.py):
metadata.request_start_ns / request_end_ns are wall-clock ns, metadata.conversation_id names the
session, `error` is set on a failed request (cancelled ones too, code 499), metrics.<tag>.value
is in display units (ms, tokens). Warm-up records are never written; requests still in flight at the SIGINT or at the end
of the grace period are not written either.

Output: one row per bucket of --bucket-s seconds (epoch-aligned, as Prometheus's 30 s scrapes),
TS = the bucket's end in UTC (yyyy-MM-dd HH:mm:ss), COMPONENT = --component, then one column per
metric, named as the AIPerf optimization pack's metric: aiperf_ + the names below (the telemetry
instance maps each column to the metric of the same name). A request belongs to the bucket in
which it ended:
  - request_success_rate / request_error_rate: requests completed / failed per second;
  - output_token_throughput: output tokens of the completed requests per second;
  - e2e_request_latency (AIPerf's request_latency), time_to_first_token (the first token,
    reasoning included), time_to_first_output_token (the first answer token: with thinking on,
    after the reasoning), inter_token_latency (per request: (e2e - TTFT) / (output tokens - 1),
    then the distribution over requests; vLLM's histogram counts every gap instead): avg, p50,
    p95 (e2e also p99) of the completed requests; the _150s columns take the same percentile
    over the requests completed in the last --window-s seconds up to the bucket's end (as the
    [150s] p95s of the Prometheus telemetry; at < 1 req/s a 30 s bucket holds few requests);
  - input / output sequence length avg and p95, reasoning tokens avg;
  - requests_in_flight: requests in flight, time-weighted over the bucket (failed ones too);
  - active_sessions: sessions whose span (first request's start to last request's end) overlaps
    the bucket (with AIPerf's --concurrency, the users holding a slot).
Percentiles are exact, linearly interpolated (numpy's default, as AIPerf's own summary). A
latency cell is empty when its bucket (or window) has no completed request. The first bucket
(before the first request's start) and the last one (cut by the stop) are partial and dropped.
Bias at the end of the run: a request still in flight at the stop has no record, so the last
buckets under-count requests_in_flight and active_sessions by those requests.
Exit 2 if the input has no usable record.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone

NS = 1_000_000_000
PREFIX = 'aiperf_'  # the pack's metric names: Akamas metric names are global, vLLM's are unprefixed

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
    v = rec.get('metrics', {}).get(tag)
    v = v.get('value') if isinstance(v, dict) else None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def load(path):
    """(start_ns, end_ns, session, ok, metrics{tag: value}) per usable record; skipped count."""
    reqs, skipped = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                md = rec['metadata']
                start, end = int(md['request_start_ns']), int(md['request_end_ns'])
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue
            if md.get('benchmark_phase', 'profiling') != 'profiling' or end < start:
                skipped += 1
                continue
            ok = not rec.get('error') and not md.get('was_cancelled')
            tags = {t for _, t, _ in LATENCIES} | {t for _, t, _ in LENGTHS}
            reqs.append((start, end, md.get('conversation_id'), ok,
                         {t: value(rec, t) for t in tags}))
    return reqs, skipped


def fmt(v):
    if v is None:
        return ''
    return '%d' % v if isinstance(v, int) else '%.6g' % v


def aggregate(reqs, bucket_s, window_s):
    b_ns, w_ns = bucket_s * NS, window_s * NS
    first = min(r[0] for r in reqs)
    last = max(r[1] for r in reqs)
    b0 = -(-first // b_ns)          # first bucket starting at or after the first request's start
    # Bucket k covers (k*b, (k+1)*b]: a request ending exactly on a boundary closes the earlier
    # one. The bucket holding the last end is cut by the stop: excluded.
    b1 = -(-last // b_ns) - 1
    by_bucket = defaultdict(list)   # bucket index -> requests that ended in it
    for r in reqs:
        by_bucket[-(-r[1] // b_ns) - 1].append(r)
    spans = {}
    for s, e, sess, _, _ in reqs:
        lo, hi = spans.get(sess, (s, e))
        spans[sess] = (min(lo, s), max(hi, e))
    n_win = max(1, round(w_ns / b_ns))
    rows = []
    for k in range(b0, b1):
        lo, hi = k * b_ns, (k + 1) * b_ns
        done = by_bucket.get(k, [])
        ok = [r for r in done if r[3]]
        win_ok = [r for j in range(k - n_win + 1, k + 1) for r in by_bucket.get(j, []) if r[3]]
        row = {
            'request_success_rate': len(ok) / bucket_s,
            'request_error_rate': (len(done) - len(ok)) / bucket_s,
            'output_token_throughput': sum(r[4]['output_sequence_length'] or 0 for r in ok) / bucket_s,
        }
        for name, tag, ps in LATENCIES:
            vals = sorted(r[4][tag] for r in ok if r[4][tag] is not None)
            wvals = sorted(r[4][tag] for r in win_ok if r[4][tag] is not None)
            row['%s_avg' % name] = avg(vals)
            for p in ps:
                row['%s_p%d' % (name, p)] = percentile(vals, p)
                row['%s_p%d_150s' % (name, p)] = percentile(wvals, p)
        for name, tag, p95 in LENGTHS:
            vals = sorted(r[4][tag] for r in ok if r[4][tag] is not None)
            row['%s_avg' % name] = avg(vals)
            if p95:
                row['%s_p95' % name] = percentile(vals, 95)
        row['requests_in_flight'] = sum(
            max(0, min(e, hi) - max(s, lo)) for s, e, _, _, _ in reqs) / b_ns
        row['active_sessions'] = sum(1 for s, e in spans.values() if s < hi and e > lo)
        rows.append((hi, row))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('records')
    ap.add_argument('out')
    ap.add_argument('--component', default='aiperf')
    ap.add_argument('--bucket-s', type=int, default=30)
    ap.add_argument('--window-s', type=int, default=150)
    a = ap.parse_args(argv)
    reqs, skipped = load(a.records)
    if not reqs:
        print('error: no usable record in %s (%d skipped)' % (a.records, skipped), file=sys.stderr)
        return 2
    rows = aggregate(reqs, a.bucket_s, a.window_s)
    cols = columns()
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['TS', 'COMPONENT'] + [PREFIX + c for c in cols])
        for end_ns, row in rows:
            ts = datetime.fromtimestamp(end_ns // NS, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            w.writerow([ts, a.component] + [fmt(row[c]) for c in cols])
    n_ok = sum(1 for r in reqs if r[3])
    print('%s: %d requests (%d failed, %d records skipped) -> %d buckets of %d s in %s'
          % (a.records, len(reqs), len(reqs) - n_ok, skipped, len(rows), a.bucket_s, a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
