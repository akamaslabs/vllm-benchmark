"""Offline estimate of the score Akamas gives one study 54 trial, from its CSV (aiperf_to_csv.py).

Usage: python3 -I score_offline.py CSV [--width 6] [--slo-ttft-ms 5000] [--slo-itl-ms 50]

The study's windowing (stability on aiperf_active_sessions, width 6 samples, `when` max) with its two
constraints: every run of `width` consecutive rows is a window; a window is valid when the average of
its TTFT p95 (150 s) and of its ITL p95 (150 s) are within the SLO (empty cells ignored; a window with
no value for either is invalid); the score is the highest average of aiperf_active_sessions over the
valid windows. Prints "score <users> window <TS>..<TS>" or "score none".
"""
import argparse
import csv
import sys


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csv')
    ap.add_argument('--width', type=int, default=6)
    ap.add_argument('--slo-ttft-ms', type=float, default=5000)
    ap.add_argument('--slo-itl-ms', type=float, default=50)
    a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.csv)))
    best = None
    for i in range(len(rows) - a.width + 1):
        w = rows[i:i + a.width]
        t = mean([num(r['aiperf_time_to_first_token_p95_150s']) for r in w])
        it = mean([num(r['aiperf_inter_token_latency_p95_150s']) for r in w])
        u = mean([num(r['aiperf_active_sessions']) for r in w])
        if t is None or it is None or u is None or t > a.slo_ttft_ms or it > a.slo_itl_ms:
            continue
        if best is None or u > best[0]:
            best = (u, w[0]['TS'], w[-1]['TS'])
    print('score none' if best is None else 'score %g window %s..%s' % best)
    return 0


if __name__ == '__main__':
    sys.exit(main())
