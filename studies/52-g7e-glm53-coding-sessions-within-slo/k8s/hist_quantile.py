"""Quantile of a vLLM Prometheus histogram between two /metrics snapshots, linearly interpolated
inside the bucket (like PromQL's histogram_quantile), printed in ms. Used by smoke.sh to compare the
server ITL (per engine step) and TPOT (per request) with the client ITL (final review I1): bucket
upper bounds alone cannot tell 50 ms from 60 ms when vLLM's buckets are 50 and 75 ms.
Usage: python3 hist_quantile.py METRIC SNAPSHOT0 SNAPSHOT1 Q   (METRIC without _bucket)
"""
import math
import re
import sys


def buckets(path, metric):
    out = {}
    pat = re.compile(re.escape(metric) + r'_bucket\{(?:.*,)?le="([^"]+)"(?:,.*)?\} (\S+)$')
    for line in open(path):
        m = pat.match(line.strip())
        if m:
            le = math.inf if m.group(1) == "+Inf" else float(m.group(1))
            out[le] = out.get(le, 0.0) + float(m.group(2))
    return out


def main():
    metric, a_path, b_path, q = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    a, b = buckets(a_path, metric), buckets(b_path, metric)
    cum = sorted((le, b[le] - a.get(le, 0.0)) for le in b)
    if not cum or cum[-1][1] <= 0:
        print("n/a")
        return
    rank = q * cum[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in cum:
        if c >= rank:
            if math.isinf(le):
                print("%.1f ms" % (prev_le * 1000))
                return
            v = prev_le + (le - prev_le) * (rank - prev_c) / max(c - prev_c, 1e-12)
            print("%.1f ms" % (v * 1000))
            return
        prev_le, prev_c = le, c


if __name__ == "__main__":
    main()
