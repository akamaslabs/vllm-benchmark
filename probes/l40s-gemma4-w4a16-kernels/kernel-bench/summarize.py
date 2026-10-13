"""Summary of the W4A16 kernel probe: one row per combination, from <dir>/<name>.json.

Columns: startup time; KV cache tokens (from the startup log); prefill wall time of a ~3k /
15k-token prompt; ms per output token at batch 1 / 16 / 64; generated tokens/s at 64; the
mixed phase at 16 workers (3k in / 150 out: e2e p50 and completed req/s); turn-2 TTFT with the
history in the prefix cache; quality (40 arithmetic problems). The last column compares
prefill 3k / ms-per-token at 16 / mixed c16 req/s with B-default (positive = better).
Usage: summarize.py <results dir>
"""
import glob
import json
import os
import re
import sys

d = sys.argv[1]
rows = {}
for p in sorted(glob.glob(os.path.join(d, '*.json'))):
    r = json.load(open(p))
    rows[r['name']] = r


def kv_tokens(name):
    try:
        m = re.findall(r'GPU KV cache size: ([\d,]+) tokens', open(os.path.join(d, name + '.kernels.txt')).read())
        return m[0].replace(',', '') if m else '-'
    except OSError:
        return '-'


def s_of(r):
    return (r.get('bench') or {}).get('summary') if r.get('started') else None


ref = s_of(next((rows[k] for k in ('B-default', 'B2-default', 'B3-default') if k in rows), {}))


def rel(s, f, lower_better):
    try:
        a, b = f(s), f(ref)
        x = a / b - 1
        return '%+.0f%%' % (100 * (-x if lower_better else x))
    except Exception:
        return 'n/a'


def f(x, fmt):
    return fmt % x if isinstance(x, (int, float)) else '-'


print('%-18s %5s %7s %6s %6s %5s %5s %5s %6s %6s %5s %6s %5s  %-16s' % (
    'name', 'start', 'kv_tok', 'pf3k', 'pf15k', 'ms@1', 'ms@16', 'ms@64', 'tok@64', 'mx_e2e', 'mx_rs', 'turn2', 'qual',
    'vs def pf/16/mx'))
for n, r in rows.items():
    s = s_of(r)
    if not r.get('started'):
        print('%-18s did not start (apply exit %s)  %s %s' % (n, r.get('apply_exit'), r.get('args'), r.get('env', '')))
        continue
    if not s:
        print('%-18s started, bench failed' % n)
        continue
    mpt = s.get('ms_per_token') or {}
    print('%-18s %5d %7s %6s %6s %5s %5s %5s %6s %6s %5s %6s %5s  %-16s' % (
        n, r['startup_s'], kv_tokens(n), f(s.get('prefill_3k_s'), '%.3f'), f(s.get('prefill_15k_s'), '%.2f'),
        f(mpt.get('1'), '%.1f'), f(mpt.get('16'), '%.1f'), f(mpt.get('64'), '%.1f'),
        f(s.get('gen_tok_per_s_64'), '%.0f'), f(s.get('mixed_c16_e2e_p50_s'), '%.2f'),
        f(s.get('mixed_c16_req_per_s'), '%.2f'), f(s.get('prefix_turn2_s'), '%.3f'), f(s.get('quality'), '%.2f'),
        '/'.join((rel(s, lambda x: x['prefill_3k_s'], True),
                  rel(s, lambda x: x['ms_per_token']['16'], True),
                  rel(s, lambda x: x['mixed_c16_req_per_s'], False)))))
