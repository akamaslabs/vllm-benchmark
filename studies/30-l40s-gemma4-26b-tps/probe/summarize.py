"""Startup probe summary for study 30: one row per combination, from results/<name>.json.

'within 15 %' (groups A- attention backends, L- linear backends, ranked with B-default, the
auto choice of both): prefill step (median of 4) <= 1.15 x the best of the group AND
generated tokens/s at 64 concurrent requests >= best / 1.15. The README's rule: a backend enters the study's
domain only if it starts and is within 15 % of the best of its group, and the parameter
enters only if at least two backends do.
'vs ref' (group M-, MTP speculative decoding): tokens/s per request at batch 1 and generated
tokens/s at 64 concurrent requests against the same configuration without MTP (K-fp8 for a
name containing fp8, B-default otherwise). MTP enters the study only if it does not lose at
64 concurrent requests. Acceptance = accepted / drafted tokens (English batch 1, English
c64, Italian c32).
"""
import glob
import json
import os
import statistics
import sys

rows = {json.load(open(p))['name']: json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.json')))}
ok = {n: r for n, r in rows.items() if r.get('started') and r.get('bench')}
# The prefill step is the MEDIAN of its 4 samples, not the mean: on 2026-10-06 one sample in
# four took ~0.9 s instead of ~0.1 s in 4 combinations out of 8 (first use of a new batch
# shape), which tripled the mean and made identical kernels look 3x apart.
for r in ok.values():
    if r['bench'].get('prefill_s'):
        r['bench']['summary']['prefill_mean_s'] = statistics.median(r['bench']['prefill_s'])
group = lambda n: n.split('-')[0]
RANKED = ('A', 'L')
best_pf, best_tp = {}, {}
for n, r in ok.items():
    s, g = r['bench']['summary'], group(n)
    if g in RANKED or g == 'B':
        for gg in (RANKED if g == 'B' else (g,)):
            best_pf[gg] = min(best_pf.get(gg, s['prefill_mean_s']), s['prefill_mean_s'])
            best_tp[gg] = max(best_tp.get(gg, s['c64_gen_tok_per_s']), s['c64_gen_tok_per_s'])


def pct(x):
    return '%+.0f%%' % (100 * x) if x is not None else 'n/a'


def acc(x):
    return '%.2f' % x if x is not None else '-'


print('%-13s %7s %7s %8s %9s %8s %-14s %-17s %s' % (
    'name', 'start_s', 'pf_s', 'tok/s@1', 'tok/s@64', 'it@32', 'within 15 %', 'acc en1/en64/it',
    'overrides'))
for n, r in rows.items():
    if n not in ok:
        print('%-13s did not start (apply exit %s)  %s' % (n, r.get('apply_exit'), r.get('overrides')))
        continue
    s, g = r['bench']['summary'], group(n)
    if g in RANKED:
        verdict = 'yes' if (s['prefill_mean_s'] <= 1.15 * best_pf[g] and s['c64_gen_tok_per_s'] >= best_tp[g] / 1.15) else 'no'
    elif g == 'M':
        ref = ok.get('K-fp8' if 'fp8' in n else 'B-default')
        if ref:
            rs = ref['bench']['summary']
            verdict = 'vs ref %s/%s' % (pct(s['tok_per_s_single'] / rs['tok_per_s_single'] - 1),
                                        pct(s['c64_gen_tok_per_s'] / rs['c64_gen_tok_per_s'] - 1))
        else:
            verdict = 'no ref'
    else:
        verdict = '-'
    print('%-13s %7d %7.3f %8.1f %9.0f %8.0f %-14s %-17s %s' % (
        n, r['startup_s'], s['prefill_mean_s'], s['tok_per_s_single'], s['c64_gen_tok_per_s'],
        s['c32_it_gen_tok_per_s'], verdict,
        '/'.join(acc(s[k]) for k in ('acceptance_single_en', 'acceptance_c64_en', 'acceptance_c32_it')),
        r.get('overrides')))
