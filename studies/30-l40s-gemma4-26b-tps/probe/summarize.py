"""Startup probe summary for study 30: one row per combination, from results/<name>.json.

'within 15 %' = prefill step AND decode TPOT at 64 sequences both within 15 % of the best
started combination OF THE SAME GROUP (name prefix: A- attention backends, L- linear
backends; B-/K-/E- are reported, not ranked). The README's rule: a backend enters the
study's domain only if it starts and is within 15 % of the best of its group, and the
parameter enters the study only if at least two backends do.
"""
import glob
import json
import os
import sys

rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.json')))]
ok = [r for r in rows if r.get('started') and r.get('bench')]
group = lambda r: r['name'].split('-')[0]
RANKED = ('A', 'L')
best_pf, best_tp = {}, {}
for r in ok:
    s_, g = r['bench']['summary'], group(r)
    # B-default is the auto choice of both groups: it sets the reference of each of them.
    if g in RANKED or g == 'B':
        for gg in (RANKED if g == 'B' else (g,)):
            best_pf[gg] = min(best_pf.get(gg, s_['prefill_mean_s']), s_['prefill_mean_s'])
            if s_['decode_c64_tpot_ms']:
                best_tp[gg] = min(best_tp.get(gg, s_['decode_c64_tpot_ms']), s_['decode_c64_tpot_ms'])
print('%-13s %7s %8s %9s %10s %10s %-11s %s' % (
    'name', 'start_s', 'pf_s', 'tpot1_ms', 'tpot64_ms', 'gen_tok/s', 'within 15 %', 'overrides'))
for r in rows:
    if not (r.get('started') and r.get('bench')):
        print('%-13s did not start (apply exit %s)  %s' % (r['name'], r.get('apply_exit'), r.get('overrides')))
        continue
    s = r['bench']['summary']
    g = group(r)
    if g in RANKED and s['decode_c64_tpot_ms']:
        within = 'yes' if (s['prefill_mean_s'] <= 1.15 * best_pf[g] and s['decode_c64_tpot_ms'] <= 1.15 * best_tp[g]) else 'no'
    else:
        within = '-'
    print('%-13s %7d %8.3f %9.1f %10s %10.0f %-11s %s' % (
        r['name'], r['startup_s'], s['prefill_mean_s'], s['tpot_single_ms'],
        '%.1f' % s['decode_c64_tpot_ms'] if s['decode_c64_tpot_ms'] else 'n/a',
        s['decode_c64_gen_tok_per_s'], within, r.get('overrides')))
