"""Kernel probe summary for study 28: one row per combination, from results/<name>.json.

'within 15 %' = prefill step AND decode TPOT at 30 sequences both within 15 % of the best
started combination OF THE SAME GROUP (the name's prefix: L- linear backends with bf16 KV
and FLASHINFER, A- attention backends, K- fp8 KV), the README's rule for keeping a backend in
the domain. Ranking across groups would let a faster fp8-KV row push valid bf16 backends out.
"""
import glob
import json
import os
import sys

rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.json')))]
ok = [r for r in rows if r.get('started') and r.get('bench')]
group = lambda r: r['name'].split('-')[0]
best_pf, best_tp = {}, {}
for r in ok:
    s_, g = r['bench']['summary'], group(r)
    best_pf[g] = min(best_pf.get(g, s_['prefill_2k_mean_s']), s_['prefill_2k_mean_s'])
    best_tp[g] = min(best_tp.get(g, s_['decode_c30_tpot_ms']), s_['decode_c30_tpot_ms'])
print('%-14s %-18s %-12s %-5s %7s %9s %9s %10s %s' % (
    'name', 'linear', 'attention', 'kv', 'start_s', 'pf2k_s', 'tpot1_ms', 'tpot30_ms', 'within 15 %'))
for r in rows:
    head = '%-14s %-18s %-12s %-5s' % (r['name'], r['linear'], r['attention'], r['kv'])
    if not (r.get('started') and r.get('bench')):
        print('%s did not start (apply exit %s)' % (head, r.get('apply_exit')))
        continue
    s = r['bench']['summary']
    within = s['prefill_2k_mean_s'] <= 1.15 * best_pf[group(r)] and s['decode_c30_tpot_ms'] <= 1.15 * best_tp[group(r)]
    print('%s %7d %9.3f %9.1f %10.1f %s' % (head, r['startup_s'], s['prefill_2k_mean_s'],
                                            s['tpot_single_ms'], s['decode_c30_tpot_ms'],
                                            'yes' if within else 'no'))
