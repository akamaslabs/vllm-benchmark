"""Study 39 probe: parse one startup, parse the rate sweep, print the summary.

  summarize.py --parse <name> <overrides> <apply_exit> <startup_s> <out_dir>
      one JSON line for phase A, from <out_dir>/<name>.log (apply_config.sh's output, which
      holds the whole vLLM log) and <out_dir>/<name>.facts.txt (node_facts.sh).
  summarize.py --sweep <sweep-job.log> <prometheus url>
      one JSON line per sweep step (STEP_RESULT lines of the sweep Job), with vLLM's own
      150 s p95s and counters over the step from Prometheus when it answers.
  summarize.py <out_dir>
      the two tables, the knee and the proposed RT_RATE / RT_RAMP_S.

Knee: the highest constant rate whose step meets the study's SLO on AIPerf's client-side
p95s (TTFT <= 3000 ms, TPOT <= 100 ms; AIPerf's inter_token_latency is a per-request TPOT,
the quantity behind vllm:request_time_per_output_token_seconds). Proposal for the study's
open-loop ramp: RT_RATE = 2 x the knee (rounded up to an integer), so the baseline crosses
its knee about halfway; RT_RAMP_S = 3000 (as study 38), i.e. 3-minute scoring windows in
which the offered rate moves by ~12 % of the knee. The proposal is a starting point for
the Smoke study, which fixes the values.
"""
import glob
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request

SLO_TTFT_MS, SLO_TPOT_MS = 3000.0, 100.0
MODEL, POD = 'qwen3-06b', 'vllm-0'


def num(s):
    return float(s.replace(',', ''))


def first(pattern, text, conv=num, group=1):
    m = re.search(pattern, text)
    return conv(m.group(group)) if m else None


def id_count(s):
    return len([x for x in s.split(',') if x.strip()]) if s is not None else None


def parse_one(name, overrides, rc, start_s, out):
    log = open(os.path.join(out, name + '.log'), errors='replace').read() if os.path.exists(os.path.join(out, name + '.log')) else ''
    fp = os.path.join(out, name + '.facts.txt')
    facts = open(fp, errors='replace').read() if os.path.exists(fp) else ''
    r = {'name': name, 'overrides': overrides, 'apply_exit': int(rc), 'started': int(rc) == 0,
         'startup_s': int(start_s),
         # render_statefulset.sh refused it: exit 2 before any cluster call.
         'rejected_by_renderer': int(rc) == 2 and 'invalid parameters' in log,
         'kv_tokens': first(r'KV cache size: ([\d,]+) tokens', log),
         'max_concurrency_x': first(r'Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x', log),
         'kv_gib': first(r'Auto set \(([\d.]+)/[\d.]+\) GiB for KV cache', log),
         'total_gib': first(r'Auto set \([\d.]+/([\d.]+)\) GiB for KV cache', log),
         'requested_gib': first(r'with ([\d.]+) GiB requested memory', log),
         'non_kv_gib': first(r'([\d.]+) GiB memory was consumed by non-kv usages', log),
         'omp_threads': id_count(first(r'local_rank=0, core ids=\[([^\]]*)\]', log, conv=str)),
         'omp_reserved': id_count(first(r'reserved_cpus=\[([^\]]*)\]', log, conv=str)),
         'ld_preload_missing': 'is not found in LD_PRELOAD' in log,
         'mempolicy_denied': bool(re.search(r'get_mempolicy|set_mempolicy.*not permitted|Operation not permitted', log)),
         'numa_nodes': first(r'NUMA node\(s\):\s*(\d+)', facts, conv=int),
         'amx': ('amx_bf16' in facts) if facts else None,
         'cpus_allowed': facts and first(r'Cpus_allowed_list:\s*(\S+)', facts, conv=str),
         'memory_max': first(r'memory\.max: (\d+)', facts, conv=int),
         'memory_current': first(r'memory\.current: (\d+)', facts, conv=int),
         'memory_peak': first(r'memory\.peak: (\d+)', facts, conv=int),
         'error': None}
    if not r['started']:
        errs = [l.strip() for l in log.splitlines()
                if re.search(r'^error:|ValueError|RuntimeError|OOMKilled|NotImplementedError', l)]
        r['error'] = errs[-1][:300] if errs else None
    return r


def prom_query(prom, q, t):
    url = prom + '/api/v1/query?' + urllib.parse.urlencode({'query': q, 'time': t})
    try:
        res = json.load(urllib.request.urlopen(url, timeout=10))['data']['result']
    except Exception:  # Prometheus is optional for the probe
        return None
    if not res:
        return None
    v = float(res[0]['value'][1])
    return None if math.isnan(v) or math.isinf(v) else v


def parse_sweep(log_path, prom):
    sel = 'model_name="%s",pod="%s"' % (MODEL, POD)
    starts = {}
    for line in open(log_path, errors='replace'):
        p = line.split(' ', 4)
        if line.startswith('STEP START '):
            starts[p[2]] = int(p[3])
        elif line.startswith('STEP_RESULT '):
            rate, t_end = p[1], int(p[2])
            j = json.loads(line.split(' ', 3)[3])
            g = lambda k, s='p95': (j.get(k) or {}).get(s)
            step = max(60, t_end - 30 - starts.get(rate, t_end - 330))
            win = '%ds' % step
            q = {
                'prom_ttft_p95_ms': 'histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{%s}[150s])))*1000' % sel,
                'prom_tpot_p95_ms': 'histogram_quantile(0.95, sum by(le)(rate(vllm:request_time_per_output_token_seconds_bucket{%s}[150s])))*1000' % sel,
                'prom_total_tok_s': 'sum(rate(vllm:prompt_tokens_total{%s}[%s])) + sum(rate(vllm:generation_tokens_total{%s}[%s]))' % (sel, win, sel, win),
                'prom_prefix_hit_pct': '100 * sum(rate(vllm:prefix_cache_hits_total{%s}[%s])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total{%s}[%s])), 1e-9)' % (sel, win, sel, win),
                'prom_kv_usage_pct': '100 * max_over_time(vllm:kv_cache_usage_perc{%s}[%s])' % (sel, win),
                'prom_preemptions_s': 'sum(rate(vllm:num_preemptions_total{%s}[%s]))' % (sel, win),
                'prom_running_max': 'max_over_time(vllm:num_requests_running{%s}[%s])' % (sel, win),
                'prom_cpu_cores': 'sum(rate(container_cpu_usage_seconds_total{pod="%s",container="vllm"}[%s]))' % (POD, win),
            }
            row = {'rate': float(rate), 't_end': t_end, 'step_s': step,
                   'ttft_p95_ms': g('time_to_first_token'), 'tpot_p95_ms': g('inter_token_latency'),
                   'req_s': g('request_throughput', 'avg'), 'total_tok_s': g('total_token_throughput', 'avg'),
                   'out_tok_s': g('output_token_throughput', 'avg'),
                   'errors': g('error_request_count', 'avg')}
            # The step's last 150 s under load, not the 30 s grace drain after it.
            row['prom_at'] = t_end - 30
            for k, qq in q.items():
                row[k] = prom_query(prom, qq, t_end - 30)
            print(json.dumps(row))


def fmt(x, f='%.0f'):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) else '-'


def summary(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, '*.json'))):
        try:
            rows.append(json.load(open(p)))
        except ValueError:
            pass
    print('Phase A: startup edges')
    print('%-14s %5s %7s %10s %7s %7s %7s %4s %4s %5s %8s  %s' % (
        'name', 'exit', 'start_s', 'kv_tokens', 'kv_GiB', 'nonkv', 'peakGiB', 'omp', 'resv', 'numa', 'flags', 'overrides'))
    for r in rows:
        if 'name' not in r:
            continue
        flags = ''.join(c for c, k in (('P', 'ld_preload_missing'), ('M', 'mempolicy_denied')) if r.get(k))
        if r.get('rejected_by_renderer'):
            flags += 'R'
        peak = r.get('memory_peak') or r.get('memory_current')
        print('%-14s %5s %7s %10s %7s %7s %7s %4s %4s %5s %8s  %s' % (
            r['name'], r['apply_exit'], r['startup_s'], fmt(r.get('kv_tokens')), fmt(r.get('kv_gib'), '%.2f'),
            fmt(r.get('non_kv_gib'), '%.2f'), fmt(peak / 2**30 if peak else None, '%.1f'),
            fmt(r.get('omp_threads')), fmt(r.get('omp_reserved')), fmt(r.get('numa_nodes')), flags or '-',
            r.get('overrides')))
        if not r.get('started') and r.get('error'):
            print('%14s   %s' % ('', r['error']))
    print('flags: P LD_PRELOAD missing, M mempolicy denied (seccomp/SYS_NICE), R refused by render_statefulset.sh')
    measured = [r for r in rows if r.get('non_kv_gib') is not None]
    if measured:
        # The largest engine RSS seen (E-gmu-low has the largest batch, the worst case).
        worst = max(measured, key=lambda r: r['non_kv_gib'])
        lo = (worst['non_kv_gib'] + 1.75) / 48
        print('gpu_memory_utilization lower bound: engine non-KV RSS up to %.2f GiB (%s) -> f >= %.3f holds one '
              '16,384-token bf16 sequence; the study domain starts at 0.15.' % (worst['non_kv_gib'], worst['name'], lo))
    sweep = os.path.join(out, 'sweep.jsonl')
    steps = [json.loads(l) for l in open(sweep) if l.strip()] if os.path.exists(sweep) else []
    print('\nPhase B: baseline rate sweep (AIPerf client p95s; prom_* = vLLM over 150 s / the step)')
    print('%6s %7s %9s %9s %9s %9s %9s %7s %6s %7s %6s %5s' % (
        'rate', 'req/s', 'tok/s', 'ttft95', 'tpot95', 'p_ttft95', 'p_tpot95', 'hit%', 'kv%', 'preem', 'run', 'SLO'))
    knee = None
    for s in steps:
        ok = (s.get('ttft_p95_ms') is not None and s.get('tpot_p95_ms') is not None
              and s['ttft_p95_ms'] <= SLO_TTFT_MS and s['tpot_p95_ms'] <= SLO_TPOT_MS)
        if ok and (knee is None or s['rate'] > knee['rate']):
            knee = s
        print('%6s %7s %9s %9s %9s %9s %9s %7s %6s %7s %6s %5s' % (
            fmt(s['rate'], '%g'), fmt(s.get('req_s'), '%.2f'), fmt(s.get('total_tok_s')), fmt(s.get('ttft_p95_ms')),
            fmt(s.get('tpot_p95_ms')), fmt(s.get('prom_ttft_p95_ms')), fmt(s.get('prom_tpot_p95_ms')),
            fmt(s.get('prom_prefix_hit_pct'), '%.1f'), fmt(s.get('prom_kv_usage_pct')),
            fmt(s.get('prom_preemptions_s'), '%.2f'), fmt(s.get('prom_running_max')), 'ok' if ok else 'over'))
    if not steps:
        print('(no sweep results)')
    elif knee is None:
        print('\nKNEE: below the lowest rate swept (%g req/s): sweep lower rates (KP_RATES).' % steps[0]['rate'])
    else:
        top = max(s['rate'] for s in steps)
        print('\nKNEE: %g req/s (%.0f total tok/s, TTFT p95 %.0f ms, TPOT p95 %.0f ms)' % (
            knee['rate'], knee.get('total_tok_s') or 0, knee['ttft_p95_ms'], knee['tpot_p95_ms']))
        if knee['rate'] >= top:
            print('the highest rate swept is still within the SLO: sweep higher rates (KP_RATES) before using the proposal.')
        print('PROPOSAL for the Smoke study: RT_RATE=%d RT_RAMP_S=3000' % math.ceil(2 * knee['rate']))


if __name__ == '__main__':
    if sys.argv[1] == '--parse':
        print(json.dumps(parse_one(*sys.argv[2:7])))
    elif sys.argv[1] == '--sweep':
        parse_sweep(sys.argv[2], sys.argv[3])
    else:
        summary(sys.argv[1])
