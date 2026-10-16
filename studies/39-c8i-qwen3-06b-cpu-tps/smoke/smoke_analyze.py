"""Recomputes study 39's Akamas score for a manual smoke run, from Prometheus.

Same queries as akamas/telemetry/prometheus.yaml (model qwen3-06b, pod vllm-0,
$DURATION$ = 30 s, TTFT / TPOT p95 over 150 s), sampled every 30 s like the telemetry
instance. Scoring as the study: windows of 6 consecutive samples (3 min); a window is valid
if time_to_first_token_p95:max <= 3000 and time_per_output_token_p95:max <= 100; the score
is the window's mean total_token_throughput, and the best valid window wins (stability
windowing on total_token_throughput, when: max). Extra columns for this study's story: the
prefix-cache hit rate, KV usage and preemptions (the KV-budget axis), CPU cores used by
vLLM and by AIPerf, and vLLM's memory working set against the 48Gi limit. No GPU columns.
Usage: python3 smoke_analyze.py <prometheus url> <start ISO> <end ISO> [out.csv]
"""
import csv
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

SLO_TTFT_MS, SLO_TPOT_MS, WIDTH = 3000, 100, 6
SEL = 'model_name="qwen3-06b", pod="vllm-0"'
Q = {
    'total_tok_s': 'sum(rate(vllm:prompt_tokens_total{%s}[30s])) + sum(rate(vllm:generation_tokens_total{%s}[30s]))' % (SEL, SEL),
    'gen_tok_s': 'sum(rate(vllm:generation_tokens_total{%s}[30s]))' % SEL,
    'req_s': 'sum(rate(vllm:request_success_total{%s}[30s]))' % SEL,
    'ttft_p95_ms': 'histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{%s}[150s])))*1000' % SEL,
    'tpot_p95_ms': 'histogram_quantile(0.95, sum by(le)(rate(vllm:request_time_per_output_token_seconds_bucket{%s}[150s])))*1000' % SEL,
    'itl_p95_ms': 'histogram_quantile(0.95, sum by(le)(rate(vllm:inter_token_latency_seconds_bucket{%s}[150s])))*1000' % SEL,
    'running': 'sum(vllm:num_requests_running{%s})' % SEL,
    'waiting': 'sum(vllm:num_requests_waiting{%s})' % SEL,
    'kv_pct': '100 * avg(vllm:kv_cache_usage_perc{%s})' % SEL,
    'hit_pct': '100 * sum(rate(vllm:prefix_cache_hits_total{%s}[2m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total{%s}[2m])), 1e-9)' % (SEL, SEL),
    'preempt_s': 'sum(rate(vllm:num_preemptions_total{%s}[30s]))' % SEL,
    'cpu_vllm': 'sum(rate(container_cpu_usage_seconds_total{pod="vllm-0", container="vllm"}[2m]))',
    'cpu_aiperf': 'sum(rate(container_cpu_usage_seconds_total{pod=~"aiperf-cpu-.*", container!=""}[2m]))',
    'mem_vllm_gib': 'max(container_memory_working_set_bytes{pod="vllm-0", container="vllm"}) / 2^30',
}


def ts(iso):
    return datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


def query_range(q, start, end):
    url = PROM + '/api/v1/query_range?' + urllib.parse.urlencode({'query': q, 'start': start, 'end': end, 'step': 30})
    r = json.load(urllib.request.urlopen(url, timeout=30))['data']['result']
    return {int(float(t)): float(v) for t, v in r[0]['values']} if r else {}


def num(x):
    return x is not None and not math.isnan(x) and not math.isinf(x)


def best_window(rows):
    """(score, index, window) of the best valid WIDTH-sample window, or None."""
    best = None
    for i in range(len(rows) - WIDTH + 1):
        w = rows[i:i + WIDTH]
        if not all(num(r['total_tok_s']) for r in w):
            continue
        tt = [r['ttft_p95_ms'] for r in w if num(r['ttft_p95_ms'])]
        tp = [r['tpot_p95_ms'] for r in w if num(r['tpot_p95_ms'])]
        if not tt or not tp or max(tt) > SLO_TTFT_MS or max(tp) > SLO_TPOT_MS:
            continue
        score = sum(r['total_tok_s'] for r in w) / WIDTH
        if best is None or score > best[0]:
            best = (score, i, w)
    return best


def main():
    t0, t1 = ts(START), ts(END)
    series = {k: query_range(q, t0, t1) for k, q in Q.items()}
    times = sorted(set().union(*[set(s) for s in series.values()]))
    rows = [{'t': datetime.fromtimestamp(t, timezone.utc).strftime('%H:%M:%S'), **{k: series[k].get(t) for k in Q}}
            for t in times]
    best = best_window(rows)
    fmt = lambda x, f='%.0f': (f % x) if num(x) else '-'
    cols = [('total_tok_s', '%.0f'), ('gen_tok_s', '%.0f'), ('req_s', '%.2f'), ('ttft_p95_ms', '%.0f'),
            ('tpot_p95_ms', '%.0f'), ('running', '%.0f'), ('waiting', '%.0f'), ('kv_pct', '%.0f'),
            ('hit_pct', '%.0f'), ('preempt_s', '%.2f'), ('cpu_vllm', '%.1f'), ('cpu_aiperf', '%.2f'),
            ('mem_vllm_gib', '%.1f')]
    print('%-9s' % 'time' + ' '.join('%8s' % h for h in (
        'tok/s', 'gen/s', 'req/s', 'ttft95', 'tpot95', 'run', 'wait', 'kv%', 'hit%', 'preem', 'cpuV', 'cpuA', 'memGiB')))
    for i, r in enumerate(rows):
        mark = '*' if best and best[1] <= i < best[1] + WIDTH else ' '
        print('%-8s%s' % (r['t'], mark) + ' '.join('%8s' % fmt(r[k], f) for k, f in cols))
    if best:
        s, i, w = best
        print('\nSCORE (best valid 3-min window, * rows): total %.0f tok/s, %.2f req/s, running max %s, '
              'TTFT p95 max %.0f ms, TPOT p95 max %.0f ms, window %s-%s'
              % (s, sum(r['req_s'] or 0 for r in w) / WIDTH, fmt(max((r['running'] for r in w if num(r['running'])), default=None)),
                 max(r['ttft_p95_ms'] for r in w if num(r['ttft_p95_ms'])),
                 max(r['tpot_p95_ms'] for r in w if num(r['tpot_p95_ms'])), w[0]['t'], w[-1]['t']))
    else:
        print('\nNo valid 3-min window: the goal would be INVALID.')
    if OUT:
        with open(OUT, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=['t'] + list(Q))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == '__main__':
    PROM, START, END = sys.argv[1:4]
    OUT = sys.argv[4] if len(sys.argv) > 4 else None
    main()
