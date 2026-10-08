"""Recomputes study 31's Akamas score for a manual smoke run, from Prometheus.

Same queries as akamas/telemetry/prometheus.yaml (model gemma4-26b-l40s-think, pod vllm-0,
$DURATION$ = 30 s, TTFT / ITL p95 over 150 s), sampled every 30 s like the telemetry
instance. Scoring as the study: windows of 6 consecutive samples (3 min); a window is valid
if time_to_first_token_p95:max <= 1500 and inter_token_latency_p95:max <= 300; the score is
the window's mean total_token_throughput, and the best valid window wins (stability
windowing on total_token_throughput, filter disabled, when: max).
Usage: python3 smoke_analyze.py <prometheus url> <start ISO> <end ISO> [out.csv]
"""
import csv
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

PROM, START, END = sys.argv[1:4]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
SEL = 'model_name="gemma4-26b-l40s-think", pod="vllm-0"'
Q = {
    'total_tok_s': 'sum(rate(vllm:prompt_tokens_total{%s}[30s])) + sum(rate(vllm:generation_tokens_total{%s}[30s]))' % (SEL, SEL),
    'gen_tok_s': 'sum(rate(vllm:generation_tokens_total{%s}[30s]))' % SEL,
    'req_s': 'sum(rate(vllm:request_success_total{%s}[30s]))' % SEL,
    'ttft_p95_ms': 'histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{%s}[150s])))*1000' % SEL,
    'itl_p95_ms': 'histogram_quantile(0.95, sum by(le)(rate(vllm:inter_token_latency_seconds_bucket{%s}[150s])))*1000' % SEL,
    'running': 'sum(vllm:num_requests_running{%s})' % SEL,
    'waiting': 'sum(vllm:num_requests_waiting{%s})' % SEL,
    'kv_pct': '100 * avg(vllm:kv_cache_usage_perc{%s})' % SEL,
    'preempt_s': 'sum(rate(vllm:num_preemptions_total{%s}[30s]))' % SEL,
    'cpu_vllm': 'sum(rate(container_cpu_usage_seconds_total{pod="vllm-0", container!=""}[2m]))',
    'cpu_aiperf': 'sum(rate(container_cpu_usage_seconds_total{pod=~"aiperf-l40s-.*", container!=""}[2m]))',
    'gpu_util_pct': 'avg(DCGM_FI_DEV_GPU_UTIL{modelName=~".*L40S.*", gpu="0"})',
    'gpu_power_w': 'max(DCGM_FI_DEV_POWER_USAGE{modelName=~".*L40S.*", gpu="0"})',
}


def ts(iso):
    return datetime.strptime(iso, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()


def query_range(q, start, end):
    url = PROM + '/api/v1/query_range?' + urllib.parse.urlencode({'query': q, 'start': start, 'end': end, 'step': 30})
    r = json.load(urllib.request.urlopen(url, timeout=30))['data']['result']
    return {int(float(t)): float(v) for t, v in r[0]['values']} if r else {}


t0, t1 = ts(START), ts(END)
series = {k: query_range(q, t0, t1) for k, q in Q.items()}
times = sorted(set().union(*[set(s) for s in series.values()]))
rows = [{'t': datetime.fromtimestamp(t, timezone.utc).strftime('%H:%M:%S'), **{k: series[k].get(t) for k in Q}} for t in times]


def num(x):
    return x is not None and not math.isnan(x) and not math.isinf(x)


best = None
for i in range(len(rows) - 5):
    w = rows[i:i + 6]
    if not all(num(r['total_tok_s']) for r in w):
        continue
    tt = [r['ttft_p95_ms'] for r in w if num(r['ttft_p95_ms'])]
    it = [r['itl_p95_ms'] for r in w if num(r['itl_p95_ms'])]
    if not tt or not it or max(tt) > 1500 or max(it) > 300:
        continue
    score = sum(r['total_tok_s'] for r in w) / 6
    if best is None or score > best[0]:
        best = (score, i, w)

fmt = lambda x, f='%.0f': (f % x) if num(x) else '-'
print('%-8s %8s %7s %6s %9s %8s %5s %5s %5s %6s %5s %5s %5s' % (
    'time', 'tok/s', 'gen/s', 'req/s', 'ttft95', 'itl95', 'run', 'wait', 'kv%', 'preem', 'cpuV', 'cpuA', 'gpu%'))
for i, r in enumerate(rows):
    mark = '*' if best and best[1] <= i < best[1] + 6 else ' '
    print('%-8s%s%8s %7s %6s %9s %8s %5s %5s %5s %6s %5s %5s %5s' % (
        r['t'], mark, fmt(r['total_tok_s']), fmt(r['gen_tok_s']), fmt(r['req_s'], '%.2f'),
        fmt(r['ttft_p95_ms']), fmt(r['itl_p95_ms']), fmt(r['running']), fmt(r['waiting']),
        fmt(r['kv_pct']), fmt(r['preempt_s'], '%.2f'), fmt(r['cpu_vllm'], '%.2f'),
        fmt(r['cpu_aiperf'], '%.2f'), fmt(r['gpu_util_pct'])))
if best:
    s, i, w = best
    print('\nSCORE (best valid 3-min window, * rows): total %.0f tok/s, %.2f req/s, running max %s, '
          'TTFT p95 max %.0f ms, ITL p95 max %.0f ms, window %s-%s'
          % (s, sum(r['req_s'] for r in w) / 6, fmt(max(r['running'] for r in w if num(r['running']))),
             max(r['ttft_p95_ms'] for r in w if num(r['ttft_p95_ms'])),
             max(r['itl_p95_ms'] for r in w if num(r['itl_p95_ms'])), w[0]['t'], w[-1]['t']))
else:
    print('\nNo valid 3-min window: the goal would be INVALID.')
if OUT:
    with open(OUT, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['t'] + list(Q))
        wr.writeheader()
        wr.writerows(rows)
