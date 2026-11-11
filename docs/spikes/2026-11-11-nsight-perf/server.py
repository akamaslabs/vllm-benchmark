#!/usr/bin/env python3
"""server.py <metrics.log> <HH:MM:SS UTC start> <step-seconds> <steps>
Per ramp step, from the sampler in ramp_capture.sh (vLLM /metrics and nvidia-smi every 15 s):
mean running and waiting requests, peak waiting, KV cache use, GPU power, and the server's
prompt and generated tokens per second over the step."""
import re, sys
path, start, step, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
samples, cur = [], None
for line in open(path):
    line = line.strip()
    if line.startswith('# '):
        cur = {'t': int(line.split()[2])}; samples.append(cur)
    elif cur is not None and line.startswith('gpu:'):
        m = re.match(r'gpu: (\d+) %, ([\d.]+) W', line)
        if m: cur['util'], cur['w'] = float(m[1]), float(m[2])
    elif cur is not None and line.startswith('vllm:'):
        m = re.match(r'vllm:(\w+)\{[^}]*\} ([-\d.e+]+)$', line)
        if m and m[1] in ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
                          'prompt_tokens_total', 'generation_tokens_total'):
            cur.setdefault(m[1], float(m[2]))
h, mi, s = map(int, start.split(':'))
day = samples[0]['t'] - samples[0]['t'] % 86400
t0 = day + h * 3600 + mi * 60 + s
mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
print('step  from      samples  running  waiting  waiting_max  kv_cache_%  gpu_W  prompt_tok/s  gen_tok/s')
for i in range(n):
    a, b = t0 + i * step, t0 + (i + 1) * step
    w = [x for x in samples if a <= x['t'] < b and 'num_requests_running' in x]
    if len(w) < 2: continue
    dt = w[-1]['t'] - w[0]['t']
    print(f"{i+1:<5} {a % 86400 // 3600:02d}:{a % 3600 // 60:02d}:{a % 60:02d}  {len(w):7d}  "
          f"{mean([x['num_requests_running'] for x in w]):7.1f}  {mean([x['num_requests_waiting'] for x in w]):7.1f}  "
          f"{max(x['num_requests_waiting'] for x in w):11.0f}  {100 * mean([x['kv_cache_usage_perc'] for x in w]):10.1f}  "
          f"{mean([x['w'] for x in w if 'w' in x]):5.0f}  "
          f"{(w[-1]['prompt_tokens_total'] - w[0]['prompt_tokens_total']) / dt:12.0f}  "
          f"{(w[-1]['generation_tokens_total'] - w[0]['generation_tokens_total']) / dt:9.0f}")
