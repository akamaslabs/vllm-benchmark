"""Runs inside vllm-0 of study 28 (kernel probe). Prints one BENCH_RESULT JSON line.

Prefill: a ~2048-token prompt (one scheduler step at max_num_batched_tokens 2048), 1 output
token, mean of 4 after one warm-up: the compute-bound side of the linear kernel.
Decode: 30 concurrent ShareGPT-like requests (~100-token prompts, 256 output tokens): the
study's regime (~25-30 in flight), which fits a 1g.16gb slice's KV even in bf16.
Client-side wall times: they include HTTP and scheduling, so compare combinations, not
absolute numbers.
"""
import json
import random
import statistics as st
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'qwen3-8b-mig'
VOCAB = ['alpha', 'river', 'stone', 'quantum', 'market', 'silver', 'engine', 'forest', 'number', 'signal',
         'orange', 'planet', 'memory', 'window', 'garden', 'rocket', 'yellow', 'bridge', 'castle', 'dragon']


def mk(seed, words):  # ~3.9 tokens per word with Qwen3's tokenizer (study 24: 1045 words ~ 4090 tokens)
    rnd = random.Random(seed)
    return ' '.join(rnd.choice(VOCAB) + str(rnd.randint(0, 999)) for _ in range(words))


def post(body):
    return urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                         {'Content-Type': 'application/json'}), timeout=600)


def prefill(seed):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': mk(seed, 523)}], 'max_tokens': 1}
    s = time.perf_counter()
    post(body).read()
    return time.perf_counter() - s


def stream(seed, words, ntok):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': mk(seed, words)}], 'max_tokens': ntok,
            'stream': True, 'ignore_eos': True}
    s = time.perf_counter()
    ttft, n = None, 0
    for line in post(body):
        if line.startswith(b'data:') and b'[DONE]' not in line:
            c = json.loads(line[5:]).get('choices') or []
            if c and (c[0].get('delta') or {}).get('content'):
                n += 1
                if ttft is None:
                    ttft = time.perf_counter() - s
    tot = time.perf_counter() - s
    return {'ttft': ttft, 'tpot': (tot - ttft) / max(n - 1, 1) if ttft else None, 'tokens': n}


out = {}
seed = int(time.time()) % 100000
prefill(seed)
iso = [prefill(seed + 10 + i) for i in range(4)]
out['prefill_2k_s'] = iso
single = [stream(seed + 300 + i, 26, 128) for i in range(3)]
out['single'] = single
time.sleep(2)
res = [None] * 30


def w(i):
    res[i] = stream(seed + 500 + i, 26, 256)


ts = [threading.Thread(target=w, args=(i,)) for i in range(30)]
s = time.perf_counter()
[t.start() for t in ts]
[t.join() for t in ts]
wall = time.perf_counter() - s
okr = [r for r in res if r and r['tokens'] and r['tpot']]
out['decode_c30'] = {'wall_s': wall, 'ok': len(okr)}
out['summary'] = {
    'prefill_2k_mean_s': st.mean(iso),
    'tpot_single_ms': 1000 * st.mean(x['tpot'] for x in single if x['tpot']),
    'decode_c30_tpot_ms': 1000 * st.mean(r['tpot'] for r in okr),
    'decode_c30_gen_tok_per_s': sum(r['tokens'] for r in okr) / wall,
}
print('BENCH_RESULT ' + json.dumps(out))
