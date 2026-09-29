"""Runs inside the vllm-pd engine container. Prints one JSON line with the results."""
import json, random, re, statistics as st, threading, time, urllib.request

VOCAB = ['alpha', 'river', 'stone', 'quantum', 'market', 'silver', 'engine', 'forest', 'number', 'signal',
         'orange', 'planet', 'memory', 'window', 'garden', 'rocket', 'yellow', 'bridge', 'castle', 'dragon']
KV = {'do_remote_decode': True, 'do_remote_prefill': False, 'remote_engine_id': None,
      'remote_block_ids': None, 'remote_host': None, 'remote_port': None}
PF, RT, DC = 'http://127.0.0.1:8100', 'http://127.0.0.1:8000', 'http://127.0.0.1:8200'


def mk(seed, words):
    random.seed(seed)
    return ' '.join(random.choice(VOCAB) + str(random.randint(0, 999)) for _ in range(words))


def post(url, body, stream=False):
    return urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                                                         {'Content-Type': 'application/json'}), timeout=600)


def prefill(seed):  # ~4090-token prompt straight to prefill-0, as the router sends it
    body = {'model': 'qwen3-8b', 'messages': [{'role': 'user', 'content': mk(seed, 1045)}], 'max_tokens': 1,
            'stream': False, 'kv_transfer_params': KV}
    s = time.perf_counter(); post(PF + '/v1/chat/completions', body).read(); return time.perf_counter() - s


def e2e(seed, words, ntok):
    body = {'model': 'qwen3-8b', 'messages': [{'role': 'user', 'content': mk(seed, words)}], 'max_tokens': ntok,
            'stream': True, 'ignore_eos': True}
    s = time.perf_counter(); r = post(RT + '/v1/chat/completions', body); ttft = None; n = 0
    for line in r:
        if line.startswith(b'data:') and b'[DONE]' not in line:
            c = json.loads(line[5:]).get('choices') or []
            if c and (c[0].get('delta') or {}).get('content'):
                n += 1
                if ttft is None: ttft = time.perf_counter() - s
    tot = time.perf_counter() - s
    return {'ttft': ttft, 'tpot': (tot - ttft) / max(n - 1, 1) if ttft else None, 'tokens': n, 'total': tot}


out = {}
seed = int(time.time()) % 100000
prefill(seed)  # warm-up (JIT, first-request effects)
iso = [prefill(seed + 10 + i) for i in range(4)]
out['prefill_isolated_s'] = iso
ov = []
for k in range(2):
    res = {}
    t = threading.Thread(target=lambda: res.__setitem__('a', prefill(seed + 100 + k))); t.start(); time.sleep(0.9)
    b = prefill(seed + 200 + k); t.join(); ov.append({'first': res['a'], 'second': b}); time.sleep(2)
out['prefill_overlap_0.9s'] = ov
time.sleep(2)
single = []
for i in range(5):
    single.append(e2e(seed + 300 + i, 1045, 64)); time.sleep(1.5)
out['e2e_single_4k_in_64_out'] = single
# Decode regime: 16 concurrent short prompts (~256 tokens), 256 output tokens each.
time.sleep(3)
res = [None] * 16
def w(i): res[i] = e2e(seed + 500 + i, 64, 256)
ts = [threading.Thread(target=w, args=(i,)) for i in range(16)]
s = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.perf_counter() - s
ok = [r for r in res if r and r['tokens']]
out['decode_c16'] = {'wall_s': wall, 'gen_tok_per_s': sum(r['tokens'] for r in ok) / wall,
                     'tpot_mean_ms': 1000 * st.mean(r['tpot'] for r in ok), 'ttft_mean_s': st.mean(r['ttft'] for r in ok),
                     'ok': len(ok)}
out['summary'] = {'prefill_iso_mean_s': st.mean(iso), 'overlap_first_mean_s': st.mean(x['first'] for x in ov),
                  'overlap_second_mean_s': st.mean(x['second'] for x in ov),
                  'e2e_ttft_mean_s': st.mean(x['ttft'] for x in single), 'e2e_tpot_mean_ms': 1000 * st.mean(x['tpot'] for x in single),
                  'decode_c16_gen_tok_per_s': out['decode_c16']['gen_tok_per_s'], 'decode_c16_tpot_ms': out['decode_c16']['tpot_mean_ms']}
print('BENCH_RESULT ' + json.dumps(out))
