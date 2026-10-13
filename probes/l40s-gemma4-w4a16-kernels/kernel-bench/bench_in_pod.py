"""Runs inside vllm-w4a16-0 (W4A16 kernel probe). Prints one BENCH_RESULT JSON line.

Derived from study 30's / study 2's bench_in_pod.py, re-aimed at the customer's load (prompts of
3k tokens median, 6k p90, 15k p99, 95k max; agentic, prefix caching on). Phases:

- prefill: unique random-word prompts of ~3k / 6k / 15k tokens (and ~95k with BENCH_LONG=1),
  max_tokens 1, median of 3 after one warm-up per length. Unique prompts so prefix caching
  (on, as at the customer) never hits. Wall time = prefill + one decode step + HTTP.
- decode: B concurrent essay requests (natural text, so MTP drafts on realistic tokens),
  256 output tokens with ignore_eos, B in 1 (3 requests one at a time), 8, 16, 32, 48, 64;
  per-request tokens/s after the first token, aggregate generated tokens/s, MTP acceptance.
- mixed: closed loop of C workers, each request = ~3k tokens of unique context + an essay
  instruction, 150 output tokens (ignore_eos), C in 8 / 16 / 32, 10 s warm + 45 s measured:
  e2e p50/p95, completed req/s, total tokens/s. Prefill and decode share the steps, as at the
  customer.
- prefix: 3 pairs of (turn 1: ~3k unique tokens, max_tokens 1; turn 2: the same + ~300 new
  tokens) -> turn-2 TTFT, the cost of a follow-up turn with the history in the prefix cache.
- quality: 40 synthetic arithmetic word problems with known answers, temperature 0, 16
  concurrent; accuracy. Guards the numerics-changing variants (W4A8 activations, fp8 KV,
  f16 accumulation) against a silent quality loss; the reference is B-default's score.
Client-side wall times include HTTP: compare combinations, not absolute numbers.
"""
import json
import os
import random
import re
import statistics as st
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:8000/v1/chat/completions'
METRICS = 'http://127.0.0.1:8000/metrics'
MODEL = 'gemma4-w4a16'
LONG = os.environ.get('BENCH_LONG') == '1'
VOCAB = ['alpha', 'river', 'stone', 'quantum', 'market', 'silver', 'engine', 'forest', 'number', 'signal',
         'orange', 'planet', 'memory', 'window', 'garden', 'rocket', 'yellow', 'bridge', 'castle', 'dragon']
TOPICS = ['the history of the printing press', 'how vaccines train the immune system', 'the economics of renewable energy',
          'why the Roman Empire declined', 'how a compiler turns code into machine instructions',
          'the role of bees in agriculture', 'the causes of inflation', 'how GPS works',
          'the life of Marie Curie', 'the water cycle', 'how neural networks learn',
          'the architecture of Gothic cathedrals', 'climate change and ocean currents',
          'the invention of the telephone', 'how the stock market works', 'the human digestive system']
TOK_PER_WORD = 4.0   # "river123"-style words, refined from the first prefill's usage
RUN = random.Random(time.time_ns())


def words(n, rnd):
    return ' '.join(rnd.choice(VOCAB) + str(rnd.randint(0, 999)) for _ in range(n))


def post(body, timeout=900):
    return urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                         {'Content-Type': 'application/json'}), timeout=timeout)


def spec_counters():
    acc = drf = None
    try:
        text = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return None, None
    for line in text.splitlines():
        if line.startswith('vllm:spec_decode_num_accepted_tokens_total'):
            acc = (acc or 0) + float(line.split()[-1])
        elif line.startswith('vllm:spec_decode_num_draft_tokens_total'):
            drf = (drf or 0) + float(line.split()[-1])
    return acc, drf


def acceptance(before, after):
    if None in before or None in after or after[1] - before[1] <= 0:
        return None
    return (after[0] - before[0]) / (after[1] - before[1])


def one_token(content):
    s = time.perf_counter()
    r = json.loads(post({'model': MODEL, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': 1}).read())
    return time.perf_counter() - s, r.get('usage', {}).get('prompt_tokens')


def prefill_len(target):
    global TOK_PER_WORD
    n = int(target / TOK_PER_WORD)
    one_token(words(n, RUN))                      # warm-up: first use of this batch shape
    samples = [one_token(words(n, RUN)) for _ in range(3)]
    toks = samples[0][1]
    if toks:
        TOK_PER_WORD = toks / n
    return {'prompt_tokens': toks, 's': [x[0] for x in samples], 'median_s': st.median(x[0] for x in samples)}


def stream(content, ntok):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': ntok,
            'stream': True, 'ignore_eos': True}
    s = time.perf_counter()
    ttft, n = None, 0
    for line in post(body):
        if line.startswith(b'data:') and b'[DONE]' not in line:
            c = json.loads(line[5:]).get('choices') or []
            d = (c[0].get('delta') or {}) if c else {}
            if d.get('content') or d.get('reasoning') or d.get('reasoning_content'):
                n += 1
                if ttft is None:
                    ttft = time.perf_counter() - s
    return {'ttft': ttft, 'chunks': n, 'wall': time.perf_counter() - s}


def essay(i):
    return 'Write a detailed essay of at least 600 words about %s, with an introduction, a body and a conclusion.' % TOPICS[i % len(TOPICS)]


def decode(b, ntok=256):
    before = spec_counters()
    if b == 1:
        res = [stream(essay(i), ntok) for i in range(3)]
        wall = sum(r['wall'] for r in res)
    else:
        res = [None] * b

        def w(i):
            res[i] = stream(essay(i), ntok)
        ts = [threading.Thread(target=w, args=(i,)) for i in range(b)]
        s = time.perf_counter()
        [t.start() for t in ts]
        [t.join() for t in ts]
        wall = time.perf_counter() - s
    okr = [r for r in res if r and r['ttft']]
    per_req = [ntok / (r['wall'] - r['ttft']) for r in okr]
    return {'b': b, 'ok': len(okr), 'gen_tok_per_s': len(okr) * ntok / wall,
            'per_request_tok_per_s': st.mean(per_req) if per_req else None,
            'ms_per_token': 1000 / st.mean(per_req) if per_req else None,
            'ttft_mean_s': st.mean(r['ttft'] for r in okr) if okr else None,
            'acceptance': acceptance(before, spec_counters())}


def mixed(c, ctx_tokens=3000, ntok=150, warm_s=10, measure_s=45):
    lock, done = threading.Lock(), []
    t0 = time.perf_counter()
    before = [None]

    def w(k):
        rnd = random.Random(time.time_ns() + k)
        i = 0
        while time.perf_counter() - t0 < warm_s + measure_s:
            content = ('Context notes: ' + words(int(ctx_tokens / TOK_PER_WORD), rnd) + '\n\n' + essay(k + i))
            i += 1
            s = time.perf_counter()
            try:
                r = json.loads(post({'model': MODEL, 'messages': [{'role': 'user', 'content': content}],
                                     'max_tokens': ntok, 'ignore_eos': True}).read())
                u = r.get('usage', {})
            except Exception:
                u = None
            e = time.perf_counter()
            with lock:
                done.append((e - t0, e - s, u))

    def snap():
        time.sleep(warm_s)
        before[0] = spec_counters()
    sn = threading.Thread(target=snap)
    sn.start()
    ts = [threading.Thread(target=w, args=(k,)) for k in range(c)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    sn.join()
    win = [d for d in done if warm_s <= d[0] <= warm_s + measure_s]
    okr = [d for d in win if d[2]]
    e2e = sorted(d[1] for d in okr)
    pt = sum(d[2].get('prompt_tokens', 0) for d in okr)
    gt = sum(d[2].get('completion_tokens', 0) for d in okr)
    return {'c': c, 'requests': len(win), 'failed': len(win) - len(okr),
            'req_per_s': len(okr) / measure_s, 'total_tok_per_s': (pt + gt) / measure_s,
            'e2e_p50_s': e2e[len(e2e) // 2] if e2e else None,
            'e2e_p95_s': e2e[int(0.95 * (len(e2e) - 1))] if e2e else None,
            'prompt_tokens_mean': pt / len(okr) if okr else None,
            'acceptance': acceptance(before[0], spec_counters()) if before[0] else None}


def prefix_counters():
    """(hits, queries) tokens of vLLM's prefix cache so far."""
    h = q = 0.0
    try:
        text = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith('vllm:prefix_cache_hits_total'):
            h += float(line.split()[-1])
        elif line.startswith('vllm:prefix_cache_queries_total'):
            q += float(line.split()[-1])
    return h, q


def prefix_pairs(ctx_tokens=3000, new_tokens=300, n=8):
    # Stage 1 saw some turn-2 requests cost as much as a cold prefill (0.2 s against 0.05 s):
    # 8 pairs and vLLM's own hit counters per turn 2 tell a miss from a slow hit.
    out = []
    for _ in range(n):
        rnd = random.Random(time.time_ns())
        hist = words(int(ctx_tokens / TOK_PER_WORD), rnd)
        t1, n1 = one_token(hist)
        c0 = prefix_counters()
        t2, n2 = one_token(hist + ' ' + words(int(new_tokens / TOK_PER_WORD), rnd))
        c1 = prefix_counters()
        hit = (c1[0] - c0[0]) if c0 and c1 else None
        out.append({'turn1_s': t1, 'turn1_prompt_tokens': n1, 'turn2_s': t2, 'turn2_prompt_tokens': n2,
                    'turn2_hit_tokens': hit})
    return {'pairs': out, 'turn2_median_s': st.median(p['turn2_s'] for p in out),
            'misses': sum(1 for p in out if p['turn2_s'] > 2 * min(q['turn2_s'] for q in out))}


def problems(n=40):
    rnd = random.Random(32)
    names = ['Anna', 'Marco', 'Luca', 'Sara', 'Giulia', 'Paolo', 'Elena', 'Davide']
    items = ['apples', 'books', 'pencils', 'boxes', 'tickets', 'bottles', 'cards', 'shirts']
    out = []
    for _ in range(n):
        nm, it = rnd.choice(names), rnd.choice(items)
        a, b, c, d = rnd.randint(20, 90), rnd.randint(2, 9), rnd.randint(3, 12), rnd.randint(5, 40)
        ans = a + b * c - d
        q = (f'{nm} has {a} {it}. {nm} buys {b} packs with {c} {it} each, then gives away {d} {it}. '
             f'How many {it} does {nm} have now? Reason step by step, then write the final answer '
             f'on the last line as "#### <number>".')
        out.append((q, ans))
    return out


def quality():
    probs = problems()
    res = [None] * len(probs)
    sem = threading.Semaphore(16)

    def w(i):
        with sem:
            q, ans = probs[i]
            try:
                r = json.loads(post({'model': MODEL, 'messages': [{'role': 'user', 'content': q}],
                                     'max_tokens': 512, 'temperature': 0}).read())
                txt = r['choices'][0]['message'].get('content') or ''
            except Exception:
                txt = ''
            m = re.findall(r'####\s*(-?\d+)', txt) or re.findall(r'(-?\d+)', txt)
            res[i] = (m[-1] == str(ans)) if m else False
    ts = [threading.Thread(target=w, args=(i,)) for i in range(len(probs))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return {'n': len(probs), 'correct': sum(bool(x) for x in res), 'accuracy': sum(bool(x) for x in res) / len(probs)}


out = {}
out['prefill'] = {str(L): prefill_len(L) for L in ([3000, 6000, 15000] + ([95000] if LONG else []))}
time.sleep(2)
out['decode'] = [decode(b) for b in (1, 8, 16, 32, 48, 64)]
time.sleep(2)
out['mixed'] = [mixed(c) for c in (8, 16, 32)]
time.sleep(2)
out['prefix'] = prefix_pairs()
out['quality'] = quality()
out['summary'] = {
    'prefill_3k_s': out['prefill']['3000']['median_s'],
    'prefill_6k_s': out['prefill']['6000']['median_s'],
    'prefill_15k_s': out['prefill']['15000']['median_s'],
    'prefill_95k_s': out['prefill'].get('95000', {}).get('median_s'),
    'ms_per_token': {str(d['b']): d['ms_per_token'] for d in out['decode']},
    'gen_tok_per_s_64': out['decode'][-1]['gen_tok_per_s'],
    'mixed_c16_e2e_p50_s': out['mixed'][1]['e2e_p50_s'],
    'mixed_c16_req_per_s': out['mixed'][1]['req_per_s'],
    'mixed_c32_total_tok_per_s': out['mixed'][2]['total_tok_per_s'],
    'prefix_turn2_s': out['prefix']['turn2_median_s'],
    'acceptance_b1': out['decode'][0]['acceptance'],
    'acceptance_b32': out['decode'][3]['acceptance'],
    'quality': out['quality']['accuracy'],
}
print('BENCH_RESULT ' + json.dumps(out))
