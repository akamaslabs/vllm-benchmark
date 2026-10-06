"""Runs inside vllm-0 of study 30 (startup probe). Prints one BENCH_RESULT JSON line.

Prefill: a ~2000-token prompt of random words, 1 output token, mean of 4 after one warm-up
(one scheduler step at max_num_batched_tokens 2048): the compute-bound side of the kernels.
Decode: natural-language prompts that ask for a long essay, 256 output tokens with
ignore_eos (fixed length; the essays rarely end earlier, so ignore_eos seldom bites), so
that speculative decoding (MTP) drafts on realistic text:
  - single: 3 requests one at a time, English (TPOT at batch 1);
  - c64: 64 concurrent English requests (TPOT and generated tokens/s near the expected
    capacity);
  - it32: 32 concurrent Italian requests (the customer's language).
With MTP on, the acceptance rate of each phase comes from vLLM's own counters
(vllm:spec_decode_num_accepted_tokens / num_draft_tokens), read before and after it.
Client-side wall times: they include HTTP and scheduling, so compare combinations, not
absolute numbers. The prompt length is reported from vLLM's usage field, not assumed.
"""
import json
import random
import statistics as st
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:8000/v1/chat/completions'
METRICS = 'http://127.0.0.1:8000/metrics'
MODEL = 'gemma4-26b-l40s'
VOCAB = ['alpha', 'river', 'stone', 'quantum', 'market', 'silver', 'engine', 'forest', 'number', 'signal',
         'orange', 'planet', 'memory', 'window', 'garden', 'rocket', 'yellow', 'bridge', 'castle', 'dragon']
TOPICS_EN = ['the history of the printing press', 'how vaccines train the immune system', 'the economics of renewable energy',
             'why the Roman Empire declined', 'how a compiler turns code into machine instructions',
             'the role of bees in agriculture', 'the causes of inflation', 'how GPS works',
             'the life of Marie Curie', 'the water cycle', 'how neural networks learn',
             'the architecture of Gothic cathedrals', 'climate change and ocean currents',
             'the invention of the telephone', 'how the stock market works', 'the human digestive system']
TOPICS_IT = ['la storia del Rinascimento italiano', 'come funziona il sistema sanitario', 'le cause della prima guerra mondiale',
             'la cucina regionale italiana', 'come si forma un vulcano', 'la vita di Leonardo da Vinci',
             'il funzionamento di una centrale idroelettrica', 'la storia della lingua italiana',
             "l'economia del turismo in Italia", 'come funzionano le batterie al litio',
             'la Divina Commedia di Dante', 'il cambiamento climatico nel Mediterraneo',
             'la storia di Venezia', "come nasce un'automobile in fabbrica", 'il sistema solare',
             'la Costituzione italiana']


def mk(seed, words):
    rnd = random.Random(seed)
    return ' '.join(rnd.choice(VOCAB) + str(rnd.randint(0, 999)) for _ in range(words))


def essay(i, lang):
    if lang == 'it':
        return 'Scrivi un saggio dettagliato di almeno 600 parole su %s, con introduzione, sviluppo e conclusione.' % TOPICS_IT[i % len(TOPICS_IT)]
    return 'Write a detailed essay of at least 600 words about %s, with an introduction, a body and a conclusion.' % TOPICS_EN[i % len(TOPICS_EN)]


def post(body):
    return urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                         {'Content-Type': 'application/json'}), timeout=600)


def spec_counters():
    """(accepted, drafted) tokens so far; (None, None) without speculative decoding."""
    acc = drf = None
    try:
        text = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:  # a probe must not die on a metrics read
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


def prefill(seed):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': mk(seed, 500)}], 'max_tokens': 1}
    s = time.perf_counter()
    r = json.loads(post(body).read())
    return time.perf_counter() - s, r.get('usage', {}).get('prompt_tokens')


def stream(content, ntok):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': ntok,
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
    # With MTP several tokens can arrive in one chunk: n counts chunks, so TPOT is per chunk
    # there; tokens/s below uses max_tokens, which ignore_eos makes exact.
    return {'ttft': ttft, 'tpot': (tot - ttft) / max(n - 1, 1) if ttft else None, 'chunks': n, 'wall': tot}


def concurrent(n, lang, ntok):
    res = [None] * n

    def w(i):
        res[i] = stream(essay(i, lang), ntok)

    before = spec_counters()
    ts = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    s = time.perf_counter()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.perf_counter() - s
    okr = [r for r in res if r and r['chunks']]
    return {'n': n, 'ok': len(okr), 'wall_s': wall,
            'gen_tok_per_s': len(okr) * ntok / wall,
            'e2e_mean_s': st.mean(r['wall'] for r in okr) if okr else None,
            'per_request_tok_per_s': st.mean(ntok / (r['wall'] - r['ttft']) for r in okr if r['ttft']) if okr else None,
            'acceptance': acceptance(before, spec_counters())}


out = {}
seed = int(time.time()) % 100000
prefill(seed)
iso = [prefill(seed + 10 + i) for i in range(4)]
out['prefill_s'] = [x[0] for x in iso]
out['prefill_prompt_tokens'] = iso[0][1]
before = spec_counters()
single = [stream(essay(i, 'en'), 128) for i in range(3)]
out['single'] = {'per_request_tok_per_s': st.mean(128 / (x['wall'] - x['ttft']) for x in single if x['ttft']),
                 'acceptance': acceptance(before, spec_counters())}
time.sleep(2)
out['c64_en'] = concurrent(64, 'en', 256)
time.sleep(2)
out['c32_it'] = concurrent(32, 'it', 256)
out['summary'] = {
    'prefill_mean_s': st.mean(out['prefill_s']),
    'tok_per_s_single': out['single']['per_request_tok_per_s'],
    'c64_gen_tok_per_s': out['c64_en']['gen_tok_per_s'],
    'c64_per_request_tok_per_s': out['c64_en']['per_request_tok_per_s'],
    'c32_it_gen_tok_per_s': out['c32_it']['gen_tok_per_s'],
    'acceptance_single_en': out['single']['acceptance'],
    'acceptance_c64_en': out['c64_en']['acceptance'],
    'acceptance_c32_it': out['c32_it']['acceptance'],
}
print('BENCH_RESULT ' + json.dumps(out))
