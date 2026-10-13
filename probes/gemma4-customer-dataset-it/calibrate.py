"""Measures the output lengths the rendered dataset makes the model produce, against the customer's.

Usage: python3 calibrate.py <rendered.jsonl> <base url, e.g. http://127.0.0.1:8000> <out.jsonl>
                            [--sample 600] [--concurrency 32] [--seed 1]
A stratified sample of the rendered requests (same share of chat / task requests and of each task
as the whole file, via its .meta.jsonl) is sent to <base>/v1/responses one by one, out of order
and without the sessions' delays: only the output length matters here, not the timing. Each
payload is sent as rendered except `stream` (false, to read `usage` in one response). Writes one
line per request (kind, task, input/output tokens, seconds) and prints the output-token
percentiles overall and per task next to the customer's (usage_stats.png, 19,204 production
requests, thinking off). Stdlib only.
"""
import argparse
import collections
import json
import random
import threading
import time
import urllib.request

TARGET = {10: 13, 25: 19, 50: 38, 75: 100, 90: 242, 95: 502, 99: 2035}


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * (len(v) - 1) + 0.5))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rendered')
    ap.add_argument('base')
    ap.add_argument('out')
    ap.add_argument('--sample', type=int, default=600)
    ap.add_argument('--concurrency', type=int, default=32)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.rendered, encoding='utf-8')]
    meta = [json.loads(line) for line in open(a.rendered + '.meta.jsonl', encoding='utf-8')]
    rnd = random.Random(a.seed)
    idx = rnd.sample(range(len(rows)), min(a.sample, len(rows)))
    todo, lock, done = list(idx), threading.Lock(), []

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop()
            body = dict(rows[i]['payload'], stream=False)
            s = time.perf_counter()
            try:
                req = urllib.request.Request(a.base.rstrip('/') + '/v1/responses', json.dumps(body).encode(),
                                             {'Content-Type': 'application/json'})
                r = json.loads(urllib.request.urlopen(req, timeout=900).read())
                u = r.get('usage') or {}
                rec = {'i': i, 'kind': meta[i]['kind'], 'task': meta[i].get('task'),
                       'input_tokens': u.get('input_tokens'), 'output_tokens': u.get('output_tokens'),
                       'reasoning_tokens': (u.get('output_tokens_details') or {}).get('reasoning_tokens'),
                       'seconds': time.perf_counter() - s, 'status': r.get('status')}
            except Exception as e:  # recorded, not fatal
                rec = {'i': i, 'kind': meta[i]['kind'], 'task': meta[i].get('task'), 'error': str(e)[:200]}
            with lock:
                done.append(rec)

    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    with open(a.out, 'w') as f:
        for rec in done:
            f.write(json.dumps(rec) + '\n')
    ok = [d for d in done if d.get('output_tokens') is not None]
    print('%d requests, %d ok, %d errors, %.0f s' % (len(done), len(ok), len(done) - len(ok), time.time() - t0))
    out = [d['output_tokens'] for d in ok]
    print('%-12s %5s ' % ('', 'n') + ' '.join('p%-5d' % p for p in TARGET))
    print('%-12s %5s ' % ('customer', '') + ' '.join('%-6d' % v for v in TARGET.values()))
    print('%-12s %5d ' % ('rendered', len(out)) + ' '.join('%-6d' % pct(out, p) for p in TARGET))
    groups = collections.defaultdict(list)
    for d in ok:
        groups[d['task']].append(d['output_tokens'])
    for g, v in sorted(groups.items()):
        print('%-12s %5d ' % (g, len(v)) + ' '.join('%-6d' % pct(v, p) for p in TARGET))
    rs = [d['reasoning_tokens'] for d in ok if d.get('reasoning_tokens')]
    print('requests with reasoning tokens: %d (must be 0: thinking is off by default)' % len(rs))


if __name__ == '__main__':
    main()
