"""Per-request output caps for the rendered replay: every payload gets max_output_tokens = the
output length the model produced for that request in a calibration run, and ignore_eos, so every
trial asks vLLM for the same tokens (study 34's load determinism test, ../README.md "Load").

Usage:
  python3 caps.py measure <dataset.jsonl> <base url> <caps.json.gz> [--users 200] [--ramp-s 6000]
                          [--concurrency 128] [--timeout-s 900]
  python3 caps.py apply   <dataset.jsonl> <caps.json.gz>
Stdlib only (runs in the AIPerf Job's python:3.12-slim).

measure (once, by calibrate_caps.sh, with natural outputs): sends every request the study's ramp
can reach, as rendered (no cap) but with stream false, to <base>/v1/responses and records
usage.output_tokens per request. Reachable: user k (in file order) starts at ramp_s * k / users
(k < users; later users at the earliest end of a session, the template's shortest think time) and
its requests follow at their delays; a request whose start, latency left out, is past ramp_s
is left out. Best effort (it assumes AIPerf opens the k-th slot no earlier than a linear ramp
would): a request a trial reaches outside the set gets its task median, and the caps file, not the
coverage, is what makes every trial ask for the same tokens. Requests are keyed by
"<session_id>/<turn>" (turn = position in its session, AIPerf's turn_index). The output also
holds the sha256 of <dataset.jsonl> and each task's median (the fallback for a request the
calibration did not reach or that failed).

apply (every trial, in the Job, after render_dataset_it.py): refuses (exit 2) a dataset whose
sha256 differs from the calibration's (another template, corpus, mix, seed or user count: the
caps would belong to other texts), then writes max_output_tokens and ignore_eos into every
payload, in place, and prints one CAPS line (counts and the caps' percentiles).
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request


def die(msg):
    print('error: ' + msg, file=sys.stderr)
    sys.exit(2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    """Rows of the rendered dataset, their keys and their meta (kind, task)."""
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    meta_path = path + '.meta.jsonl'
    if not os.path.exists(meta_path):
        die('%s missing (render_dataset_it.py writes it next to the dataset)' % meta_path)
    meta = [json.loads(line) for line in open(meta_path, encoding='utf-8')]
    if len(meta) != len(rows):
        die('%s: %d lines for %d requests' % (meta_path, len(meta), len(rows)))
    turns, keys = collections.Counter(), []
    for r in rows:
        sid = r['session_id']
        keys.append('%s/%d' % (sid, turns[sid]))
        turns[sid] += 1
    return rows, keys, meta


def reachable(rows, users, ramp_s):
    """Indexes of the requests a ramp of `users` over `ramp_s` s can send (an upper bound)."""
    sessions = collections.OrderedDict()
    for i, r in enumerate(rows):
        sessions.setdefault(r['session_id'], []).append(i)
    think = {s: sum(rows[i].get('delay', 0) for i in idx) / 1000 for s, idx in sessions.items()}
    earliest_end = min(think.values())
    out = []
    for k, (s, idx) in enumerate(sessions.items()):
        t = ramp_s * k / users if k < users else earliest_end
        for i in idx:
            t += rows[i].get('delay', 0) / 1000
            if t >= ramp_s:
                break
            out.append(i)
    return out


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * (len(v) - 1) + 0.5))] if v else None


def measure(a):
    rows, keys, meta = load(a.dataset)
    todo = reachable(rows, a.users, a.ramp_s)
    n_all = len(todo)
    print('caps measure: %d requests of %d reachable (users %d over %d s), concurrency %d'
          % (n_all, len(rows), a.users, a.ramp_s, a.concurrency), flush=True)
    todo.reverse()  # pop() takes them in file order: a session's history is cached for its next turn
    lock, caps, errors, reasoning = threading.Lock(), {}, collections.Counter(), [0]

    def send(i):
        body = dict(rows[i]['payload'], stream=False)
        req = urllib.request.Request(a.base.rstrip('/') + '/v1/responses', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        r = json.loads(urllib.request.urlopen(req, timeout=a.timeout_s).read())
        u = r.get('usage') or {}
        return u.get('output_tokens'), (u.get('output_tokens_details') or {}).get('reasoning_tokens') or 0

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop()
            out = None
            for attempt in (1, 2):  # one retry: a transient error must not cost a cap
                try:
                    out, rt = send(i)
                    break
                except Exception as e:  # recorded below, the request falls back to its task median
                    err = type(e).__name__
            with lock:
                if out is None:
                    errors[err] += 1
                else:
                    caps[keys[i]] = out
                    reasoning[0] += rt > 0

    t0 = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    [t.start() for t in threads]
    next_progress = time.time() + 60
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        if time.time() < next_progress:
            continue
        next_progress += 60
        with lock:
            done = len(caps) + sum(errors.values())
        el = time.time() - t0
        eta = el / done * (n_all - done) if done else 0
        print('%s caps measure: %d / %d done, %d errors, %.0f s elapsed, ~%.0f s left'
              % (time.strftime('%H:%M:%S', time.gmtime()), done, n_all, sum(errors.values()), el, eta), flush=True)
    by_task = collections.defaultdict(list)
    idx = {k: i for i, k in enumerate(keys)}
    for k, n in caps.items():
        by_task[meta[idx[k]].get('task') or meta[idx[k]]['kind']].append(n)
    out = {'dataset_sha256': sha256(a.dataset), 'requests': len(rows), 'reachable': n_all,
           'users': a.users, 'ramp_s': a.ramp_s, 'measured': len(caps), 'errors': dict(errors),
           'requests_with_reasoning': reasoning[0], 'seconds': round(time.time() - t0),
           'task_median': {t: pct(v, 50) for t, v in sorted(by_task.items())}, 'caps': caps}
    with gzip.open(a.out + '.tmp', 'wt', encoding='utf-8') as f:
        json.dump(out, f)
    os.replace(a.out + '.tmp', a.out)
    v = list(caps.values())
    print('caps measure: %d measured, %d errors %s, %d with reasoning tokens (must be 0), %.0f s'
          % (len(caps), sum(errors.values()), dict(errors), reasoning[0], time.time() - t0))
    print('caps measure: output tokens p10 %s p25 %s p50 %s p75 %s p90 %s p95 %s p99 %s max %s'
          % tuple(pct(v, p) for p in (10, 25, 50, 75, 90, 95, 99, 100)))
    print('caps measure: task medians %s' % out['task_median'])
    if not caps:
        die('no request measured')


def apply(a):
    with gzip.open(a.caps, 'rt', encoding='utf-8') as f:
        c = json.load(f)
    got = sha256(a.dataset)
    if got != c['dataset_sha256']:
        die('dataset %s (sha256 %s) is not the calibrated one (%s): another template, corpus, mix, '
            'seed or user count; recalibrate (calibrate_caps.sh)' % (a.dataset, got[:12], c['dataset_sha256'][:12]))
    rows, keys, meta = load(a.dataset)
    medians, caps = c['task_median'], c['caps']
    fallback_all = pct(list(caps.values()), 50)
    n_cal = n_med = 0
    applied = []
    for r, k, m in zip(rows, keys, meta):
        if k in caps:
            n, n_cal = caps[k], n_cal + 1
        else:
            n, n_med = medians.get(m.get('task') or m['kind'], fallback_all), n_med + 1
        n = max(1, int(n))
        r['payload']['max_output_tokens'] = n
        r['payload']['ignore_eos'] = True
        applied.append(n)
    with open(a.dataset + '.tmp', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(a.dataset + '.tmp', a.dataset)
    print('CAPS: %d requests, %d capped at their calibrated length, %d at their task median; '
          'max_output_tokens p50 %s p90 %s p99 %s max %s; ignore_eos on every request'
          % (len(rows), n_cal, n_med, pct(applied, 50), pct(applied, 90), pct(applied, 99), max(applied)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('measure')
    m.add_argument('dataset')
    m.add_argument('base')
    m.add_argument('out')
    m.add_argument('--users', type=int, default=200)
    m.add_argument('--ramp-s', type=int, default=6000)
    m.add_argument('--concurrency', type=int, default=128)
    m.add_argument('--timeout-s', type=int, default=900)
    p = sub.add_parser('apply')
    p.add_argument('dataset')
    p.add_argument('caps')
    a = ap.parse_args()
    measure(a) if a.cmd == 'measure' else apply(a)


if __name__ == '__main__':
    main()
