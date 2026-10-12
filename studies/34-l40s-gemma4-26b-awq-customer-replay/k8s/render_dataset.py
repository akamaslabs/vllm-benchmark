"""Renders the customer's anonymized AIPerf dataset template into a replayable mooncake_trace file.

Usage: python3 render_dataset.py <template.jsonl[.gz]> <corpus.txt> <users> <out.jsonl> [seed]

The template (the customer's dataset_u000.template.jsonl, not in git: see ../load/README.md) is
their AIPerf replay dataset with every text replaced by a placeholder of the same length:
{"session_id": "user_0000", "delay": 521, "payload": {"model": ..., "instructions":
"{{instructions: 3113 chars}}", "input": [{"role": "user", "content": "{{user: 30 chars}}"}, ...]}}.
AIPerf 0.11.0 sends a mooncake `payload` verbatim and renders nothing, so this script fills every
placeholder with text of that many characters cut from <corpus.txt> (AIPerf's own
shakespeare.txt, the corpus of its synthetic prompts in studies 32-33), at an offset drawn from
the seed and a key:
  - instructions: the key is the length alone, shared by every user. The customer reports many
    repeated prompts in the data behind the template; two instructions of the same length are
    taken to be the same prompt (a guess: the template does not say).
  - user / assistant messages and string inputs: the key is (user, role, length), so a
    conversation's history repeats the same text in each later request of that user (its
    prefix is cached as in production), while two users never share it.
The template's sessions are replayed in order, `users` of them: user k is template session
k mod S (S sessions in the template) with its own texts, named <session_id>_c<k // S>. AIPerf
samples trace conversations sequentially and starts the next one when a session ends, so
`users` should exceed the run's concurrency (render_job.sh). The `text` field, "{{text:
omitted}}" in the template (an output format, structured output in their data), is dropped:
its content is unknown. Everything else (model, temperature, top_p, store, stream, delay) is
kept as the customer wrote it. Exit 2 on a placeholder this script does not know.
"""
import gzip
import hashlib
import json
import re
import sys

PLACEHOLDER = re.compile(r'\{\{(\w+): (\d+) chars\}\}')
OMITTED = re.compile(r'\{\{(\w+): omitted\}\}')


def die(msg):
    print('error: ' + msg, file=sys.stderr)
    sys.exit(2)


def main():
    if len(sys.argv) not in (5, 6):
        die('usage: render_dataset.py <template> <corpus> <users> <out> [seed]')
    template, corpus_path, users, out = sys.argv[1:5]
    seed = sys.argv[5] if len(sys.argv) == 6 else '42'
    if not users.isdigit() or int(users) < 1:
        die("users '%s' is not a positive integer" % users)
    users = int(users)
    opener = gzip.open if template.endswith('.gz') else open
    with opener(template, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    corpus = ' '.join(open(corpus_path, encoding='utf-8').read().split())
    if len(corpus) < 200000:
        die('corpus %s too short (%d chars)' % (corpus_path, len(corpus)))

    sessions = {}
    for r in rows:
        sessions.setdefault(r['session_id'], []).append(r)
    order = list(sessions)

    def text(key, n):
        h = int.from_bytes(hashlib.sha256(('%s|%s' % (seed, key)).encode()).digest()[:8], 'big')
        start = h % len(corpus)
        s = corpus[start:start + n]
        while len(s) < n:
            s += ' ' + corpus[:n - len(s) - 1]
        return s[:n]

    def fill(value, user):
        if isinstance(value, str):
            m = PLACEHOLDER.fullmatch(value)
            if not m:
                if '{{' in value:
                    die('unknown placeholder %r' % value[:80])
                return value
            name, n = m.group(1), int(m.group(2))
            key = ('instructions', n) if name == 'instructions' else (user, name, n)
            return text(key, n)
        if isinstance(value, list):
            return [fill(v, user) for v in value]
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if isinstance(v, str) and OMITTED.fullmatch(v):
                    continue
                out[k] = fill(v, user)
            return out
        return value

    n_req = 0
    with open(out, 'w', encoding='utf-8') as f:
        for u in range(users):
            src = order[u % len(order)]
            sid = '%s_c%d' % (src, u // len(order))
            for r in sessions[src]:
                rec = {'session_id': sid}
                if 'delay' in r:
                    rec['delay'] = r['delay']
                rec['payload'] = fill(r['payload'], u)
                if any(k in rec['payload'] for k in ('max_output_tokens', 'max_tokens')):
                    die('session %s carries an output cap' % src)
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                n_req += 1
    print('rendered %d users (%d template sessions), %d requests -> %s' % (users, len(order), n_req, out))


if __name__ == '__main__':
    main()
