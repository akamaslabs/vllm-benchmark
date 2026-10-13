"""Renders the customer's anonymized AIPerf template with Italian texts, so the model answers real
questions and tasks and its output lengths are natural (no cap, no ignore_eos).

Usage: python3 render_dataset_it.py <template.jsonl[.gz]> <corpus_it.json.gz> <users> <out.jsonl>
                                    [--seed 42] [--mix mix.json]
Stdlib only (runs where the load generator runs). Writes <out.jsonl> (mooncake_trace, AIPerf
sends each `payload` verbatim) and <out.jsonl>.meta.jsonl (one line per request: session, kind,
task), which calibrate.py uses to split the output lengths by request type.

Same replay rules as study 34's render_dataset.py: user k replays template session k mod S with
its own texts (session id <session_id>_c<k // S>); delays, model, temperature, top_p, store and
stream are kept; the `text` field ("{{text: omitted}}", a structured-output format) is dropped.

The template has two request shapes (README.md, "The template"); each is filled so that it means
something:
  - chat (instructions + message list): `instructions` = an assistant preamble + SQuAD-it
    paragraphs as numbered documents, cut to the placeholder's length. Key = the length alone,
    shared by every user (as study 34: same length taken as the same prompt). User messages =
    SQuAD-it questions about those documents, the closest in length; assistant messages (the
    history: fixed text, never generated) = passages of the same documents. Key = (user,
    instructions length, role, length), so a conversation's history repeats in its later turns.
  - task (a single string `input`): a task prompt + news articles (Il Post, Fanpage) cut to the
    length. The task (title, category, keywords, one sentence, paragraph, detailed summary,
    rewrite, translation of long texts) is drawn from --mix by the key (user, length), so a repeated input repeats exactly.
The mix and the preamble decide the output lengths: calibrate.py measures them against the
customer's (p50 38, p90 242, p99 2035 tokens) and the mix is tuned until they match.
Exit 2 on a placeholder this script does not know.
"""
import argparse
import gzip
import hashlib
import json
import re
import sys

PLACEHOLDER = re.compile(r'\{\{(\w+): (\d+) chars\}\}')
OMITTED = re.compile(r'\{\{(\w+): omitted\}\}')

PREAMBLE = ("Sei un assistente virtuale. Rispondi in italiano alle domande dell'utente in modo chiaro "
            "e completo, spiegando il contesto, usando solo le informazioni contenute nei documenti "
            "seguenti. Se un'informazione non è nei documenti, dillo.\n\nDocumenti:\n")

TASKS = {  # name: (prompt, weight, max input chars or 0 = any, min input chars)
    'category': ("Classifica il seguente testo in una di queste categorie: politica, economia, cronaca, "
                 "sport, cultura, tecnologia, salute, esteri. Rispondi solo con la categoria.\n\n", 0.12, 0, 0),
    'title': ("Scrivi un titolo di massimo 10 parole per il seguente testo. Rispondi solo con il "
              "titolo.\n\n", 0.25, 0, 0),
    'keywords': ("Estrai cinque parole chiave dal seguente testo, separate da virgole.\n\n", 0.12, 0, 0),
    'sentence': ("Riassumi il seguente testo in una sola frase.\n\n", 0.20, 0, 0),
    'paragraph': ("Riassumi il seguente testo in un paragrafo di 3-5 frasi.\n\n", 0.17, 0, 0),
    'detailed': ("Scrivi un riassunto dettagliato del seguente testo, con un elenco puntato dei punti "
                 "principali.\n\n", 0.07, 0, 0),
    'rewrite': ("Correggi eventuali errori e riscrivi integralmente il seguente testo in un italiano "
                "chiaro e formale.\n\n", 0.04, 16000, 0),
    # The long tail (their p99 2035, max 8608 output tokens): output about as long as a long input.
    'translate': ("Traduci integralmente in inglese il seguente testo.\n\n", 0.25, 0, 8000),
}


def die(msg):
    print('error: ' + msg, file=sys.stderr)
    sys.exit(2)


class Renderer:
    def __init__(self, corpus, seed, mix):
        self.c, self.seed = corpus, seed
        self.questions_of = {}
        for i, (ctx, _) in enumerate(corpus['qa']):
            self.questions_of.setdefault(ctx, []).append(i)
        self.tasks = []
        for name, (prompt, weight, max_in, min_in) in TASKS.items():
            w = mix.get(name, weight)
            if w > 0:
                self.tasks.append((name, prompt, w, max_in, min_in))
        self.docs_cache = {}

    def h(self, *key):
        return int.from_bytes(hashlib.sha256(('%s|%r' % (self.seed, key)).encode()).digest()[:8], 'big')

    @staticmethod
    def cut(s, n):
        if len(s) <= n:
            return s
        i = s.rfind(' ', 0, n)
        return s[:i if i > n * 0.9 else n]

    def run_of(self, pool, start, n, sep):
        """Consecutive items of pool from start, joined, until n characters."""
        out, total, i = [], 0, start
        while total < n:
            item = pool[i % len(pool)]
            out.append(item)
            total += len(item) + len(sep)
            i += 1
        return out

    def documents(self, n):
        """Instructions of n characters and the SQuAD-it paragraphs they hold."""
        if n not in self.docs_cache:
            ctxs = self.c['contexts']
            start = self.h('instructions', n) % len(ctxs)
            body = n - len(PREAMBLE)
            idx, total, i = [], 0, start
            while total < max(body, 1):
                idx.append(i % len(ctxs))
                total += len(ctxs[i % len(ctxs)]) + 20
                i += 1
            text = PREAMBLE + '\n'.join('[%d] %s' % (k + 1, ctxs[j]) for k, j in enumerate(idx))
            self.docs_cache[n] = (self.cut(text, n), idx)
        return self.docs_cache[n]

    def question(self, docs, user, n):
        pool = [q for d in docs for q in self.questions_of.get(d, [])] or range(len(self.c['qa']))
        near = sorted(pool, key=lambda q: (abs(len(self.c['qa'][q][1]) - n), self.h(user, n, q)))[:5]
        return self.c['qa'][near[self.h('q', user, n) % len(near)]][1]

    def passage(self, docs, user, n):
        ctxs = self.c['contexts']
        d = docs[self.h('a', user, n) % len(docs)]
        text = ' '.join(ctxs[(d + k) % len(ctxs)] for k in range(1 + n // 400))
        start = self.h('a-start', user, n) % max(1, len(ctxs[d]) // 2)
        dot = text.find('. ', start)
        text = text[dot + 2:] if 0 <= dot < len(text) - n else text
        return self.cut(text, n)

    def task(self, user, n):
        ok = [t for t in self.tasks if (not t[3] or n <= t[3]) and n >= t[4]]
        r = (self.h('task', user, n) % 10 ** 6) / 10 ** 6 * sum(t[2] for t in ok)
        for name, prompt, w, _, _ in ok:
            r -= w
            if r < 0:
                break
        body_n = max(40, n - len(prompt))
        pool = self.c['articles'] if body_n > 250 else self.c['summaries']
        start = self.h('body', user, n) % len(pool)
        body = self.cut('\n\n'.join(self.run_of(pool, start, body_n, '\n\n')), body_n)
        return name, prompt + body


def main():
    p = argparse.ArgumentParser()
    p.add_argument('template')
    p.add_argument('corpus')
    p.add_argument('users', type=int)
    p.add_argument('out')
    p.add_argument('--seed', default='42')
    p.add_argument('--mix', help='JSON file {task: weight}, overrides the default weights')
    a = p.parse_args()
    if a.users < 1:
        die('users must be positive')
    opener = gzip.open if a.template.endswith('.gz') else open
    with opener(a.template, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with gzip.open(a.corpus, 'rt', encoding='utf-8') as f:
        corpus = json.load(f)
    mix = json.load(open(a.mix)) if a.mix else {}
    unknown = set(mix) - set(TASKS)
    if unknown:
        die('unknown tasks in --mix: %s' % ', '.join(sorted(unknown)))
    r = Renderer(corpus, a.seed, mix)

    sessions = {}
    for row in rows:
        sessions.setdefault(row['session_id'], []).append(row)
    order = list(sessions)

    def length(value):
        m = PLACEHOLDER.fullmatch(value) if isinstance(value, str) else None
        if not m:
            if isinstance(value, str) and '{{' in value:
                die('unknown placeholder %r' % value[:80])
            return None
        return m.group(1), int(m.group(2))

    n_req = 0
    with open(a.out, 'w', encoding='utf-8') as f, open(a.out + '.meta.jsonl', 'w', encoding='utf-8') as fm:
        for u in range(a.users):
            src = order[u % len(order)]
            sid = '%s_c%d' % (src, u // len(order))
            for row in sessions[src]:
                pl = {k: v for k, v in row['payload'].items() if not (isinstance(v, str) and OMITTED.fullmatch(v))}
                if any(k in pl for k in ('max_output_tokens', 'max_tokens')):
                    die('session %s carries an output cap' % src)
                meta = {'session_id': sid}
                if isinstance(pl.get('input'), str):
                    ph = length(pl['input'])
                    if ph:
                        meta['task'], pl['input'] = r.task(u, ph[1])
                    meta['kind'] = 'task'
                else:
                    ph = length(pl.get('instructions'))
                    docs = []
                    if ph:
                        pl['instructions'], docs = r.documents(ph[1])
                    instr_n = ph[1] if ph else 0
                    msgs = []
                    for msg in pl.get('input', []):
                        msg = dict(msg)
                        ph_m = length(msg.get('content'))
                        if ph_m:
                            role, n = ph_m
                            if role == 'assistant':
                                msg['content'] = r.passage(docs or [0], (u, instr_n), n)
                            elif role in ('user', 'input'):
                                msg['content'] = r.question(docs, (u, instr_n), n)
                            else:
                                die('unknown message placeholder %r' % role)
                        msgs.append(msg)
                    pl['input'] = msgs
                    meta['kind'] = 'chat'
                    meta['task'] = 'qa'
                rec = {'session_id': sid}
                if 'delay' in row:
                    rec['delay'] = row['delay']
                rec['payload'] = pl
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fm.write(json.dumps(meta) + '\n')
                n_req += 1
    print('rendered %d users (%d template sessions), %d requests -> %s' % (a.users, len(order), n_req, a.out))


if __name__ == '__main__':
    main()
