"""Builds the Italian text corpus that render_dataset_it.py fills the customer's template with.

Usage: python3 build_corpus.py <dir with the parquet files> <out corpus_it.json.gz> [--articles N]

Sources (Hugging Face, parquet conversion of the train split; download commands in README.md):
  - crux82/squad_it (SQuAD v1.1 translated to Italian): paragraphs + short questions + answers.
    The paragraphs become the documents of the chat requests' `instructions`, the questions their
    user messages. Consecutive paragraphs come from the same Wikipedia article, so a run of them
    reads as one topic.
  - ARTeLab/ilpost and ARTeLab/fanpage (Italian news, CC BY 4.0): article bodies (`source`) are
    the texts of the single-string task requests; their summaries (`target`) fill short texts.
Needs pyarrow (the renderer itself is stdlib only). The corpus is derived data: not in git.
"""
import argparse
import gzip
import json
import os
import random

import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('out')
    p.add_argument('--articles', type=int, default=16000, help='news articles kept (half per source)')
    a = p.parse_args()

    s = pq.read_table(os.path.join(a.src, 'squad_it_train.parquet')).to_pydict()
    contexts, ctx_index, qa = [], {}, []
    for c, q in zip(s['context'], s['question']):
        c, q = ' '.join(c.split()), ' '.join(q.split())
        if c not in ctx_index:
            ctx_index[c] = len(contexts)
            contexts.append(c)
        if 10 <= len(q) <= 300:
            qa.append([ctx_index[c], q])

    rnd = random.Random(42)
    articles, summaries = [], []
    for name in ('ilpost', 'fanpage'):
        t = pq.read_table(os.path.join(a.src, '%s_train.parquet' % name)).to_pydict()
        idx = rnd.sample(range(len(t['source'])), min(a.articles // 2, len(t['source'])))
        for i in sorted(idx):
            src, tgt = ' '.join(t['source'][i].split()), ' '.join(t['target'][i].split())
            if len(src) >= 300:
                articles.append(src)
            if len(tgt) >= 40:
                summaries.append(tgt)

    corpus = {'contexts': contexts, 'qa': qa, 'articles': articles, 'summaries': summaries,
              'sources': ['crux82/squad_it train', 'ARTeLab/ilpost train', 'ARTeLab/fanpage train']}
    with gzip.open(a.out, 'wt', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False)
    print('contexts %d, questions %d, articles %d, summaries %d -> %s (%.1f MB)' % (
        len(contexts), len(qa), len(articles), len(summaries), a.out, os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
