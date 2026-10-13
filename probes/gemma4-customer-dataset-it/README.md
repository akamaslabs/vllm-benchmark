# gemma4-customer-dataset-it (probe, not a study)

**Status:** DONE 2026-10-13. Feeds study 35 (`../../studies/35-l40s-gemma4-26b-awq-replay-users/`).

## Why

The customer's replay dataset arrives as a **template**: every text replaced by a placeholder of
its length. Filled with filler text (study 34: Shakespeare) the model answers nonsense at lengths
of its own; with an output cap and `ignore_eos` every request does fixed work, but the text after
EOS is degenerate (MTP's acceptance on it is not real) and the cap is hard to defend to the
customer. This probe fills the template with **real Italian texts** whose answers come out, by
calibration, with **the customer's output-length distribution** (their production numbers,
`usage_stats.png`, 19,204 requests, thinking off), with no cap.

## The template

Two request shapes (10 sessions, 2848 requests):

| Shape | Requests | Structure | Characters |
|---|---|---|---|
| chat | 1544 (54 %) | `instructions` + history (1-9 messages) + a user question | instructions p50 9,457, p90 25,989, p99 45,852 (892 lengths); questions p50 36-41; history replies p50 325, p90 1,590 |
| task | 1304 (46 %) | one string `input` | p10 186, p50 1,035, p90 12,125, p99 61,167 |

## How it is filled (`render_dataset_it.py`, stdlib only)

- **chat:** a short assistant preamble + SQuAD-it paragraphs (`crux82/squad_it`) as numbered
  documents in `instructions` (same length -> same text for every user); user messages = SQuAD-it
  questions about those documents; the history's replies = passages of the same documents.
- **task:** a task prompt + Il Post / Fanpage articles (`ARTeLab/ilpost`, `ARTeLab/fanpage`, CC BY
  4.0) cut to length. Tasks and weights: title 0.25, one sentence 0.20, paragraph 0.17, category
  0.12, keywords 0.12, detailed summary 0.07, rewrite 0.04 (inputs up to 16k characters),
  translation into English 0.25 (inputs of 8k characters or more only: the long tail).
- Same placeholder -> same text (the history repeats in later turns, prefixes are cached as in
  the template); delays, model name, sampling and `stream` kept; `text` dropped; no output cap.
- `build_corpus.py` (pyarrow) builds the 17 MB corpus from the three datasets' train parquet files.

## Calibration (`calibrate.py`, 2026-10-13)

The customer's compose on vLLM 0.29.0, one L40S (vllm-bench-gatling, `k8s/`), requests sampled
from 10 rendered users, sent one by one to `/v1/responses` (not streaming), 32 concurrent.
Reasoning tokens: 0 in every run (thinking off by default).

| Output tokens | n | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|
| **Customer (production)** | 19,204 | 13 | 19 | 38 | 100 | 242 | 502 | 2035 | 132 |
| v1: "concise" preamble, no long task | 600 | 14 | 19 | 32 | 63 | 179 | 451 | 920 | |
| v2: "complete" preamble, rewrite to 16k chars | 600 | 15 | 23 | 44 | 97 | 302 | 461 | 1057 | |
| **v3 (kept): + translation of long texts, detailed 0.10 -> 0.07** | 900 | 14 | 21 | 43 | 91 | 224 | 445 | 2518 | 146 |

v3 per task, output tokens p50 (p90): qa 48 (146), title 15 (19), category 3 (33), keywords 17 (21),
sentence 49 (76), paragraph 121 (142), detailed 456 (748), rewrite 430 (1352), translate 3501 (6698).
Input tokens p50 / p90 / p99: v3 1569 / 6042 / 12419; their synthetic dataset 1354 / 5933 / 13186
(tiktoken); their real one 1464 / 8064 / 17732 (the template is 10 of their 100 users).

**Limits:** the contents are ours (the structure, lengths, sessions, pauses and reuse are theirs);
which texts are equal is guessed from lengths; configurations that change numerics (fp8 KV) can
shift the generated lengths a little.

## Reproduce

```bash
# parquet files (train split): crux82/squad_it, ARTeLab/ilpost, ARTeLab/fanpage, as
#   <dir>/{squad_it,ilpost,fanpage}_train.parquet (URLs in ../../studies/35-*/k8s/05-job_template.yaml)
python3 build_corpus.py <dir> corpus_it.json.gz                          # needs pyarrow
python3 render_dataset_it.py dataset_u000.template.jsonl corpus_it.json.gz 10 r10.jsonl
PARAMS=$PWD/k8s/compose.params.env KUBE_CONTEXT=lab-vllm-bench-gatling bash k8s/apply_config.sh
kubectl --context lab-vllm-bench-gatling -n llm-serving port-forward pod/vllm-w4a16-0 18000:8000 &
python3 calibrate.py r10.jsonl http://127.0.0.1:18000 cal.jsonl --sample 900
```

`k8s/` is the W4A16 kernel probe's serving setup (`../l40s-gemma4-w4a16-kernels/k8s/`) with the
served name `aiservices_gemma`. Data files (template, corpus, renders, calibration records) are not
in git (`.gitignore`).
