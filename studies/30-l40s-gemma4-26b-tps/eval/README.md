# eval/ — study 30 accuracy check (baseline vs best)

**Status:** DONE 2026-10-08: built, smoke-tested and run (12:33-13:49 UTC) after the user
stopped study 30. Verdict: **no degradation** on GSM8K Platinum and IFEval (see "Results").

## Question

**Does the best configuration found by study 30 answer worse than the baseline, and by how
much?** Study 30 scores tokens/s only. Three of its 13 parameters can change what the model
answers. The other ten change nothing but the batch-dependent numerical noise.

| Parameter | Why it can change the answers |
|---|---|
| `kv_cache_dtype=fp8` | **the prime suspect.** The RedHat checkpoint ships no KV-cache scales, so vLLM stores K/V in fp8 e4m3 with scale 1.0, and says so at start: "Using fp8 data type to store kv cache ... it may cause accuracy drop without a proper scaling factor" (`../probe/results/K-fp8.log`). RedHat's card validates the FP8 weights, not an fp8 KV cache. |
| `spec_method=mtp` | lossless by construction (greedy verification keeps the target's argmax; rejection sampling keeps its distribution), but the K+1-token verify step runs different kernel shapes, and a bug would show here. |
| `linear_backend` | `auto` (Cutlass) and `torch` compute the same W8A8 FP8 product with different kernels; `marlin` is W8A16 (activations stay bf16), if anything closer to bf16. |

## Tool and reference

**lm-evaluation-harness 0.4.13** (EleutherAI, 2026-08-31), model type
`local-chat-completions` against the **deployed** vLLM (`vllm-0`, the StatefulSet that
`../k8s/apply_config.sh` starts), so every tuned flag is in effect. Never `--model vllm`:
it starts its own engine and ignores the study's flags.

Reference: RedHat validated this checkpoint with lm-eval on a vLLM server
(<https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic>, "Without thinking"
column, which matches study 30's fixed `enable_thinking: false`):

| Task (0-shot, chat template) | bf16 | FP8 (this checkpoint) |
|---|---|---|
| GSM8K Platinum, strict-match | 95.43 | 95.37 |
| IFEval, prompt-level strict | 89.96 | 89.34 |
| IFEval, instruction-level strict | 93.21 | 92.69 |

RedHat ran a neuralmagic fork of lm-eval; the two task files used here
(`gsm8k-platinum-cot-llama.yaml`, `ifeval.yaml` + `utils.py`) have the same git blob in
the fork's `main` and upstream `v0.4.13`, so upstream reproduces their task definitions.
Not used (decided 2026-10-08): MMLU-Pro (`mmlu_pro_chat` exists only in the fork, ~40 min
per configuration), lighteval's MATH-500 / AIME / GPQA / LiveCodeBench (long sampled
generations, nothing they add for this question).

## Tasks

- **`gsm8k_platinum_cot_llama`**, 1209 problems (`madrylab/gsm8k-platinum`, test),
  `--num_fewshot 0` (the task defaults to 8). Multi-step reasoning, ~200-300 generated
  tokens per answer: the decode path where an fp8 KV cache and MTP act. Primary metric
  `exact_match,strict-match` (the card's); `flexible-extract` reported.
- **`ifeval`**, 541 prompts (`google/IFEval`). Primary metric `prompt_level_strict_acc`;
  `inst_level_strict_acc` reported.

## Protocols

Both with `--apply_chat_template`, `--log_samples`, `--seed 1234`, model_args
`model=gemma4-26b-l40s,base_url=http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200,max_length=4096`
(the card's string with the served name and the study's context), and
**`max_gen_toks=2048`**. The explicit `max_gen_toks` is the one flag that must not be
dropped: `gsm8k_platinum_cot_llama` sets none, and lm-eval's API default (256 tokens) would
cut most chains of thought.

- **greedy** (every comparison): `do_sample=false,temperature=0`. Deterministic except for
  batch-dependent numerics, so a difference is attributable item by item.
- **card** (anchor, baseline only, once): `do_sample=true,temperature=1.0,top_p=0.95,
  top_k=64,seed=1234`, the card's sampling.

**`max_model_len` stays 4096**, the study's fixed flag: the point is the deployment as
tuned. The card used 69632 with `max_gen_toks=32000`. vLLM 0.29.0 rejects (HTTP 400,
`VLLMValidationError` in `vllm/renderers/params.py`) a request whose prompt exceeds 4096 -
`max_tokens`; with 2048 that leaves 2048 prompt tokens, against 0-shot prompts of a few
hundred. An answer cut at 2048 tokens counts as wrong in both configurations; the runner
counts them (vLLM's `vllm:request_success_total{finished_reason="length"}` before and after
each task) and the comparison reports them.

**Limit of the result:** these contexts stay under ~2.5k tokens, as study 30's own traffic
does (ShareGPT prompts p99 794 tokens), and an fp8 KV cache's error grows with context
length. "No degradation" here says nothing about long-context use.

## Configurations

Each one is a `params.env` for `../k8s/apply_config.sh`, the startup probe's pattern
(`../probe/probe.sh`: a base file plus overrides). Every configuration gets its own vLLM
start (`apply_config.sh` always scales to 0 and back).

| Run | params.env | Protocols | Why |
|---|---|---|---|
| `baseline-a` | `configs/baseline.params.env` (experiment 1's values) | greedy, card | the reference; card = the anchor |
| `best` | `configs/best.params.env` | greedy | the question |
| `best-kv-auto` | best + `KV_CACHE_DTYPE=auto` | greedy | isolates the fp8 KV cache (only if best uses fp8) |
| `best-no-mtp` | best + `SPEC_METHOD=none SPEC_TOKENS=0` | greedy | isolates MTP (only if best uses MTP) |
| `baseline-b` | `configs/baseline.params.env` | greedy | the noise floor: same configuration, new start |

`configs/best.params.env` is **experiment 28** (8418.13 total tokens/s, +234 %, goal
VALID): `akamas describe study` "best configuration" on 2026-10-08, the study stopped after
31 experiments. It uses an fp8 KV cache and MTP (K=3), so both ablations run;
`linear_backend=torch` is W8A8 like the baseline's `auto`. A
`linear_backend` ablation is added only if the two above do not explain a degradation.

## Decision rule (fixed before the best configuration is evaluated)

For each task, on the primary metric, paired by `doc_id`:

- **Delta** = best - baseline-a, with a 95 % CI from a paired bootstrap (10,000 resamples,
  fixed seed), and an exact McNemar p-value on the discordant pairs. The same for
  baseline-b - baseline-a (the noise floor) and for each ablation against baseline-a.
- **Margin:** **1.0 point on GSM8K Platinum, 2.0 points on IFEval.** One verdict per task:

  | CI lower bound | CI upper bound | Verdict |
  |---|---|---|
  | above -margin | >= 0 | **no degradation** |
  | above -margin | < 0 | **measurable drop within the margin** (report the delta) |
  | <= -margin | < 0 | **degradation** (report the delta with its CI) |
  | <= -margin | >= 0 | **inconclusive** |

- The margins: 1 point on GSM8K Platinum is ~1 % relative, the usual "99 % recovery" bar
  (the card's FP8 recovers 99.9 %); IFEval has 541 prompts, and with ~25 discordant pairs a
  paired delta has a standard error of ~0.9 points, so a 1-point margin could never be met.
- **Flips** (right -> wrong, wrong -> right) are listed for every pair; best's flip count is
  read against baseline-b's: about the same number means numerical noise, not a lever.
- **Anchor check, before any comparison:** baseline-a under the card protocol should land
  within 2 standard errors of the card's FP8 column. That column is a mean of 3 seeds, so
  the gap from one run has ~1.15x a single run's error: GSM8K Platinum 95.37 +- 1.4,
  IFEval prompt-strict 89.34 +- 3.1. Outside that, find out why (prompt or chat-template
  mismatch, truncations) before reading any delta; it is a prompt to investigate, not an
  automatic fail.
- **Truncations:** more than 1 % of a task's answers ending on `length` in any run means
  `max_gen_toks` is too small for this model; stop and rerun with a larger one.

## Pieces

| File | Does |
|---|---|
| `configs/baseline.params.env` | experiment 1's values (study YAML, `baseline` step) |
| `configs/best.params.env` | from the export, at the end of the study |
| `eval_job_template.yaml` | Job on `system-m8a` (as AIPerf): waits for `vllm-0`'s `/health`, `pip install "lm_eval[api,ifeval]==0.4.13"` in `python:3.12-slim`, runs the tasks under the run's protocols, snapshots vLLM's `finished_reason` counters around each task, writes everything under `/benchmarks/eval/<run>/` on the `aiperf-results` PVC (HF datasets cached on `hf-cache`), then a `DONE` marker and a 30-minute sleep for the copy |
| `render_eval_job.sh` | renders the template (`@RUN@`, `@PROTOCOLS@`); exit 2 on bad input or a leftover token, as `../k8s/render_job.sh` |
| `run_eval.sh` | per run: base + overrides -> `params.env`, `apply_config.sh`, render + apply the Job, wait for `DONE`, `kubectl cp` to `results/<run>/`, delete the Job; skips an ablation the best configuration does not need; refuses to start while an `aiperf-l40s` Job exists (a live experiment); ends with vLLM at 0 |
| `compare.py` | stdlib only: reads the `samples_*.jsonl` of two runs, pairs by `doc_id`, computes delta / CI / McNemar / flips / truncations, applies the rule above; writes `results/comparison.md` |
| `results/<run>/` | lm-eval's `results_*.json`, `samples_*.jsonl.gz`, the vLLM finished-reason counts, the apply log |

## Tests (before any GPU time)

- `tests/test_compare.sh`: hand-made samples files with known deltas, discordant counts
  and verdicts (no degradation / degradation / inconclusive), plus the refusals (different
  `doc_id` sets, different task versions).
- `tests/test_render_eval_job.sh`: tokens rendered, bad input rejected, nothing left over.
- `tests/test_run_eval.sh`: with a `kubectl` stub (as `../k8s/tests/stub/kubectl`): the run
  order, the skipped ablations, the refusal while an `aiperf-l40s` Job exists.
- **Cluster smoke** (`run_eval.sh --limit 20 baseline-a`): the real vLLM, 20 items per
  task under both protocols. It checks the flags, the datasets download, the output layout,
  the samples format `compare.py` reads, and the `finished_reason` counters. It replaces the
  local dry run against a mock planned at first: the GPU was free from 2026-10-08, the day
  study 30 was stopped. The `max_tokens`-above-context 400 is checked once during the full
  run, while vLLM is up. The `finished_reason` labels were read on 2026-10-08 from
  `vllm-0`'s `/metrics`: `stop`, `length`, `abort`, `error`, `repetition`.

**Smoke, 2026-10-08 12:25-12:32 UTC** (`run_eval.sh --limit 20 baseline-a`, exit 0, ~2 min
of evaluation after a ~5 min vLLM start). Layout as in "Pieces", one `results_*.json` and
one `samples_*.jsonl.gz` per protocol and task. Sample keys: `doc_id`, `filter`,
`exact_match` (GSM8K, one line per filter, `strict-match` and `flexible-extract`);
`prompt_level_strict_acc` (bool) and `inst_level_strict_acc` (list of bools) under filter
`none` (IFEval). Counters: +20 `stop`, +0 `length` per task. 20-item scores, the same under
both protocols: GSM8K Platinum 100 (strict and flexible), IFEval prompt-strict 85,
inst-strict 90. The card run did sample (0 of 40 GSM8K responses and 2 of 20 IFEval
responses identical to greedy); the equal scores are a coincidence at n=20. `compare.py` on
the smoke (baseline-a copied as baseline-b): delta 0, no flips, both anchors flagged, as
expected at n=20. lm-eval logs "Tokenized requests are disabled. Context + generation length
is not checked": the context limit is enforced by vLLM only.

**The 400, 2026-10-08 12:38 UTC** (baseline-a up, `max_tokens` 4096 on a 68-character
prompt): `400 ... This model's maximum context length is 4096 tokens. However, you requested
4096 output tokens ...`. vLLM rejects, it does not clamp: a request too long for the context
would fail visibly (lm-eval retries, then errors), never be silently shortened.

## When

`apply_config.sh` scales vLLM to 0: running this while study 30 is RUNNING kills the live
experiment. It runs after the optimize step ends (or in an agreed pause), with the
g6e.xlarge node up and tagged `AlwaysOn` if it runs past 17:00 UTC. Budget: 5 starts x
~5-6 min, ~5-7 min of greedy evaluation per run (~1750 requests, ~0.5 M generated tokens),
~6 min for the card run: **~70-80 min, ~2.5 USD** of g6e.xlarge.

## Results

Full run 2026-10-08 12:33-13:49 UTC (`run_eval.sh`, exit 0; `results/run.log`): five vLLM
starts, ~5 min each, and 5-10 min of evaluation per greedy run (best: 5 min, the fastest).
`results/comparison.md` is `compare.py`'s full output; `results/<run>/` holds lm-eval's
`results_*.json`, the gzipped samples, the logs and the `params.env` each run was started
with. No warnings: both anchors within tolerance, at most 5 of 541 answers (0.9 %) ending on
`length` in any run.

**Anchor** (baseline-a, card protocol): GSM8K Platinum **95.37** (card 95.37), IFEval
prompt-strict **89.28** (card 89.34). The setup reproduces RedHat's evaluation, so the deltas
below measure the configuration, not the harness.

**Accuracy, greedy** (primary metric; secondary in brackets):

| Run | GSM8K Platinum strict (flexible) | IFEval prompt-strict (inst-strict) | Truncated GSM8K / IFEval |
|---|---|---|---|
| baseline-a | 95.20 (95.20) | 88.72 (92.33) | 0 / 3 |
| baseline-b | 95.37 (95.37) | 88.54 (92.21) | 0 / 3 |
| **best** (exp 28) | **95.86** (95.86) | **89.65** (92.81) | 0 / 3 |
| best-kv-auto | 95.20 (95.20) | 88.91 (92.57) | 1 / 3 |
| best-no-mtp | 95.62 (95.62) | 89.46 (93.05) | 0 / 5 |

**Paired against baseline-a** (B - A, 95 % paired-bootstrap CI, flips lost / gained, exact
McNemar p):

| Task | B | Delta | 95 % CI | Lost / gained | p | Verdict |
|---|---|---|---|---|---|---|
| GSM8K Platinum | best | +0.66 | [+0.17, +1.24] | 2 / 10 | 0.039 | no degradation |
| GSM8K Platinum | best-kv-auto | +0.00 | [-0.50, +0.41] | 4 / 4 | 1 | no degradation |
| GSM8K Platinum | best-no-mtp | +0.41 | [-0.17, +0.99] | 4 / 9 | 0.27 | no degradation |
| GSM8K Platinum | baseline-b (noise) | +0.17 | [-0.25, +0.66] | 3 / 5 | 0.73 | no degradation |
| IFEval | best | +0.92 | [-0.55, +2.40] | 6 / 11 | 0.33 | no degradation |
| IFEval | best-kv-auto | +0.18 | [-1.48, +1.85] | 11 / 12 | 1 | no degradation |
| IFEval | best-no-mtp | +0.74 | [-1.11, +2.59] | 11 / 15 | 0.56 | no degradation |
| IFEval | baseline-b (noise) | -0.18 | [-0.55, +0.00] | 1 / 0 | 1 | no degradation |

**Verdict: the best configuration does not degrade either task.** Both CIs sit entirely
above the margins (-1.0 GSM8K, -2.0 IFEval); the point estimates are positive.

- **Not a gain either.** GSM8K's +0.66 (p 0.039) is partly baseline-a's luck: 5 of best's 12
  flips are problems that also flip between the two baseline starts (doc 146, 604, 712, 916,
  1153), and against baseline-b the delta is +0.50, CI [+0.00, +0.99], p 0.11 (computed ad
  hoc, not in `comparison.md`). IFEval against baseline-b: +1.11, CI [-0.37, +2.59], p 0.21.
  Read: same accuracy as the baseline, within noise.
- **Flips against the noise floor.** The same configuration restarted flips 8 GSM8K problems
  and 1 IFEval prompt; best flips 12 and 17, every ablation 8-26. Any change of numerics
  (fp8 KV, MTP's verify shapes, torch vs Cutlass, batch shapes) moves a few borderline answers
  each way, balanced or slightly in best's favour; none of the three levers shows a one-sided
  loss.
- **fp8 KV cache without scales** (best vs best-kv-auto): 95.86 vs 95.20 and 89.65 vs 88.91,
  no sign of the accuracy drop vLLM's startup warning allows for, at these context lengths
  (prompt + answer under ~2.5k tokens; see "Limit of the result").
- **MTP, K=3** (best vs best-no-mtp): 95.86 vs 95.62 and 89.65 vs 89.46: lossless, as expected.
