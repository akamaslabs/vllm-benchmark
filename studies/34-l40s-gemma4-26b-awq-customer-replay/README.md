# 34-l40s-gemma4-26b-awq-customer-replay

**Status:** DESIGNED, local only (not pushed, not created on Akamas). Waits for study 33 to
finish or be stopped (same node, namespace, pod and Job name) and for the dataset template on
the toolbox (`load/README.md`).
**Dates:** scaffolded 2026-10-12 as a copy of study 33, the day the customer sent their load.

## Objective

**Study 32's setup (thinking on) under the customer's own load: how many requests per second
does one L40S complete with their Docker Compose setup of Gemma 4 26B-A4B when their users'
sessions are replayed with their own AIPerf command, and which vLLM configuration beats their
compose?** The rehearsal of the PoC's optimization, with the script and the dataset they will
run on site (there with the full dataset, 100 users with real texts; here with its anonymized
template, rendered).

On 2026-10-12 the customer sent `run_benchmark.sh` (their AIPerf command) and
`dataset_u000.template.jsonl` (their replay dataset with every text replaced by its length), and
wrote that their dataset's calls do not ask for reasoning but that they would keep it on, since
production cannot disable it (other use cases need it); the user chose thinking on.

## What changes from study 33

| Item | Study 33 | Study 34 | Where |
|---|---|---|---|
| Thinking | off (`enable_thinking` false) | **on** (`enable_thinking` true, study 32's) | `k8s/01-statefulset_template.yaml` |
| Served model name | `gemma4-26b-awq-nothink` | **`aiservices_gemma`** (the name their payloads carry; AIPerf sends them verbatim) | StatefulSet, Job, `run_test.sh`, `akamas/components/vllm.yaml` |
| API | `/v1/chat/completions` | **`/v1/responses`** (their `--endpoint-type responses`) | `k8s/05-job_template.yaml` |
| Load | AIPerf synthetic multi-turn chat, open-loop rate ramp 0 -> 6 req/s | **their `run_benchmark.sh`, unchanged**: `mooncake_trace` payloads, `--concurrency` users ramping 1 -> 100 over 6000 s, closed loop, sticky user sessions, think times capped at 60 s, seed 42; our values in its `benchmark.env` | `k8s/run_benchmark.sh`, `05-job_template.yaml`, `render_job.sh`, `run_test.sh` |
| Dataset | AIPerf's synthetic pool | **their template rendered** into 120 users (`k8s/render_dataset.py`), template on the toolbox, not in git | `load/`, `k8s/render_dataset.py` |
| Job warm-up | 8 synthetic chat requests | none (Apply config's warm-up stays) | `k8s/05-job_template.yaml` |
| Goal, constraint, parameters, steps | | unchanged (constraint provisional, "Open") | `akamas/34-L40S-Gemma4-AWQ-Replay.yaml` |

**What vLLM 0.29.0 does with their payloads** (read in the image, 2026-10-12): on `/v1/responses`
Gemma 4 thinks only for a request with `reasoning.effort` or through the server default
`--default-chat-template-kwargs '{"enable_thinking": true}'`
(`entrypoints/openai/responses/protocol.py` `build_chat_params`, `serving.py`
`_effective_chat_template_kwargs`; the CLI flag reaches the Responses handler,
`entrypoints/generate/api_router.py`); their payloads carry no `reasoning`, so with their 0.25.1
compose (reasoning parser only) their calls do not think. Thinking on here assumes their 0.29.0
deployment sets that default ("Open"). The trace streams as `response.reasoning_text.delta`,
the answer as `response.output_text.delta`; AIPerf 0.11.0 counts them apart (reasoning tokens,
time to the first output token).

## Stack & versions

- **Akamas version:** Studio 4.1.0 (`akamas41.lab.akamas.io`, namespace `akamas-41`).
- **Optimization packs:** vLLM **1.12.0** (component type `vLLM`; installed on the 4.1 server,
  `akamas list optimization-pack` 2026-10-12), GPU 1.4.0, Kubernetes 1.9.0 (metrics only).
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a` as `aiservices_gemma`, StatefulSet
  `vllm` (pod `vllm-0`) in namespace `llm-l40s`. Fixed flags: `--max-model-len=96000`,
  `--enable-auto-tool-choice`, `--tool-call-parser=gemma4`, `--reasoning-parser=gemma4`,
  `--default-chat-template-kwargs '{"enable_thinking": true}'`. Sampling: their payloads'
  temperature 0.1 (9 requests 0) and top_p 1; top_k from the checkpoint's
  `generation_config.json` (64). Pod: CPU request 3500m, no CPU limit, memory 28 GiB. The
  customer moved to vLLM 0.29.0 on their L40S on 2026-10-12 (their email), the version here.
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), node group
  `llm-serving-l40s-1xl`, one **g6e.xlarge** (1x NVIDIA L40S 48 GB, 4 vCPU / 32 GiB).
  Provisioning: `infra/` (study 30's, unchanged).
- **Load generator:** AIPerf 0.11.0 on the `system-m8a` node with the customer's flags ("Load").
  Their AIPerf version is not known (their script runs it from their own venv).
- **Telemetry:** Prometheus, 30 s samples, ServiceMonitor `vllm-l40s` (shared, already
  deployed), dcgm-exporter.
- **Model and KV cache (study 32's measurement on this checkpoint):** model 16.39 GiB, KV cache
  22.01 GiB = 708,768 tokens at `gpu_memory_utilization` 0.90 in bf16; 1,262,011 tokens in fp8
  with the MTP drafter.

## Parameters tuned

Study 32's 13, same domains: `gpu_memory_utilization` 0.80-0.94, `max_num_seqs` 16-512,
`max_num_batched_tokens` 2496-16384 (vision tower loaded), `kv_cache_dtype` {auto, fp8},
`performance_mode`, `optimization_level` 1-3, `scheduling_policy`, `async_scheduling`,
`max_cudagraph_capture_size` 16-512, `block_size` 16-128 (ordinal), `linear_backend` {auto,
triton, humming}, `spec_method` {none, mtp}, `spec_tokens` 0-4 (rationale: study 32's README,
"Parameters tuned"). **parameterConstraints:** `max_num_batched_tokens >= max_num_seqs`;
`spec_method != "none" || spec_tokens == 0`; `spec_method == "none" || spec_tokens > 0`.
**Not tuned:** as study 33 (`max_model_len`, prefix caching, the vision tower, tool calling,
`enforce_eager` false, no NCCL settings on one GPU). The vision tower stays loaded: the customer
reports a few image requests in production (not in the dataset).

## Load

**Their command**: `k8s/run_benchmark.sh` is their script byte for byte (sha256 checked by
`k8s/tests/test_render_job.sh`); the Job gives it what it expects (its `benchmark.env`, AIPerf
0.11.0 in `../llm-coding-benchmark/.venv`) and runs it with `NAME=<epoch>`. It runs
`aiperf profile --endpoint-type responses --input-file <dataset> --custom-dataset-type
mooncake_trace --concurrency USERS --concurrency-ramp-duration RAMP --benchmark-duration DURATION
--benchmark-grace-period GRACE --inter-turn-delay-cap-seconds CAP --random-seed 42
--connection-reuse-strategy sticky-user-sessions --use-server-token-count --streaming
--export-level records --ui simple`, `--tokenizer` from TOKENIZER (no revision flag in it: the
tokenizer comes from the checkpoint's main branch; token counts are vLLM's anyway). Their
`benchmark.env` was not sent; ours: ENDPOINT vllm-0, MODEL `aiservices_gemma`, USERS 100 (their
full dataset's users), RAMP = DURATION 6000 s, CAP 60 s (their dataset is capped there
already), GRACE 60 s.

**What it does** (AIPerf 0.11.0 source, checked 2026-10-12): the mooncake `payload` is sent
verbatim (`workers/inference_client.py`), so `--model` names only the tokenizer and the
model in the payloads (`aiservices_gemma`) is what vLLM must serve. `--concurrency` counts
sessions: a user holds its slot for its whole session (`timing/concurrency.py`), sends a request,
waits for the whole reply, then the trace's `delay`, then the next request
(`timing/strategies/request_rate.py` `handle_credit_return`, used by the concurrency phase): a
closed loop, the users of the SLA. Trace sessions are taken in order (`base_trace_loader.py`:
sequential); a finished session gives its slot to the next.

**Their dataset** (`load/README.md`): 10 users (`user_0000`-`0009`) of their 100, each ~1 h of
activity, 212-416 requests (mean 285), think times median 0.5 s (chains of calls) with 12 % at
the 60 s cap, mean 13 s. Per request median ~1,500 tokens, mean ~2,300, p99 ~13,000, max
~27,000 (characters / 4, an estimate); 54 % carry `instructions` and the conversation history.

**The render** (`k8s/render_dataset.py`, in the Job every trial, seed 42): each placeholder gets
text of its length from AIPerf's `shakespeare.txt` (the corpus of studies 32-33's synthetic
prompts). Instructions of one length are the same text for every user (their "many repeated
prompts", a guess: the template does not say which texts are equal); user, assistant and input
texts are per user, the same within a user, so a conversation's history repeats as in
production. User k replays template session k mod 10 with its own texts: 120 users for 100
concurrent, so no session repeats. `text` (an output format, 71 requests) is dropped. Measured
locally: 120 users = 34,176 requests, 311 MB, rendered in ~1 s; AIPerf sent its first request
~8 s after the start (`k8s/tests/dry_run_replay.sh` on their template, 100 users).

**Ramp sizing, PROVISIONAL.** With thinking on, a reply is ~1,500 tokens and ~20 s at low load
(study 32), so a user cycles in ~20 s + 13 s: ~0.03 req/s. Study 32's knees (compose ~0.59
req/s, best ~1.3) then fall near ~20 and ~45 users: inside a ramp to 100 (~1 user a minute),
the compose's at ~1/5 of it, the best near its middle. Experiment 1 checks it.

**Not like production:**
- The texts are Shakespeare, not their prompts: the model reasons and answers about them, so
  reply lengths are the model's on filler text (study 32: ~1,500 tokens with reasoning), not
  theirs (their history's answers, thinking off: median ~80 tokens).
- Which texts repeat is guessed (above); prefix-cache hits can differ from production either way.
- 10 template sessions behind 120 users: the same think-time patterns, 12 times over (with
  different texts, started at different times by the ramp).

### Load determinism test (2026-10-13, manual, outside Akamas)

**Question:** how much does the same load vary between two runs with the same seed, and what makes
it repeat? AIPerf 0.11.0 source: `--random-seed` already fixes everything AIPerf draws itself
(`common/random_generator.py`, one derived RNG per component: dataset, samplers, intervals, ramps),
and the render is seeded too. What still varies comes from the server: the closed loop (a user's
next request waits for the whole reply), the cut at `--benchmark-duration`, and the reply lengths
(the payloads carry no output cap).

**Setup:** a test pod in `llm-l40s` (`python:3.12-slim`, AIPerf 0.11.0, the Job's node); vLLM
0.29.0 still on study 33's last trial (thinking **off**, MTP 4 tokens, served name
`gemma4-26b-awq-nothink`), not this study's configuration. Their template rendered with
`k8s/render_dataset.py` (20 users, seed 42), kept to a 300 s window (318 requests; user k starts at
3k s, each next request 3 s + its `delay` later), the model in the payloads rewritten to the
served name. Their script's flags (`/v1/responses`, sticky sessions, server token counts,
streaming, seed 42); closed loop `--concurrency 20 --concurrency-ramp-duration 60
--benchmark-duration 300`. Three variants, two runs each:

| Variant | Requests (run 1 / 2) | Same output tokens | Send-time diff p90 / max | e2e mean (ms) | e2e p90 (ms) | MTP acceptance | Running seqs (avg) |
|---|---|---|---|---|---|---|---|
| natural (their payloads) | 257 / 231, 227 shared | 9 of 227 | 4.4 s / 20.6 s | 5111 / 4893 | 8917 / 8858 | 0.60 / 0.71 | 6.3 / 7.7 |
| capped: `max_output_tokens` 128 + `ignore_eos`, closed loop | 388 / 387 | 387 of 387 | 1.3 s / 2.3 s | 1191 / 1174 | 1667 / 1553 | 0.41 / 0.42 | 1.4 / 1.4 |
| capped + `timestamp` per record (`--fixed-schedule`) | 318 / 318 | 318 of 318 | 0.00 s / 0.00 s | 1132 / 1169 | 1525 / 1596 | not read | not read |

(Capped closed-loop runs wrapped the dataset: the 300 s window's sessions ended at ~160 s and
AIPerf's sequential sampler started them again, ~70 requests a run; the full sessions of the
study, ~1 h each, do not wrap.)

**Findings:**
- **Their replay as is does not repeat its load:** run 2 sent 11 % fewer requests and 18 % fewer
  output tokens; 9 replies in 227 had the same length (temperature 0.1: the model stops at
  different points), and MTP acceptance moved from 0.60 to 0.71. The e2e looks stable (~4 %) over
  two different workloads.
- **`max_output_tokens` + `ignore_eos` work on `/v1/responses`** (vLLM 0.29.0; checked by hand
  too: `status: incomplete`, `incomplete_details.reason: max_output_tokens`, exactly N output
  tokens, also at temperature 0.7). `max_output_tokens` alone is only a ceiling (a 10-token reply
  stays 10). Not checked on their 0.25.1.
- **With the cap the requests and their tokens repeat exactly; with `--fixed-schedule` the send
  times repeat too** (0.00 s apart). What is left is the server's own noise: ~3-5 % on e2e mean and
  p90.
- **`ignore_eos` lowers MTP acceptance** (0.41 against 0.60-0.71): the tokens after the EOS are
  hard to draft. The capped runs' lower ITL (~8 ms against ~12.6 ms) comes from the lighter load
  (1.4 sequences running against 6-8), not from MTP.
- **A fixed cap of 128 is not their load:** natural replies here were ~420 tokens on average
  (p50 ~440, p90 ~730), so the capped load was ~3x lighter (e2e ~1.2 s against ~5 s).
- **`--fixed-schedule` is not a fully open loop** (`timing/strategies/fixed_schedule.py`): first
  turns go at their timestamps, a later turn at its timestamp only once the previous reply is
  back. A server slower than the gap between timestamps delays the session again.
- AIPerf refuses `--benchmark-grace-period` without `--benchmark-duration` (so not with a fixed
  schedule alone).

**If the study needs a repeatable load (not adopted yet):** a per-request `max_output_tokens`
close to the natural length (from the reply the next request of the session carries in its
history, the customer's real one; with thinking on, to be sized separately), so that few tokens
follow the EOS and MTP acceptance stays realistic; `timestamp`s with a realistic gap between
turns; the users at the SLA measured afterwards with their closed-loop replay on the best
configurations. `VLLM_BATCH_INVARIANT=1` was ruled out: it changes the system being tuned (slower
kernels the customer will not run) and fixes only the reply contents, not the closed loop or the
duration cut.

## Steps

As study 33: `baseline` (the compose: `gpu_memory_utilization` 0.90, `max_num_seqs` 64; the other
11 not rendered), `baseline repeat`, presets `kv fp8`, `kv fp8 mtp2`, `vllm defaults`, `kv fp8
large batch`, `study 30 best`, `study 31 best`, `study 32 best` (study 32's experiment 10, still
its best valid when it ended at 61 experiments: 0.633 completed req/s, +280 %, thinking on as
here), `optimize` (AKAMAS, 0 init, 60 experiments, `maxFailedExperiments` 20).

**KPIs (8):** as study 32 (Total throughput, MTP acceptance, Completed requests, TTFT P95 150s,
ITL P95 150s, KV cache usage, Preemption, Running requests).

**Goal:** maximize `vllm.request_success_rate` with `vllm.e2e_request_latency_p95 <= 30000`
averaged over the scored window (study 33's). Provisional: completed requests/s stands for the
users served (it grows with them until the GPU saturates), and the customer's SLA ("e2e <= 2-3 s
at Kong") cannot be an e2e with thinking on (~20 s per reply at low load).

## Expected results (written before the start)

- Completed requests/s near study 32's at the knee (thinking on, inputs of a similar size: mean
  ~2,300 tokens here, ~3,400 there), with more prefix-cache hits (shared instructions, history).
- `study 32 best` (fp8 KV, MTP K=3, `humming`, 408 sequences) at or near the top again.
- Users at the knee: ~20 for the compose, ~45 or more for the best (provisional sizing above).

## Risks

- **Thinking on may not be their production.** If their 0.29.0 deployment does not set
  `enable_thinking` true, their calls do not think and this study measures ~1,400 tokens per
  reply that production never generates: the flag is one line in
  `k8s/01-statefulset_template.yaml` (false = study 33's), but changing it means a new study.
- `/v1/responses` on vLLM 0.29.0 was exercised against the mock only: experiment 1 is its first
  real run (first-trial checks, `akamas/README.md`).
- The reply lengths come from filler text ("Load"): a configuration's ranking can still hold,
  the absolute users cannot be quoted as theirs.
- The closed loop never builds an unbounded queue: past the knee latency rises with the users
  instead; the watchdog (TTFT p95 > 3 s or ITL p95 > 600 ms for 120 s, study 32's) ends the run.

## Open (to ask the customer)

1. The vLLM 0.29.0 command line of their L40S deployment: `--default-chat-template-kwargs`
   with `enable_thinking` true or not (decides thinking on or off here).
2. The other 90 users' templates (no text in them): no rendered user would repeat a session.
3. An id in each placeholder for identical texts: the renderer would reproduce their repeated
   prompts instead of guessing them from the length.
4. Output tokens per request (`usage.output_tokens`) and the 71 `text` schemas; their
   `benchmark.env` (USERS, DURATION, RAMP, CAP, GRACE) and AIPerf version; how `delay` was
   measured (from the end of the previous reply, as AIPerf replays it, or between starts).
5. The SLA: first token or full reply, which percentile, per request or per user action (a
   chain of 2-4 calls).

## Runbook

1. Study 33 finished or stopped (the user decides; `akamas finish study
   "33-L40S-Gemma4-AWQ-NoThinking" --workspace default`), exported, its Job deleted.
2. The template on the toolbox (`load/README.md`, "On the toolbox").
3. Commit, push, `git pull` in `/work/vllm-benchmark` on the toolbox.
4. `akamas/README.md` "Setup & run": create in dependency order, wait 120 s, start.
5. Experiment 1 checks (`akamas/README.md`): the render line, `/v1/responses` 200s, thinking
   on, the knee inside the ramp.

**Offline checks (2026-10-12):** `python3 akamas/check_offline.py` 0 failures; `bash
k8s/tests/test_*.sh` 0 failures (8 suites; `test_render_dataset.sh` with `CORPUS=` AIPerf's
`shakespeare.txt`); `bash k8s/tests/dry_run_replay.sh` 0 failures, on the fixture (6 users over
60 s) and on their template (100 users over 120 s: 528 requests, every payload rendered, model
`aiservices_gemma`, 12 reasoning tokens per mock reply counted, time to the first answer token
after TTFT); then again on the fixture after the switch to their unchanged script (their run
folder written, `run.log` and records in it). Not testable here: `stop_aiperf.sh`'s process
scan reads `/proc` (Linux), so experiment 1 checks it (`akamas/README.md`).

## Running notes

- 2026-10-12: scaffolded from study 33 the evening the customer's script and template arrived;
  thinking on (the user's choice after their email); `smoke/` and the synthetic dry run dropped.
  First draft rewrote their AIPerf command inside the Job; the user asked to use their script
  as is: the Job now wraps `run_benchmark.sh` unchanged.
- 2026-10-13: study 33 stopped by the user; its AIPerf Job was still running (rate ramp to 6
  req/s on `vllm-0`) and was deleted. Load determinism test by hand on the free GPU ("Load
  determinism test" above); the test pod `aiperf-determinism-test` deleted afterwards, its records
  with it (the numbers above are all that was kept).

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
