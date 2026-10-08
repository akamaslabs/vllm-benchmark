# 31-l40s-gemma4-26b-tps-thinking

**Status:** TODO. Scaffolded 2026-10-08 from study 30; runs after study 30 on the same L40S
node. Not created on Akamas yet; the smoke run (length run + ramp) comes first.
**Dates:** —

## Objective

**Study 30 with Gemma 4's thinking mode on: what changes once the model reasons before it
answers?** Same model, GPU, vLLM image, 13 parameters, goal, SLA, windowing, KPIs and steps as
study 30 (`../30-l40s-gemma4-26b-tps/README.md`). Only the serving flags that turn thinking on
and the requests' output budget differ, so the two studies compare setting by setting.

- **Goal:** maximize vLLM's **total token throughput** (`vllm.total_token_throughput` =
  prompt + generation tokens/s; with thinking on, generation counts reasoning and answer
  tokens alike).
- **Constraints (study 30's chat SLA):** TTFT p95 <= 1500 ms and ITL p95 <= 300 ms, both p95
  over 150 s, read with `:max` over the scored window. TTFT is now the time to the first
  *reasoning* token (see below).
- **Questions:** (1) how far the knee moves, in req/s and in tokens/s, when every request
  reasons; (2) whether the levers of study 30 (fp8 KV, MTP, batch caps) keep their rank, and
  whether the optimizer finds a different best configuration; (3) how long Gemma 4 reasons on
  ShareGPT prompts, and how long a user waits for the first answer token at the knee.

## What changes with thinking on

| Item | Study 30 | Study 31 | Where |
|---|---|---|---|
| Chat template | `enable_thinking` false | `--default-chat-template-kwargs '{"enable_thinking": true}'`: `<|think|>` at the top of a system turn, the thought channel left open. The template defaults to false (`enable_thinking \| default(false)`), so the flag is required | `k8s/01-statefulset_template.yaml` |
| Reasoning parser | none | `--reasoning-parser=gemma4` (vLLM 0.29.0 `Gemma4ParserReasoningAdapter`): the trace streams in the `reasoning` field, the answer in `content` | same |
| `max_model_len` | 4096 | **16384, provisional** (the length run confirms it): prompt + trace + answer | same |
| Output budget | AIPerf's ShareGPT `max_tokens` = the length of the ShareGPT reply (aiperf 0.11.0 `dataset/loader/sharegpt.py`) | **no `max_tokens`**: the Job strips it from a copy of the cache (`inputs-<model>-nomax.json`); vLLM caps each request at `max_model_len` - prompt | `k8s/05-job_template.yaml` |
| AIPerf warm-up | 60 s at concurrency 4 | 8 requests at concurrency 4: with a 60 s window and requests that reason longer than that, AIPerf 0.11.0 exits non-zero ("No profile results to export") and the trial fails (reproduced by `k8s/tests/dry_run_thinking.sh`) | `k8s/05-job_template.yaml` |
| Served model name | `gemma4-26b-l40s` | `gemma4-26b-l40s-think` (separate vLLM series and ShareGPT cache) | StatefulSet, components, scripts |
| Ramp | 0 -> 40 req/s over 6000 s | **0 -> 6 req/s over 6000 s, provisional** (from the smoke run) | workflow, `k8s/run_test.sh` |

**Why not keep ShareGPT's `max_tokens`:** with thinking on, the reasoning trace uses up the
reply's token budget, and almost every request would stop mid-thought without an answer. That
would measure truncated reasoning, not a thinking workload (decided with the user,
2026-10-08: "ragionamento completo").

**TTFT semantics:** vLLM's `time_to_first_token` histogram, which the constraint, the scoring
and the watchdog read, times the first generated token, now a reasoning token. The SLA stays
study 30's so the two studies compare. The time to the first *answer* token (what a chat
user waits for) is measured only by AIPerf, client side (`time_to_first_output_token`, which
reads the parser's `reasoning` field): it is reported by the length run and by any run that
completes (`LENGTHS` lines in the Job log), not scored.

**Unchanged:** the model and checkpoint, vLLM 0.29.0, the g6e.xlarge node and pod sizing,
`--language-model-only`, `--no-enable-prefix-caching`, the 13 tuned parameters with their
domains and the three `parameterConstraints`, the 8 KPIs, the steps (baseline x2, three
presets, 60 AKAMAS), the ramp's shape (gamma arrivals, smoothness 4, seed 30), the watchdog
(2x the SLA for 120 s), the scoring (best valid 3-minute window). Checked by loading both
manifests (`akamas/README.md`, "Resources").

## Stack & versions

- **Akamas version:** Studio 4.1.0 (`akamas41.lab.akamas.io`, namespace `akamas-41`).
- **Optimization packs:** vLLM **1.12.0** (component type `vLLM`), GPU 1.4.0, Kubernetes
  1.9.0 (metrics only), as on the 4.1 server for study 30.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  **`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`** as `gemma4-26b-l40s-think`, StatefulSet `vllm`
  (pod `vllm-0`) in namespace `llm-l40s` (study 30's namespace and objects: the two studies run
  one after the other on the same node). Fixed flags: `--language-model-only`,
  `--no-enable-prefix-caching`, `--max-model-len=16384`, `--default-chat-template-kwargs
  '{"enable_thinking": true}'`, `--reasoning-parser=gemma4`. Pod: CPU request 3500m, no CPU
  limit, memory 28 GiB, not tuned. Sampling: AIPerf sends none, so vLLM applies the
  checkpoint's `generation_config.json` (temperature 1.0, top_k 64, top_p 0.95): trace lengths
  vary from request to request and from trial to trial.
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), node group
  `llm-serving-l40s-1xl`, one **g6e.xlarge** (1x NVIDIA L40S 48 GB, Ada SM 8.9, 4 vCPU / 32
  GiB). Provisioning: `infra/` (study 30's, unchanged).
- **Load generator:** AIPerf 0.11.0, ShareGPT tokenized once with the Gemma 4 tokenizer and
  cached, replayed **without `max_tokens`**, `--request-rate R --request-rate-ramp-duration D
  --arrival-pattern gamma --arrival-smoothness 4 --random-seed 30`, on the `system-m8a` node.
- **Telemetry:** Prometheus, 30 s samples, ServiceMonitor `vllm-l40s`, dcgm-exporter (study
  30's release, already covering this node role).
- **Model and KV cache:** study 30's README, "The model": KV cache 63,988 tokens in bf16 and
  127,625 in fp8 at `gpu_memory_utilization` 0.92 (study 30's probe). 25 of the 30 layers use a
  1024-token sliding window, so past ~1k tokens a request's KV grows only through the 5
  full-attention layers.

## Parameters tuned

Study 30's 13 parameters, domains and constraints, unchanged (rationale, probe results and the
left-out list: study 30's README, "Parameters tuned" and "Startup probe"): `gpu_memory_utilization`
0.80-0.94, `max_num_seqs` 16-512, `max_num_batched_tokens` 512-16384, `kv_cache_dtype` {auto,
fp8}, `performance_mode`, `optimization_level` 1-3, `scheduling_policy`, `async_scheduling`,
`max_cudagraph_capture_size` 16-512, `block_size` 16-128 (ordinal), `linear_backend` {auto,
torch, marlin}, `spec_method` {none, mtp}, `spec_tokens` 0-4. **parameterConstraints:**
`max_num_batched_tokens >= max_num_seqs`; `spec_method != "none" || spec_tokens == 0`;
`spec_method == "none" || spec_tokens > 0`.

The thinking flags are fixed, not tuned: the study measures one serving mode, the one a
customer turning thinking on would deploy.

## Load

- **Open loop, linear rate ramp (study 27/30):** gamma arrivals (smoothness 4), seed 30 (study
  30's), 8 requests at concurrency 4 before the measured
  run (discarded), the watchdog (`k8s/run_test.sh`) ends the test past 2x the SLA for 120 s.
- **ShareGPT without `max_tokens`:** every request reasons and answers until the model stops,
  or until `max_model_len` - prompt.
- **R and D: provisional 0 -> 6 req/s over 6000 s.** Estimated before any measurement: study
  30's baseline broke at ~10 req/s with ~300 generated tokens per request; if a request
  generates several times more, the knee falls by about the same factor, to ~1-2 req/s. Study
  30's rule (R ~ 4 K, D such that the baseline crosses its knee ~25 min into the ramp) gives
  R ~ 6 and D = 6000 s. **The smoke run measures K and sets R = 4 K, D = 6000 s** (with D fixed
  at 6000 s, the baseline crosses at 6000 x K / R = 1500 s whatever K is); then the workflow's
  command and `k8s/run_test.sh`'s default change together. At 1.5 req/s a 3-minute window
  holds ~270 requests (~1800 in study 30), and each request lasts tens of seconds, so the
  window's tokens/s is smoother than its req/s.
- **Scoring:** as study 30: `stability` windowing on `vllm.total_token_throughput`, 6 samples
  (3 min), filter disabled, `when: max`.

## Steps

As study 30: `baseline` (vLLM 0.29.0 defaults, every parameter written out), `baseline repeat`
(noise), `kv fp8`, `kv fp8 large batch` (fp8, gmu 0.94, 512 seqs, 8192 batched tokens,
throughput mode), `kv fp8 mtp2` (fp8 + MTP K=2), `optimize` (AKAMAS, 0 init, 60 experiments,
`maxFailedExperiments` 20). **KPIs (8, Italian names as the repo convention):** Throughput
totale, Accettazione MTP, Richieste completate, TTFT P95 150s, ITL P95 150s, KV cache in uso,
Preemption, Richieste in esecuzione. Generated tokens per request (the length of the
workload at the window) is not a KPI (8 is the limit): it is `decode_token_throughput /
request_success_rate` from the telemetry, to compute at recap time.

## Expected results (written before the start)

Hypotheses, not measurements:

- **Traces of several hundred to a few thousand tokens** on ShareGPT prompts (median prompt 41
  tokens), against ~300 generated tokens per request in study 30; the knee falls to ~1-2 req/s.
- **The KV cache binds earlier, in requests in flight:** each request holds its trace's KV
  for tens of seconds, so fewer requests fit than study 30's ~220 (bf16). **fp8 KV should gain
  more than in study 30** (+87 % there), since the knee is KV-bound with fewer, longer requests.
- **Total tokens/s at the knee below study 30's** (2.5-4.7k): a smaller decode batch uses the
  GPU less efficiently per step; prefill becomes a negligible share.
- **MTP gains more than in study 30:** it pays most at small batches, and the batch at the knee
  is smaller; acceptance on reasoning text is unknown (study 30's probe: 0.60 at K=2 on
  English essays).
- `max_num_batched_tokens` matters less (short prompts, decode-dominated steps); `max_num_seqs`
  matters only if fp8 + high `gpu_memory_utilization` lift the KV limit above it.
- TTFT (queueing behind a full KV cache) stays the binding constraint; ITL stays far below 300
  ms.

## Risks

- **Trace length unknown until the length run:** too long a tail (runaway traces up to
  16384 tokens) holds slots and KV; too low a cap truncates. The length run counts the
  requests near the cap; lower or raise `--max-model-len` from it (no KV is reserved for the
  cap, so lowering it does not change the KV size, only the tail).
- **Numerics-changing settings change the workload:** `kv_cache_dtype` fp8 and
  `linear_backend` (marlin is W8A16) can change what the model generates, hence the trace
  lengths, not only the speed. With sampling at temperature 1.0 lengths vary anyway; generated
  tokens per request at the scored window (above) shows whether a configuration's load moved.
- **The score counts reasoning tokens:** tokens/s at the knee measures GPU capacity, as in
  study 30; a configuration that made the model reason longer would serve fewer req/s at the
  same tokens/s. Read "Richieste completate" next to the score.
- **Host CPU (4 vCPU):** the reasoning parser runs per token in the API server. Study 30's
  smoke had vLLM's container at <= 0.54 cores at ~3900 tokens/s; the smoke run checks it again
  (`cpu_vllm` in `smoke_analyze.py`).
- **The `LENGTHS` summary only prints when AIPerf completes:** the watchdog stops a trial's Job
  past the knee, so study trials do not print it; the length run does.
- **Sequencing with study 30:** same node, namespace, pod name `vllm-0` and Job name. Study 30
  must be finished and its load Job deleted before the smoke run, or the two would fight over
  the GPU.
- **Capacity, nightly stop, platform restarts:** as study 30 (g6e pool, `AlwaysOn` tags, the
  17:00 UTC restart of 2026-10-06 that killed a RunTest).

## Runbook

1. **Study 30 finished** (`akamas finish study 30-L40S-Gemma4-TPS` when done, export, its load
   Job deleted: `kubectl -n llm-l40s delete job -l app=aiperf-l40s`).
2. **Node up + AlwaysOn** (if study 30 left it at 0): `AWS_PROFILE=lab
   ./infra/eks/gpu-nodegroup.sh --up --always-on`; `AWS_PROFILE=lab ./infra/eks/provision.sh`
   (a no-op on study 30's layer).
3. **Smoke run** (workstation, ~30-60 min): `mkdir -p /tmp/smoke31 && SMOKE_OUT=/tmp/smoke31
   caffeinate -i bash smoke/smoke_manual.sh`. It applies the baseline, runs the length run
   (prints the `LENGTHS` lines), then a ramp over 1800 s until the watchdog fires; the ramp's
   top rate comes from the length run (2.5 x 2900 / mean output tokens per request, so the
   knee falls mid-ramp; `SMOKE_RATE` overrides it). To look at the lengths before the ramp:
   `SMOKE_PHASES=length`, then `SMOKE_PHASES=ramp SMOKE_SKIP_APPLY=1` (same vLLM, no restart).
   Then `python3 smoke/smoke_analyze.py http://127.0.0.1:19090 <RUNTEST START> <RUNTEST
   END> /tmp/smoke31/timeseries.csv > /tmp/smoke31/summary.txt` with a Prometheus
   port-forward, and copy the results into `smoke/results/`.
4. **Decide, with the user:** `--max-model-len` from the length run (it lives in three places:
   `k8s/01-statefulset_template.yaml`, the `15300` threshold of the `LENGTHS` check in
   `k8s/05-job_template.yaml` (max_model_len - 1024 prompt tokens - template), and `k8s/tests/test_render_statefulset.sh`); K from the ramp
   (low bound: "Richieste completate" of the scored window; high bound: the rate at the
   `watchdog: over` line, `R_smoke x (t_over - t_start) / 1800`; K = midpoint), then R = 4 K
   and D = 6000 s.
   Edit `k8s/run_test.sh` (default), `akamas/31-L40S-Gemma4-TPS-Thinking-Workflow.yaml`
   (command; RunTest timeout above D + 1500 s), and this README; re-run
   `bash k8s/tests/test_*.sh` and `python3 akamas/check_offline.py`.
5. **Sync:** commit and push (user), `git pull` on the 4.1 toolbox.
6. **Create and start:** commands in `akamas/README.md`, "Setup & run".

## Running notes

—

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
