# 32-l40s-gemma4-26b-awq-thinking

**Status:** RUNNING on Akamas 4.1 (`32-L40S-Gemma4-AWQ-Thinking`, started 2026-10-09 15:35 UTC),
**without the probe and the smoke run** (decided by the user, once study 31 was stopped): `linear_backend`'s domain comes
from vLLM's source, the ramp (0 -> 2 req/s over 6000 s) and the load profile are the
pre-smoke estimates, and the baseline's not-rendered defaults come from vLLM's config classes.
The first experiments check them ("Runbook" steps 3-5 become checks on experiments 1-2).
**Dates:** scaffolded, created and started on Akamas 4.1 2026-10-09.

## Objective

**A customer's Docker Compose setup of Gemma 4 26B-A4B, thinking on, on our L40S: how many
tokens per second does it serve within a chat SLA, and which vLLM configuration beats their
compose?** A rehearsal of a customer engagement on the same GPU model: their checkpoint, their
serving flags, a load shaped on their figures, and vLLM 0.29.0 (their compose runs 0.25.1; they
are moving to 0.29.0). Studies 30/31's machinery (goal, SLA, ramp, watchdog, scoring, KPIs) is
kept so the results read the same way.

- **Goal (decided with the user 2026-10-09):** maximize **completed requests per second**
  (`vllm.request_success_rate`), the throughput a user sees, instead of studies 30/31's total
  tokens/s. The windowing picks the 3-minute window with the most completed requests/s.
- **Constraint:** e2e p95 (150 s, `:max` over the scored window) **<= 30 s**
  (`vllm.e2e_request_latency_p95:max <= 30000`, the metric is in ms). Studies 30/31's TTFT / ITL
  constraints are dropped. Warned before the start: with thinking on, study 31's e2e was p50
  ~36-42 s and p95 ~111-113 s (Prometheus, 6 h and 24 h), so most configurations may violate it,
  and violations count against `maxFailedExperiments` (20). The goal can be changed on the
  running study with `akamas update study` without losing the history.
- **Questions:** (1) where the compose's knee is, and how far each lever moves it (fp8 KV, MTP,
  the 64-sequence cap, batch budgets); (2) whether the int4 checkpoint changes which levers
  matter compared with study 31's FP8 one (kernels, KV room, MTP acceptance on an int4
  target); (3) how much prefix caching gives on a load that reuses prefixes; (4) what
  configuration to bring to the customer, and how far above their compose.

## What changes from study 31

| Item | Study 31 | Study 32 | Where |
|---|---|---|---|
| Checkpoint | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` (FP8, 26.7 GiB) | **`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a`** (W4A16 int4 g32, compressed-tensors, 16.0 GiB; dense MLP, router, vision tower, lm_head in bf16; no KV scales) | `k8s/01-statefulset_template.yaml` |
| `max_model_len` | 16384 | **96000** (the compose's) | same |
| Tool calling | no | `--enable-auto-tool-choice --tool-call-parser=gemma4` (the compose's; the load sends no tools) | same |
| Vision tower | skipped (`--language-model-only`) | **loaded** (the compose does not pass the flag) | same |
| Prefix caching | off | **vLLM's default** (the compose passes no flag; the startup log says which) | same |
| Environment | — | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `RAY_DISABLE_NODE_DISCOVERY=1` (the compose's) | same |
| Thinking | on, server-wide | on, server-wide (their clients would send `enable_thinking` per request: same prompt) | same |
| Served model name | `gemma4-26b-l40s-think` | `gemma4-26b-awq-think` (not in the compose; our scripts key on it) | StatefulSet, components, scripts |
| Load | ShareGPT replay, one turn per request (prompts: mean 81 tokens, p50 43, p99 847, study 31's vLLM histogram) | **synthetic multi-turn chat with history** (input per request median ~3000 / p90 ~6000 tokens, the customer's figures) | `k8s/05-job_template.yaml`, `k8s/render_job.sh` |
| Baseline | vLLM 0.29.0 defaults, every parameter rendered | **the compose**: `gpu_memory_utilization` 0.90, `max_num_seqs` 64; the 10 parameters it does not pass are not rendered (vLLM's defaults) | `akamas/32-L40S-Gemma4-AWQ-Thinking.yaml`, `k8s/render_statefulset.sh` |
| `linear_backend` | {auto, torch, marlin} (FP8 kernels) | {auto, triton, humming} (int4 kernels that can run here, from vLLM 0.29.0's source) | manifest, `k8s/params.env.template` |
| Ramp | 0 -> 4 req/s over 6000 s | 0 -> 2 req/s over 6000 s, **provisional** until the smoke run | workflow, `k8s/run_test.sh` |
| Goal and constraints | total tokens/s; TTFT p95 <= 1500 ms, ITL p95 <= 300 ms | **completed requests/s; e2e p95 <= 30 s** (the e2e p95 query takes a 150 s window, 30 s before) | manifest, `akamas/telemetry/prometheus.yaml` |

**Unchanged:** vLLM 0.29.0, the g6e.xlarge node and pod sizing, the gemma4 reasoning parser,
the other 12 tuned parameters with their domains and the three `parameterConstraints`, the 8
KPIs, the windowing's shape (6 samples, filter disabled, `when: max`, now on completed
requests/s), the watchdog (TTFT p95 > 3 s or ITL p95 > 600 ms for 120 s: it still ends the
ramp past the knee, though they are no longer constraints).

## Stack & versions

- **Akamas version:** Studio 4.1.0 (`akamas41.lab.akamas.io`, namespace `akamas-41`).
- **Optimization packs:** vLLM **1.12.0** (component type `vLLM`), GPU 1.4.0, Kubernetes
  1.9.0 (metrics only), as on the 4.1 server for studies 30/31.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a` as `gemma4-26b-awq-think`, StatefulSet
  `vllm` (pod `vllm-0`) in namespace `llm-l40s` (studies 30/31's namespace and objects: the
  studies run one after the other on the same node). Fixed flags: `--max-model-len=96000`,
  `--enable-auto-tool-choice`, `--tool-call-parser=gemma4`, `--reasoning-parser=gemma4`,
  `--default-chat-template-kwargs '{"enable_thinking": true}'`. The checkpoint's chat
  template defaults `enable_thinking` to false, as RedHat's. Sampling: AIPerf sends none, so
  vLLM applies the checkpoint's `generation_config.json` (temperature 1.0, top_k 64, top_p
  0.95). Pod: CPU request 3500m, no CPU limit, memory 28 GiB, not tuned.
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), node group
  `llm-serving-l40s-1xl`, one **g6e.xlarge** (1x NVIDIA L40S 48 GB, Ada SM 8.9, 4 vCPU / 32
  GiB, 200 GB disk, 144 GB free on 2026-10-09). Provisioning: `infra/` (study 30's, unchanged).
- **Load generator:** AIPerf 0.11.0, synthetic multi-turn chat ("Load"), tokenizer of the served
  checkpoint at its revision (`tokenizer.json` byte-identical to RedHat's and Google's),
  `--request-rate R --request-rate-ramp-duration D --arrival-pattern gamma --arrival-smoothness 4
  --random-seed 30`, on the `system-m8a` node.
- **Telemetry:** Prometheus, 30 s samples, ServiceMonitor `vllm-l40s`, dcgm-exporter (study
  30's release, already covering this node role).
- **Model and KV cache (expected, the probe measures it):** study 31's FP8 checkpoint loaded
  24.72 GiB and left 13.47 GiB of KV at `gpu_memory_utilization` 0.92 (170,836 tokens). The int4
  checkpoint's ~16 GiB, minus the compose's lower 0.90 and the vision tower's profiling, should
  leave ~20 GiB (~+50 %). 25 of the 30 layers use a 1024-token sliding window, so past ~1k
  tokens a request's KV grows only through the 5 full-attention layers.

## Parameters tuned

Study 31's 13 parameters, domains and constraints, but `linear_backend` (rationale for the
rest: study 30's README, "Parameters tuned"): `gpu_memory_utilization` 0.80-0.94,
`max_num_seqs` 16-512, `max_num_batched_tokens` 512-16384, `kv_cache_dtype` {auto, fp8},
`performance_mode`, `optimization_level` 1-3, `scheduling_policy`, `async_scheduling`,
`max_cudagraph_capture_size` 16-512, `block_size` 16-128 (ordinal), **`linear_backend` {auto,
triton, humming}**, `spec_method` {none, mtp}, `spec_tokens` 0-4.
**parameterConstraints:** `max_num_batched_tokens >= max_num_seqs`; `spec_method != "none" ||
spec_tokens == 0`; `spec_method == "none" || spec_tokens > 0`.

**`linear_backend` on an int4 checkpoint.** vLLM 0.29.0 runs W4A16 layers on its
mixed-precision kernels (`vllm/model_executor/kernels/linear/__init__.py`, `_POSSIBLE_KERNELS`
for CUDA: CutlassW4A8, Machete, AllSpark, Marlin, Conch, Exllama, TritonW4A16, Humming). On SM
8.9 `auto` takes Marlin (the first two need SM 9.0). `torch` has no mixed-precision kernel and
falls back to `auto` (study 30's best used it on FP8), `marlin` is `auto`'s own pick: neither is
in the domain. Of the others, `can_implement` (`mixed_precision/*.py`) rules out Exllama (float16
activations only; Gemma 4 runs bf16) and Conch (needs the `conch` package, not in vLLM's
`requirements/cuda.txt`): with either, the engine would fail at startup. TritonW4A16 (bf16 ok)
and Humming (`humming-kernels` is in the requirements) remain; their speed is for the study to
find (the study started without the probe, decided by the user 2026-10-09).
The MoE experts run on the fused-MoE path whatever this is (`moe_backend` is not in the pack).

**Not tuned, fixed by the compose:** `max_model_len` (96000), prefix caching (default),
the vision tower (loaded), tool calling. `enforce_eager` fixed false (study 30).

## Load

Synthetic **multi-turn chat with history** (`k8s/05-job_template.yaml`), shaped on the
customer's figures for the input per request: **median ~3000 tokens, p90 ~6000**, history
included (2026-10-09). AIPerf 0.11.0's synthetic mode keeps each conversation's history (default
context mode `deltas_without_responses`): turn k carries the shared system prompt, the previous
user messages and the model's previous replies, plus a new message. Two facts checked in the
source and in `k8s/tests/dry_run_multiturn.sh` (2026-10-09, AIPerf 0.11.0 against a mock):

- **The history keeps the reasoning.** `build_assistant_turn` takes each streamed chunk's
  content, or its reasoning when the chunk has no content, so a reply in the history is
  reasoning + answer (a Gemma chat client would drop the reasoning). The profile is calibrated
  on the token counts that result.
- **The rate counts requests.** At each arrival AIPerf sends a queued next turn of an open
  conversation if there is one, else starts a new one, so the ramp is in requests/s as in
  studies 27-31 (dry run: 354 requests in a 180 s ramp to 4 req/s, ~360 expected).

**Profile:** a 1000-token system prompt shared by every conversation (instructions, tool
definitions), a new user message of 300 +- 150 tokens per turn, 3 +- 1 turns per conversation,
15 +- 5 s of think time between a reply and the next turn. Simulated with study 31's reply lengths
(reasoning + answer: mean 1501, p50 1511, p90 2157 tokens): turn k's input is ~1000 + 300 k +
1500 (k - 1), i.e. ~1300, ~3100, ~4900, ~6700 tokens for k = 1..4, and over all requests median
~3150, p90 ~6340, mean ~3440 tokens (20,000 simulated conversations). The length run measures it
with this checkpoint's replies (`input_sequence_length` in the `LENGTHS` lines) and the profile
is retuned if it misses. Conversations come from a pool generated from the seed, drawn without
repetition (`--num-dataset-entries` = R x D / 4 + 1, at least 100), so no conversation's first
turn is served from the prefix cache twice.

- **No `max_tokens`** (no `--output-tokens-mean`): every request reasons and answers until it
  stops, capped by vLLM at `max_model_len` - prompt (study 31's choice). The warm-up checks the
  wire.
- **Open loop, linear rate ramp (study 27/30/31):** gamma arrivals (smoothness 4), seed 30, 8
  single-turn requests at concurrency 4 before the measured run (discarded), the watchdog
  (`k8s/run_test.sh`) ends the test past 2x the SLA for 120 s.
- **R and D: provisional 0 -> 2 req/s over 6000 s** (estimate: study 31's knee of ~1 req/s with
  ~1.6k tokens per request, scaled to ~4.9k tokens (3.4k prompt + 1.5k reply) and the compose's
  64 sequences: baseline K ~ 0.5 req/s). The smoke run sets R so that both the compose and the
  `kv fp8 large batch` preset reach their knee mid-ramp: the spread between them is expected to
  be wider than study 30's 3.3x (64 sequences against hundreds, bf16 against fp8 KV).
- **Scoring:** as studies 30/31: `stability` windowing on `vllm.total_token_throughput`, 6
  samples (3 min), filter disabled, `when: max`.
- **Not exercised:** tool calls (the flags parse every reply, no request carries tools),
  images, the customer's own prompts. When their AIPerf script arrives, it replaces this profile.

## Steps

- `baseline`: **the compose.** `gpu_memory_utilization` 0.90, `max_num_seqs` 64 and
  `max_num_batched_tokens` 2048 (vLLM's default on a < 70 GB GPU) are rendered; the other 10
  parameters are in `doNotRenderParameters`, so Akamas writes them empty and
  `k8s/render_statefulset.sh` passes no flag: vLLM picks its defaults, as with the compose
  (decided with the user 2026-10-09). Known cost (study 27): a step with
  `doNotRenderParameters` never reaches the optimizer engine.
- `baseline repeat` (noise): the same values, every one written out (the user's choice): the
  defaults vLLM resolves when the baseline leaves them out (vLLM 0.29.0's config classes:
  capture size 128, block 16, async on, fcfs, balanced, O2), so it reaches the optimizer.
- Presets: `kv fp8` (compose + fp8 KV), `kv fp8 mtp2` (compose + fp8 + MTP K=2), `vllm
  defaults` (studies 30/31's baseline: 0.92, 256 sequences; the cap removed), `kv fp8 large
  batch` (fp8, 0.94, 512 sequences, 8192 batched tokens, throughput mode), `study 30 best`
  (study 30's experiment 28 with `linear_backend` auto instead of torch), `study 31 best` (study
  31's best at its stop, 3936 total tokens/s, +185.67 %: fp8, MTP K=3, 0.94, 502 sequences, 16384
  batched tokens, O1, throughput mode, capture 189; `linear_backend` auto instead of torch).
- `optimize`: AKAMAS, 0 init, 60 experiments, `maxFailedExperiments` 20.

**KPIs (8):** as study 31 (Total throughput, MTP acceptance, Completed requests, TTFT P95 150s,
ITL P95 150s, KV cache usage, Preemption, Running requests). The prefix cache hit rate is in the
telemetry (`vllm.prefix_cache_hit_rate`), not a KPI (8 is the limit).

## Expected results (written before the start)

Hypotheses, not measurements:

- **The compose's knee is set by its 64 sequences**, not by the KV cache: 64 requests of ~4.9k
  tokens need ~310k tokens of KV before prefix sharing, close to what ~20 GiB of bf16 KV holds.
  The first lever is the cap; `vllm defaults` (256) should beat the compose clearly.
- **fp8 KV gains less than in study 31 at 64 sequences, more once the cap rises.**
- **Prefill matters again:** with ~3.4k-token prompts the prefill share of each step is no
  longer negligible; `max_num_batched_tokens` and `performance_mode` should matter more than in
  study 31, TTFT binds earlier, and prefix caching (system prompt + history) saves a large part
  of each prompt's prefill.
- **Int4 kernels:** Marlin W4A16 reads half the weight bytes of FP8, faster decode at small
  batches; at large batches it computes in bf16 without the FP8 tensor cores, so the gain over
  study 31 may shrink at the knee. The probe compares the two checkpoints row by row.
- **MTP acceptance on an int4 target** is unknown; the drafter is bf16 and was trained against
  the bf16 model.

## Risks

- **Synthetic prompts:** AIPerf's synthetic text is drawn from a corpus, not real questions;
  with thinking on, the model may reason oddly on it (trace lengths). The length run is the
  gate; the fallback is a real-text multi-turn dataset built from ShareGPT's user messages.
- **Runaway traces:** with `max_model_len` 96000 and no `max_tokens` a looping trace can hold a
  slot for up to ~2 h. The `LENGTHS` check counts requests within ~1000 tokens of the cap; if the
  length run shows any, decide a cap with the user.
- **Numerics-changing settings change the workload** (study 31): fp8 KV and the linear kernel
  change what the model generates, hence the reply lengths, and here also the next turns'
  prompts (the history holds the replies). Generated tokens per request at the scored window
  shows whether a configuration's load moved.
- **The e2e constraint may be infeasible:** with thinking on, a reply of ~1500 tokens (study
  31: reasoning + answer) at 40-75 ms per token takes 60-110 s, beyond 30 s; study 31's e2e p95 was
  ~111-113 s. If the baselines and the first presets all violate it, the optimize step can stop
  at 20 violations: change the threshold (or the goal) with `akamas update study`. The e2e also
  moves with the reply length: settings that change numerics (fp8 KV, kernels) can change it
  without being slower per token; generated tokens per request at the window tell them apart.
- **`doNotRenderParameters` on Akamas 4.1:** that a not-rendered parameter is written as an
  empty string was seen on Akamas 3.7 (study 20). If 4.1 writes the literal `${vllm.x}` token
  instead, `render_statefulset.sh` refuses it (unsubstituted tokens) and the first trial fails
  before vLLM starts: the fix is one line there. Checked on the first trial ("Setup & run").
- **The int4 checkpoint's accuracy** with fp8 KV (no KV scales) at long contexts is unverified;
  the best configuration gets study 30's accuracy check (`../30-l40s-gemma4-26b-tps/eval/`)
  before it goes to the customer.
- **Host CPU (4 vCPU):** reasoning and tool parsers run per token in the API server; the smoke
  run checks vLLM's CPU (`cpu_vllm` in `smoke_analyze.py`).
- **Sequencing with study 31:** same node, namespace, pod name `vllm-0` and Job name. Study 31
  must be finished and its load Job deleted before the probe.
- **Capacity, nightly stop, platform restarts:** as studies 30/31 (g6e pool, `AlwaysOn` tags).

## Runbook

1. **Study 31 finished** (user: `akamas finish study 31-L40S-Gemma4-TPS-Thinking`, export, its
   load Job deleted: `kubectl -n llm-l40s delete job -l app=aiperf-l40s`). Read its best
   configuration for the `study 31 best` preset.
2. **Node up + AlwaysOn** (if left at 0): `AWS_PROFILE=lab ./infra/eks/gpu-nodegroup.sh --up
   --always-on`; `AWS_PROFILE=lab ./infra/eks/provision.sh` (a no-op on study 30's layer).
3. **Probe** (workstation, ~60-75 min, `probe/README.md`): `mkdir -p /tmp/probe32 &&
   KP_OUT=/tmp/probe32/results caffeinate -i nohup bash probe/probe.sh > /tmp/probe32/probe.log
   2>&1 &`. Decide with the user: `linear_backend`'s categories (manifest,
   `k8s/params.env.template` comment), whether MTP and fp8 KV start on the int4 checkpoint, the
   presets that start. Reconcile, before `akamas create`, the baseline's recorded values and the
   compose-based presets (`kv fp8`, `kv fp8 mtp2`) with what `B-compose.kernels.txt` and
   `M-mtp2.kernels.txt` show vLLM resolved (capture size 128 / 384 expected from vLLM 0.29.0's
   config classes; block size, async scheduling, scheduling policy and performance mode are not
   logged at their defaults: 16 / on / fcfs / balanced per the same classes). Copy the results
   into `probe/results/`.
4. **Smoke run** (workstation, ~2 h, `smoke/README.md`): `mkdir -p /tmp/smoke32 &&
   SMOKE_OUT=/tmp/smoke32/compose caffeinate -i bash smoke/smoke_manual.sh`, then
   `SMOKE_OUT=/tmp/smoke32/large SMOKE_CONFIG=large SMOKE_PHASES=ramp
   SMOKE_LEN_LOG=/tmp/smoke32/compose/length_run.log caffeinate -i bash smoke/smoke_manual.sh`;
   `python3 smoke/smoke_analyze.py http://127.0.0.1:19090 <RUNTEST START> <RUNTEST END>
   <dir>/timeseries.csv > <dir>/summary.txt` for each, with a Prometheus port-forward.
5. **Decide, with the user:** the profile (if the length run's input misses median ~3000 / p90
   ~6000: `PROMPT` / `TURNS` in `k8s/05-job_template.yaml`, `k8s/tests/test_render_job.sh`), a
   reply cap if runaway traces show up, R from both knees (low bound: the compose's knee x 4;
   high bound: the large preset's knee at mid-ramp), D = 6000 s. Edit `k8s/run_test.sh`
   (default), `akamas/32-L40S-Gemma4-AWQ-Thinking-Workflow.yaml` (command; RunTest timeout
   above D + 1500 s), and this README; re-run `bash k8s/tests/test_*.sh` and `python3
   akamas/check_offline.py`.
6. **Sync:** commit and push (user), `git pull` on the 4.1 toolbox.
7. **Create and start:** commands in `akamas/README.md`, "Setup & run" (check the first trial's
   `params.env`: the 10 not-rendered keys empty).

## Startup probe

Not run yet.

## Smoke run

Not run yet.

## Running notes

- 2026-10-09: study 31 stopped by the user (FINISHED, 23 experiments, best 3936 total tokens/s,
  +185.67 %), its leftover load Job deleted; its best added as the `study 31 best` preset. Goal
  changed with the user to completed requests/s under e2e p95 <= 30 s (warned: study 31's e2e
  p95 was ~111-113 s). Started without probe and smoke run: on experiments 1-2 check the
  `LENGTHS`-free signs that matter (the input per request from the KV and prompt-token metrics,
  whether the ramp reaches the knee, the not-rendered keys empty in `params.env`, the resolved
  capture size in the Apply config log).
- 2026-10-09: scaffolded from study 31 (`k8s/`, `akamas/`, `smoke/`, `infra/`) and study 30's
  probe. Offline: `bash k8s/tests/test_*.sh` (6 suites) and `bash probe/tests/test_summarize.sh`
  pass, `python3 akamas/check_offline.py` 0 failures, `bash k8s/tests/dry_run_multiturn.sh`
  passes (AIPerf 0.11.0 against the mock: no `max_tokens`, one shared system prompt, history
  with reasoning, input lengths reported, the ramp followed).

- 2026-10-09: created on Akamas 4.1 from the toolbox (commit 1fb98da). The first create was
  refused: a parameter cannot be both in a step's `values` and in its `doNotRenderParameters`
  (the baseline now has only its three rendered values; `check_offline.py` checks it). The server
  lists the 9 steps; started 15:35:39 UTC, experiment 1 (baseline) RUNNING. First trial checked:
  `params.env` has the 10 not-rendered keys empty (Akamas 4.1 renders them empty, as 3.7), and
  vLLM started with the compose's flags only (`--gpu-memory-utilization=0.90000
  --max-num-seqs=64 --max-num-batched-tokens=2048 --no-enforce-eager` besides the fixed ones),
  downloading the AWQ checkpoint.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
