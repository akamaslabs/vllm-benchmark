# 33-l40s-gemma4-26b-awq-no-thinking

**Status:** RUNNING on Akamas 4.1 since 2026-10-12 (created and started after study 32 finished:
same node, namespace, pod and Job name).
**Dates:** scaffolded 2026-10-12 as a copy of study 32.

## Objective

**Study 32 with thinking off: how many requests per second does one L40S complete with a
customer's Docker Compose setup of Gemma 4 26B-A4B when the model does not reason, and which vLLM
configuration beats their compose?** Same checkpoint, serving flags, load profile, parameters,
goal and machinery as study 32
(`../32-l40s-gemma4-26b-awq-thinking/README.md`); the variable is `enable_thinking`. Asked by the
user on 2026-10-12, to see how the deployment behaves with thinking off.

- **Goal (study 32's, decided with the user 2026-10-09):** maximize **completed requests per
  second** (`vllm.request_success_rate`); the windowing picks the 3-minute window with the most
  completed requests/s.
- **Constraint:** e2e p95 (150 s) **<= 30 s**, now **averaged over the scored window**
  (`vllm.e2e_request_latency_p95 <= 30000`, no aggregation = Akamas's default `avg`, per the
  live Goal & Constraints docs; user, 2026-10-12). Study 32 took the worst sample (`:max`): a
  window now passes if its six 150 s p95 samples average under 30 s, so a single late spike no
  longer invalidates it.
- **Questions:** (1) how far the compose's knee and the best configuration's move when every
  generated token is an answer token; (2) whether the levers that won with thinking on (MTP,
  fp8 KV, many sequences) still win with ~620-token replies and shorter histories; (3) how the
  result compares with study 32 at the same SLA, i.e. what thinking costs the customer in
  requests/s.

## What changes from study 32

| Item | Study 32 | Study 33 | Where |
|---|---|---|---|
| Thinking | on, server-wide (`{"enable_thinking": true}`) | **off**, server-wide (`{"enable_thinking": false}`, the chat template's own default, written out) | `k8s/01-statefulset_template.yaml` |
| Served model name | `gemma4-26b-awq-think` | `gemma4-26b-awq-nothink` (keeps the two studies' vLLM series apart in Prometheus) | StatefulSet, `akamas/components/vllm.yaml`, scripts |
| e2e constraint | `vllm.e2e_request_latency_p95:max <= 30000` | `vllm.e2e_request_latency_p95 <= 30000` (avg over the window) | `akamas/33-L40S-Gemma4-AWQ-NoThinking.yaml`, `smoke/smoke_analyze.py` |
| Ramp | 0 -> 2 req/s over 6000 s | 0 -> **6** req/s over 6000 s, **provisional** ("Load") | workflow, `k8s/run_test.sh` |
| Presets | six | seven: **`study 32 best`** added (experiment 10) | manifest |
| Akamas names | `32-L40S-Gemma4-AWQ-Thinking` (+ `-Workflow`), `vLLM_Benchmark_32_L40S_Gemma4_AWQ_Thinking`, `Prometheus_32_...` | `33-L40S-Gemma4-AWQ-NoThinking` (+ `-Workflow`), `vLLM_Benchmark_33_L40S_Gemma4_AWQ_NoThinking`, `Prometheus_33_L40S_Gemma4_AWQ_NoThinking` | `akamas/` |
| AIPerf results | `/benchmarks/study32`, `/work/aiperf-exports/32-l40s-gemma4-26b-awq-thinking/` | `/benchmarks/study33`, `/work/aiperf-exports/33-l40s-gemma4-26b-awq-no-thinking/` | `k8s/05-job_template.yaml`, `k8s/stop_aiperf.sh` |

**Unchanged:** vLLM 0.29.0, the checkpoint and revision, the compose's fixed flags
(`--max-model-len 96000`, tool calling, `--reasoning-parser=gemma4`, vision tower loaded, prefix
caching at its default, the two environment variables), the g6e.xlarge node and pod sizing, the
13 tuned parameters with their domains and three `parameterConstraints`, the 8 KPIs, the
windowing, the AIPerf load profile, the watchdog (TTFT p95 > 3 s or ITL p95 > 600 ms for 120 s),
the four-task workflow with study 32's `Stop AIPerf` (2026-10-12). Not copied: study 32's
`probe/` (never run) and `results/`.

**Not added: NCCL settings** (asked by the user, 2026-10-12). With one GPU (tensor, pipeline and
data parallel all 1) vLLM runs no collective, so NCCL environment variables cannot change
anything, and the vLLM pack (1.12.0) declares no NCCL parameter anyway: its only
interconnect knob, `disable_custom_all_reduce`, switches the tensor-parallel all-reduce kernel,
inert at TP 1. Each would be a dead dimension for the optimizer.

**TTFT and TTFO:** with thinking off, AIPerf's time to the first output token equals its TTFT
(AIPerf 0.11.0, `TimeToFirstOutputTokenMetric`: "for models without reasoning, TTFO and TTFT are
equivalent"), and vLLM's TTFT histogram (the watchdog's, the KPI's) times the first answer token.
The client-side TTFO exporter discussed on 2026-10-12 is not needed for this study.

## Stack & versions

- **Akamas version:** Studio 4.1.0 (`akamas41.lab.akamas.io`, namespace `akamas-41`).
- **Optimization packs:** vLLM **1.12.0** (component type `vLLM`; installed on the 4.1 server,
  `akamas list optimization-pack` 2026-10-12), GPU 1.4.0, Kubernetes 1.9.0 (metrics only).
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a` as `gemma4-26b-awq-nothink`, StatefulSet
  `vllm` (pod `vllm-0`) in namespace `llm-l40s`. Fixed flags: `--max-model-len=96000`,
  `--enable-auto-tool-choice`, `--tool-call-parser=gemma4`, `--reasoning-parser=gemma4`,
  `--default-chat-template-kwargs '{"enable_thinking": false}'`. Sampling: AIPerf sends none, so
  vLLM applies the checkpoint's `generation_config.json` (temperature 1.0, top_k 64, top_p
  0.95). Pod: CPU request 3500m, no CPU limit, memory 28 GiB.
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), node group
  `llm-serving-l40s-1xl`, one **g6e.xlarge** (1x NVIDIA L40S 48 GB, 4 vCPU / 32 GiB).
  Provisioning: `infra/` (study 30's, unchanged).
- **Load generator:** AIPerf 0.11.0 on the `system-m8a` node, study 32's synthetic multi-turn
  chat ("Load").
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
**Not tuned:** as study 32 (`max_model_len`, prefix caching, the vision tower, tool calling,
`enforce_eager` false), plus NCCL (above).

## Load

Study 32's profile, unchanged (`k8s/05-job_template.yaml`): a 1000-token system prompt shared by
every conversation, a new user message of 300 +- 150 tokens per turn, 3 +- 1 turns per
conversation, 15 +- 5 s between a reply and the next turn, history kept (turn k carries the
previous user messages and replies), no `max_tokens`, gamma arrivals (smoothness 4), seed 30.
Only thinking changes, so the comparison with study 32 is one variable.

- **Shorter inputs, a consequence:** a reply in the history is now the answer only. Study 32's
  answers (AIPerf records of 2026-10-12: answer tokens mean 622, p50 637, p90 829; reasoning
  mean 901) give turns of ~1300, ~2200, ~3100, ~4000 tokens for k = 1..4, mean ~2400, against
  study 32's ~3440 and the customer's median ~3000 / p90 ~6000. If the customer's figures matter
  more than the comparison, retune `TURNS` (more turns) in a follow-up study; the `LENGTHS`
  lines of a completed run and vLLM's prompt-token histogram measure it.
- **Answer lengths without thinking are not known:** Gemma 4 may answer longer when it does not
  reason first; experiment 1 measures it (generated tokens per request).
- **R = 6 req/s over 6000 s, provisional.** Study 32's knees (no queue): compose median ~0.59
  req/s, best 1.3 req/s (experiment 10), with ~1520 generated tokens per request. Decode-bound,
  so scaled by ~1520 / 620: compose ~1.4 req/s, best ~3.2 req/s. Study 32's rule (R >= the
  compose's knee x 4; the best preset's knee near mid-ramp) gives 5.6-6.4: **6**. The
  conversation pool is R x D / 4 + 1 = 9001 (`render_job.sh`). If experiment 1's knee is far
  from ~1.4 req/s, rescale R before the presets run (workflow recreated, see "Runbook").
- **Scoring:** stability windowing on `vllm.request_success_rate`, 6 samples (3 min), filter
  disabled, `when: max`; a window is valid if its e2e p95 samples average <= 30 s.

## Steps

- `baseline`: the compose (`gpu_memory_utilization` 0.90, `max_num_seqs` 64 rendered; the other
  11 not rendered, so vLLM picks its defaults). Never reaches the optimizer engine (study 27).
- `baseline repeat`: the same values, every one written out (vLLM 0.29.0's resolved defaults).
- Presets, as study 32: `kv fp8`, `kv fp8 mtp2`, `vllm defaults`, `kv fp8 large batch`,
  `study 30 best`, `study 31 best`; plus **`study 32 best`**: study 32's experiment 10 (gmu
  0.938, 408 sequences, 2496 batched tokens, fp8 KV, balanced, O1, fcfs, async off, capture 512,
  block 16, `humming`, MTP K=3), 0.633 completed req/s, +280 % over its baseline, the best
  valid experiment on 2026-10-12 (study 32 was running experiment 62 of 68; its experiment 49 scored
  1.107 but violated the constraint). If study 32 ends with another best, update the preset
  before the create.
- `optimize`: AKAMAS, 0 init, 60 experiments, `maxFailedExperiments` 20.

**KPIs (8):** as study 32 (Total throughput, MTP acceptance, Completed requests, TTFT P95 150s,
ITL P95 150s, KV cache usage, Preemption, Running requests).

## Expected results (written before the start)

Hypotheses, not measurements:

- **Roughly 2-3x study 32's completed requests/s** at every configuration, since decode is
  ~40 % of study 32's tokens per request and prefill ~70 % (shorter histories). Less if the
  answers grow without the reasoning.
- **The e2e constraint binds near the knee**, not at the start of the ramp: ~620 tokens at
  16-40 ms per token is 10-25 s.
- **MTP:** acceptance may differ on answers than on reasoning traces (study 32 measured it with
  thinking on); with shorter replies its share of the e2e matters less.
- **The 64-sequence cap** of the compose binds even harder (more requests per second, each
  shorter): `max_num_seqs` stays the first lever.

## Risks

- **The ramp may be mis-sized** (R = 6 is an estimate): too low and the best configurations
  never reach their knee, so the scores saturate at the offered load; too high and the compose's
  knee comes early in the ramp (coarser windows). Experiment 1 checks it.
- **Toolbox room:** `/work` is 1 GiB (778 MB free on 2026-10-12); at 6 req/s each run's records
  are ~2-3x study 32's 10-26 MB. Copy and delete them every few experiments
  (`akamas/README.md`, "Stop AIPerf"); below 150 MB free `stop_aiperf.sh` saves nothing (the
  trial still counts).
- **The four-task workflow is new on the server:** `alwaysRun` / `critical` match the live docs,
  but no study has run `Stop AIPerf` under Akamas yet; the first trial shows it.
- **Synthetic prompts and numerics-changing settings:** as study 32 ("Risks" there).
- **Sequencing with study 32:** same node, namespace, pod name `vllm-0` and Job name.
- **Capacity, nightly stop, platform restarts:** as studies 30-32 (g6e pool, `AlwaysOn` tags).

## Runbook

1. **Study 32 finished** (by itself at 68 experiments, or `akamas finish study
   "32-L40S-Gemma4-AWQ-Thinking"`), exported, its load Job deleted (`kubectl -n llm-l40s delete
   job -l app=aiperf-l40s`). Re-read its best valid experiment: if it changed, update the `study
   32 best` preset (manifest, this README, `akamas/README.md`) and re-run `python3
   akamas/check_offline.py`.
2. **Node up + AlwaysOn** (if left at 0): `AWS_PROFILE=lab ./infra/eks/gpu-nodegroup.sh --up
   --always-on`.
3. **Sync:** commit and push (user), `git pull` on the 4.1 toolbox.
4. **Create and start:** commands in `akamas/README.md`, "Setup & run".
5. **Experiment 1 checks** (`akamas/README.md`, "First trial checks"): thinking really off, the
   answer length, the knee inside the ramp, the not-rendered keys empty, `Stop AIPerf` saving
   the records. If R must change: `akamas finish study`, edit the workflow command and
   `k8s/run_test.sh`, push, pull, delete and recreate the workflow and the study.

Offline checks: `bash k8s/tests/test_*.sh` (7 suites), `python3 akamas/check_offline.py`,
`bash k8s/tests/dry_run_multiturn.sh` (network, ~8 min).

## Running notes

- 2026-10-12: created and started on Akamas 4.1 (commit 854d7f9) after study 32 finished (61
  experiments, best 0.633 req/s = the `study 32 best` preset). Experiment 1 (baseline):
  `params.env` with the 11 not-rendered keys empty; vllm-0 started with
  `default_chat_template_kwargs {'enable_thinking': False}`, served as gemma4-26b-awq-nothink,
  prefix caching on, KV cache 22.01 GiB = 708,768 tokens (gmu 0.90), ready in ~3.5 min
  (compile 83 s); the AIPerf Job started right after the warm-up.
- 2026-10-12: scaffolded from study 32 (`akamas/`, `k8s/` with `stop_aiperf.sh`, `smoke/`,
  `infra/`; not `probe/`, not `results/`). Thinking off, the e2e constraint averaged, the ramp
  to 6 req/s, a `study 32 best` preset. NCCL settings not added (one GPU). Offline: `python3
  akamas/check_offline.py` 0 failures (its study glob was still `32-*.yaml` after the copy,
  which skipped the study checks silently: fixed, and a check added that it finds exactly one
  study), `bash k8s/tests/test_*.sh` 7 suites 0 failures, `bash k8s/tests/dry_run_multiturn.sh`
  0 failures (AIPerf 0.11.0 against the mock streaming content only: no `max_tokens`, one shared
  system prompt, history without reasoning, `LENGTHS` with no reasoning tokens and TTFT =
  time to the first output token, the ramp followed, ratio 2.34).

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
