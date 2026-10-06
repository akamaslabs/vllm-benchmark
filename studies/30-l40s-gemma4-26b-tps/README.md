# 30-l40s-gemma4-26b-tps

**Status:** TODO — scaffolded 2026-10-05; 2026-10-06 07:18 UTC the g6e.8xlarge pool was
empty, so the study moved to a g6e.xlarge node (`llm-serving-l40s-1xl`, brought up the same
morning). Nothing is on the Akamas server yet.
**Dates:** 2026-10-06 –

## Objective

**How many tokens per second can one NVIDIA L40S serve with Gemma 4 26B-A4B within a chat
SLA, and which base vLLM settings get there?** A base-configuration study like studies 0-1:
one GPU, no parallelism of any kind, studies 0-1's vLLM parameters, on a model and a GPU
not studied here before (the first Gemma model, the first L40S).

- **Goal:** maximize vLLM's **total token throughput** (prompt + generation tokens/s, the
  "TPS" of studies 11/12/25: `vllm.total_token_throughput` = `prefill_token_throughput +
  decode_token_throughput`).
- **Constraints (SLA of studies 1 and 25-28):** TTFT p95 <= 1500 ms and ITL p95 <= 300 ms,
  both p95 over 150 s, read with `:max` over the scored window.
- **Load:** study 27's open-loop linear rate ramp (Cereda's load, studies 27/29), on
  ShareGPT instead of 4096/256 synthetic prompts (see "Load").

## Stack & versions

- **Akamas version:** 3.7.x.
- **Optimization packs:** vLLM **1.12.0** (component type `vLLM`), GPU pack (metrics only;
  1.4.0 installed since 2026-09-30), Kubernetes pack (metrics only; 1.9.0-dev on the server).
  Check on the instance before `akamas create`.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  **`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`** as `gemma4-26b-l40s`, StatefulSet `vllm`
  (one pod, `vllm-0`) in namespace `llm-l40s`. Fixed flags: `--language-model-only`,
  `--no-enable-prefix-caching`, `--max-model-len=4096`, `--default-chat-template-kwargs
  '{"enable_thinking": false}'`. Pod: CPU request 3500m with no CPU limit, memory 28 GiB (request = limit), not tuned.
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), node group
  **`llm-serving-l40s-1xl`**: one **g6e.xlarge** (1x NVIDIA L40S 48 GB, Ada SM 8.9,
  4 vCPU / 32 GiB, 1.861 USD/h on demand). The study was designed for a g6e.8xlarge (same
  GPU, 32 vCPU / 256 GiB, node group `llm-serving-l40s-8xl`, created 2026-10-05 and left at
  0); on 2026-10-06 07:18 UTC only g6e.xlarge had capacity, in all three AZs, and the user
  chose it. Node up 07:24 UTC in us-east-2a: `nvidia-smi` reports NVIDIA L40S, driver
  580.178.04, 46068 MiB, 350 W, compute capability 8.9, SM max 2520 MHz; allocatable 3920m
  CPU / 29.9 GiB. Provisioning: `infra/`.
- **Load generator:** AIPerf 0.11.0 (`pip` in `python:3.12-slim`), `--public-dataset
  sharegpt` tokenized once with the Gemma 4 tokenizer and cached, `--request-rate R
  --request-rate-ramp-duration D --arrival-pattern gamma --arrival-smoothness 4
  --random-seed 30`, `AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10`, on the `system-m8a` node.
- **Telemetry:** Prometheus (kube-prometheus-stack), 30 s samples, vLLM `/metrics` through
  ServiceMonitor `vllm-l40s`, dcgm-exporter (shared release, this node role added: see
  "Morning runbook"). TTFT / ITL p95 computed over 150 s.

### The model

- `google/gemma-4-26B-A4B-it`: MoE, 26B total / ~4B active (128 experts, top-8), 30 layers
  (25 sliding-window layers, window 1024, head dim 256, 8 KV heads; 5 full-attention layers,
  head dim 512, 2 KV heads), 262144-token vocabulary, multimodal (vision tower).
- **bf16 does not fit:** 48.07 GiB of safetensors against the L40S's 48 GB (~45 GiB
  usable). The RedHat FP8-dynamic checkpoint (26.67 GiB, ungated, apache-2.0, validated by
  RedHat on vLLM 0.24) quantizes the transformer blocks' linear layers to FP8 (per-channel
  weights, per-token dynamic activations); vision tower, embeddings, lm_head and router stay
  bf16. NVFP4 is not an option on Ada.
- vLLM 0.29.0 supports it (`Gemma4ForConditionalGeneration`). Its `Gemma4Config` forces
  `TRITON_ATTN` on this GPU: the two head dims (256, 512) would mix backends, FA2/FA3 stop at
  256 and FA4 does not exist on Ada.
- Expected KV cache (estimate, the probe measures it): ~41 GiB of budget at
  `gpu_memory_utilization` 0.92, minus ~26 GiB of weights and ~2 GiB of activations and
  graphs, ~13 GiB of KV: ~225 KB per token in bf16 for requests shorter than the 1024-token
  window, so ~55-60k tokens, ~100 ShareGPT requests in flight; about twice that in fp8.

## Parameters tuned

Studies 0-1's base set, re-checked against vLLM 0.29.0's source and this model. Every
parameter goes through `k8s/params.env.template` -> `k8s/render_statefulset.sh`, which
writes booleans as `--x` / `--no-x` (vLLM 0.29.0 parses them with `BooleanOptionalAction`
and rejects `--x=false`, study 27's note).

| Parameter | Domain | Baseline (vLLM 0.29.0 default) | Why |
|---|---|---|---|
| `vllm.gpu_memory_utilization` | 0.80-0.94 | 0.92 | sets the KV left after ~26 GiB of weights; 0.94 still leaves ~2.7 GiB for the CUDA context and the sampler warm-up (0.5 GiB at 512 sequences with this vocabulary) |
| `vllm.max_num_seqs` | 16-512 | 256 | admission cap; the KV cache is expected to bind first |
| `vllm.max_num_batched_tokens` | 512-16384 | 2048 | prefill budget per step: TTFT against ITL stalls |
| `vllm.kv_cache_dtype` | auto, fp8 | auto | twice the KV tokens (fp8_e5m2 left out: a near-duplicate) |
| `vllm.performance_mode` | balanced, interactivity, throughput | balanced | vLLM 0.29.0 preset (CUDA graph sizes, batching) |
| `vllm.optimization_level` | 0-3 | 2 | 0 = no compilation, no CUDA graphs |
| `vllm.enforce_eager` | true, false | false | |
| `vllm.scheduling_policy` | fcfs, priority | fcfs | AIPerf sends one priority, so priority ~ fcfs; kept as in studies 0-1 |
| `vllm.async_scheduling` | true, false | true | |
| `vllm.max_cudagraph_capture_size` | 16-512 | 512 | vLLM's default is min(2 x max_num_seqs, 512); clamped by vLLM to the max batch tokens |
| `vllm.block_size` | 16, 32, ..., 128 (ordinal) | 16 | TRITON_ATTN takes any multiple of 16 |

**parameterConstraints:** `max_num_batched_tokens >= max_num_seqs` (vLLM 0.29.0 raises
`ValueError` otherwise). Study 16's sampler-warm-up guard is not needed: with this domain
box it cannot bind (0.94 x 44.99 GiB + 512 x 0.001 GiB = 42.8 < ~43.5 GiB).

**Left out, with the reason:**

- **All parallelism** (`tensor/pipeline/data/decode_context/prefill_context_parallel_size`,
  `enable_expert_parallel`): one GPU.
- **`attention_backend`** and **`linear_backend`**: decided by the startup probe. A backend
  enters only if it starts and serves within 15 % of the best of its group, and the
  parameter enters only if at least two do (study 28's rule). For attention, TRITON_ATTN is
  expected to be the only one on SM 8.9.
- **`disable_cascade_attn`**: no effect (cascade attention needs prefix caching, which is off,
  and the FlashAttention backend).
- **`tokenizer_mode`**: `slow` cannot load (the checkpoint ships `tokenizer.json` only), and
  `hf` is what `auto` selects for a non-Mistral model.
- **`spec_method` / `spec_tokens`**: no speculative decoding in a base-configuration study.
- `enable_prefix_caching` off and `max_model_len` 4096 as fixed flags (repo convention for
  ShareGPT replay).

## Load

- **Open loop, linear rate ramp (study 27):** `--request-rate R
  --request-rate-ramp-duration D`, gamma arrivals (smoothness 4: the same mean as Poisson, a
  quarter of the variance of the intervals), fixed seed 30, so every trial receives the same
  arrival sequence. 60 s at concurrency 4 before the measured run (discarded).
- **ShareGPT, not 4096/256:** a chat SLA (TTFT 1.5 s) on chat traffic, as studies 1 and
  25-28. Study 27's long fixed prompts went with its 10 s TTFT limit.
- **Watchdog** (`k8s/run_test.sh`): every 15 s it reads vllm-0's TTFT and ITL p95 over 150 s
  from Prometheus; past 2x the SLA (3000 ms / 600 ms) for 120 s it ends the test with
  success. Past the capacity of an open loop the queue only grows, so every later window is
  invalid anyway. Armed 150 s after the measured run starts. The fail-fast guards of studies
  27/28 stay (Job failed, pod restarted or replaced, no completion for 15 min, deadline).
- **Scoring:** `stability` windowing on `vllm.total_token_throughput`, 6 samples (3 min),
  `maxStdDev` 300000000 (filter disabled), `when: max`: the valid window with the most
  tokens/s, i.e. the last 3 minutes before the SLA breaks.
- **R and D:** defaults `RT_RATE=30`, `RT_RAMP_S=4500` (0.4 req/s per minute), sized on a
  guessed baseline knee K of ~10 req/s. The rule, applied after the smoke run: **R ~ 3 K
  (rounded up to 5), D = 4500 s**. Then the baseline crosses its knee at ~25 min, a 2x
  configuration at ~50 min; a 3-minute window spans ~12 % of the baseline's K and ~6 % of a
  2x configuration's. Arrival noise is small here: ~1800 requests in a window at 10 req/s,
  against ~90 in study 27. A configuration better than R/K is censored at R (the ramp ends
  without breaking the SLA; RunTest says so in its log): raise R if that happens.

## Steps

| # | Step | Values | Why |
|---|---|---|---|
| 1 | baseline | vLLM 0.29.0 defaults (table above), every parameter written out | the reference point |
| 2 | baseline repeat | the same | first measure of the noise (study 27: ~±4 %) |
| 3 | kv fp8 | baseline + `kv_cache_dtype` fp8 | twice the KV tokens: the expected largest single lever |
| 4 | kv fp8 large batch | fp8, gmu 0.94, `max_num_seqs` 512, `max_num_batched_tokens` 8192, `performance_mode` throughput | the large-batch corner |
| 5 | optimize | AKAMAS, 0 init experiments, 60 experiments, `maxFailedExperiments` 20 | as studies 27-29 |

Every baseline/preset renders every parameter (study 27's note: a `doNotRenderParameters`
step never reaches the optimizer engine). **Smoke study** `30-L40S-Gemma4-TPS-Smoke`: the
baseline only, with a steep ramp (0 -> 40 req/s over 15 min).

**KPIs (8, Italian names as the repo convention):** Throughput totale, Token generati,
Richieste completate (the req/s of the scored window = the knee), TTFT P95 150s, ITL P95
150s, KV cache in uso, Preemption, Potenza GPU.

## Startup probe (before the smoke run)

`probe/probe.sh`, outside Akamas, 15 vLLM starts on the L40S (~75-90 min, the first one also
downloads the model). Per combination it runs `k8s/apply_config.sh` and a short benchmark
inside the pod: prefill step of a ~2000-token prompt; long natural-language essays (256
tokens) one at a time, 64 concurrent in English and 32 concurrent in Italian; the MTP
acceptance rate per phase from vLLM's counters. It answers what cannot be verified from the
source:

- **B-default:** Gemma 4 FP8 MoE loads and serves on Ada with vLLM 0.29.0 (fused-MoE FP8 path),
  the log line "forcing TRITON_ATTN backend", `Model loading took`, `GPU KV cache size`.
- **K-fp8:** an fp8 KV cache with TRITON_ATTN on this model.
- **A-flashinfer / A-fa / A-triton:** which attention backends start (expected: TRITON_ATTN only).
- **L-cutlass / L-triton / L-marlin / L-torch:** which FP8 linear kernels start, and their
  speed against `auto`.
- **E-memory** (gmu 0.94, 512 sequences, 16384 batched tokens, fp8), **E-eager** (eager, O0,
  no async, priority, block 128, interactivity, 16 sequences), **E-o3** (O3, throughput,
  block 48, capture 16, gmu 0.80): the corners of the domains.
- **M-mtp2 / M-mtp4 / M-mtp2-fp8:** Gemma 4's MTP speculative decoding (drafter
  `google/gemma-4-26B-A4B-it-assistant`, 0.78 GiB, shares the target's KV cache; vLLM
  `--spec-method mtp --spec-model ... --spec-tokens K`). Does it start on Ada, what does it
  gain at batch 1 and at 64 concurrent requests against the same configuration without it,
  and how many drafted tokens are accepted on English and on Italian text. Decided with the
  user 2026-10-06: MTP enters the study as a variable (`spec_method` {none, mtp},
  `spec_tokens`, none <-> 0 constraint, one preset) only if it does not lose at 64
  concurrent requests; otherwise it is left to a later latency study with the customer's
  SLA and prompts. Never always-on: its gain depends on the load.

Decision after it: `Model loading took` should be ~26 GiB, the text weights only: with
`--language-model-only` vLLM 0.29.0 skips the vision tower (`_mark_tower_model` drops tower
modules when every multimodal limit is 0; checked in the source). Backends per the 15 % rule (edit both studies, `k8s/params.env.template`,
re-run `akamas/check_offline.py`); if an E- corner fails, narrow that domain; write the
measured KV size and the summary table here.

## Morning runbook (2026-10-06)

1. **Node up + AlwaysOn** (the study runs past 17:00 UTC; tag changes on shared AWS
   resources are the user's to run): `AWS_PROFILE=lab ./infra/eks/gpu-nodegroup.sh --up`
   (done 2026-10-06 morning, g6e.xlarge), then `AWS_PROFILE=lab ./infra/eks/gpu-nodegroup.sh
   --always-on`. `system-2b` (AIPerf, Prometheus) already carries `AlwaysOn=true` on its
   instance and ASG. On an `InsufficientInstanceCapacity`: probe the g6e sizes again
   (`../17-g7e-speculative-decoding-goodput/infra/eks/gpu-capacity-fallback.sh probe
   g6e.xlarge g6e.2xlarge ...`).
2. **Cluster layer:** `AWS_PROFILE=lab ./infra/eks/provision.sh` (namespace, PVCs, Services,
   ServiceMonitor).
3. **GPU telemetry:** `helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring
   --version 4.8.3 --reuse-values=false -f k8s/monitoring/dcgm-exporter-values.yaml`, then
   check a dcgm-exporter pod runs on the L40S node and Prometheus returns
   `DCGM_FI_DEV_GPU_TEMP{modelName=~".*L40S.*"}`.
4. **Startup probe** (~60-75 min), from this folder on the workstation:
   `mkdir -p /tmp/probe30 && KP_OUT=/tmp/probe30/results caffeinate -i nohup bash probe/probe.sh > /tmp/probe30/probe.log 2>&1 &`
   (or on the toolbox with `setsid nohup`, see `probe/README.md`; macOS has no `setsid`). Apply its decisions (above) and
   copy the results into `probe/results/`.
5. **Sync:** commit and push (user), `git pull` on the toolbox.
6. **Smoke study** (~35-50 min, the first trial also builds the ShareGPT cache): commands in
   `akamas/README.md`. Check, as study 27: the AIPerf log shows the ramp starting; the RunTest
   log shows the watchdog lines and ends with success; the KPIs are all filled. The trial is
   most likely VALID with a low score (study 27's smoke: a steep ramp scores a window well
   before the knee); INVALID by its constraints is acceptable too, as long as the watchdog
   fired. **Read the knee K two ways:** the low bound is "Richieste completate" of the scored
   window; the high bound is the rate at the RunTest line `watchdog: over`,
   `40 x (t_over - t_start) / 900` req/s with `t_start` the `measured run started` line (the
   watchdog is at 2x the SLA, so it fires past the knee). Take K as the midpoint.
7. **R / D** from K (rule in "Load"); if they differ from 30 / 4500, edit the workflow,
   push, pull, delete and recreate it.
8. **Delete the smoke study, create the study, wait 2 min, start it.**

## Budget

~35-85 min per experiment (vLLM start ~5-8 min with the model on the node: torch.compile and
CUDA-graph capture are slower on 4 vCPU; pip ~2 min, warm-up 1 min, the ramp until the
watchdog, 2-4 min of watchdog hold): 64 experiments ~2.5-3 days, **~130 USD** of g6e.xlarge
(1.861 USD/h), plus ~2 h of probe and smoke. The
optimize step can be stopped earlier if the best configuration plateaus (study 27).

## Expected results (written before the start)

Hypotheses, not measurements:

- The baseline is **KV-bound**: ~100 ShareGPT requests in flight in bf16; past that the
  scheduler queues and TTFT crosses 1.5 s. Knee guessed at ~8-15 req/s.
- **fp8 KV is the largest single lever** (twice the requests in flight), as in study 0
  (fp8 KV in the winner) and study 27 (fp8 KV in every top configuration).
- With fp8, the limit moves to the decode step: an MoE step at large batch reads nearly
  all 128 experts (~25 GB of FP8 weights), ~30 ms at the L40S's 864 GB/s, so ITL stays far
  below 300 ms and TTFT (queueing) binds again.
- `max_num_batched_tokens` above 2048 helps TTFT at high rate (ShareGPT prompts are short,
  several per step); `enforce_eager` / O0 lose clearly (no CUDA graphs on a decode-heavy
  load); `scheduling_policy` and `block_size` are inside the noise.

## Risks

- **Host CPU and memory (g6e.xlarge, 4 vCPU / 32 GiB):** at ~10-20 req/s of streaming
  ShareGPT the vLLM API server (tokenization, SSE) and the busy-looping engine core share
  ~3.6 cores. If they saturate, the score measures the CPU, not the GPU settings. The probe's
  64-request decode and the smoke run show it (KPIs CPU of `container`, and a GPU that is
  idle while requests queue); if it binds, stop and move to a larger g6e size. RAM: ~27 GB of
  model files through the page cache next to a ~5 GiB vLLM process under a 28 GiB limit
  (page cache is reclaimable; an OOM would show at the first start).
- **Capacity:** the g6e pool changed overnight (8xlarge on 2026-10-05, only xlarge on
  2026-10-06); a replaced node may not come back in the same size.
- **Gemma 4 on Ada:** the FP8 MoE path and the tokenizer in AIPerf (transformers >= 4.56,
  `GemmaTokenizer`) are unverified until the probe and the smoke run.
- **Censoring:** a configuration above R is scored at R (see "Load").
- **Domain corners:** a corner that does not start costs ~5-8 min and one of the 20 allowed
  failures; the probe's E- starts check them first.
- **Shared dcgm-exporter:** this study's GPU queries filter on `modelName=~".*L40S.*"`; the
  release now scrapes three node roles. Studies 0-17 (pod `.*`) would mix GPUs if resumed.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
