# 28-g7-4500-mig-min-cost-fixed-load

**Status:** RUNNING (started 2026-10-02 at 13:0x UTC, see akamas/README.md) — scaffolded 2026-10-02 (`492b032`),
kernel probe and calibration done 2026-10-02, R = 3.3 req/s confirmed with the user.
Checkpoint after the three presets (~2 h): if half a GPU in bf16 passes with a large margin,
propose stopping and restarting at a higher R.
**Dates:** 2026-10-02 –

## Objective

**Does the service fit in half a GPU?** For a fixed chat traffic, find the cheapest
deployment (MIG slice + CPU + RAM of the pod) that still holds the SLO. If the cheapest
valid configuration uses a `1g.16gb` slice, the answer is yes.

This is ROADMAP section D study #4 (MIG right-sizing, Goal B) in its inverted shape:
**minimize resources subject to a fixed load**, instead of study 26's "maximize
throughput per slice". Study 26 measured how much each MIG size sustains (closed loop,
Qwen3-4B); this study asks the question a capacity planner asks, for a model that is
tight on a slice.

- **Goal:** minimize the estimated hourly cost of one tenant (see "Goal").
- **Constraints (SLO, same values as study 26):** TTFT p95 <= 1500 ms and ITL p95 <=
  300 ms over 150 s, held for the whole measured window (`:max`), and at least 95 % of
  the offered requests completed.
- **Load:** a **fixed open-loop rate**, not a ramp and not concurrency levels (see "Load").

### Why Qwen3-8B-FP8 makes the question open

Measured on study 27 (L4, same checkpoint, vLLM 0.29.0): `kv_cache_size_tokens` 56720
at `gpu_memory_utilization` 0.86285 of 22.49 GiB, bf16 KV at 147,456 B/token (36 layers
x 8 KV heads x 128 x K,V x 2 B), so weights + runtime overhead ~ 11.6 GiB. A `1g.16gb`
slice has 15.66 GiB: at `gpu_memory_utilization` 0.90-0.95 that leaves **~2.5-3.5 GiB of
KV, ~18-24k tokens in bf16, ~36-48k in fp8**. Qwen3-4B (study 26) would fit easily (~56k
tokens of KV per slice, ~160 closed-loop users within this SLO): the answer would be
known in advance.

**Measured by the kernel probe (2026-10-02):** tighter than the estimate. On a `1g.16gb`
slice vLLM reports `Model loading took 8.88 GiB`, `Available KV cache memory: 1.98 GiB` and
`GPU KV cache size: 14,384 tokens` (bf16, `gpu_memory_utilization` 0.90): 3.5 requests of
4096 tokens.

## Stack & versions

- **Akamas version:** 3.7.x.
- **Optimization packs:** GPU pack 1.4.0 (`mig_profile`), vLLM pack 1.12.0
  (`linear_backend`, `attention_backend`), Kubernetes pack
  (installed build 1.8.0-dev in study 26; `Kubernetes Container` `cpu_limit` /
  `memory_limit` to be checked on the instance before `akamas create`).
- **Workload under test:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as
  `qwen3-8b-mig` (a name no other study's telemetry filters on), `--max-model-len 4096`, prefix caching off. StatefulSet `vllm` in namespace
  `gpu-sharing`, one pod per MIG instance (study 26's layout).
- **Cluster / hardware:** node group `llm-serving-g7-4500`, 1x g7.4xlarge (16 vCPU /
  64 GiB, NVIDIA RTX PRO 4500 Blackwell Server Edition 32 GB, 165 W, MIG `1g.16gb` x2 or
  `2g.32gb` x1). The node group is at 0 since 2026-10-01: scale it to 1 and tag node and
  ASG `AlwaysOn=true` before starting (nightly EC2 stop at 17:00 UTC). Provisioning:
  this study's `infra/`, copied from study 26.
- **Load generator:** AIPerf 0.11.0, ShareGPT (`--public-dataset sharegpt`, cached as in
  study 26), `--request-rate R` with no ramp, `--arrival-pattern gamma
  --arrival-smoothness 4`. One Job per serving replica, each with its own fixed
  `--random-seed` (28 for `vllm-0`, 29 for `vllm-1`): the same sequence in every trial,
  but the two slices do not receive their bursts at the same instant.
- **Telemetry:** Prometheus (kube-prometheus-stack, dcgm-exporter on the g7 node),
  30 s samples; study 26's metric catalog, with the TTFT / ITL p95 over 150 s (see
  "Goal").

## The MIG layout and the busy neighbour

| `mig_profile` | GPU share of the tenant | Pods | Load |
|---|---|---|---|
| `none` | 1 (whole GPU, MIG off) | `vllm-0` | R on `vllm-0` |
| `1g.16gb` | 0.5 | `vllm-0` (under test) + `vllm-1` (neighbour) | R on each |

MIG isolates compute and memory bandwidth but **not power**: in study 25 one slice alone
served ~13 % more than with its neighbour busy (2378 vs 2076 output tok/s). A tenant in a
shared cluster has a busy neighbour, so the second slice runs an identical replica (same
StatefulSet, same parameters) with the same traffic. Goal and constraints read
`vllm-0` only; `vllm-1` is reported as a KPI so a starved or dead neighbour is visible.

`2g.32gb` is left out: it costs the same as `none` and served 3-6 % less in study 26.
Two `1g.16gb` replicas for one tenant are left out too: they cost a whole GPU, like
`none`, with twice the CPU/RAM.

## Load

**Fixed open-loop rate.** Every configuration receives exactly the same traffic: the same
rate and, with the fixed seed, the same arrival sequence. In a closed loop an
under-provisioned configuration slows its users down and receives less traffic (it
"unloads itself"), which is the bias to avoid when the optimizer is pushed towards fewer
resources. A fixed rate also avoids the ramp effects found in study 27 (the dead time
before the first request, the window lag, the max over windows).

- **Target R = 3.3 req/s**: "100 concurrent chat users", each sending one message every
  ~30 s (think and reading time included). With ShareGPT's lengths that is ~25-30
  requests in flight. Rendered from `RT_RATE` by the RunTest script, so it can change
  without touching the template.
- **Duration:** a 1 min warm-up at concurrency 4 (a separate AIPerf run, as in study 26;
  ~1 req/s, well below R), then **13 min at R**. Akamas scores a 12 min window inside the
  13 min (see "Goal", windowing).
- **Addressing:** each Job targets its own pod through the StatefulSet's headless
  Service `vllm-headless` (`vllm-0.vllm-headless.gpu-sharing.svc.cluster.local`, `vllm-1...`),
  not study 26's load-balanced Service `vllm`. With `mig_profile` `none` only the `vllm-0` Job runs.
- **Cold-start spike out of every window:** the first requests a fresh vLLM serves hit
  first-use kernel JIT (TTFT p95 33.6 s in study 25's phase 0, against ~0.1 s 20 s
  later). A 150 s p95 remembers such a request for 150 s, so it must never fall inside the
  scored window or the watchdog's view. `apply_config.sh` therefore ends by sending 8
  short requests to every replica; the RunTest task then spends ~2 min on `pip install`
  and 1 min on the AIPerf warm-up, so the spike is > 150 s old when the measured run
  starts.
- **Watchdog (fail fast):** armed 150 s after the 13 min run starts (the AIPerf log
  prints `MEASURED RUN START`), so nothing from the warm-up is in its 150 s view. If
  `vllm-0`'s TTFT p95 (150 s) stays above 3000 ms or its ITL p95 above 600 ms for 120 s
  (2x the SLO), the test ends early; the earliest it can fire is ~4.5 min into the run. The execution succeeds and the goal is
  INVALID: with no 12 min window of full traffic, the success-rate constraint fails on its
  own, on top of the latency ones. The next experiment starts up to ~8 min earlier, which
  matters because the optimizer will explore many under-provisioned configurations.
- Study 27's guards stay: the trial fails if a Job fails, a serving container restarts or
  is replaced, or no request completes for 15 min.
- **Verified locally (AIPerf 0.11.0 against a mock endpoint, 2026-10-02):** the fixed-mode
  arguments with an `inputs_json` dataset give `pattern=gamma, rate=3.3, smoothness=4.0`,
  198 requests in 60 s, the first one 0.29 s after the start (no dead time without a ramp),
  interval CV 0.47 (0.50 expected for smoothness 4).

### Calibration (before the study)

A separate Akamas study with two presets runs the **study 27 ramp** (0 -> 12 req/s over
40 min, gamma, watchdog) once on the whole GPU and once on half a GPU with a busy
neighbour, bf16 KV, generous CPU/RAM. It measures each layout's maximum rate within the
SLO and checks the whole pipeline end to end. Decision after it:

- R between the half-GPU and the whole-GPU capacity: the GPU dimension decides, as
  intended.
- R well below the half-GPU capacity: the answer is "yes, it fits"; the study still runs
  and the interesting part becomes CPU/RAM and the vLLM settings.
- R above the whole-GPU capacity: the target is not reachable on this node; change R
  before starting.

**Calibration results (2026-10-02, study `28-G7-4500-MIG-Min-Cost-Calibration`,
`results/calibration-export.tar.gz`):** both VALID.

| Step | Score (max req/s on `vllm-0` within the SLO, 3-min window) | What ended it |
|---|---|---|
| baseline: whole GPU, bf16, 7 cores / 28000 MB | 11.56 | the ramp's end (12 req/s): no limit reached. At 11.4 req/s TTFT p95 ~90 ms, ITL p95 ~35 ms, KV 18 %: the whole GPU holds well above 12 req/s |
| half GPU bf16 (busy neighbour) | **3.67** | the KV cache: at ~3.7 req/s KV 100 %, preemptions, waiting queue up to ~200, TTFT p95 to 34-74 s; ITL p95 stayed at ~49 ms |

- Half a GPU in bf16 is KV-bound (14,384 tokens), not decode-bound: fp8 KV should move its
  limit well above 3.7 req/s. Below ~3.4 req/s TTFT p95 stayed under 0.2 s, with KV 50-75 %
  and a few preemptions (0.16/s) from 2.8 req/s.
- The neighbour (`vllm-1`, same ramp, seed 29) behaved the same; the GPU sat at its 165 W
  cap, as in study 26.
- `vllm:request_success_total` by `finished_reason`: no `abort` or `error` in either run, so
  the success-rate constraint needs no filter.
- **R stays 3.3 req/s** (confirmed with the user 2026-10-02): 90 % of the half-GPU bf16 capacity measured on the ramp. A ramp
  measures a quasi-static limit, and the study holds R for 13 min, so half a GPU in bf16 at
  defaults is borderline: the GPU dimension, the KV dtype and the CPU/RAM floor all matter,
  as intended.

### Kernel probe (before the calibration)

Outside Akamas, on one `1g.16gb` slice (the tight case), as study 24's `kernel-bench/`:
start vLLM with each candidate `linear_backend` (`auto`, `cutlass`,
`flashinfer_cutlass`, `deep_gemm`, `marlin`, `triton`, and any other the pack lists for
FP8) and each `attention_backend` (`FLASHINFER`, `FLASH_ATTN`, `TRITON_ATTN`), with bf16
and fp8 KV. Record which combinations start, which kernel vLLM reports it actually
selected, and the prefill step (2048-token prompt, mean of 4) and the decode step at the
study's regime (30 concurrent ShareGPT-like requests: ~100-token prompts, 256 output tokens,
which fits a slice's KV even in bf16). Keep in the domain the backends that start and are
within ~15 % of the best of their group; note the rest in this README.

**Results (2026-10-02, 09:00-09:48 UTC, `kernel-probe/results/`):** one replica on a
`1g.16gb` slice, the other slice idle, 7 cores / 28000 MB, `gpu_memory_utilization` 0.90,
`max_num_seqs` 256, `max_num_batched_tokens` 2048. Client-side wall times (HTTP included).

| Combination | Kernel vLLM selected | Prefill 2k (s) | TPOT 1 req (ms) | TPOT 30 req (ms) | Decision |
|---|---|---|---|---|---|
| linear `auto`, FLASHINFER, bf16 | DeepGemmFp8BlockScaledMMKernel | 0.257 | 22.9 | 27.3 | kept (baseline) |
| linear `deep_gemm` | DeepGemmFp8BlockScaledMMKernel | 0.251 | 22.9 | 27.3 | dropped: the same kernel as `auto` |
| linear `cutlass` | CutlassFp8BlockScaledMMKernel | 0.288 | 22.9 | 27.5 | kept (+12 % prefill vs auto) |
| linear `triton` | TritonFp8BlockScaledMMKernel | 0.405 | 24.3 | 28.7 | dropped (+58 % prefill) |
| linear `marlin` | MarlinFP8ScaledMMLinearKernel | 0.617 | 23.0 | 28.1 | dropped (+140 % prefill) |
| linear `flashinfer_cutlass` | — | — | — | — | does not start: "FlashInfer block-scale FP8 GEMM is not available" |
| linear `humming` | — | — | — | — | does not start: `pynvml.NVMLError_NoPermission` in the container |
| attention FLASH_ATTN, bf16 | DeepGemm + FLASH_ATTN | 0.266 | 22.9 | 27.3 | kept |
| attention TRITON_ATTN, bf16 | DeepGemm + TRITON_ATTN | 0.301 | 22.7 | 27.0 | kept |
| attention `auto`, bf16 | selects FLASH_ATTN | 0.268 | 22.9 | 27.2 | not a category (= FLASH_ATTN) |
| FLASHINFER, fp8 KV | DeepGemm + FLASHINFER | 0.252 | 22.9 | 26.2 | — (fp8 is the `kv_cache_dtype` parameter) |
| TRITON_ATTN, fp8 KV | DeepGemm + TRITON_ATTN | 0.280 | 22.8 | 25.7 | — |
| FLASH_ATTN, fp8 KV | — | — | — | — | does not start: "FP8 KV cache requires FA3 on SM90 or FA4 on SM100": constraint kept |

- Decode is the same with every kernel (TPOT 22.7-24.3 ms alone, 25.7-28.7 ms at 30): it is
  memory-bound; the linear kernels differ on prefill only. fp8 KV shortens the decode step
  at 30 sequences by 4-5 %.
- `tuned_kernel_configs` stays out: Triton is not competitive here.
- **vLLM's own footprint**, every start of the probe (13, pod `vllm-0`): working set peak
  5.08 GiB, RSS 4.88 GiB (12.40 GiB with the page cache of the model files), CPU peak 1.12
  cores. Hence `container.memory_limit` >= 8500 MB (1.3 x 5.08 GiB + 1 GiB `/dev/shm`,
  rounded up) and the 2-core floor kept.
- Startup 119-394 s (the first one also downloaded the model onto the node).

## Parameters tuned

| Parameter | Domain | Baseline | Why |
|---|---|---|---|
| `gpu0.mig_profile` | `none`, `1g.16gb` | `none` | the GPU share: whole or half |
| `container.cpu_limit` | 2000-7000 millicores | 7000 | request = limit (Guaranteed QoS). Below 2 cores vLLM V1 starves: its engine-core process busy-loops on one core |
| `container.memory_limit` | 8500-28000 MB | 28000 | request = limit. Floor from the kernel probe's measured footprint ("Kernel probe"); below it, OOMKilled at load |
| `vllm.gpu_memory_utilization` | 0.80-0.95 | 0.90 | of the MIG instance under MIG: sets the KV left after the weights |
| `vllm.kv_cache_dtype` | `auto`, `fp8` | `auto` | fp8 doubles the KV tokens: likely the lever that makes the 8B fit in half a GPU |
| `vllm.max_num_seqs` | 16-256 | 256 | admission cap; bounded in practice by the KV cache |
| `vllm.max_num_batched_tokens` | 1024-8192 | 2048 | prefill chunk per step: TTFT vs ITL |
| `vllm.linear_backend` | `auto` (DeepGEMM), `cutlass` | `auto` | FP8 GEMM kernel; the others are slower or do not start here ("Kernel probe") |
| `vllm.attention_backend` | `FLASHINFER`, `FLASH_ATTN`, `TRITON_ATTN` | `FLASHINFER` | all three start and are within 15 % ("Kernel probe") |

The vLLM parameters do not change the cost: they decide whether a cheaper configuration
holds the SLO (fp8 KV, a faster kernel, a smaller `gpu_memory_utilization` margin...).
They are the same knobs as studies 24-27, on one aggregated `vllm` component: each
slice runs one vLLM, there is no prefill/decode split here. Two differences from those
studies:

- **Kernel domains come from a probe on this GPU, not from study 24.** Study 24
  measured the backends on an L4 (SM 8.9); this is an RTX PRO 4500 Blackwell (SM 12.0),
  where other backends exist (the vLLM pack lists `cutlass`, `flashinfer_cutlass`,
  `deep_gemm`, `flashinfer_b12x`...) and some L4 ones may not apply to Qwen3-8B-FP8's
  block-quantized FP8. Only the backends that start and serve correctly enter the domain
  (see "Kernel probe").
- **`tuned_kernel_configs` is left out.** Study 24's tuned Triton FP8 configs are files
  for `device_name=NVIDIA_L4`; vLLM would not load them here. They would be re-tuned for
  this GPU (study 24's `tune_fp8_block.py`) only if the probe shows `triton` competitive.
- The probe confirmed that `FLASH_ATTN` rejects an fp8 KV cache on SM 12.0 as on the L4
  ("FP8 KV cache requires FA3 on SM90 or FA4 on SM100"): study 24's constraint is kept, `attention_backend !=
  "FLASH_ATTN" || kv_cache_dtype == "auto"`.

CPU and memory per replica: two replicas at the domain maxima request 14 cores / 56 GB
of the node's 16 vCPU / 64 GiB. The node's allocatable is lower than that (kubelet
reserve and the dcgm-exporter / device-plugin / node-exporter DaemonSets): read it with
`kubectl describe node` once the node is up, and lower the CPU maximum if two replicas at
7000 m do not fit, rather than adding a cross-replica constraint. Parameter
domains must fit inside the installed component types' domains (checked with `akamas
describe` before `akamas create`).

## Goal

Minimize the **estimated hourly cost of the tenant** (`vllm-0`), in USD/h, from AWS
on-demand prices in us-east-2 (AWS Pricing API, 2026-10-02):

| Resource | Price | How it is derived |
|---|---|---|
| GPU | 2.0683 USD/h per whole GPU | g7.4xlarge 3.04208 minus m8a.4xlarge 0.97376 (same 16 vCPU / 64 GiB without GPU) |
| vCPU | 0.04522 USD/h | from c8a.4xlarge (16 vCPU / 32 GiB, 0.86216) and r8a.4xlarge (16 / 128 GiB, 1.27808) |
| Memory | 0.0043325 USD/GiB-h | same pair: (1.27808 - 0.86216) / 96 GiB |

```
cost = 2.0683 * vllm_r0.active_gpus
     + 0.04522 * container.container_cpu_limit / 1000
     + 0.0043325 * container.container_memory_limit / 1073741824
```

`active_gpus` reads the pod's `nvidia.com/gpu` request over the node's allocatable
(1 with `none`, 0.5 per replica with `1g.16gb`, study 26's query); `container` is scoped
to pod `vllm-0`. Examples: whole GPU, 7 cores, 28 GB: ~2.50 USD/h; half GPU, 7 cores,
28 GB: ~1.46; half GPU, 2 cores, 8.5 GB: ~1.16. The GPU dominates; CPU and RAM separate
configurations with the same slice (2 -> 7 cores moves the cost by ~0.23 USD/h).

Constraints (all on `vllm_r0`):

- `time_to_first_token_p95:max <= 1500`
- `inter_token_latency_p95:max <= 300`
- `request_success_rate:avg >= 3.135` (0.95 x R; catches a dead or erroring replica,
  whose failed requests do not reach the latency histograms)

**The two p95 metrics are computed over 150 s in this study's telemetry.** vLLM pack
1.12.0 declares `time_to_first_token_p95_150s` / `inter_token_latency_p95_150s` only on
the `vLLM_PD_Topology` component type, not on `vLLM`, which this study uses. So the
telemetry instance redefines `time_to_first_token_p95` and `inter_token_latency_p95`
with a `[150s]` rate window instead of `[$DURATION$]` (as study 26 redefined
`active_gpus`). At 3.3 req/s a 150 s p95 holds ~500 requests; a 30 s one ~100, and the
`:max` over 24 of them would act as a p99. TODO (pack): declare the `_150s` metrics on
the `vLLM` component type too, then switch back to their own names.

**Windowing: stability, not trim.** A trim window is anchored at the start of the
trial, which includes the Apply config task (MIG reconfiguration and vLLM start,
~14 min on this node): ~10 min of zero traffic would land in the scored window and fail
the success-rate constraint for every configuration. Instead, as in studies 26-27:
`stability` on `vllm_r0.request_success_rate`, `width` 24 samples (12 min at 30 s),
`maxStdDev` very large (the filter is disabled), `when: max`. The window with the most
completions is the 12 min inside the 13 min run at R (the warm-up is ~1 req/s), and the
`:max` constraints apply over exactly that span.

## KPIs (at most 8, Akamas 3.7)

| KPI | Formula | Why |
|---|---|---|
| Memoria usata | `container.container_memory_working_set` (`:max`) | the real footprint against the memory limit: the working set, not `container_memory_usage_bytes`, which counts the page cache of the ~9 GiB of model files (the cost itself is the experiment score) |
| TTFT P95 150s | `vllm_r0.time_to_first_token_p95` (`:max`, 150 s in this telemetry) | constraint |
| ITL P95 150s | `vllm_r0.inter_token_latency_p95` (`:max`, 150 s in this telemetry) | constraint |
| Richieste completate | `vllm_r0.request_success_rate` | constraint |
| KV cache in uso | `vllm_r0.kv_cache_usage_avg` | how close the slice is to its KV limit |
| CPU usata | `container.container_cpu_used` | how much of the CPU limit is really used |
| Richieste vicino | `vllm_r1.request_success_rate` | the neighbour really carried R |
| Temperatura GPU | `gpu0.gpu_temp` | study 26's ~5 % thermal drift is pass/fail noise at the SLO boundary |

## Steps

| # | Step | `mig_profile` | CPU / RAM | KV dtype | Why |
|---|---|---|---|---|---|
| 1 | baseline | none | 7000 m / 28000 MB | auto | whole GPU, generous resources: must pass, or R is not reachable |
| 2 | half GPU bf16 | 1g.16gb | 7000 / 28000 | auto | the slice at defaults |
| 3 | half GPU fp8 | 1g.16gb | 7000 / 28000 | fp8 | twice the KV tokens |
| 4 | half GPU fp8 lean | 1g.16gb | 2000 / 8500 | fp8 | the cheapest corner of the space |
| 5 | optimize | — | — | — | up to 60 AKAMAS experiments (9 parameters), no init experiments, step stops after 20 failed |

Other vLLM parameters in the presets: `gpu_memory_utilization` 0.90, `max_num_seqs`
256, `max_num_batched_tokens` 2048, `linear_backend` `auto`, `attention_backend`
`FLASHINFER`. Step names follow Akamas' pattern (no digit first, no hyphens or dots).

**The goal is cost only (decided 2026-10-02).** The study answers "what is the cheapest
configuration that holds R", vLLM settings included. Among configurations with the same
resources it does not rank the vLLM settings (no latency or headroom term in the goal):
the reported best is the cheapest valid one, not the one with the most margin. A
headroom tie-breaker, or a follow-up study maximizing capacity on the winning slice, was
considered and not chosen.

**Budget:** ~30 min per experiment (MIG reconfiguration and vLLM start ~14 min on this
node as in study 26, then the AIPerf Job start, 1 min of warm-up and 13 min at R; less
when the watchdog ends a failing trial): up to 64 experiments ~32 h, plus ~2 h of kernel
probe and ~2 h of calibration, ~36 h at ~3.3 USD/h (g7.4xlarge + load-generator node)
~ 120 USD. Optimize step as studies 27/29 (set 2026-10-02): `optimizer: AKAMAS`,
`numberOfInitExperiments: 0`, `numberOfExperiments: 60`, `maxFailedExperiments: 20`
(failed workflows and constraint violations both count).

## Before starting

- Scale `llm-serving-g7-4500` to 1 and tag the instance and its ASG `AlwaysOn=true`.
- Read the node's allocatable CPU / memory (see "Parameters tuned").
- The model cache of the vLLM pods is a `hostPath` on the GPU node (study 26's choice: the
  node group spans three AZs, an EBS volume would pin it to one), so a node scaled up
  from 0 starts empty: the first vLLM start downloads `Qwen/Qwen3-8B-FP8` (~9-10 GiB) onto
  the node. Let the kernel probe pay it (it runs first), and expect it again whenever the
  node is replaced.
- Check on the Akamas instance: GPU pack 1.4.0, vLLM pack 1.12.0, and the Kubernetes
  pack's `Kubernetes Container` `cpu_limit` / `memory_limit` domains.
- Run the kernel probe and write the `linear_backend` / `attention_backend` domains.
- Run the calibration study, decide R, then create and start the study.

## Expected results (written before the start)

Hypotheses, not measurements:

- The whole GPU holds 3.3 req/s with a wide margin (study 26: the whole GPU served ~18
  req/s of ShareGPT with Qwen3-4B; the 8B reads about twice the weights per step).
- Half a GPU in bf16 is KV-bound: 14,384 tokens (measured) against ~25-30 requests in
  flight of a few hundred tokens each should still fit, with little headroom.
- fp8 KV passes more easily and lets `gpu_memory_utilization` go lower.
- The cheapest valid configuration is half a GPU with fp8 KV, ~2-3 cores and the lowest
  memory that loads the model, ~1.2 USD/h against ~2.5 for the baseline.
- If half a GPU fails, the likely binding constraint is ITL (decode on half the SMs,
  under the shared 165 W cap), not the KV cache.
- The kernels matter only where a slice is close to the SLO: a faster `linear_backend`
  can let a configuration with fewer cores or bf16 KV pass, but it does not change the
  cost of one that already passes.

## Risks

- g7 capacity: the g7.4xlarge had capacity in every AZ on 2026-09-29, but nothing
  guarantees it on the day.
- OOMKilled or CPU-starved trials cost ~8 min each and fail the experiment; the presets
  give the optimizer valid points first.
- The cost goal is deterministic given the parameters: only the constraints carry
  measurement noise, so near the SLO boundary a configuration can pass or fail by noise.
  A finalist should be re-run before it is reported.
- The Kubernetes pack build on the instance must expose `cpu_limit` / `memory_limit` on
  `Kubernetes Container`, inside the domains above.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
