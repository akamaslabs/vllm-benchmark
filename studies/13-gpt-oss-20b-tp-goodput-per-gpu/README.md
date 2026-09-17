# 13-gpt-oss-20b-tp-goodput-per-gpu

**Status:** STOPPED after 16 of 100 experiments — superseded by
[`14-gpt-oss-20b-parallelism-and-kv-goodput-per-gpu`](../14-gpt-oss-20b-parallelism-and-kv-goodput-per-gpu/README.md),
which reuses this study's system, telemetry instance and workflow and imports all 16
experiments.
**Dates:** Scaffolded 2026-09-15; ran 2026-09-15 19:42 UTC -> 2026-09-16; stopped 2026-09-16
(the last state read live from the instance was RUNNING with 16 experiments finished and the
17th in progress; the stop is per study 14's manifest and was not re-verified — the shared
toolbox CLI session lost its Administrator role).

> **Outcome in one line.** Best experiment 16: **2345.31 tokens/s per active GPU (+18.49%**
> over the 1979.41 baseline) with TP2/DP1, expert parallelism on, `kv_cache_dtype` auto,
> `gpu_memory_utilization` 0.80, `max_num_seqs` 1024, `max_num_batched_tokens` 7781,
> `block_size` 96, optimization level 3. It was stopped early because its Sobol initial
> design **perfectly aliased `kv_cache_dtype == fp8` with `enable_expert_parallel == false`**
> across all 15 optimizer experiments, so neither effect is identifiable — see study 14's
> README for the contingency table and the fix. The full recap (Results/Conclusions below)
> is still to be written with the `study-recap` skill.

**Note on the `kpis` block** added to the manifest on 2026-09-16: it was never applied to
the running study (there is no `akamas update` verb for `kpis`, and re-creating the study
would have discarded the 16 experiments study 14 imports). Study 14 carries an 8-KPI block
of its own.

## Objective

**Same goal, model, hardware and 14 parameters as studies 10, 11 and 12 — but every
configuration must shard the model over at least two L4s with tensor parallelism.** The
question is no longer "is one GPU enough?" (studies 10-12 let the optimizer answer that by
offering TP1) but "given that we spread gpt-oss-20b over 2-4 L4s, which way of doing it —
TP2 on two GPUs, TP2 x DP2 or TP4 on all four, with or without expert parallelism — gives
the highest throughput per GPU actually used, without breaching the latency SLA?"

```
maximize  (vLLM.prefill_token_throughput + vLLM.decode_token_throughput) / vLLM.active_gpus
subject to  vLLM.time_to_first_token_p95 <= 1500 ms,  vLLM.inter_token_latency_p95 <= 300 ms
```

`active_gpus` (GPUs with > 1 GiB of framebuffer in use, i.e. TP x DP) is 2 for TP2/DP1 and
4 for TP2/DP2 and TP4/DP1, so the three topologies are compared on the same per-GPU footing.

### KPIs (added 2026-09-16)

The manifest carries an explicit `kpis` block with the five metrics the goal and the two
constraints already reference — the same set Akamas derives by itself when the block is
absent. It is written out only to pin each metric's direction, which the UI otherwise has
no way to infer for `active_gpus`:

| KPI | direction | why |
|---|---|---|
| `vLLM.active_gpus` | minimize | it is the goal's divisor: at equal goodput, fewer GPUs wins |
| `vLLM.decode_token_throughput` | maximize | goal numerator |
| `vLLM.prefill_token_throughput` | maximize | goal numerator (and the windowing metric) |
| `vLLM.inter_token_latency_p95` | minimize | SLA constraint (<= 300 ms) |
| `vLLM.time_to_first_token_p95` | minimize | SLA constraint (<= 1500 ms) |

`name` is omitted so each KPI's UI label defaults to the metric name, and `aggregation` is
omitted so it stays `avg` over the stability window — the two p95 metrics are already
quantiles computed inside the Prometheus query, and `avg` is how the constraints evaluate
them too. The score is still the goal formula and the optimizer still reads only goal and
constraints; per the 3.7 docs the `kpis` block drives the UI (the "Best `<name>`" badges and
the per-KPI columns).

## What changes vs. studies 10 and 12

| | study 10 | study 12 | **this study** |
|---|---|---|---|
| `tensor_parallel_size` | [1, 4], != 3 | [1, 4], != 3 | **[2, 4]**, != 3 → {2, 4} |
| `data_parallel_size` | [1, 4] | [1, 4] | **[1, 2]** |
| valid topologies | 7 | 7 | **3**: TP2/DP1, TP2/DP2, TP4/DP1 |
| `parameterConstraints` | 5 | 5 | **5**: EP-needs-more-than-one-GPU dropped (always true), **sampler-warmup guard added** |
| load | 12 levels, 150->1024 x 300 s | constant 512 x 600 s | **12 levels, 150->1024 x 300 s** (study 10's) |
| windowing | stability, `is: max` | `trim: [2m, 30s]` | **stability, `is: max`** (study 10's block) |
| experiment wall-clock | ~75 min | ~15.5 min | **~75 min** |
| optimize | 200 / 40 | 45 (400 / 20 in the manifest) | **100 / 20** (~5 days) |
| baseline | TP1 defaults + `gpu_memory_utilization` 0.90 | idem + `kv_cache_dtype: fp8` | **TP2** + defaults + `gpu_memory_utilization` 0.90 |
| `apply_config.sh` | 20-min rollout wait | idem | **+ crash-loop fail-fast** |

Everything else — goal, SLA, the other 12 parameters and their domains, the 9 components,
the 108-metric telemetry, the workflow shape and timeouts, the restart guard in
`run_test_goodput.sh`, the deployment template and the model-specific AIPerf dataset cache
— is byte-identical to study 12.

## Why TP >= 2 only

Study 12's own baseline showed that a single L4 saturates early on this model: with bf16
KV cache one GPU holds ~119k tokens of KV, 256 running sequences fill 92% of it, and the
engine-side TTFT p95 at 512 concurrent requests was 38 s. Studies 10-12 leave it to the
optimizer to discover that TP1 loses at high load; this study removes TP1 from the search
altogether so that all of its ~100 experiments are spent comparing the multi-GPU layouts
against each other, per GPU. `data_parallel_size` is narrowed to [1, 2] because with
TP >= 2 and `TP x DP <= 4` no larger value can be valid — the narrower domain only spares
the optimizer proposals the constraint would reject.

The baseline has to render `tensor_parallel_size: 2` explicitly: the pack default is 1,
outside this study's domain. It is otherwise vLLM's own defaults on two GPUs (TRITON_ATTN,
auto KV dtype, `max_num_seqs` 256, optimization level 2, no EP) with
`gpu_memory_utilization` 0.90 as in studies 10-12. `kv_cache_dtype` goes back to
unrendered: study 12 pinned fp8 because its constant 512-request load made a bf16
single-GPU baseline infeasible; under the ramp the stability window scores the level where
the baseline peaks, and at TP2 the per-GPU KV budget is ~12 GiB for half the heads, so bf16
capacity is roughly four times study 12's single-GPU figure anyway.

## Why the ramp and `stability` windowing are back

Study 12 replaced the ramp with one constant level because 512 concurrent requests was the
measured point that separates a single L4 (queues from 302) from TP2/DP2 (queue-free at
512). With TP >= 2 that single discriminating level does not exist: the aggregate capacity
of the valid topologies spans 2x to 4x one GPU, so any constant level either leaves TP4
unsaturated or drowns TP2/DP1. Study 10's 12-level ramp (150 -> 1024, 300 s per level)
walks through every regime, and the `stability` window — 6 samples around the
prefill-throughput peak, `is: max` — reads each configuration at its own peak. The ramp's
known artefacts (AIPerf reconnects between levels, ~30 s with nothing in flight; the window
may straddle a level change) are far cheaper on 300 s levels than they were on study 11's
90 s levels, and `maxStdDev: 3e8` still disables the stability filter as in studies 10/11
(documented limitation; the restart guard covers the crash-after-peak case).

The cost is time: ~67 min of load per trial, ~75 min per experiment against study 12's
~15. Hence optimize 100/20 (~5 days) rather than study 12's 45 or study 10's 200/40.

## The sampler-warmup memory guard

Study 12's experiment 12 (TP1/DP4/EP, `gpu_memory_utilization` 0.89995, `max_num_seqs`
1024, `enforce_eager`) never came up: every start ended ~70 s in with a CUDA OOM on all
four GPUs, and the trial burned the 20-minute progress deadline in CrashLoopBackOff. The
OOM was **not** in weight loading or KV-cache allocation — both had succeeded — but in
`compile_or_warm_up_model` → `warmup_kernels` → `sample_tokens` → FlashInfer
`top_k_top_p_sampling_from_logits` → `torch.softmax`:

| | value |
|---|---|
| allocation that failed | 824,180,736 B = 786 MiB |
| = `max_num_seqs` x vocab x fp32, rounded to 2 MiB | 1024 x 201,088 x 4 = 823,656,448 B |
| torch memory per L4 | 22.03 GiB |
| budget at 0.89995 | 19.83 GiB (8.99 weights + non-torch, 0.44 "peak activation", 10.40 KV cache) |
| headroom left after the KV cache | ~2.3 GiB |
| PyTorch allocated at the OOM | 21.31 GiB, i.e. ~3 fp32 logits copies already live, the 4th failing |

vLLM's profiler sizes the KV cache to fill the budget and its 0.44 GiB "peak activation"
cannot contain even one 786 MiB copy: the sampler warmup is outside its accounting. This
is the same failure mode `ROADMAP.md` section C records from study 0 (Qwen2.5-7B, A10G,
`max_num_seqs` 917/1016 near the top of `gpu_memory_utilization`); gpt-oss-20b's 201k
vocabulary makes it 32% worse per sequence. Tensor parallelism does not relax it: the
headroom after the KV cache is `(1 - gpu_memory_utilization) x 22.03 GiB` whatever the
topology, and every TP rank samples over the full logits (assumption — see the smoke test).

The new `parameterConstraint` keeps 5 fp32 copies of the logits inside that headroom, with
0.4 GiB reserved for the CUDA context (5 x 201,088 x 4 B = 0.00375 GiB per sequence; 4
copies were live at the OOM, the 5th is margin):

```
vLLM.gpu_memory_utilization * 22.03 + vLLM.max_num_seqs * 0.00375 <= 21.63
```

| `gpu_memory_utilization` | `max_num_seqs` admitted |
|---|---|
| 0.80 | whole domain (≤ 1068) |
| 0.85 | ≤ 774 |
| 0.90 | ≤ 480 |

The baseline (0.90, `max_num_seqs` at vLLM's default 256) passes. The inherited
weights-shard constraint uses DCGM's 22.5 GiB `FB_TOTAL`; the guard uses the 22.03 GiB
torch sees — the 0.5 GiB gap is memory torch never gets. Two smoke-test configurations
(`tp2-guard-090` = 0.90/470 replaying experiment 12's other settings at TP2, `tp2-guard-085`
= 0.85/760) sit just inside the line and are the empirical check that the TP-invariance
assumption holds; if either OOMs in `warmup_kernels`, lower the 21.63 constant before
starting the study. The shape is reusable for any model: replace 0.00375 with
`5 x vocab_size x 4 B` in GiB.

## Workflow guards

- **Restart guard** (`k8s/run_test_goodput.sh`, inherited from study 10): the vLLM
  container's `restartCount` is read before and after the AIPerf job; if it grew, the
  previous container's logs are dumped and the trial FAILS.
- **Crash-loop fail-fast** (`k8s/apply_config.sh`, new): the rollout is polled in 30 s
  slices instead of one 1500 s wait, and as soon as the vllm container of a pod created by
  this rollout has restarted twice the script dumps the logs and exits 1. A deterministic
  startup crash now costs ~3 min instead of the 20-minute progress deadline (which stays as
  the backstop for a pod that never becomes ready without crashing). Pods are filtered by
  creation time so the previous trial's terminating pod cannot trip the guard.
- **Full logs to stdout** in both scripts (vLLM container logs after the rollout, AIPerf
  logs after the job, previous-container logs on any restart), unchanged from study 12.

## Stack & versions

- **Akamas:** 3.7.1 (`akamas.lab.akamas.io`, workspace `default`), CLI in the `toolbox` pod
  (`kubectl -n akamas exec deploy/toolbox -c toolbox -- akamas ...`; login expires roughly
  daily). The resources below were created there on 2026-09-15.
- **Optimization packs:** vLLM **1.9.1** (installed, `akamas list optimization-pack`
  2026-09-15; local checkout `~/akamas/offline/optimization-packs/vllm`, branch
  `feature/agentic-scheduling-knobs-1.9.1`, used to re-check every domain), GPU **1.2.0**,
  Kubernetes **1.8.0-dev**. `stream_interval` (1.9.x) stays optional and commented out.
- **Workload under test:** `vllm/vllm-openai:v0.29.0`, model `openai/gpt-oss-20b` served
  as `gpt-oss-20b`, pinned flags `--attention-backend TRITON_ATTN --reasoning-parser
  openai_gptoss --max-model-len 32768 --no-enable-prefix-caching --enable-mfu-metrics`
  (rationale in the template's header). On the L4 the Marlin MXFP4 MoE kernel keeps the
  expert weights 4-bit (12.8 GiB checkpoint, 6.4 GiB per GPU at TP2, 3.2 GiB at TP4).
- **Cluster / hardware:** EKS `vllm-bench` (us-east-2), node group `llm-serving-l4`
  (1x `g6.12xlarge`, 4x L4 24 GB, no NVLink — TP traffic goes over PCIe), see
  `infra/README.md`.
- **Load generator:** NVIDIA AIPerf **0.11.0**, closed-loop concurrency sweep
  `150,179,213,253,302,359,428,509,606,722,860,1024` x 300 s (study 10's), streaming chat
  completions, ShareGPT replay from the model-specific cache file
  (`inputs-gpt-oss-20b.json`), `--extra-inputs reasoning_effort:low`,
  `--goodput "time_to_first_token:1500 inter_token_latency:300"`.
- **Telemetry:** Prometheus (`kube-prometheus-stack`, 5 s scrape on the `vllm` and
  `dcgm-exporter` ServiceMonitors), instance `Prometheus_13_GPT_OSS_20B_TP`, the same 108
  metrics as studies 9-12. `kv_cache_capacity_*` from the sidecar are approximate for this
  model (hybrid sliding-window KV layout).

## Parameters tuned

| Parameter | Domain (⊂ vLLM pack 1.9.1) | Baseline | Note |
|---|---|---|---|
| `vLLM.gpu_memory_utilization` | [0.80, 0.90] | **0.90** (rendered) | coupled to `max_num_seqs` by the warmup guard |
| `vLLM.max_num_seqs` | [16, 1024] | unrendered (vLLM default, 256 here) | ≤ 774 at 0.85, ≤ 480 at 0.90 |
| `vLLM.max_num_batched_tokens` | [256, 8192] | unrendered | must be ≥ `max_num_seqs` |
| `vLLM.kv_cache_dtype` | auto, fp8, fp8_e4m3 | unrendered (auto) | back to study 10's convention |
| `vLLM.performance_mode` | balanced, interactivity, throughput | unrendered | |
| `vLLM.optimization_level` | [0, 3] | unrendered (2) | |
| `vLLM.enforce_eager` | true, false | unrendered | |
| `vLLM.scheduling_policy` | fcfs, priority | unrendered | |
| `vLLM.async_scheduling` | true, false | unrendered | |
| `vLLM.max_cudagraph_capture_size` | [1, 1024] | unrendered | |
| `vLLM.block_size` | 16 … 128 (ordinal) | unrendered (16) | |
| **`vLLM.tensor_parallel_size`** | **[2, 4]**, != 3 | **2** (rendered) | the point of the study |
| **`vLLM.data_parallel_size`** | **[1, 2]**, TP x DP <= 4 | unrendered (1) | |
| `vLLM.enable_expert_parallel` | true, false | unrendered (false) | meaningful on every trial now (>= 2 ranks) |

Constraints (5): `max_num_batched_tokens >= max_num_seqs`; `TP x DP <= 4`; `TP != 3`;
`gpu_memory_utilization * 22.5 - 12.8 / TP >= 4` (weights shard + KV headroom, always true
here); `gpu_memory_utilization * 22.03 + max_num_seqs * 0.00375 <= 21.63` (sampler warmup).

## Prerequisites before this study can be started

1. **No other study may be running against `Deployment/vllm`** (`llm-serving`) and
   `Job/aiperf-benchmark` (`llm-benchmark`). Study 12 was still running when this study was
   scaffolded and had **finished by the evening of 2026-09-15 (13 experiments)**, so the
   deployment is free; check with `akamas list studies` before the smoke test and the start.
2. **Toolbox checkout and key:** commit this folder (plus `studies/README.md` and
   `ROADMAP.md`) and `git pull` in `/work/vllm-benchmark` on the toolbox — the workflow's
   three tasks and the smoke test reference files under that path. The SSH key is
   **already in place** at `/work/vllm-benchmark/studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/id_rsa`
   (copied from study 12's on 2026-09-15 so that `akamas create workflow` would accept the
   path; never in git — see `.gitignore`).
3. **Smoke test:** `bash studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/smoke_test.sh` on the
   toolbox — six configurations, must end with `0 failed configuration(s)`. The two
   `tp2-guard-*` rows are the empirical check of the warmup guard (see above).
4. **Model cache PVC** `vllm-model-cache-gptoss` and the other one-time objects are already
   applied for study 12 on this node (same manifests); re-apply is idempotent.
5. **Packs:** vLLM 1.9.1, GPU 1.2.0, Kubernetes 1.8.0-dev installed — already true on
   2026-09-15 (`akamas describe optimization-pack vLLM | grep -E 'version|active_gpus'`).

## Setup & run

All commands run in the `toolbox` pod (`kubectl -n akamas exec -it deploy/toolbox -c toolbox
-- bash`), from `/work/vllm-benchmark`, after `akamas login`.

```bash
# 0. one-time Kubernetes objects (idempotent; identical to study 12's)
kubectl apply -f studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/01-pvc-model-cache.yaml
kubectl apply -f studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/00-pvc.yaml
kubectl apply -f studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/06-hf-cache-pvc.yaml
kubectl apply -f studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/02-service.yaml
kubectl apply -f studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/04-kv-cache-exporter-configmap.yaml
bash studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/smoke_test.sh        # must end with "0 failed configuration(s)"

# 1. Akamas resources, typed form, dependency order (DONE on 2026-09-15 from a copy of
#    akamas/ in the toolbox's /tmp — re-run only after `akamas delete` of the same names)
akamas create system            studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/system.yaml
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/vllm.yaml               "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/container.yaml          "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/cluster.yaml            "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/gpu0.yaml               "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/gpu1.yaml               "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/gpu2.yaml               "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/gpu3.yaml               "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/container_loadtest.yaml "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create component         studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/cluster_loadtest.yaml   "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create telemetry-instance studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/telemetry/prometheus.yaml         "vLLM_Benchmark_13_GPT_OSS_20B_TP"
akamas create workflow          studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/13-GPT-OSS-20B-TP-Goodput-Per-GPU-Workflow.yaml
akamas create study             studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/13-GPT-OSS-20B-TP-Goodput-Per-GPU.yaml

#    or, bulk form (every file self-describes kind:/system:; same dependency order applies):
akamas create -f studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/

# 1b. MOOT as of 2026-09-16 — kept for the record only. The `kpis` block was added to this
#     manifest after the study had already run 16 experiments, and `akamas update study` has
#     no verb for it; re-creating the study would have discarded the experiments study 14
#     imports, so it was never applied. Study 14 carries its own 8-KPI block.
#     (Original note: delete and re-create the study only — the system, components,
#     telemetry instance and workflow are untouched.)
akamas delete study "13-GPT-OSS-20B-TP-Goodput-Per-GPU"
akamas create study  studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/13-GPT-OSS-20B-TP-Goodput-Per-GPU.yaml

# 2. check, then start (only once study 12 is finished — see Prerequisites)
akamas describe study "13-GPT-OSS-20B-TP-Goodput-Per-GPU"      # expect 14 parameters, 5 parameterConstraints, 5 KPIs, 2 steps
akamas start study "13-GPT-OSS-20B-TP-Goodput-Per-GPU"
akamas list experiment "13-GPT-OSS-20B-TP-Goodput-Per-GPU"
```

There is no `akamas update` verb for domains, constraints, KPIs, windowing or steps: to change
any of them, `akamas delete study "13-GPT-OSS-20B-TP-Goodput-Per-GPU"` and re-create it
(the system, components, telemetry instance and workflow can stay). Only the `goal` can be
edited in place on a running study.

## Known caveats

- **No optimization trial has run yet**, and the smoke test has not run either (study 12
  owned the deployment while this study was being scaffolded on 2026-09-15). In particular the warmup guard's TP-invariance
  assumption and the fail-fast's behaviour under a real crash loop are unverified until
  `tp2-guard-090`/`tp2-guard-085` and the first trials run.
- The ramp was calibrated on 1x A10G with Qwen2.5-7B; with TP >= 2 the first levels
  under-load every topology and only the top levels discriminate. If TP4 still shows no
  peak at 1024, the ceiling (and `max_num_seqs`'s domain) would have to grow together.
- `maxStdDev: 3e8` disables the stability filter (inherited limitation); the window is the
  6 samples around the prefill peak.
- Reasoning tokens inflate `decode_token_throughput` relative to a non-reasoning model:
  the goal measures engine tokens/s per GPU, not "useful answer tokens".
- Absolute scores are **not** comparable with studies 10/11/12 (different offered load
  and window per study, and a different baseline topology here); only the within-study
  ranking is meaningful.
- Shares `llm-serving/vllm`, `llm-benchmark/aiperf-benchmark`, the `hf-cache` and
  `aiperf-results` PVCs with studies 7-12 — one study at a time.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
