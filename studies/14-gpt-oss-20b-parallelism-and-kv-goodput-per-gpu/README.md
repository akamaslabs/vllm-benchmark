# 14-gpt-oss-20b-parallelism-and-kv-goodput-per-gpu

**Status:** RUNNING (created and started on the live instance on 2026-09-16, outside this
repo; this folder is the tracking record)
**Dates:** Created 2026-09-16, successor to `13-gpt-oss-20b-tp-goodput-per-gpu`

> **This study owns only its own manifest on the instance.** It reuses study 13's system
> (`vLLM_Benchmark_13_GPT_OSS_20B_TP`), its telemetry instance (`Prometheus_13_GPT_OSS_20B_TP`)
> and its workflow (`13-GPT-OSS-20B-TP-Goodput-Per-GPU-Workflow`) — that is the precondition
> for study 13's imported experiments to be comparable with the new ones. Their YAML is kept
> in `akamas/` as byte-identical copies (see `akamas/README.md`). As a consequence the
> running workflow executes `studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/*` on the
> toolbox: the `k8s/` and `infra/` folders here are a **byte-identical snapshot for the
> record**, not the files the study reads. Editing them changes nothing — see `k8s/README.md`.

## Objective

Identical goal, SLA and windowing to study 13 — deliberately, so the 16 imported
experiments sit on the same scale as the new ones:

```
maximize  (vLLM.prefill_token_throughput + vLLM.decode_token_throughput) / vLLM.active_gpus
subject to  vLLM.time_to_first_token_p95 <= 1500 ms,  vLLM.inter_token_latency_p95 <= 300 ms
```

Same model (`openai/gpt-oss-20b`), same 4x L4 node, same 12-level AIPerf ramp
(150 -> 1024 x 300 s) and same `stability` windowing on the prefill peak.

## What changes vs. study 13

| | study 13 | **this study** |
|---|---|---|
| `tensor_parallel_size` | [2, 4] | **[1, 4]** (pack allows [1, 16]) |
| `data_parallel_size` | [1, 2] | **[1, 4]** (pack allows [1, 8]) |
| initial design | Sobol bootstrap (10 points) | **10 explicit `preset` steps** + `numberOfInitExperiments: 0` |
| prior history | none | **baseline + experiments 2-16 of study 13 imported** |
| `parameterConstraints` | 5 | 5, unchanged |
| KPIs | 5 | 8 |
| `parametersSelection` | 14 | 14, same set |

Everything else — goal, constraints, windowing, workflow, benchmark, the other 12
parameter domains — is unchanged.

## Why: the kv_cache_dtype / enable_expert_parallel alias

Study 13's Sobol initial design produced a **perfect confound** between the KV cache dtype
and expert parallelism. Reconstructed from the study export on 2026-09-16, across all 15
optimizer experiments:

| | `kv_cache_dtype` | experiments |
|---|---|---|
| `enable_expert_parallel: true` | `auto` or `fp8_e4m3`, never `fp8` | 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 16 |
| `enable_expert_parallel: false` | `fp8`, always | 3, 7, 10, 13 |

`kv_cache_dtype == fp8` holds if and only if `enable_expert_parallel == false`: no
experiment can separate the two effects. Presets S3 (`fp8` + EP on) and S4 (`auto` + EP
off) are exactly the two missing cells; S10 breaks it a third time.

Sobol starts at `initial_offset = 0` and is deterministic, so the degeneracy is a property
of the first points of the sequence, not bad luck — hence the move to explicit presets.

## The 10 presets

Every preset pins **all 14 parameters**: a parameter left out of `values` is not rendered
and falls back to vLLM's own default, which would silently make two presets differ on
something nobody chose.

| # | purpose | key config |
|---|---|---|
| S1 | anchor — reproduce study 13's current best (exp. 16, 2345.3) | tp2 dp1, auto, EP on, seqs 1024, batched 7781, block 96, opt 3 |
| S2 | S1 + quantised KV, one variable changed | as S1 but `fp8_e4m3` |
| S3 | **alias breaker 1** — `fp8` with EP **on** | as S1 but `fp8`, EP on |
| S4 | **alias breaker 2** — `auto` with EP **off** | as S1 but EP off |
| S5 | main hypothesis: quantised KV + full batch + fast prefill | as S2 but block 32, opt 0 |
| S6 | the empty box: no tensor parallelism | tp1 dp1, rest as S5 |
| S7 | four independent replicas, no cross-GPU collectives | tp1 dp4, EP off, rest as S5 |
| S8 | lockstep diagnostic | tp2 dp2, EP off, rest as S5 |
| S9 | control for S8 | tp2 dp2, EP on, rest as S5 |
| S10 | space-filling coverage, Sobol index 1026 | tp4 dp1, `fp8` + EP on, seqs 325, batched 1003, gpu_mem 0.8447, block 128, eager |

At ~75 min per experiment the 10 presets cost ~12.5 h before the optimizer starts.

## Steps

```
baseline    from study 13, experiment 1                (imported, nothing re-run)
bootstrap   from study 13, experiments 2..16           (imported; 17 was ABORTED, no score)
S1..S10     10 preset steps                            (~12.5 h)
optimize    AKAMAS, 100 experiments / 20 failures, numberOfInitExperiments: 0
```

Experiment 3 of study 13 is `CONSTRAINTS_VIOLATED` but `FINISHED` **with a score**, so it
imports cleanly — the bootstrap crash study 9 hit (`NullPointerException` in
`refreshNormalization`, Airflow retrying forever) only affects experiments with no score.

`numberOfInitExperiments: 0` is required: any value > 0 makes the campaign service run its
own Sobol bootstrap on top of the presets, landing back on the degenerate points.

### Study 13's state at import time (2026-09-16)

| exp | score | status | | exp | score | status |
|---|---|---|---|---|---|---|
| 1 (baseline) | 1979.41 | FINISHED | | 9 | 1248.88 | FINISHED |
| 2 | 1891.40 | FINISHED | | 10 | 1302.11 | FINISHED |
| 3 | 1697.82 | CONSTRAINTS_VIOLATED | | 11 | 1541.93 | FINISHED |
| 4 | 2319.20 | FINISHED | | 12 | 2291.82 | FINISHED |
| 5 | 2152.65 | FINISHED | | 13 | 2088.39 | FINISHED |
| 6 | 1953.41 | FINISHED | | 14 | 2182.70 | FINISHED |
| 7 | 1917.82 | FINISHED | | 15 | 2331.85 | FINISHED |
| 8 | 2215.50 | FINISHED | | 16 | **2345.31** (+18.49%) | FINISHED |

Study 13 ran 2026-09-15 19:42 UTC onwards and was stopped after experiment 16 (experiment
17 aborted with the step, no score — excluded from the bootstrap list).

## KPIs — two of the eight are not collected

The manifest declares 8 KPIs. Checked against the live telemetry instance
(`Prometheus_13_GPT_OSS_20B_TP`, exported 2026-09-16):

| KPI | metric | collected? |
|---|---|---|
| Batch raggiunto | `vLLM.num_requests_running` | yes |
| Throughput totale | `vLLM.total_token_throughput` | **NO** |
| Utilizzo KV cache | `vLLM.kv_cache_usage_avg` | yes |
| Preemption (picco) | `vLLM.preemption_rate` (`aggregation: max`) | yes |
| TTFT P95 | `vLLM.time_to_first_token_p95` | yes |
| Tempo in coda P95 | `vLLM.request_queue_time_p95` | **NO** |
| ITL P95 | `vLLM.inter_token_latency_p95` | yes |
| Banda memoria GPU0 | `gpu0.gpu_dram_active` | yes |

`total_token_throughput` and `request_queue_time_p95` are declared by vLLM pack 1.9.1 and
bound to the `vLLM` component type, but the telemetry instance this study reuses has no
Prometheus query for either, so both KPIs will stay empty. Fixing it means adding the two
queries and re-creating the telemetry instance — there is no `akamas update` verb for it,
and doing so mid-study is not advisable. `total_token_throughput` is recoverable in
analysis as `prefill_token_throughput + decode_token_throughput` (the goal's own
numerator); the queue time is not recoverable from what is collected.

## Known caveats

- **`enable_expert_parallel` is unconstrained at one rank.** Study 12 carried
  `enable_expert_parallel == "false" || tensor_parallel_size > 1 || data_parallel_size > 1`
  because EP over a single rank is a no-op. That constraint is absent here (study 13 had
  dropped it as always-true under TP >= 2), and TP1/DP1 is now reachable: the optimizer can
  propose TP1/DP1 with EP on and EP off as two distinct configurations that are physically
  identical. Preset S6 is one of them (tp1 dp1, EP on — inert). Costs experiments, does not
  break anything.
- **The presets sit inside all 5 `parameterConstraints`**, checked numerically: the
  sampler-warmup guard at `gpu_memory_utilization` 0.80 / `max_num_seqs` 1024 gives
  17.624 + 3.84 = 21.46 <= 21.63, i.e. **0.17 GiB of margin**. S7 (tp1 dp4) is the closest
  analogue of study 12's experiment 12, which OOMed in `warmup_kernels` — at 0.90, not 0.80.
- **`k8s/` and `infra/` here are a snapshot**, not the live files (see the box at the top).
- The `results/` folder is empty until the study finishes and is exported.

## Stack & versions

Identical to study 13 — see
[`../13-gpt-oss-20b-tp-goodput-per-gpu/README.md`](../13-gpt-oss-20b-tp-goodput-per-gpu/README.md),
"Stack & versions". In short: `openai/gpt-oss-20b` on vLLM 0.29.0, 1x `g6.12xlarge`
(4x NVIDIA L4 24GB, no NVLink), AIPerf load generator, vLLM optimization pack **1.9.1**,
GPU pack **1.2.0**, Kubernetes pack **1.8.0-dev**.

## Setup & run

The study was created and started outside this repo on 2026-09-16. The system, components,
telemetry instance and workflow already existed (study 13 created them on 2026-09-15) and
are **not** re-created; their YAML sits in `akamas/` as byte-identical copies of study 13's
(added 2026-09-16 so the folder is complete — see `akamas/README.md`), which is also why
`akamas create -f akamas/` must not be used here. All commands run in the `toolbox` pod
(`kubectl -n akamas exec -it deploy/toolbox -c toolbox -- bash`, or `~/bin/toolbox-ssh`),
from `/work/vllm-benchmark`, after `akamas login`.

```bash
# study 13 must be stopped first — it owns Deployment/vllm on the node
akamas finish study "13-GPT-OSS-20B-TP-Goodput-Per-GPU"

# the only resource this study creates
akamas create study studies/14-gpt-oss-20b-parallelism-and-kv-goodput-per-gpu/akamas/14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU.yaml

akamas describe study "14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU"   # expect 14 parameters, 5 parameterConstraints, 8 KPIs, 13 steps
akamas start study "14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU"
akamas list experiment "14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU"  # within minutes: 1..16 imported, S1 running

# when it finishes
akamas export study "14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU" \
  studies/14-gpt-oss-20b-parallelism-and-kv-goodput-per-gpu/results/export.tar.gz
```

Only the `goal` can be updated in place on a running study. Changing
`parametersSelection`, `parameterConstraints`, `kpis`, `windowing` or `steps` requires
deleting and re-creating the study, which loses its history — so a change of plan means a
study 15 that bootstraps this one, exactly as this study bootstraps 13.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
