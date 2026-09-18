# akamas/ — resources for study 15

**Created:** 2026-09-17

Every resource here belongs to this study alone. Unlike study 14 — which reused study
13's system, telemetry instance and workflow so that its imported experiments stayed
comparable — this study shares nothing: the model is different, so study 13/14's
experiments are not on the same scale and there is nothing to import.

## What it optimizes

```
maximize  (vLLM.prefill_token_throughput + vLLM.decode_token_throughput) / vLLM.active_gpus
subject to  vLLM.time_to_first_token_p95 <= 1500 ms
            vLLM.inter_token_latency_p95 <=  300 ms
```

Goodput per GPU *actually holding weights*, so a configuration is not rewarded merely for
occupying all four L4s. `active_gpus` is DCGM-sourced and genuinely varies here (2 or 4):
the model does not fit one card.

## Versions

| | |
|---|---|
| Akamas | 3.7.x |
| vLLM optimization pack | **1.9.1** (`INSTALLED`, verified on the instance 2026-09-17) |
| GPU optimization pack | **1.2.0** (`INSTALLED`, verified 2026-09-17) |
| Kubernetes optimization pack | 1.8.0-dev (carried over from study 13/14 — re-verify) |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (29.03 GiB, FP8 e4m3, block 128x128) |
| Serving image | `vllm/vllm-openai:v0.29.0` |
| Load generator | AIPerf 0.11.0, ShareGPT replay, 12-level closed-loop ramp 24 → 1024 |
| Telemetry | Prometheus (`kube-prometheus-stack`), 30 s duration |

## Files

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_15_Qwen3_30B_A3B` |
| `components/vllm.yaml` | `vLLM` (type `vLLM`) — the tuned component |
| `components/gpu0..gpu3.yaml` | `gpu0`-`gpu3` (type `GPU`) — one per physical L4, `$GPU$` = device index |
| `components/cluster.yaml`, `cluster_loadtest.yaml` | `Kubernetes Cluster`, GPU node and load-test node |
| `components/container.yaml`, `container_loadtest.yaml` | `Kubernetes Container`, vLLM's pod and AIPerf's pod |
| `telemetry/prometheus.yaml` | telemetry instance `Prometheus_15_Qwen3_30B_A3B` |
| `15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU-Workflow.yaml` | workflow (3 tasks: render → apply+rollout → AIPerf) |
| `15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU.yaml` | the study manifest — 16 parameters, 7 constraints, 8 KPIs, 17 steps |

## Steps

1. **baseline** — TP4/DP1 + expert parallelism, `gpu_memory_utilization` 0.85, everything
   else left to vLLM's defaults via `doNotRenderParameters`. It cannot be the pack/vLLM
   default (TP1/DP1 does not fit this model) and it cannot be imported (different model).
2. **15 presets** — the initial design, replacing Sobol's, with `numberOfInitExperiments: 0`.
   They cover 10 of the 11 topologies reachable on a 4-GPU node (all but `TP1/DP2/PP2`),
   including both head-to-head TP-vs-PP pairs at equal `active_gpus` (S4 vs S11 on 2 GPUs,
   S1 vs S12 on 4), and all four cells of the `kv_cache_dtype` × `enable_expert_parallel`
   table — which is exactly what study 13's Sobol head failed to do (ROADMAP.md section C).
3. **optimize** — AKAMAS, 100 experiments, `maxFailedExperiments: 20`.

≈ 85 min per experiment → roughly 6.8 days of node time for the whole study (116 experiments).

## Three things to settle before starting

1. **`tp4-noep` in `../k8s/smoke_test.sh`.** The last `parameterConstraint` ("expert
   parallelism is required once the model is split over 4 ranks") is an inference from the
   checkpoint's `weight_block_size [128, 128]` against `moe_intermediate_size` 768 — not a
   measurement. If that smoke configuration starts fine, **delete the constraint**.
2. **`pp2` / `pp4` in `../k8s/smoke_test.sh`.** Pipeline parallelism on its own is
   documented upstream, but PP combined with DP on vLLM V1 was not verified from the
   source. If a PP+DP trial later fails, add `pipeline_parallel_size == 1 ||
   data_parallel_size == 1` to the manifest.
3. **The SSH key.** The workflow references
   `/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/id_rsa`
   on the toolbox host. It is **not** in this repo and must never be committed
   (`.gitignore`: `studies/*/akamas/id_rsa`); copy the toolbox's own key into place there.

There are no other placeholders: hostnames (`toolbox`), the Prometheus address, namespaces
and node roles are the real values this cluster already uses.

## Setup & run

Dependency order matters — Akamas resolves `system:`/`workflow:` references by name at
creation time. From the repo root:

```bash
# 1. resources, in order
akamas create system            studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/system.yaml
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/vllm.yaml             vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/gpu0.yaml             vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/gpu1.yaml             vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/gpu2.yaml             vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/gpu3.yaml             vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/cluster.yaml          vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/cluster_loadtest.yaml vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/container.yaml        vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create component         studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/components/container_loadtest.yaml vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create telemetry-instance studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/telemetry/prometheus.yaml       vLLM_Benchmark_15_Qwen3_30B_A3B
akamas create workflow          studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU-Workflow.yaml
akamas create study             studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU.yaml

# bulk alternative (every file self-describes its `kind:`; the same order still applies)
akamas create -f studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/

# 2. only after the k8s prerequisites and the smoke test (see ../k8s/README.md)
akamas start study "15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU"
akamas describe study "15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU"
```

On this environment the CLI lives in the toolbox pod, so each command is prefixed with
`kubectl -n akamas exec deploy/toolbox -c toolbox -- ` and paths are relative to
`/work/vllm-benchmark` (verified working on 2026-09-17 for `list`/`describe`).

Only `goal` can be edited on a running study (`akamas update study`); changing
`parametersSelection`, `windowing` or `steps` requires a new study.
