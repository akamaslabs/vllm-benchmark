# akamas/ — resources for study 16

**Created:** 2026-09-18

> **This study owns only its own manifest.** It reuses study 15's system
> (`vLLM_Benchmark_15_Qwen3_30B_A3B`), telemetry instance (`Prometheus_15_Qwen3_30B_A3B`)
> and workflow (`15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU-Workflow`) — that is the
> precondition for study 15's imported experiments to sit on the same scale, exactly as
> study 14 did with study 13. `system.yaml`, `components/`, `telemetry/` and the workflow
> file here are **byte-identical copies kept for the record**: do NOT re-create them, they
> already exist on the instance. As a consequence the running workflow executes
> `studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/k8s/*` on the toolbox, so this
> folder's `k8s/` and `infra/` are snapshots too — editing them changes nothing.

Successor of study 15, which was stopped on 2026-09-18 after its S2 preset died with a
CUDA out-of-memory in the sampler warmup. Study 15's folder is left untouched as the
record of that first run; everything measured there that has a valid score is imported
here rather than re-run.

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

1. **baseline** — **imported** from study 15 experiment 1 via `from`, so the score scale is
   unchanged (607.187 tokens/s per GPU: TP4/DP1, `gpu_memory_utilization` 0.85, vLLM
   defaults). Not re-run: it would cost 75 minutes and shift every imported point.
2. **bootstrap** — study 15's other experiments that finished with a valid score.
   **Verify the list before creating**: the manifest ships `[2, 4]`, written while the AWS
   endpoint was unreachable from the workstation. Experiments 3 and 6 failed and must not
   be imported.
3. **13 presets** — S1 and S3 are not repeated (imported above); the rest run with
   `max_num_seqs` 256 instead of 768. Together with the imported points they still cover
   10 of the 11 topologies reachable on a 4-GPU node, both head-to-head TP-vs-PP pairs
   (S4 vs S11 on 2 GPUs, the imported S1 vs S12 on 4) and all four cells of the
   `kv_cache_dtype` × `enable_expert_parallel` table.
4. **optimize** — AKAMAS, 100 experiments, `maxFailedExperiments: 20`.

≈ 85 min per experiment → roughly 6.8 days of node time for the whole study (116 experiments).

## What changed vs study 15

| | study 15 | **this study** |
|---|---|---|
| warmup constraint | `gmu × 22.03 + seqs × 0.00283 <= 21.63` | **`gmu × 22.03 + seqs × 0.0006 <= 19.7`** |
| presets' `max_num_seqs` | 768 | **256** |
| baseline | run (TP4/DP1, gmu 0.85) | **imported** from 15 exp 1 |
| prior history | none | **bootstrap** of 15's valid experiments |
| presets | 15 | 13 (S1 and S3 imported) |

The constraint is recalibrated on two measured points instead of an assumed copy count,
and the ceiling drops from the full 22.03 GiB framebuffer to 19.7 = 22.03 minus the
~2.3 GiB of CUDA context and NCCL buffers that live outside `gpu_memory_utilization`'s
budget. The lever is `max_num_seqs` and not `gpu_memory_utilization` because the failing
allocation scales linearly with the first and not at all with the second, while lowering
the second would starve the 2-GPU layouts of KV cache. Full detail in the manifest header
and in ROADMAP.md section C.

## Three things to settle before starting

0. **Study 15's experiment 6 failed and its log was never read** (AWS endpoint unreachable
   from the workstation at the time). Read it first: if it is another sampler OOM the fix
   here covers it; if the 2-GPU layouts are failing on KV cache instead, S4/S5/S6/S8/S9/S11
   need `kv_cache_dtype` pinned to `fp8_e4m3` rather than a smaller batch.
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

System, components, telemetry instance and workflow **already exist** (study 15 created
them) — do not re-create them. Only the study manifest is new:

```bash
# verify which of study 15's experiments have a valid score, and fix the bootstrap list
akamas list experiments "15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU"

# stop study 15 if it is still running, then create and start this one
akamas stop study "15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU"
akamas create study /work/vllm-benchmark/studies/16-qwen3-30b-a3b-parallelism-goodput-per-gpu-rerun/akamas/16-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU.yaml
akamas start study "16-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU"
akamas describe study "16-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU"
```

On this environment the CLI lives in the toolbox pod, so each command is prefixed with
`kubectl -n akamas exec deploy/toolbox -c toolbox -- ` and paths are relative to
`/work/vllm-benchmark` (verified working on 2026-09-17 for `list`/`describe`).

Only `goal` can be edited on a running study (`akamas update study`); changing
`parametersSelection`, `windowing` or `steps` requires a new study.
