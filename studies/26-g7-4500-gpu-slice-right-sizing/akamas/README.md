# akamas/ — 26-G7-4500-GPU-Slice-Right-Sizing

**Created:** 2026-09-29, with the `akamas-study-manager` plugin conventions, from study 25's
resources (same system shape, workflow and telemetry). **Not created on the Akamas server
yet** — validated offline only (see "Validation").

## What it optimizes

A sweep, not an optimization: goodput **per unit of GPU** —
`(vllm.prefill_token_throughput + vllm.decode_token_throughput) / vllm.active_gpus`,
`active_gpus` being the fraction of the RTX PRO 4500 the vLLM pods hold — under TTFT
p95 <= 1500 ms and ITL p95 <= 300 ms, `stability` windowing, for one piece of the GPU
alone, the same piece with its neighbour busy, and the whole GPU. Design: `../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | 3.7.x |
| Optimization pack **GPU** | **1.3.0** (`sharing_mode`), branch `feature/gpu-sharing-mode` of the `nvidia-gpu` repo, rebased onto the installed 1.2.0 — **not installed yet** |
| Optimization pack **vLLM** | 1.12.0 (installed) |
| Kubernetes pack | the installed build (`Kubernetes Container`, `Kubernetes Cluster`, `Kubernetes Workload`) |
| Serving | `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-4B-Instruct-2507-FP8` served as `qwen3-4b` |
| Load generator | AIPerf 0.11.0, ShareGPT replay |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter 4.6.0-4.8.3 |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_26_G7_4500_GPU_Slice` |
| `components/vllm.yaml` | `vllm` (vLLM) — both replicas; the only component goal/constraints/windowing read |
| `components/vllm_r0.yaml`, `vllm_r1.yaml` | per-replica views (`vllm_r1` empty with one replica) |
| `components/vllm_workload.yaml` | `vllm_workload` (Kubernetes Workload) — carries `replicas` (1-2), no telemetry |
| `components/gpu0.yaml` | `gpu0` (GPU) — carries `sharing_mode` |
| `components/cluster.yaml`, `container.yaml`, `cluster_loadtest.yaml`, `container_loadtest.yaml` | node and container views (Kubernetes pack) |
| `telemetry/prometheus.yaml` | `Prometheus_26_G7_4500_GPU_Slice`, study 25's 117 metrics with `active_gpus` redefined as the GPU fraction |
| `26-G7-4500-GPU-Slice-Right-Sizing-Workflow.yaml` | `Write config` -> `Apply config` (60 m) -> `RunTest` (105 m), scripts in `../k8s/` |
| `26-G7-4500-GPU-Slice-Right-Sizing.yaml` | study `26-G7-4500-GPU-Slice-Right-Sizing` |

**Steps:** `baseline` (exclusive x1), `MIG one slice alone`, `MIG both slices busy`,
`MPS one client alone`, `MPS both clients busy`, `exclusive repeat` — all at vLLM 0.90 /
256 / 2048 / 1, every parameter rendered; no optimize step. **parameterConstraints:**
exclusive implies one replica. **Placeholders left:** none (host `toolbox`, user `akamas`,
key `/home/akamas/.ssh/id_rsa`, as every study here).

## Validation

Offline, 2026-09-29, same checks as study 25: component types, parameter names and
domain subsets (`replicas` [1,2] inside the Workload type's [0,1024]; `sharing_mode`
categories inside GPU 1.3.0's), metrics referenced vs produced, `$KEY$` placeholders,
`${component.param}` tokens (6, all in `parametersSelection`), step names, 8 KPIs. Not yet
validated by `akamas create` on the server.

## Setup & run

Prerequisites: study 25's "Before starting" (GPU pack 1.3.0, dcgm-exporter on
`llm-serving-g7-4500`, `infra/eks/provision.sh`, toolbox sync), and study 25 not running.

From the toolbox, in `/work/vllm-benchmark`:

```bash
A=studies/26-g7-4500-gpu-slice-right-sizing/akamas
SYS=vLLM_Benchmark_26_G7_4500_GPU_Slice

akamas create system             $A/system.yaml
akamas create component          $A/components/vllm.yaml               $SYS
akamas create component          $A/components/vllm_r0.yaml            $SYS
akamas create component          $A/components/vllm_r1.yaml            $SYS
akamas create component          $A/components/vllm_workload.yaml      $SYS
akamas create component          $A/components/gpu0.yaml               $SYS
akamas create component          $A/components/cluster.yaml            $SYS
akamas create component          $A/components/container.yaml          $SYS
akamas create component          $A/components/container_loadtest.yaml $SYS
akamas create component          $A/components/cluster_loadtest.yaml   $SYS
akamas create telemetry-instance $A/telemetry/prometheus.yaml          $SYS
akamas create workflow           $A/26-G7-4500-GPU-Slice-Right-Sizing-Workflow.yaml
akamas create study              $A/26-G7-4500-GPU-Slice-Right-Sizing.yaml

akamas start study "26-G7-4500-GPU-Slice-Right-Sizing"
```

Bulk alternative (every file carries `kind:` and `system:`):

```bash
akamas create -f studies/26-g7-4500-gpu-slice-right-sizing/akamas/
akamas start study "26-G7-4500-GPU-Slice-Right-Sizing"
```

Monitoring: `akamas list experiment "26-G7-4500-GPU-Slice-Right-Sizing"`, `akamas log
--study "26-G7-4500-GPU-Slice-Right-Sizing" --log-level INFO` (never `-d`), and at the end
`akamas export study "26-G7-4500-GPU-Slice-Right-Sizing"
studies/26-g7-4500-gpu-slice-right-sizing/results/export.tar.gz`.
