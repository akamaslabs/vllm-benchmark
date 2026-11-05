# akamas/ — study 47's Akamas resources

Created 2026-11-05 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.0**
(`akamas-41`). Goal, design and calibration are in `../README.md` and
`docs/superpowers/specs/2026-11-05-study-47-kv-cache-offload-design.md`.

## Versions

- **Optimization packs on akamas-41 (2026-11-05):** vLLM **1.15.0** (adds
  `kv_offloading_size`, `kv_offloading_backend` and the offload metrics; MR
  https://gitlab.com/akamas/optimization-packs/vllm/-/merge_requests/12), Kubernetes **1.9.0**.
- **Workload:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as `qwen3-8b`.
- **Load generator:** AIPerf 0.11.0 (`python:3.12-slim` pod on the GPU node).
- **Telemetry:** Prometheus (kube-prometheus-stack, in-cluster
  `kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090`), vLLM scraped every 5 s
  by the ServiceMonitor `vllm-kv-offload` (`../k8s/monitoring/servicemonitor.yaml`).

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_47_G7_KV_Offload` |
| `components/vllm.yaml` | component (vLLM) | `vllm`: every tuned parameter; goal, constraint, KPIs |
| `components/container.yaml` | component (Kubernetes Container) | vLLM container: CPU throttling guard, memory incl. the offload buffer |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_47_G7_KV_Offload` (67 metrics: study 46's vLLM and container queries + 4 offload metrics) |
| `47-G7-KV-Offload-Workflow.yaml` | workflow | Write config → Apply config → Load |
| `47-G7-KV-Offload-TTFT.yaml` | study | `47-G7-KV-Offload-TTFT` |

## Study

- **Goal:** minimize `vllm.time_to_first_token_p95` (server-side histogram) over the `Load` task,
  trimmed `[360s, 90s]` (steady state).
- **Constraint:** `vllm.inter_token_latency_p95 <= 50` ms. Errors ≥ 1 % or < 95 % of the sent
  requests completed fail the trial in `run_load.sh` (exit 6).
- **Parameters:**

  | Parameter | Domain |
  |---|---|
  | `vllm.kv_offloading_size` | 0–28 GiB (0 = off) |
  | `vllm.kv_cache_dtype` | auto, fp8 |
  | `vllm.gpu_memory_utilization` | 0.80–0.94 |
  | `vllm.max_num_seqs` | 16–256 |
  | `vllm.max_num_batched_tokens` | 1024–16384 |
  | `vllm.block_size` | 16, 32, 64, 128 |

  `parameterConstraints`: mnbt ≥ mns.
- **Steps:**
  1. `baseline`: vLLM defaults, no tuned flag. The values record what vLLM resolved by itself
     (gmu 0.92, mns 256, mnbt 2048, block 16, kv auto).
  2. Presets: `Offload only` (24 GiB), `FP8 KV only`, `Offload and FP8`, then `Baseline repeat`
     as the drift check.
  3. `optimize`: AKAMAS, 4 init, 25 experiments, 10 failures max.
- **No placeholders.** Hosts, keys and paths are the toolbox's, as in studies 45/46.

## Setup & run

From the toolbox (`kubectl -n akamas-41 exec -it deploy/toolbox -c toolbox -- bash`), with the
study copied to `/work/study47/47-g7-kv-cache-offload-ttft`:

```
cd /work/study47/47-g7-kv-cache-offload-ttft/akamas
akamas create system             system.yaml
akamas create component          components/vllm.yaml      vLLM_Benchmark_47_G7_KV_Offload
akamas create component          components/container.yaml vLLM_Benchmark_47_G7_KV_Offload
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_47_G7_KV_Offload
akamas create workflow           47-G7-KV-Offload-Workflow.yaml
akamas create study              47-G7-KV-Offload-TTFT.yaml

akamas start study "47-G7-KV-Offload-TTFT"
```

Bulk alternative, same dependency order (every file carries its `kind:`):
`akamas create -f system.yaml`, then `akamas create -f components/`,
`akamas create -f telemetry/`, `akamas create -f 47-G7-KV-Offload-Workflow.yaml`, and
`akamas create -f 47-G7-KV-Offload-TTFT.yaml`.

Before starting the study:

- bring the node up: `AWS_PROFILE=lab ../infra/eks/gpu-nodegroup.sh --up --always-on`;
- tear it down at the end: `--down`.

If the start hangs on "Could not start the study" (Akamas 4.1 DAG timeout), run
`akamas finish study` and then `yes y | akamas start study`.
