# akamas/ — study 48's Akamas resources

Created on 2026-11-06 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.0**
(`akamas-41`). The files are copied from study 47 and modified. The goal, the design and the
ramp rules are in:

- `../README.md`;
- `docs/superpowers/specs/2026-11-06-study-48-kv-offload-users-within-slo-design.md`.

## Versions

- **Optimization packs on akamas-41:**
  - vLLM **1.15.0**, which has the offload parameters and metrics (MR
    https://gitlab.com/akamas/optimization-packs/vllm/-/merge_requests/12);
  - Kubernetes **1.9.0**.
- **Workload:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as `qwen3-8b`.
- **Load generator:** AIPerf 0.11.0, in a `python:3.12-slim` pod on the GPU node with 4 CPU and
  8 GiB.
- **Telemetry:** Prometheus (kube-prometheus-stack).
  - Address: `kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090`, in the
    cluster.
  - The ServiceMonitor `vllm-kv-offload` scrapes vLLM every 5 s. It is shared with study 47:
    same namespace, `kv-offload`.

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_48_G7_KV_Offload_Users` |
| `components/vllm.yaml` | component (vLLM) | `vllm`: every tuned parameter; goal, constraints, KPIs |
| `components/container.yaml` | component (Kubernetes Container) | the vLLM container |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_48_G7_KV_Offload_Users` (study 47's 67 metrics, unchanged) |
| `48-G7-KV-Offload-Users-Workflow.yaml` | workflow | Write config → Apply config → Ramp → Hold |
| `48-G7-KV-Offload-Users.yaml` | study | `48-G7-KV-Offload-Users` |

## Study

- **Goal:** maximize `64 * vllm.request_success_rate - vllm.time_to_first_token_p95 / 1000`
  over the `Hold` task, trimmed `[620s, 90s]`.
  - It reads as the users held by the Hold task.
  - The TTFT term, in seconds, only breaks ties between configurations at the same level.
- **SLO:** the Ramp judges client TTFT p95 ≤ 500 ms at each level. It also requires ≥ 95 % of
  the sent requests completed, errors < 1 %, and the expected number of requests sent.
- **Constraints, as a server-side safety net on the Hold window:**
  - `vllm.time_to_first_token_p95 <= 600`;
  - `vllm.inter_token_latency_p95 <= 120`.
- **Parameters:** same as study 47.

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
  1. `baseline`: the vLLM defaults, rendered explicitly. `block_size` is not rendered.
  2. Presets:
     - `Offload only` (24 GiB);
     - `FP8 KV only`;
     - `Offload and FP8`;
     - `Baseline repeat`.
  3. `optimize`: AKAMAS, 3 initial experiments, 10 experiments, 5 failures max.
- **No placeholders.** Hosts, keys and paths are the toolbox's, as in studies 45–47.

## Setup & run

Open a shell in the toolbox:

```
kubectl -n akamas-41 exec -it deploy/toolbox -c toolbox -- bash
```

Copy the study to `/work/study48/48-g7-kv-offload-users-within-slo`, then:

```
cd /work/study48/48-g7-kv-offload-users-within-slo/akamas
akamas create system             system.yaml
akamas create component          components/vllm.yaml      vLLM_Benchmark_48_G7_KV_Offload_Users
akamas create component          components/container.yaml vLLM_Benchmark_48_G7_KV_Offload_Users
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_48_G7_KV_Offload_Users
akamas create workflow           48-G7-KV-Offload-Users-Workflow.yaml
akamas create study              48-G7-KV-Offload-Users.yaml

akamas start study "48-G7-KV-Offload-Users"
```

Bulk alternative, in the same dependency order. Every file carries its `kind:`.

1. `akamas create -f system.yaml`
2. `akamas create -f components/`
3. `akamas create -f telemetry/`
4. `akamas create -f 48-G7-KV-Offload-Users-Workflow.yaml`
5. `akamas create -f 48-G7-KV-Offload-Users.yaml`

Node group:

- before the study: `AWS_PROFILE=lab ../infra/eks/gpu-nodegroup.sh --up --always-on`;
- at the end: `--down`.

If the start hangs on "Could not start the study" (Akamas 4.1 DAG timeout), run
`akamas finish study` and then `yes y | akamas start study`.
