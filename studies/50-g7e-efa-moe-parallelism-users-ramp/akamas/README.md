# akamas/ — study 50's Akamas resources

Created on 2026-11-10 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.0**
(`akamas-41`). Adapted from studies 48 (ramp and hold, goal) and 49 (per-GPU DCGM join, two
AIPerf clients). Goal and design: `../README.md` and
`docs/superpowers/specs/2026-11-09-study-50-g7e-efa-moe-parallelism-design.md`.

**Status:** see `../README.md`.

## Versions

- **Optimization packs on akamas-41** (checked with `akamas describe optimization-pack` on
  2026-11-10): vLLM **1.15.0**, GPU **1.4.0**, Kubernetes **1.9.0**.
- **Workload:** AWS Deep Learning Container
  `763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.29.0-gpu-py312-cu130-ubuntu24.04-ec2`
  (vLLM 0.29.0, NCCL 2.29.7, aws-ofi-nccl 1.18.0), `openai/gpt-oss-120b` served as
  `gpt-oss-120b`, `--max-model-len 16384`, prefix caching on.
- **Load generator:** AIPerf 0.11.0, multi-turn chat, two client pods on the `m8a-2xlarge` node.
- **Telemetry:** Prometheus (kube-prometheus-stack). Scrapes:
  - vLLM, via ServiceMonitor `vllm-moe-efa`, every 5 s;
  - dcgm-exporter (Helm release in `monitoring`, node role `llm-serving-g7e-8xl-efa` added);
  - the study's EFA exporter, via ServiceMonitor `efa-exporter-50`, every 5 s;
  - node-exporter and kube-state-metrics, already on the cluster.

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_50_G7E_EFA_MoE_Parallelism` |
| `components/vllm.yaml` | vLLM | `vllm`: both pods (`vllm-[ab]-0`); nine tuned parameters, the goal |
| `components/vllm_workload.yaml` | Kubernetes Workload | `vllm_workload`: `replicas` (1 = one vLLM, 2 = two independent ones) |
| `components/gpu_a.yaml`, `gpu_b.yaml` | GPU | one per physical GPU (`role: a` / `b`): DCGM and the node's EFA bytes |
| `components/node_a.yaml`, `node_b.yaml` | Kubernetes Node | the two GPU nodes: CPU, memory, ENA traffic, TCP retransmits |
| `components/container.yaml` | Kubernetes Container | the vLLM containers (CPU-throttle guard) |
| `components/container_loadtest.yaml` | Kubernetes Container | the two AIPerf clients |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_50_G7E_EFA_MoE_Parallelism`, 115 metrics |
| `50-G7E-EFA-MoE-Parallelism-Workflow.yaml` | workflow | Write config → Apply config → Ramp → Hold |
| `50-G7E-EFA-MoE-Parallelism.yaml` | study | `50-G7E-EFA-MoE-Parallelism` |

Telemetry notes (the header of `telemetry/prometheus.yaml` has the details):

- **vLLM percentiles are the worse pod's** (`max` over the pods): with two API servers a
  fleet-wide percentile would hide the slower node. Rates and counters are summed.
- **DCGM** series carry `hostname` = node name. Each GPU component joins them to
  `study50_node_role{role=~"$ROLE$"}`, exported by the EFA exporter, so `gpu_a` is the GPU of the
  node labelled `study50/role=a` whatever its name.
- **EFA bytes** have no metric in any installed pack. They are stored under the GPU pack's
  `gpu_nvlink_tx_bytes` / `gpu_nvlink_rx_bytes` (the GPU-to-GPU link of this study is EFA, not
  NVLink). TODO for whoever manages the packs: a network-fabric metric pair.
- **`vllm.active_gpus`** counts the Ready vLLM pods (1 on the one-GPU layout, 2 elsewhere).

## Study

- **Goal:** maximize `64 * vllm.request_success_rate - vllm.time_to_first_token_p95 / 1000`.
  - The first term is the users held: each user sends one turn every 64 s.
  - The second term is the TTFT p95 tie-break, in seconds.
  - The window is the `Hold` task, trimmed `[620s, 90s]`.
- **Constraints, server side:** TTFT p95 ≤ 600 ms and ITL p95 ≤ 200 ms. The 500 ms SLO itself is
  judged client-side by the `Ramp` task (`../k8s/run_ramp.sh`).
- **Parameters (10):** `vllm_workload.replicas`, `vllm.tensor_parallel_size`,
  `vllm.data_parallel_size`, `vllm.pipeline_parallel_size`, `vllm.enable_expert_parallel` pick
  the layout; `vllm.gpu_memory_utilization`, `vllm.max_num_seqs`,
  `vllm.max_num_batched_tokens`, `vllm.kv_cache_dtype`, `vllm.moe_backend` are the same in every
  step (0.90, 512, 8192, auto, auto).
- **Parameter constraints (3):** `max_num_batched_tokens >= max_num_seqs`; replicas × TP × DP ×
  PP ≤ 2 (two GPUs); expert parallelism only with TP or DP at 2.
- **Steps:** presets only, no optimize step.

  | Step | Layout (`../k8s/lib_layout.sh`) | replicas / TP / DP / PP / EP |
  |---|---|---|
  | `baseline` | single: one GPU, node b idle | 1 / 1 / 1 / 1 / false |
  | `Two replicas` | replicas: two independent vLLM | 2 / 1 / 1 / 1 / false |
  | `DP two` | dp2, external load balancing | 1 / 1 / 2 / 1 / false |
  | `DP two expert parallel` | dp2ep | 1 / 1 / 2 / 1 / true |
  | `TP two` | tp2 | 1 / 2 / 1 / 1 / false |
  | `TP two expert parallel` | tp2ep | 1 / 2 / 1 / 1 / true |
  | `PP two` | pp2 | 1 / 1 / 1 / 2 / false |

- **8 KPIs:** completed requests per s (users = 64 × this), tokens per second, TTFT p95, ITL p95,
  prompt share from the prefix cache, EFA bytes sent by node a, utilization of GPU a and GPU b.
  Every other metric stays as telemetry for analysis.
- **Not in Akamas:** the TCP controls (`NCCL_ENV="NCCL_NET=Socket"` on TP2 and DP2 + expert
  parallel) are run by hand after the presets with `../k8s/smoke.sh`. No pack parameter exists
  for the NCCL transport.

## Setup & run

From the toolbox (`kubectl -n akamas-41 exec -it deploy/toolbox -c toolbox -- bash`), with the
study copied to `/work/study50/50-g7e-efa-moe-parallelism-users-ramp`:

```
cd /work/study50/50-g7e-efa-moe-parallelism-users-ramp/akamas
SYS=vLLM_Benchmark_50_G7E_EFA_MoE_Parallelism
akamas create system system.yaml
for c in vllm vllm_workload gpu_a gpu_b node_a node_b container container_loadtest; do
  akamas create component components/$c.yaml $SYS
done
akamas create telemetry-instance telemetry/prometheus.yaml $SYS
akamas create workflow 50-G7E-EFA-MoE-Parallelism-Workflow.yaml
akamas create study 50-G7E-EFA-MoE-Parallelism.yaml

akamas start study "50-G7E-EFA-MoE-Parallelism"
```

Bulk alternative, in the same dependency order:

1. `akamas create -f system.yaml`
2. `akamas create -f components/`
3. `akamas create -f telemetry/`
4. `akamas create -f 50-G7E-EFA-MoE-Parallelism-Workflow.yaml`
5. `akamas create -f 50-G7E-EFA-MoE-Parallelism.yaml`

If the start fails with "Could not start the study", run `akamas finish study`, then
`echo y | akamas start study`.

Before the start: both GPU nodes Ready and labelled, the model on both nodes, and the smoke
checks of `../README.md` passed.
