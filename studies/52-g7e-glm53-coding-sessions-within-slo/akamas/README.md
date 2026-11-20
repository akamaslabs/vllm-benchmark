# akamas/ — study 52's Akamas resources

Created on 2026-11-17 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.3**
(`akamas-41`), from study 51's resources adapted to one g7e.24xlarge spot with 4 GPUs. Goal and
method: `../README.md`. Design: `docs/superpowers/specs/2026-11-17-study-52-glm53-coding-sessions-design.md`.

**Status:** files written and checked offline (cross-references, domains against the packs, every
PromQL query run against the cluster's Prometheus: 152 queries, 0 errors). **Not created on
akamas-41 yet**: waiting for the user's go, after the smoke (`../k8s/smoke.sh`).

## Versions

- **Optimization packs on akamas-41** (checked with `akamas list optimization-pack` and
  `akamas describe optimization-pack` on 2026-11-17): vLLM **1.15.0**, GPU **1.4.0**, Kubernetes
  **1.9.0**. Every parameter and metric used here exists in those versions.
- **Workload:** `vllm/vllm-openai:v0.34.0`, `canada-quant/GLM-5.3-Flash-W4A16-MTP` (4-bit, revision
  5681f7c2; decided 2026-11-18, the FP8 checkpoint leaves no KV room at TP4 on 96 GB GPUs) served as
  `zai-org/GLM-5.3-Flash`, TP4, `--language-model-only`, `--max-model-len 262144`.
- **Load generator:** AIPerf 0.11.0, two clients (`aiperf-client-0/1`) on the `m8a-2xlarge` node.
- **Telemetry:** Prometheus (kube-prometheus-stack). Scrapes: vLLM via ServiceMonitor
  `vllm-glm53-coding` every 5 s (`../k8s/00-base.yaml`); dcgm-exporter (Helm revision 30 covers
  the node role `llm-serving-g7e-24xl-spot`); node-exporter and kube-state-metrics (node labels,
  incl. `label_node_role`), already on the cluster.

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_52_G7E_GLM53_Coding` |
| `components/vllm.yaml` | vLLM | `vllm`: the Deployment's pod; the 10 tuned parameters, the goal |
| `components/vllm_workload.yaml` | Kubernetes Workload | `vllm_workload`: the Deployment `vllm` |
| `components/gpu0.yaml` … `gpu3.yaml` | GPU | one per physical GPU (DCGM `gpu` index 0-3) |
| `components/node_gpu.yaml` | Kubernetes Node | the g7e.24xlarge spot |
| `components/node_load.yaml` | Kubernetes Node | the m8a-2xlarge load node |
| `components/container.yaml` | Kubernetes Container | the vLLM container |
| `components/container_loadtest.yaml` | Kubernetes Container | the two AIPerf clients |
| `components/cluster.yaml` | Kubernetes Cluster | the whole cluster |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_52_G7E_GLM53_Coding`, 152 metrics |
| `52-G7E-GLM53-Coding-Workflow.yaml` | workflow | Write config → Apply config → Ramp → Hold |
| `52-G7E-GLM53-Coding-Sessions.yaml` | study | `52-G7E-GLM53-Coding-Sessions` |

Telemetry notes (the header of `telemetry/prometheus.yaml` has the details):

- vLLM queries are scoped to `namespace="glm53-coding"`.
- GPU and Node queries join to the node role through kube-state-metrics' `kube_node_labels`
  (`label_join`, no `$1` next to the placeholders); no role exporter of our own.
- `k8s_node_network_in_bytes_details` / `_out_bytes_details` keep the `device` label (per interface).

## Study

- **Goal:** maximize `30 * vllm.request_success_rate` (sessions held: one turn per session every
  30 s). Window: the `Hold` task, trimmed `[520s, 90s]`.
- **Constraints, server side:** TTFT p95 <= 6000 ms and TPOT p95 <= 80 ms (per request, the
  server-side equivalent of AIPerf's ITL). The SLO itself (5 s,
  50 ms) is judged client side by the `Ramp` task (`../k8s/run_ramp.sh`).
- **Parameters (10):** `kv_offloading_size` 0–512, `gpu_memory_utilization` 0.85–0.97,
  `max_num_seqs` 8–128, `max_num_batched_tokens` 2048–16384, `spec_tokens` 0–5, `moe_backend`
  {auto, marlin, humming, triton}, `max_cudagraph_capture_size` 16–512, `async_scheduling`,
  `performance_mode`, `optimization_level` 1–3. No DCP: no MLA backend supports it on SM120.
- **Parameter constraint (1):** `max_num_batched_tokens >= max_num_seqs`.
- **Steps:** `baseline` (gmu 0.95, MTP 5 tokens, the rest not rendered = vLLM defaults: the target's
  arguments), presets `Offload 256`, `MTP three`, `MTP one`, `Baseline repeat`, then
  `optimize` (10 experiments, 3 initial).
- **8 KPIs:** completed requests per s, output tokens per second, TTFT p95, ITL p95, prompt from
  cache incl. offload, MTP acceptance, KV cache usage, GPU 0 utilization (ITL is read as TPOT p95).
- **After the smoke** (TODO in the study YAML): pin `max_num_seqs` in the baseline if v0.34.0's
  default exceeds the Mamba block count at TP4; set `spec_tokens` 0 and drop the MTP presets if MTP
  shows no acceptance; narrow `moe_backend` to the smoke's
  candidates.

## Setup & run

From the toolbox (`kubectl -n akamas-41 exec -it deploy/toolbox -c toolbox -- bash`), with the
study copied to `/work/study52/52-g7e-glm53-coding-sessions-within-slo` (tar, as studies 45-51):

```bash
cd /work/study52/52-g7e-glm53-coding-sessions-within-slo/akamas
akamas create system             system.yaml
for f in components/*.yaml; do akamas create component "$f" vLLM_Benchmark_52_G7E_GLM53_Coding; done
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_52_G7E_GLM53_Coding
akamas create workflow           52-G7E-GLM53-Coding-Workflow.yaml
akamas create study              52-G7E-GLM53-Coding-Sessions.yaml
akamas describe study 52-G7E-GLM53-Coding-Sessions

akamas start study 52-G7E-GLM53-Coding-Sessions
```

Bulk alternative (every file has `kind:` and `system:`), same dependency order:
`akamas create -f system.yaml`, then `akamas create -f components/`, `akamas create -f telemetry/`,
`akamas create -f 52-G7E-GLM53-Coding-Workflow.yaml`, `akamas create -f 52-G7E-GLM53-Coding-Sessions.yaml`.

Before `start`: GPU node up (`../infra/eks/nodegroups.sh up`), load node up (node group
`m8a-2xlarge`), smoke passed (`bash ../k8s/smoke.sh`), and on Akamas 4.1 constraints cannot be
edited while the study runs (finish + update + start restarts from zero).

No placeholders are left in these files.
