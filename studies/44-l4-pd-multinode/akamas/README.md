# akamas/ — study 44 tappa C (`44-C-L40S-A10G`)

Created 2026-10-29. Status on akamas-41: system, 3 components, telemetry instance, workflow and
study **created, not started** (study state CREATED).

## What it optimizes

Qwen/Qwen3-8B-FP8 served by vLLM 0.29.0 on two **fixed** single-GPU nodes in us-east-2b:

| GPU | Node group | Instance | $/h on demand (us-east-2) |
|---|---|---|---|
| NVIDIA L40S 48 GB | `llm-serving-l40s-pd` | g6e.2xlarge (8 vCPU, 64 GiB) | 2.24208 |
| NVIDIA A10G 24 GB | `llm-serving-a10g-pd` | g5.2xlarge (8 vCPU, 32 GiB) | 1.212 |

Akamas chooses the **topology** and the **vLLM flags of each GPU**:

- `pd_topology.pd_prefill_instances = 1`: P/D, prefill on the L40S, decode on the A10G,
  NixlConnector (pull), KV over TCP through host memory (`kv_buffer_device=cpu`);
- `pd_topology.pd_prefill_instances = 0`: both GPUs serve whole requests, the router sends
  each request to the one with fewer requests in flight (`PD_ROUTER_POLICY=least`).

**Goal:** maximize `pd_topology.total_token_throughput / 3.454`, router tokens/s per $/h (the
hardware is fixed, so the cost is a constant), with TTFT p95 ≤ 10 s and ITL p95 ≤ 75 ms, both p95
over 150 s at the router, read with `:max`. Windowing: study 27's stability window (6 × 30 s)
on the throughput maximum. Load: study 27's AIPerf 0.11.0 open-loop ramp, 0 → 2.4 req/s over
7200 s, gamma arrivals (smoothness 4), seed 18, 4096 in / 256 out, ended by the watchdog
(TTFT p95 > 20 s or ITL p95 > 225 ms for 120 s).

**Why:** tappa B measured P/D L40S → A10G at 5343.6 tokens/s (1.307 req/s), limited by the
A10G's ITL, i.e. 1547 tokens/s per $/h, against 2445 for one aggregated L4 on a g6.xlarge
(tappa A). Tappa C asks whether tuning, or aggregating both GPUs, closes that gap.

## Versions

- Akamas 4.1.0 (akamas-41), CLI in the toolbox.
- vLLM optimization pack **1.14.0** (installed on akamas-41; component types `vLLM`,
  `vLLM_PD_Topology`).
- vLLM `vllm/vllm-openai:v0.29.0`, model `Qwen/Qwen3-8B-FP8`.
- Telemetry: Prometheus (kube-prometheus-stack in `monitoring`), provider `Prometheus`,
  `duration: 30`.

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_44_L40S_A10G_PD` |
| `components/pd_topology.yaml` | `vLLM_PD_Topology`: the router's view (TTFT/ITL/E2E, delivered tokens), `active_gpus` |
| `components/vllm_l40s.yaml` | `vLLM`: the L40S pod (prefill in P/D, replica when aggregated) |
| `components/vllm_a10g.yaml` | `vLLM`: the A10G pod (decode in P/D, replica when aggregated); carries the shared KV dtype |
| `telemetry/prometheus.yaml` | `Prometheus_44_L40S_A10G_PD`: study 27's catalog filtered to the vLLM/PD metrics (53 queries, all accepted by the live Prometheus on 2026-10-29) |
| `44-C-L40S-A10G-Workflow.yaml` | Write config (FileConfigurator `../k8s/40-c-config_template.yaml`) → Apply config (`../k8s/apply_config_c.sh`) → RunTest (`../k8s/run_test_c.sh`) |
| `44-C-L40S-A10G.yaml` | the study |

No GPU (DCGM) or Kubernetes components: dcgm-exporter does not run on the L40S/A10G node
groups. Known gap: `kv_cache_capacity_*` / `kv_cache_used_*` of the two vLLM components stay
empty (the router republishes them under the tappa A/B served names); `kv_cache_usage_*` work.

## Parameters

| Parameter | Domain | Baseline |
|---|---|---|
| `pd_topology.pd_prefill_instances` | 0, 1 | 1 (P/D) |
| `vllm_l40s.gpu_memory_utilization` | [0.6, 0.92] | 0.9 |
| `vllm_l40s.max_num_seqs` | [8, 256] | 128 |
| `vllm_l40s.max_num_batched_tokens` | [2048, 16384] | 8192 |
| `vllm_l40s.linear_backend` | marlin, humming, triton | triton |
| `vllm_a10g.gpu_memory_utilization` | [0.8, 0.92] | 0.9 |
| `vllm_a10g.max_num_seqs` | [8, 256] | 128 |
| `vllm_a10g.max_num_batched_tokens` | [2048, 16384] | 8192 |
| `vllm_a10g.kv_cache_dtype` | auto, fp8 | fp8 |

Pinned in the templates: FLASHINFER on both GPUs (FlashAttention 2 rejects an fp8 KV cache,
and P and D must match for NIXL), A10G linear backend Marlin (no FP8 tensor cores on Ampere),
NixlConnector, `kv_buffer_device=cpu`, `max_model_len` 8192, no prefix caching.

Constraints: batched ≥ seqs on each GPU; activation headroom
`gmu × usable GiB + batched × 0.00016 ≤ usable GiB` (study 27's rule; L40S 44.39, A10G 22.03).

Steps (30 experiments): `baseline PD` (= tappa B2), `baseline PD repeat`, 3 presets
(`aggregated both B4 B5 flags`, `PD decode 32 seqs`, `aggregated both L40S triton`), then 25
AKAMAS experiments (`numberOfInitExperiments: 0`, `maxFailedExperiments: 8`). ~1 h per
experiment, ~$3.45/h: ~30 h and ~$105 for the whole study.

## Setup & run (from the toolbox of akamas-41)

The workflow runs the scripts from a copy of this study folder on the toolbox's persistent
volume, `/work/study44/44-l4-pd-multinode` (not from `/work/vllm-benchmark`: study 44 is not
pushed yet). Copy it from a laptop with:

```bash
cd studies
COPYFILE_DISABLE=1 tar --no-xattrs -czf - --exclude='results/*' 44-l4-pd-multinode \
  | kubectl -n akamas-41 exec -i deploy/toolbox -c toolbox -- sh -c 'mkdir -p /work/study44 && tar -xzf - -C /work/study44'
```

Create (done on 2026-10-29; repeat only after a delete):

```bash
cd /work/study44/44-l4-pd-multinode/akamas
akamas create system system.yaml
akamas create component components/pd_topology.yaml vLLM_Benchmark_44_L40S_A10G_PD
akamas create component components/vllm_l40s.yaml vLLM_Benchmark_44_L40S_A10G_PD
akamas create component components/vllm_a10g.yaml vLLM_Benchmark_44_L40S_A10G_PD
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_44_L40S_A10G_PD
akamas create workflow 44-C-L40S-A10G-Workflow.yaml
akamas create study 44-C-L40S-A10G.yaml
```

Bulk alternative (same dependency order applies): `akamas create -f /work/study44/44-l4-pd-multinode/akamas/`.

Before starting, from a laptop with `AWS_PROFILE=lab`:

```bash
studies/44-l4-pd-multinode/infra/nodegroups.sh up l40s
studies/44-l4-pd-multinode/infra/nodegroups.sh up a10g
```

Start, monitor, stop:

```bash
akamas start study 44-C-L40S-A10G
akamas list experiment 44-C-L40S-A10G
akamas finish study 44-C-L40S-A10G --workspace default
```

If `start` reports "Could not start the study" and the study stays RUNNING with no
experiment (late Airflow DAG on 4.1), `akamas finish study` and `echo y | akamas start study`.

After an edit: the study's goal can be changed with `akamas update study 44-C-L40S-A10G
44-C-L40S-A10G.yaml`; parameters, windowing or steps need `akamas delete study
44-C-L40S-A10G` and a new `akamas create study`. Workflow/system/component/telemetry edits need
the matching `akamas delete ...` + `akamas create ...`. Script/template edits (`../k8s/`) only
need the folder copied to the toolbox again.

**Not run yet:** the C-specific scripts (`apply_config_c.sh`, `run_test_c.sh`,
`41-c-serving_template.yaml`, the router's `least` policy) passed `bash -n`, a server-side
dry run of both topologies and an offline test of the routing logic, but no real experiment.
Watch the first experiment (`baseline PD`) to its end before leaving the study alone.
