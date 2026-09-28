# 23-L4-PD-New-Constraint-CPU-KV

**Status:** RUNNING (created and started 2026-09-28)
**Dates:** created 2026-09-28

> Disaggregation-only optimization with a "no queue" goal. It **reuses study 20's system,
> telemetry instance and workflow** (the precedent of studies 14, 16 and 21), so the imported
> baseline sits on the same scale. `akamas/system.yaml`, `components/`, `telemetry/` and
> the workflow file are copies of study 20's, kept for the record. Do NOT create them
> again. The running workflow executes `studies/20-l4-pd-disaggregation-tuned/k8s/*` on the
> toolbox, so this folder has no `k8s/`.

## Why this study exists

**It is study 22 with the KV buffer pinned to host memory (`kv_buffer_device=cpu`).**
Everything else is identical: goal, constraints, the other 11 domains, load. Study 22's
first two random experiments drew `NixlPushConnector` + `kv_buffer_device=cuda`:

| Study 22 exp | Topology, KV | KV transfer | TTFT avg | Outcome |
|---|---|---|---|---|
| 3 | 1P1D, fp8 (303 MB) | 9.5 s, growing to 147 s | 10 s -> 152 s | no request completes after ~70 min, AIPerf timeout (80 min) -> ERROR |
| 4 | 2P1D, bf16 (606 MB) | ~60 s | ~71 s | same path, stopped |

This node has no GPU P2P, so a `cuda` buffer has to go GPU -> host -> GPU in small pieces.
Pull + `cuda` was already ~1.4 s per bf16 transfer in study 18. With `cpu` the same
transfers take 20-50 ms. Push + `cpu` was smoke-tested on 2026-09-28: 10 requests, 20-43 ms
transfers, TTFT ~1.48 s. On Akamas 3.7 the parameter space of a started study cannot be
edited, hence a new study. Study 22's P1D1 preset (exp. 2, **1382.61, +31%** over the
imported baseline) is inside this domain and is bootstrapped.

Background carried over from study 22:

Studies 20 and 21 left three findings:
1. **Study 20 never searched the disaggregated space.** Its KV-capacity constraints left
   0.07% of the feasible space with `pd_prefill_instances > 0`. All 51 random and optimizer
   experiments were aggregated (best 1630 tokens/s/GPU, 0P3D fp8).
2. **The prefill ran on the wrong FP8 kernel.** On the L4 (SM 8.9) vLLM 0.29.0 picks
   `MarlinFP8ScaledMMLinearKernel` for this block-quantized checkpoint. Marlin is
   weight-only W8A16: it does not use the FP8 tensor cores. With Marlin and Humming
   disabled on the prefill, the selection falls to `TritonFp8BlockScaledMMKernel` (W8A8 FP8).
   The A/B test of 2026-09-28 (2P1D, ~4000-token prompts):
   - a single prefill takes **1.116 s** instead of 1.755 s (-36%);
   - TTFT goes from 2.12 s to 1.47 s;
   - saturated prefill throughput is about 3340 instead of about 2150 tokens/s per prefill
     GPU (short test).
3. **Scoring by a TTFT SLA alone hides where a topology saturates.** The queue of the
   saturated role is the direct signal.

## Objective

```
maximize  pd_topology.total_token_throughput / pd_topology.active_gpus        (router)
subject to  pd_topology.time_to_first_token_p95 <= 10000 ms   (router)
            pd_topology.inter_token_latency_p95 <=    75 ms   (router)
            vllm_prefill.num_requests_waiting   <=     1      (sum over prefill instances)
            vllm_decode.num_requests_waiting    <=     1      (sum over decode instances)
```

Windowing: the most stable 6-sample window at the `pd_topology.total_token_throughput` peak.
The score is the throughput at the highest load where both roles still keep up.

Two things to keep in mind when reading the queue constraints:
- **Decode "waiting" includes requests whose KV is still being pulled.** NIXL's slow
  transfers (300 ms to 1 s, 10-30% of them on this node) add to it.
- **Prefill "waiting" only grows once the token budget per step is full.** vLLM runs the
  prompts that fit in `max_num_batched_tokens` in parallel, and those count as running.

## Stack & versions

As study 20 (`../20-l4-pd-disaggregation-tuned/README.md`):
- **Akamas:** 3.7.x.
- **Optimization pack:** vLLM pack 1.11.0.
- **Model and runtime:** `Qwen/Qwen3-8B-FP8` on `vllm/vllm-openai:v0.29.0`.
- **Node:** 1x g6.12xlarge (4x L4, PCIe, no P2P).
- **Load:** AIPerf 0.11.0, 4096 in / 256 out, 6 levels x 600 s.
- **Telemetry:** Prometheus.

One difference from every earlier study:
- **Prefill instances get
  `VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel`.**
  It comes from the serving template, `prefill.env` in `pd-config`, since commit b29754b.
- **Decode instances keep Marlin.**
- **The imported baseline (aggregated, decode-only processes) ran on Marlin.** An aggregated
  run with the Triton kernel has not been measured yet.

## Search space and constraints

| Parameter | Domain |
|---|---|
| `pd_prefill_instances` / `pd_decode_instances` | [1, 3] / [1, 3], with P + D <= 4 |
| `pd_kv_connector` | NixlConnector, NixlPushConnector |
| `pd_kv_buffer_device` | cpu (pinned: see above) |
| prefill / decode `gpu_memory_utilization` | [0.8, 0.92] |
| prefill / decode `max_num_seqs` | [8, 512] |
| prefill / decode `max_num_batched_tokens` | [512, 16384] |
| prefill / decode `kv_cache_dtype` | auto, fp8 (the same on both roles) |

- **Constraints:** P + D <= 4; batched tokens >= seqs on each role; the same KV dtype on
  both roles.
- **Dropped on purpose:** study 20's KV-capacity caps and its pin on the connector.

## Steps

1. `baseline from study 20`: imported from study 20's experiment 1 (0P2D, vLLM defaults,
   1056.02). Not re-run.
2. `bootstrap study 22 P1D1`: study 22's experiment 2 (1P1D; 128 seqs and 8192 batched
   tokens on both roles; bf16 KV), 1382.61. Not re-run.
3. `random`: 9 RANDOM experiments.
4. `optimize`: 100 AKAMAS experiments, 20 failures max.

About 65 min per experiment.

## Setup & run

From the toolbox: `kubectl -n akamas exec -it deploy/toolbox -c toolbox -- bash`, then
`cd /work/vllm-benchmark/studies/23-l4-pd-new-constraint-cpu-kv/akamas`. The system, the
components, the telemetry instance and the workflow already exist (study 20). Only the
study is created:

```bash
akamas create study 23-L4-PD-New-Constraint-CPU-KV.yaml
akamas start study "23-L4-PD-New-Constraint-CPU-KV"
```

After the run, export straight away (Prometheus keeps 10 days):
`akamas export study "23-L4-PD-New-Constraint-CPU-KV" studies/23-l4-pd-new-constraint-cpu-kv/results/export.tar.gz`.

## Placeholders

None.

## Results

_Not run yet._
