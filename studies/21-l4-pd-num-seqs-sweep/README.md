# 21-L4-PD-Num-Seqs-Sweep

**Status:** TODO (created 2026-09-28, not started: study 20 is still running on the same node)
**Dates:** created 2026-09-28

> A controlled sweep, not an optimization. It **reuses study 20's system, telemetry instance
> and workflow** (the precedent of studies 14 and 16), so the imported baseline and study
> 20's own 1P1D result sit on the same scale. `akamas/system.yaml`, `components/`,
> `telemetry/` and the workflow file are copies of study 20's, kept for the record. Do NOT
> create them again. The running workflow executes
> `studies/20-l4-pd-disaggregation-tuned/k8s/*` on the toolbox, so this folder has no `k8s/`.

## Why this study exists

The question comes up when reading study 20's P1D1 experiment (exp. 3, 690.46 tokens/s/GPU).
The decode instance used only ~21% of its KV cache, so it looked as if a higher decode
`max_num_seqs` (24 there) would let it do more.

Study 20's own per-level data says the opposite: the cap was never reached.

| concurrency | decode running | decode waiting | decode KV | prefill waiting | req/s |
|---|---|---|---|---|---|
| 8 | 5.8 | 0.2 | 18% | 0 | 0.5 |
| 16 | 6.7 | 0.2 | 21% | 5.8 | 0.5 |
| 32 | 6.8 | 0.3 | 22% | **21.7** | 0.5 |

- **The prefill is the bottleneck.** The single prefill L4 runs at 100% utilization, at its
  72 W cap (71.6 W measured). It prefills a 4108-token prompt about every 2 s, so it
  delivers ~0.5 req/s.
- **The KV usage follows from that rate (Little's law).** 0.5 req/s x ~13 s of decode per
  request (256 tokens at ~50 ms) is ~6.5 requests in the decode. At ~4350 tokens each out of
  137,808 tokens of KV, that is ~21%.

This study measures it directly. It keeps everything of study 20's P1D1 fixed and steps only
the decode `max_num_seqs`. **Expected result: throughput, TTFT, ITL, decode running
requests and decode KV usage all flat across the six steps.**

## Objective

Same goal, SLA and windowing as study 20, so the numbers compare one to one:

```
maximize  (pd_topology.prefill_token_throughput + pd_topology.decode_token_throughput) / pd_topology.active_gpus
subject to  pd_topology.time_to_first_token_p95 <= 5000 ms   (router)
            pd_topology.inter_token_latency_p95 <=   75 ms   (router)
```

## Stack & versions

Identical to study 20 (see `../20-l4-pd-disaggregation-tuned/README.md`):
- **Akamas:** 3.7.x.
- **Optimization pack:** vLLM pack **1.11.0** (component type `vLLM_PD_Topology`).
- **Model and runtime:** `Qwen/Qwen3-8B-FP8` on `vllm/vllm-openai:v0.29.0`.
- **Node:** 1x g6.12xlarge (4x L4). The 1P1D presets use 2 GPUs.
- **KV transfer:** NixlConnector, pull, `kv_buffer_device=cpu`.
- **Load:** AIPerf 0.11.0, 4096 in / 256 out, 6 levels x 600 s (2, 4, 8, 16, 24, 32).
- **Telemetry:** Prometheus (kube-prometheus-stack), study 20's telemetry instance
  `Prometheus_20_L4_PD_Disaggregation`.

## Steps

| # | Step | Topology | Decode `max_num_seqs` | Note |
|---|---|---|---|---|
| 1 | `baseline from study 20` | 0P2D | vLLM default (256) | imported (study 20 exp. 1, 1056.02), not re-run |
| 2 | `decode seqs 16` | 1P1D | 16 | already above the ~7 working point |
| 3 | `decode seqs 24 study 20 rerun` | 1P1D | 24 | re-run of study 20's P1D1 (690.46): run-to-run noise |
| 4 | `decode seqs 32` | 1P1D | 32 | just above the KV-capacity cap (~31 full requests in fp8) |
| 5 | `decode seqs 64` | 1P1D | 64 | |
| 6 | `decode seqs 128` | 1P1D | 128 | pack default |
| 7 | `decode seqs 256 vLLM default` | 1P1D | 256 | vLLM default |

- **Fixed in every preset:** prefill 16 seqs / 4160 batched tokens (one whole prompt per
  step), fp8 KV on both roles, gmu 0.9 on both roles, decode batched tokens 2048,
  NixlConnector, `cpu` buffer.
- **No optimize step.** Six experiments of ~65 min each, so ~6.5 h of node time (~30 USD).
- **Study 20's KV-capacity `parameterConstraints` are dropped on purpose,** so the steps
  above 29 are allowed. The physical and NIXL constraints stay.

KPIs (8, the Akamas 3.7 limit) are chosen to show *why* nothing moves:
- throughput, TTFT p95 and ITL p95 at the router;
- decode running requests (peak) and decode KV usage;
- prefill waiting requests (peak) and prefill queue time p95;
- decode preemptions (peak).

## Reading the result

**If the prediction holds**, throughput stays at ~690 tokens/s/GPU, with the decode at
~7 running requests and ~21% KV, and prefill waiting up to ~22 at concurrency 32. The
steps from 64 upward also never preempt: their extra slots are never used.

**To make disaggregation faster on this node, the prefill side must grow, not the decode
cap:**
- **More prefill instances per decode:** study 20's P2D1 and a 3P1D.
- **Batching more prompts per prefill step:** `max_num_batched_tokens` above 4160. This is
  unlikely to help much, because a single prompt already saturates the L4.

## Setup & run

From the toolbox: `kubectl -n akamas exec -it deploy/toolbox -c toolbox -- bash`, then
`cd /work/vllm-benchmark/studies/21-l4-pd-num-seqs-sweep/akamas`. The system, the 11
components, the telemetry instance and the workflow **already exist** (study 20). Only the
study is created:

```bash
akamas create study 21-L4-PD-Num-Seqs-Sweep.yaml
# equivalently: akamas create -f 21-L4-PD-Num-Seqs-Sweep.yaml

# Only after study 20 has been stopped: both drive the same Deployment vllm-pd.
akamas stop study "20-L4-PD-Disaggregation-Tuned"
akamas start study "21-L4-PD-Num-Seqs-Sweep"
```

The study has no dependency to re-create, since its resources exist (study 20). If study
20's resources were ever deleted, create them from study 20's folder, in its README order.

After the run, export straight away (Prometheus keeps 10 days):
`akamas export study "21-L4-PD-Num-Seqs-Sweep" studies/21-l4-pd-num-seqs-sweep/results/export.tar.gz`.

## Placeholders

None. Study 20's UUID (`552141b9-1e05-49ca-a298-878c95bc16ec`) is in the baseline step.

## Results

_Not run yet._
