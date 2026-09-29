# 26-g7-4500-gpu-slice-right-sizing

**Status:** TODO — built locally 2026-09-29, not synced to the toolbox, not created in
Akamas (see "Before starting").
**Dates:** —

## Objective

How much of a GPU does a small model actually need? Study 25 asks whether splitting one
RTX PRO 4500 between replicas beats one replica on the whole GPU, always using every
piece. This study asks the right-sizing question instead: **one piece alone** — a MIG
`1g.16gb` slice or one MPS client, with the other half idle — against the same piece
with **its neighbour busy**, and against the whole GPU.

**Goal:** maximize goodput **per unit of GPU** — aggregate
`vllm.prefill_token_throughput + vllm.decode_token_throughput` divided by
`vllm.active_gpus`, the fraction of the physical GPU the replicas hold (1 for exclusive or
both pieces, 0.5 for one piece) — under TTFT p95 <= 1500 ms and ITL p95 <= 300 ms,
`stability` windowing. Same shape as study 21's per-GPU goal.

It is a **sweep, not an optimization**: a baseline and five presets, no optimizer step,
because the answer is a small table, not a search. Chosen with the user on 2026-09-29
over folding a replica-count parameter into study 25: with a per-unit goal the optimizer
would favour one slice alone for a reason that only holds while the other half is idle.

**Why it can go either way.** Phase 0 (study 25 README) measured one slice alone at 2378
tokens/s at 128 users = 4756 per GPU, **+4 %** over exclusive (4560): with the neighbour
idle the GPU drew ~135 W at a full 2400 MHz, while exclusive sat at its 165 W cap with the
SM clock throttled to ~1.8 GHz. With both slices busy each delivered ~2076 (**-9 %**). So
half a GPU is worth slightly more than half — but only if the other half is not used.

## Stack & versions

Identical to study 25 (`../25-g7-4500-gpu-sharing-goodput/README.md`, "Stack &
versions"): Akamas 3.7.x; GPU pack **1.3.0** (`sharing_mode`, not installed yet), vLLM
pack 1.12.0, Kubernetes pack (installed; this study also uses its `Kubernetes Workload`
type for `replicas`); `vllm/vllm-openai:v0.29.0` serving
`Qwen/Qwen3-4B-Instruct-2507-FP8` as `qwen3-4b`; node group `llm-serving-g7-4500`
(1x g7.4xlarge, RTX PRO 4500 Blackwell Server Edition 32 GB, 165 W, driver 595.91.07);
AIPerf 0.11.0 ShareGPT, 60 s warm-up, 12 levels `16..768` x 300 s; Prometheus, 117
metrics.

Same node, same namespace (`gpu-sharing`), same Kubernetes resource names and the same
GPU sharing layer as study 25 (`infra/`, idempotent) — **studies 25 and 26 must never run
at the same time.**

## Parameters and steps

| Step | `gpu0.sharing_mode` | `vllm_workload.replicas` | GPU fraction | What it measures |
|---|---|---|---|---|
| baseline | exclusive | 1 | 1 | whole GPU, one replica |
| MIG one slice alone | mig | 1 | 0.5 | a hard half, neighbour idle |
| MIG both slices busy | mig | 2 | 1 | a hard half, neighbour busy (per slice = goal / 2) |
| MPS one client alone | mps | 1 | 0.5 | a soft half (50 % SMs, half memory), neighbour idle |
| MPS both clients busy | mps | 2 | 1 | a soft half, neighbour busy |
| exclusive repeat | exclusive | 1 | 1 | drift check at the end |

vLLM parameters are pinned in every step to vLLM's defaults on this GPU
(`gpu_memory_utilization` 0.90 — halved by `apply_config.sh` under MPS —, `max_num_seqs`
256, `max_num_batched_tokens` 2048, `stream_interval` 1), exactly study 25's
head-to-head presets, so its MIG/MPS two-piece presets and this study's rows compare
like with like. `time_slicing` is excluded: a time-sliced "half" has no memory or compute
isolation, it is not a slice. One `parameterConstraint`: exclusive implies one replica.

Budget: 6 experiments x ~70-80 min ~= 7.5 h of node time, ~23 USD at 3.04 USD/h.

## Design

Everything is study 25's (`k8s/apply_config.sh`, `k8s/run_test_goodput.sh`,
`k8s/05-job.yaml`, telemetry), with two differences:

1. `REPLICAS` comes from `vllm_workload.replicas` (rendered into `params.env`) instead of
   following from the mode; `apply_config.sh` still waits for the node to advertise the
   mode's full number of pieces, then starts only `REPLICAS` of them, and rejects
   exclusive x2 and time_slicing x1.
2. `vllm.active_gpus` is redefined as the GPU fraction held by the vLLM pods,
   `sum(kube_pod_container_resource_requests{resource="nvidia_com_gpu", namespace=
   "gpu-sharing", pod=~"$POD$"}) / node allocatable nvidia.com/gpu` — 0.5 with one
   replica on one of two MIG slices, checked live on 2026-09-29.

Caveats to read the results with: the MPS client's cap is the device plugin's
100/replicas default active-thread percentage, so one MPS client alone still gets only
half the SMs (a real "half"), unlike exclusive; and under MIG an idle slice still leaves
its power budget to the busy one, which is exactly the effect this study isolates.

## Before starting

Same as study 25's "Before starting" (GPU pack 1.3.0 with an Administrator login;
dcgm-exporter scraping `llm-serving-g7-4500`, today pinned to the L4 node for study 24;
no overlap with another study on `system-m8a`; toolbox sync; `infra/eks/provision.sh`),
plus: study 25 must not be running.

Setup & run commands: `akamas/README.md`.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
