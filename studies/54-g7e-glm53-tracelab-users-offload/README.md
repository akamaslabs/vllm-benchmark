# 54 — GLM-5.3-Flash: coding users held per replica, CPU KV offload, TraceLab load (g7e dress rehearsal)

**Status:** TODO (built 2026-11-20; not created on Akamas yet)
**Dates:** – 

Design: `docs/superpowers/specs/2026-11-20-study-54-glm53-tracelab-users-offload-design.md`.
Plan: `docs/superpowers/plans/2026-11-20-study-54-glm53-tracelab-users-offload.md`.

## Objective

How many concurrent coding users one GLM-5.3-Flash replica holds within the SLO (client TTFT p95
<= 5 s and ITL p95 <= 50 ms, both over a trailing 150 s): with the target's arguments (baseline), with
CPU KV offload presets (native 128 / 256 / 512 GiB, LMCache 512 GiB), and with an Akamas-optimized
configuration (30 experiments). The study we will run at the target on their H200; here it runs end to
end on the g7e as a dress rehearsal. Follows studies 52 (coding sessions within the SLO) and 53 (CPU KV
offload for idle sessions: TTFT of a resumed session 22.9 s -> 1.73 s from 128 GiB).

**Score:** the users holding a lane (`aiperf.aiperf_active_sessions`) in the best valid 3-minute
window. Users enter linearly (one ramp per client) and never leave; a lane replays real TraceLab
sessions one after the other, developer pauses included. The run ends at the knee (watchdog), and the
CSV Akamas reads stops at the first sustained SLO breach, so no window after the knee can score.

## Stack & versions

- **Akamas version:** 4.1.3 (akamas-41, chart 1.9.4).
- **Optimization packs:** vLLM (component type `vLLM`; studies 52/53 ran on 1.15.0), AIPerf 1.0.0,
  GPU 1.4.0, Kubernetes 1.9.0 — confirmed with `akamas describe component-type` on 2026-11-20.
- **Workload under test:** `vllm/vllm-openai:v0.34.0`, checkpoint
  `canada-quant/GLM-5.3-Flash-W4A16-MTP` rev 5681f7c2 (the target's FP8 leaves no KV cache on 96 GB
  GPUs), TP4, the target's arguments (`--kv-cache-dtype fp8 --max-model-len 262144 --tool-call-parser
  glm47 --reasoning-parser glm47 --enable-auto-tool-choice`) plus `--language-model-only`; MTP 5,
  gmu 0.95 in the baseline. LMCache 0.5.8rc1 (installed at start) for the LMCache preset.
- **Cluster / hardware:** `vllm-bench` (EKS, us-east-2), node group `llm-serving-g7e-24xl-spot`
  (g7e.24xlarge spot: 4x RTX PRO 6000 Blackwell 96 GB, 96 vCPU, 1 TiB), see `infra/README.md`. Load
  node m8a.2xlarge (`node-role=m8a-2xlarge`).
- **Load generator:** AIPerf 0.13.0, two client pods (`k8s/02-aiperf-clients.yaml`), each replaying
  its own pool of 600 TraceLab sessions (`load/README.md`) with an 8,000-token shared system prompt;
  invocation in `k8s/client_run.sh` (`--custom-dataset-type tracelab --no-fixed-schedule
  --concurrency N --concurrency-ramp-duration R --inter-turn-delay-cap-seconds 1200 --extra-inputs
  ignore_eos:true`). Ramp, SLO and watchdog values in `k8s/workload.env`.
- **Data:** TraceLab (SyFI Lab, University of Washington, <https://tracelab.cs.washington.edu>),
  v0.0.2, CC BY 4.0.
- **Telemetry:** Prometheus (vLLM, DCGM, kubelet/cAdvisor, node exporter: the full catalog) and a CSV
  File instance for the client-side AIPerf metrics (30 s buckets, `k8s/aiperf_to_csv.py`).

## Parameters tuned

| Parameter | Domain | Baseline |
|---|---|---|
| `kv_offloading_size` (GiB) | [0, 512] | not set (no offload) |
| `gpu_memory_utilization` | [0.85, 0.97] | 0.95 |
| `max_num_batched_tokens` | [2048, 32768] | not set (vLLM 8192) |
| `max_num_seqs` | [16, 1024] | not set (vLLM 1024) |
| `spec_tokens` (MTP) | [0, 5] | 5 |
| `kv_offloading_backend` | presets only (native; lmcache in `LMCache 512`) | native |

Steps: `baseline`, `Native 128`, `Native 256`, `Native 512`, `LMCache 512`, `optimize` (30, AKAMAS).

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
