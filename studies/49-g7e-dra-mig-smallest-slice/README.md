# 49-g7e-dra-mig-smallest-slice

**Status:** TODO. Built locally on 2026-11-06. Not yet run: it waits for a g7e.4xlarge
capacity probe (needs the user's go) and the smoke checks below.

- **Design:** `docs/superpowers/specs/2026-11-06-study-49-g7e-mig-smallest-slice-design.md`
  (approved 2026-11-06).
- **Plan:** `docs/superpowers/plans/2026-11-06-study-49-g7e-mig-smallest-slice.md`.
- **Builds on:** study 46 (`studies/46-g7e-dra-mig-users-per-dollar/`).

## Objective

A customer serves a small model (~8B) on a whole H100. This lab case shows that **the same
users, within the same SLA, fit in a fraction of the GPU** once vLLM is tuned. The rest of the
GPU stays free for other vLLM workloads, each requested through Kubernetes DRA as a MIG slice.

- **Goal:** the smallest slice that serves **128 users** within **TTFT p95 ≤ 1.5 s and ITL p95
  ≤ 300 ms**. The slice is 1/4 (`1g.24gb`), 1/2 (`2g.48gb`) or the whole GPU, and the customer's
  vLLM is tuned with it.
- **The other slices are busy during every measurement.** Neighbour vLLMs serve 64 users per
  quarter of GPU. MIG does not isolate power, so a slice measured next to idle neighbours would
  look better than it is.
- **The whole GPU competes:** it is the baseline, the customer today. A slice wins only by
  meeting the SLA.
- **Reported:**
  - GPU left free;
  - users per fraction (`128 / fraction`);
  - the neighbours' latency.

The RTX PRO 6000 (AWS g7e) stands in for the customer's H100. Its MIG profiles differ from the
H100's.

## Stack & versions

- **Hardware:** one g7e.4xlarge, node group `llm-serving-g7e-4xl`.
  - NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB;
  - 16 vCPU, 128 GiB;
  - 4.00 USD/h.

  Load node: `m8a-2xlarge` (m8a.2xlarge). EKS `vllm-bench` 1.36.5, us-east-2. See
  `infra/README.md`.
- **MIG through DRA:** `dra-driver-nvidia-gpu` with DynamicMIG, as study 46. Every vLLM pod
  has its own ResourceClaim (`k8s/00-claims.yaml`).
- **Server:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as `qwen3-8b-dra`,
  `--max-model-len 4096`, prefix caching off.
  - Customer: the tuned flags.
  - Neighbours: `--gpu-memory-utilization 0.90` and the rest at vLLM's defaults.
  - CPU and memory split evenly between the vLLM pods, with requests equal to limits.
- **Load:** AIPerf 0.11.0, closed loop on ShareGPT, the 10k-conversation subset from study 46.
  - Customer: 128 users.
  - Neighbours: 64 users each on `1g.24gb`, 128 on `2g.48gb`.
  - 20 s warm-up, then 600 s measured, all clients at once.
- **Akamas:** 4.1.0 with packs GPU 1.4.0, vLLM 1.15.0 and Kubernetes 1.9.0. See
  `akamas/README.md`.
- **Telemetry:** Prometheus.
  - vLLM is scraped every 5 s.
  - **NVIDIA DCGM metrics** for the whole GPU (`gpu0`), the customer slice (`gpu_customer`)
    and the neighbours (`gpu_neighbours`). The slices are matched through the slice-roles
    exporter (`k8s/03-slice-roles.yaml`): DCGM has no pod label under DRA.
  - At the end, the DCGM series are pulled from Prometheus into `results/`.

## Parameters tuned

| Parameter | Domain |
|---|---|
| `gpu0.mig_profile` | none, 2g.48gb, 1g.24gb |
| `vllm.gpu_memory_utilization` | 0.80–0.94 |
| `vllm.max_num_seqs` | 16–1024 |
| `vllm.max_num_batched_tokens` | 1024–16384 (≥ `max_num_seqs`) |
| `vllm.kv_cache_dtype` | auto, fp8 |
| `vllm.performance_mode` | balanced, interactivity, throughput |
| `vllm.optimization_level` | 1–3 |
| `vllm.async_scheduling` | true, false |
| `vllm.max_cudagraph_capture_size` | 16–1024 |
| `vllm.block_size` | 16, 32, 64, 128 |
| `vllm.watermark` | 0–0.10 (study 46's guard against a dropped exponent) |
| `vllm.scheduler_reserve_full_isl` | true, false |
| `vllm.stream_interval` | 1–16 |

Steps:

1. Baseline: whole GPU, defaults.
2. Half GPU with the same defaults.
3. Quarter GPU with the same defaults.
4. Whole-GPU repeat.
5. Optimize: 15 experiments.

Cost estimate: ~19 experiments × ~30 min ≈ 10 h ≈ 40 USD of g7e, plus ~5 USD for the load node.

## Before starting: smoke checks

Run these on the cluster, after the node is up, with the user's go:

1. **GPU instance ids.** For each of `none`, `2g.48gb` and `1g.24gb`, `apply_config.sh` must
   print a GPU instance id per pod. `nvidia-smi -q` must show `GPU Instance ID` inside a MIG pod.
   The slice-roles exporter must serve the mapping.
2. **DCGM join.**
   - `gpu_customer.gpu_fb_total` must return ~24k, ~48k or ~98k MiB in the three layouts.
   - `gpu0` must return power and temperature.
   - Check whether whole-GPU framebuffer series exist next to the per-instance ones in MIG
     mode: if they do, `gpu0.gpu_fb_total` double counts. It is not used by the goal.
3. **One short Load** (`LOAD_SECONDS=120`) per layout. Every client must return a result, and
   `RESULT customer` / `RESULT neighbours` must be printed.
4. **dcgm-exporter** runs on the 4xlarge node, and is restarted after each re-layout.
   `apply_config.sh` waits for it, then checks the customer-slice framebuffer in Prometheus
   before the load (exit 7 if empty).
5. **Empty KPIs:** in the whole-GPU layout (baseline and repeat), "Neighbours TTFT P95" and
   `gpu_neighbours` are empty because there are no neighbours. Check that Akamas 4.1 still
   scores those experiments.

## Results

TODO (study-recap at the end).

## Conclusions

TODO.
