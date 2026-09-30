# 26-g7-4500-gpu-slice-right-sizing

**Status:** RUNNING — redesigned 2026-09-30 as a MIG-only right-sizing sweep (see "Why MIG
only"); `apply_config.sh` tested in the toolbox on all three profiles, then created and
started on Akamas 2026-09-30 09:01 UTC (GPU pack 1.4.0).
**Dates:** 2026-09-30 –

## Objective

How much of a GPU does a small model need to hold its SLO? In enterprise GPU clusters
the unit of allocation is a **MIG partition**: time-slicing and MPS give no memory or
fault isolation between tenants and are not used there. So the parameter to explore is
the **MIG partition size**, and the answer is a capacity table: how much traffic each
size sustains within the SLO — from which the smallest partition for any target traffic
can be read.

**Parameter:** `gpu0.mig_profile` (GPU pack 1.4.0) on one RTX PRO 4500 Blackwell:

| `mig_profile` | What runs | Replicas |
|---|---|---|
| `none` | the whole GPU, MIG off — the non-MIG reference | 1 |
| `2g.32gb` | the whole GPU as one MIG instance (what MIG itself costs) | 1 |
| `1g.16gb` | two half-GPU instances, **both serving** | 2 |

This GPU offers only these two MIG sizes (NVIDIA MIG User Guide, Table 8: `1g.16gb` x2
or `2g.32gb` x1), so the right-sizing answer here is coarse — half or whole. A100/H100
(1g..7g, up to 7 instances) or RTX PRO 6000 (1g/2g/4g) would give a finer table; the
parameter and the scripts are written for that (the profile list comes from `nvidia-smi
mig -lgip`), only the study's category list would change.

**The GPU is always fully partitioned, one replica per MIG instance.** A slice is
therefore always measured with busy neighbours, as in a shared cluster. That matters:
MIG isolates compute and memory bandwidth but **not power**. Measured in study 25: one
`1g.16gb` slice alone ran unthrottled at 2400 MHz / ~135 W and served 2378 output
tokens/s at 128 users, while with both slices busy the GPU sat at its 165 W cap and each
slice served ~2076 (study 25 README, phase 0). The alone number is kept as the
optimistic bound; the study measures the realistic one.

**Goal:** maximize aggregate `vllm.prefill_token_throughput +
vllm.decode_token_throughput` under TTFT p95 <= 1500 ms and ITL p95 <= 300 ms. The
result to read is the **capacity per slice** = aggregate / number of instances, and the
concurrency at which the SLO still holds.

**Windowing on total throughput** (`vllm.total_token_throughput`, stability width 6,
`when: max`), not on prefill as in study 25: re-scoring study 25 showed the
prefill-ranked window sits at high concurrency where decode slows, under-scoring the
split modes by up to 15 % (study 25 README).

## Why MIG only, and why presets only

Decided 2026-09-30 after review by a colleague: time-slicing and MPS are not usable in an
enterprise context, MIG is — the question is right-sizing, not the sharing technique.
With only two MIG sizes on this GPU an optimizer adds little for its ~25 h, so the study
is a sweep of seven presets (~9 h). If one size turns out interesting, a follow-up can
run the optimizer on that size alone.

## Stack & versions

- **Akamas** 3.7.x; **GPU pack 1.4.0** (`mig_profile`, installed 2026-09-30), vLLM pack
  1.12.0, Kubernetes pack (installed build 1.8.0-dev).
- **Workload:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-4B-Instruct-2507-FP8` as
  `qwen3-4b`, `--max-model-len 4096`, prefix caching off; StatefulSet `vllm` in namespace
  `gpu-sharing`, one pod per MIG instance.
- **Hardware:** node group `llm-serving-g7-4500`, 1x g7.4xlarge, NVIDIA RTX PRO 4500
  Blackwell Server Edition 32 GB, 165 W, driver 595.91.07 (`infra/README.md`). Same node,
  namespace and GPU sharing layer as study 25 — the two studies never run together.
- **Load:** AIPerf 0.11.0, ShareGPT, 60 s warm-up at 64 users, 12 levels 16..768 x 300 s
  (`k8s/05-job.yaml`). With `1g.16gb` each slice gets half the users.
- **Telemetry:** Prometheus, 117 metrics (study 25's catalog); placeholder keys have no
  underscore (`$GPUMODEL$`, `$NODEROLE$`) because Akamas 3.7 does not substitute keys
  such as `gpu_model` / `node_role` (found on study 25, 2026-09-30).

## Steps (all presets, no optimizer)

| # | Step | `mig_profile` | Replicas | `max_num_seqs` | Why |
|---|---|---|---|---|---|
| 1 | baseline | none | 1 | 256 | whole GPU, no MIG |
| 2 | MIG whole GPU seqs 256 | 2g.32gb | 1 | 256 | the cost of MIG itself |
| 3 | MIG whole GPU seqs 512 | 2g.32gb | 1 | 512 | 159k KV tokens leave room for bigger batches |
| 4 | MIG half GPU seqs 256 | 1g.16gb | 2 | 256 | half GPU at defaults |
| 5 | MIG half GPU seqs 128 | 1g.16gb | 2 | 128 | 56k KV tokens per slice: less KV pressure, lower ITL |
| 6 | MIG half GPU seqs 384 | 1g.16gb | 2 | 384 | the slice pushed to its KV limit |
| 7 | no MIG repeat | none | 1 | 256 | drift check |

Other vLLM parameters: `gpu_memory_utilization` 0.90 (of the MIG instance under MIG),
`max_num_batched_tokens` 2048, `stream_interval` 1. Budget ~7 x 75 min ~= 9 h, ~27 USD.

## How an experiment is applied (`k8s/apply_config.sh`)

1. FileConfigurator renders `params.env` (five parameters); the script refuses an
   unsubstituted or empty value.
2. Scale vLLM to 0, delete its pods, stop any MPS daemon left by study 25.
3. Destroy every MIG instance. `none`: MIG off. A profile: MIG on, look up the profile ID
   and free count in `nvidia-smi mig -lgip`, create that many instances with compute
   instances (`mig -cgi <id>,<id>,... -C`), check the count.
4. Device-plugin config `exclusive` (none) or `mig` (migStrategy single), wait until the
   node advertises one `nvidia.com/gpu` per instance.
5. Restart dcgm-exporter on the node (it re-reads the MIG layout).
6. Render the StatefulSet with one replica per instance; `OrderedReady` start;
   crash-loop fail-fast; full logs to the Akamas task output.

`k8s/run_test_goodput.sh` fails the trial as soon as the Job fails, a replica restarts or
is replaced, or no request completes for 15 min (study 25's guards).

## Before starting

- Study 25 finished (done 2026-09-30 08:30 UTC) and the node left neutral.
- GPU pack 1.4.0 installed (done).
- dcgm-exporter covers `llm-serving-g7-4500` (done in study 25, helm revision 22).
- Toolbox sync, then `akamas create -f` (commands in `akamas/README.md`) and start (done
  2026-09-30 09:01 UTC; node and load-generator instances and ASGs tagged AlwaysOn).

## Running notes

- **Experiment 1 (baseline, 2026-09-30 09:01-10:15 UTC): 5926 tok/s, VALID, every metric
  collected** (`missingMetrics` empty; gpu0 36/36, cluster and cluster_loadtest 14/14 —
  the letters-only placeholder keys work).
- **Thermal drift against study 25:** the same configuration scored 6202 in study 25's
  baseline (2026-09-29 21:56 UTC). Same 165 W and same window load (127-128 running), but
  the GPU ran at 87.5 °C instead of 76.3 °C and the SM clock at 1673 instead of 1768 MHz
  (Prometheus, both scored windows): -5 % clock, -4.5 % throughput. Compare the MIG
  profiles against this study's own baseline and its `no MIG repeat` step, not study 25,
  and keep an eye on `gpu0.gpu_temp`.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
