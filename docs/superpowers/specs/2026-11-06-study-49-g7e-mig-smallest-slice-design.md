# Study 49 — Smallest MIG slice that serves the same users (design)

**Date:** 2026-11-06 · **Status:** approved 2026-11-06 (user: "ok va bene"), build after study 48 and a g7e capacity probe · **Study folder (to
create):** `studies/49-g7e-dra-mig-smallest-slice/`

## Intent and the why

A customer runs a small model (~8B) on a whole H100. The GPU is far bigger than the model needs.
We want a lab case showing that the **same users, within the same SLA, fit in a fraction of the
GPU** once vLLM is tuned. The rest of the GPU is then free for other vLLM workloads, each
requested through Kubernetes DRA as a MIG slice.

The README sentence this study should let us write:

> *"128 chat users within TTFT p95 ≤ 1.5 s / ITL p95 ≤ 300 ms need X of an RTX PRO 6000 instead of
> the whole GPU, leaving Y free, with the other slices busy serving other tenants."*

Study 46 asked "how many users does each split serve in total". This study asks the customer's
question instead: **"what is the smallest slice that serves my load?"**

## Fairness: the split must win on the data, not by construction

The user wants the split to be the best configuration, because a split frees GPU for other
workloads. The goal below rewards exactly that, and nothing else:

- **Goal:** the smallest GPU fraction that serves the fixed customer load within the SLA.
- **The whole GPU competes.** It is the baseline: the customer's setup today.
  - If a slice serves the load, the slice wins because it uses less GPU.
  - If no slice can, the whole GPU wins, and the study says so. We need to know that before
    telling a customer.
- **The other slices are busy during every measurement.** MIG isolates memory and compute, not
  power (study 26). A slice measured next to idle neighbours would look better than it is in
  real use.

Study 46 suggests the split will win honestly at 128 users:

| Configuration | Users within SLA (study 46) | Fraction |
|---|---|---|
| `2g.48gb` (1/2 GPU) | 256 per replica | 1/2 |
| `1g.24gb` (1/4 GPU) | 64 per replica, poorly tuned (2 of 4 optimize runs crashed) | 1/4 |

So 1/2 should pass and 1/4 is the open question the optimizer answers.

## Decided (with the user, 2026-11-06)

- **Lab case on an RTX PRO 6000 Blackwell 96 GB (AWS g7e)**, standing in for the customer's
  H100. The README says so.
- **Fractions:** 1/4 (`1g.24gb`), 1/2 (`2g.48gb`), 1/1 (whole GPU). The MIG layout is created
  through DRA DynamicMIG, as in study 46.
- **Customer load: 128 users**, with study 46's SLA:
  - TTFT p95 ≤ 1500 ms;
  - ITL p95 ≤ 300 ms;
  - ≥ 95 % of requests completed.
- **vLLM tuned along with the slice:** study 46's parameter set (below).
- **Goal: smallest fraction that meets the SLA at 128 users** (my recommendation, accepted).
  The "users per fraction" view the colleague suggested is reported as a KPI in the README.

## My calls (the user can override)

- **Stack as in study 46:**
  - `Qwen/Qwen3-8B-FP8` on vLLM 0.29.0, `--max-model-len 4096`, prefix caching off;
  - AIPerf 0.11.0 closed-loop on ShareGPT, on the separate load node group `m8a-2xlarge`;
  - node group `llm-serving-g7e-4xl` (**g7e.4xlarge**, 16 vCPU, 4.00 USD/h). Study 46 moved
    off the 2xlarge after 14–16 % CPU throttling with four replicas. Settled after the builder
    check. If g7e.4xlarge is short, the study waits; the 2xlarge is not a fallback.
  - NVIDIA DRA driver with DynamicMIG;
  - per-replica client-side SLA check.

  Study 46's infra, apply script and AIPerf jobs are reused as far as possible.
- **Layout per experiment:**
  - `none`: one replica on the whole GPU, which serves the customer;
  - `2g.48gb`: two replicas, one customer and one neighbour;
  - `1g.24gb`: four replicas, one customer and three neighbours.
- **Load:**
  - **Customer replica:** 128 concurrent users, closed loop, fixed. No ramp: the question is
    whether this slice serves this load.
  - **Neighbours:** the same model with vLLM defaults, at **64 users per quarter of GPU**: 64 on
    a `1g.24gb`, 128 on a `2g.48gb`. 64 is what a `1g.24gb` sustained in study 46, so every
    neighbour is busy in proportion to its size (settled after the builder check).
  - **Neighbours must really be busy:** a trial fails if a neighbour vLLM restarts, or its
    AIPerf run errors ≥ 1 % or completes < 95 %. Measuring next to an idle neighbour is the
    first anti-goal (settled after the builder check).
- **Duration:** 20 s warm-up, then 10 min measured at 128 users; study 46's windowing pattern.
- **Constraints (Akamas, absolute, on the customer replica only):**
  - TTFT p95 ≤ 1500 ms;
  - ITL p95 ≤ 300 ms;
  - completed ≥ 95 %, enforced client-side (the trial fails otherwise).

  Neighbour SLAs are logged and reported, but they are not part of the goal.
- **Goal formula:** minimize `fraction + TTFT p95 (ms) / 1,000,000`.
  - `fraction` is 0.25, 0.5 or 1. The TTFT term (≤ 0.0015 under the SLA) only breaks ties
    inside one fraction.
  - Akamas goals read metrics, not parameters, so the builder must expose `fraction` as a
    measured metric. Two candidates:
    - the customer replica's MIG framebuffer from DCGM. This is unlikely to work: study 46
      found dcgm-exporter has no pod label under DRA.
    - **Chosen after the builder check:** a tiny static exporter in the study namespace. It is
      a pod serving `study49_gpu_fraction <value>` over HTTP, rewritten by the apply script
      for each experiment and scraped by a ServiceMonitor. No Pushgateway is needed.
- **Parameters** (study 46's set; domains re-checked against the installed vLLM 1.15.0 / GPU
  1.4.0 packs):

  | Parameter | Domain |
  |---|---|
  | `gpu0.mig_profile` | none, 2g.48gb, 1g.24gb |
  | `gpu_memory_utilization` | [0.80, 0.94] |
  | `max_num_seqs` | [16, 1024], with mnbt ≥ mns |
  | `max_num_batched_tokens` | [1024, 16384] |
  | `kv_cache_dtype` | auto, fp8 |
  | `performance_mode` | balanced, interactivity, throughput |
  | `optimization_level` | 1–3 |
  | `async_scheduling` | true, false |
  | `max_cudagraph_capture_size` | 16–1024 |
  | `block_size` | 16, 32, 64, 128 |
  | `watermark` | 0–0.10 (with study 46's tiny-real guard) |
  | `scheduler_reserve_full_isl` | true, false |
  | `stream_interval` | 1–16 |

  The tuned values apply only to the customer replica. The neighbours run vLLM defaults: they
  stand for other tenants we do not control.
- **Steps:**
  1. `baseline`: whole GPU, vLLM defaults (gmu 0.90 as in study 46). This is the customer
     today.
  2. Presets:
     - 1/2 with the same defaults;
     - 1/4 with the same defaults;
     - baseline repeat.

     A preset for "1/4 with study 46's best small-slice settings" was dropped: no `1g.24gb` run
     in study 46 reached 128 users. If a 1/4 configuration passes early, the optimize step
     still runs: it then improves TTFT margin at 1/4.
  3. `optimize`: **15 experiments**.
- **Cost:** ~25–30 min per experiment, i.e. MIG re-layout, vLLM start and 10 min of load.
  ~19 experiments ≈ 10 h ≈ 40 USD of g7e.4xlarge plus ~5 USD for the m8a load node.

## NVIDIA metrics for later analysis (user, 2026-11-06)

The study collects NVIDIA DCGM metrics so the results can be analysed afterwards, not only
scored. Most of them are telemetry metrics rather than KPIs, because a study holds at most 8
KPIs. They are visible in the Akamas UI and in the export.

- **Whole GPU, `gpu0` component (GPU pack 1.4.0):** study 46's DCGM catalog.
  - power usage and power limit;
  - GPU and memory temperature;
  - power and thermal violation time;
  - `GPU_UTIL`;
  - `PROF_SM_ACTIVE`, `PROF_SM_OCCUPANCY`, `PROF_PIPE_TENSOR_ACTIVE`, `PROF_PIPE_FP16_ACTIVE`,
    `PROF_DRAM_ACTIVE`;
  - framebuffer used and total;
  - PCIe replay.
- **Per MIG instance:** dcgm-exporter publishes one series per instance (`GPU_I_ID`,
  `GPU_I_PROFILE`), after an exporter restart following each re-layout, as study 46 does.
  - The study adds per-slice queries for the **customer slice** and for the **neighbour
    slices**: SM active, tensor active, DRAM active, framebuffer used.
  - Under DRA the series carry no pod label (study 46), so the apply script must make the
    customer replica identifiable. Either pin the customer to a known `GPU_I_ID`, or record the
    customer's MIG UUID and map it to the DCGM `UUID`/`GPU_I_ID` labels. The builder chooses
    and verifies it in a smoke run.
- **At the end:** the full DCGM series for the study window are pulled from Prometheus into
  `results/` (Prometheus keeps ~10 days), next to the Akamas export.
- **Prerequisite:** dcgm-exporter's node affinity includes the g7e node group. Study 46 did
  this; re-check it on the 4xlarge node group.

## Goals

1. Find the smallest fraction (1/4, 1/2, 1/1) that serves 128 users within the SLA with busy
   neighbours, and the vLLM settings that make it pass.
2. Report the free GPU left, and users per fraction (`128 / fraction`), as the efficiency
   figure.
3. Report how the neighbours fared, so "the rest is usable" is shown, not assumed.

## Non-goals

- Running the customer's real model or an H100.
- Mixed layouts (one `2g.48gb` + two `1g.24gb`), MPS or time-slicing.
- Tuning the neighbours.

## Anti-goals (a failure even if it runs)

- A slice that "wins" with idle neighbours.
- A goal that scores a slice ahead of the whole GPU when the slice misses the SLA.
- A result driven by the load node or by vLLM CPU starvation (study 46's checks stay).

## Prerequisites (block the study)

1. **g7e capacity in us-east-2.** It was short repeatedly in September and October, and study
   46's ODCR has expired. Probe before building.
2. The DRA driver and node group from study 46 still deploy on Kubernetes 1.36.
3. A measured `fraction` metric for the goal (see "Goal formula").

## Left to the builder

- How neighbours are deployed and loaded: extra StatefulSet replicas and AIPerf jobs, as in
  study 46.
- The KPI list (≤ 8): fraction, customer TTFT/ITL p95, customer tokens/s, neighbour TTFT p95,
  GPU power, customer-slice SM active, vLLM CPU throttle. Every other DCGM metric stays as
  telemetry (see "NVIDIA metrics").
- Exact windowing trim.
