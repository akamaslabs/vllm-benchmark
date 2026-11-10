# 49-g7e-dra-mig-smallest-slice

**Status:** FINISHED on Akamas 4.1 (akamas-41, 2026-11-07 19:44 → 2026-11-08 03:10 UTC, 23
experiments, ~35 USD of g7e) **but invalid for its goal: the customer's tuned flags were never
applied** (every experiment ran the customer at vLLM's defaults; only the MIG layout changed). The
rendering bug is fixed and tested (`k8s/lib_render.sh`), the scripts were synced to the toolbox
on 2026-11-10 (render and apply tests green there too), smoke check 6 passed (below), and the
rerun was started on 2026-11-10 17:51 UTC as a new study, `49-G7E-MIG-Smallest-Slice-Rerun`
(`akamas/49-G7E-MIG-Smallest-Slice-Rerun.yaml`: same system, workflow, goal and steps), and
**stopped by hand at 19:15 UTC during experiment 5**: the user wanted study 50 to run that
night, not this rerun. Only the four default presets finished (they repeat the first run: whole
94.97, half 47.37, quarter 23.65 CONSTRAINTS_VIOLATED, whole repeat 94.97); no tuned experiment
completed, so **the open question (does tuning recover the quarter) is still open and the rerun
is still pending**. Both studies stay on Akamas. See Results for the first run.

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

Run on the cluster after the node is up (`k8s/smoke.sh` in the toolbox covers 1–4; 5 is read
from Akamas once the baseline is scored). **Done on 2026-11-07, all passed:**

| Layout | GPU instance ids (pod = DCGM) | Customer FB in Prometheus | 120 s load, customer 128 users (defaults) | Neighbours |
|---|---|---|---|---|
| `1g.24gb` | 3 / 4 5 6 | 24 192 MiB | TTFT p95 18 626 ms, ITL p95 106 ms, 518 req, 0 err: SLA missed | 3 × 64 users, TTFT p95 512–604 ms |
| `2g.48gb` | 1 / 2 | 48 512 MiB | TTFT p95 824 ms, ITL p95 64 ms, 1044 req: SLA met | 1 × 128 users, 952 ms |
| `none` | – | 97 249 MiB | TTFT p95 462 ms, ITL p95 34 ms, 1787 req: SLA met | – |

In MIG mode DCGM publishes only the per-instance series (no whole-GPU framebuffer), so
`gpu0.gpu_fb_total` does not double count. The first try failed on scheduling: study 46's
`vllm-0` (56 GiB, 7.7 CPU) and two AIPerf clients in `dra-mig` had come back with the node and
were scaled to 0.

The quarter slice with vLLM defaults (bf16 KV, ~9.6 GiB of KV ≈ 70k tokens) queues 128 in-flight
requests: the optimization has to recover it (FP8 KV ≈ 140k tokens, batching).

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
   `gpu_neighbours` are empty because there are no neighbours. **Passed:** Akamas 4.1 scored
   experiment 1 (baseline) VALID, 94.97 GiB (97 249 MiB / 1024 + the TTFT tie-break); the
   neighbour KPI simply has no value.
6. **The running customer pod carries the tuned flags** (added 2026-11-08, after the first run).
   `kubectl -n dra-mig-49 get pod vllm-customer-0 -o jsonpath='{.spec.containers[0].args}'` must
   list every flag printed by `apply_config.sh` ("customer vLLM flags: ..."), and
   `vllm:cache_config_info{pod="vllm-customer-0"}` in Prometheus must show the experiment's
   `gpu_memory_utilization`, `cache_dtype` and `block_size`. `apply_config.sh` now checks the
   rendered customer block before `kubectl apply` and the pod's args after the rollout (exit 8).
   **Not passed by the first run:** checks 1–5 looked at the layout, the DCGM join and the load,
   never at the pod's effective configuration, and the `RENDER_ONLY` tests stop before the
   manifest is rendered.
   **Passed on 2026-11-10 before the rerun:** whole GPU with `gpu_memory_utilization 0.85`,
   `kv_cache_dtype fp8`, `block_size 32`, `max_num_seqs 200`, `max_num_batched_tokens 4096`:
   `apply_config.sh` exit 0 (rollout 154 s), the pod's args list all five flags, and
   `vllm:cache_config_info{pod="vllm-customer-0"}` reports gmu 0.85 / fp8 / block 32 /
   31 291 GPU blocks (the first run always showed 0.92 / auto / 16).

## Results

Raw data in `results/`: `export.tar.gz` (Akamas export, md5 0c4cd0ea…), `experiments.csv` (per
experiment: step, state, score, the parameters **Akamas intended**, the Load-window aggregates,
the client-side RESULT lines, the KV capacity the customer actually ran with) and
`dcgm-prometheus.json.gz` (the raw DCGM series of the study window, per GPU instance, 15 s step,
pulled from Prometheus after the study).

### Data limitation: the customer ran vLLM's defaults in every experiment

`k8s/01-vllm_template.yaml`'s header comment mentioned `@TUNED_ARGS@` literally, and
`apply_config.sh` took the indent of the tuned flags from the *first* line matching the token:
that comment. Every flag was therefore rendered as a YAML comment, the placeholder line was
deleted, the "no token left" check passed, and the customer StatefulSet got only the fixed args
(`--model`, `--served-model-name`, `--port`, `--host`, `--no-enable-prefix-caching`,
`--max-model-len 4096`). Evidence: the StatefulSet's args after the last experiment, and
`vllm:cache_config_info{pod="vllm-customer-0"}` reporting `gpu_memory_utilization` 0.92,
`cache_dtype` auto and `block_size` 16 in all 23 experiments (the neighbours, whose
`--gpu-memory-utilization=0.90` is fixed in the template, report 0.9). `params.env` was rendered
correctly by Akamas every time, so the parameter columns in `experiments.csv` are what the
optimizer asked for, not what ran.

Consequences:

- **Only `gpu0.mig_profile` was really tuned.** The 13 vLLM parameters had no effect, so Akamas's
  "best configuration" (2g.48gb, fp8, `max_num_seqs` 1024, …) is one of 14 tied half-GPU runs
  separated by the TTFT tie-break only; it means nothing.
- **The study's question, whether tuning recovers the quarter slice, is unanswered.** What the
  data settles is the three layouts at vLLM's defaults, repeated 3 / 14 / 5 times.
- Customer and neighbours did not run the same settings: vLLM's own default (0.92 per
  `cache_config_info`) against the neighbours' 0.90.
- Fixed on 2026-11-08: `k8s/lib_render.sh` replaces only the line holding the token alone,
  `apply_config.sh` checks the rendered customer block before the apply and the pod's args after
  the rollout (smoke check 6), `k8s/test_render.sh` (12 cases) covers the real template, a comment
  mentioning the token and a pod at defaults. The toolbox still has the old scripts.

### What the 23 experiments measured (customer at vLLM's defaults, 128 users, 600 s)

| Layout | Exps | Score (GiB) | Goal state | Client TTFT p95 | Client ITL p95 | Server TTFT p95 | Server ITL p95 | Tokens/s | Running / waiting | KV usage | Customer KV capacity | Neighbours (client TTFT p95) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| whole GPU | 1, 4, 6 | 94.970 | VALID | 170–173 ms | 33.4–33.6 ms | 233 ms | 69–70 ms | 12 660–12 750 | 127 / 0 | 19 % | 542 768 tokens | – |
| `2g.48gb` (half) | 2, 5, 8, 9, 13–23 | 47.374–47.375 | VALID | 380–407 ms | 69.9–72.7 ms | 424–545 ms | 138–142 ms | 6 450–6 540 | 127 / 0 | 44–45 % | 232 544 tokens | 1 × 128 users: 379–413 ms |
| `1g.24gb` (quarter) | 3, 7, 10–12 | 23.646–23.648 | CONSTRAINTS_VIOLATED | 27.97–29.35 s | 106.8–108.1 ms | 21.6–23.6 s | 318–321 ms | 2 620–2 710 | 88–90 / 37–39 | 98 % | 73 392 tokens | 3 × 64 users: 506–533 ms |

- **Repeatability:** the three whole-GPU runs and the fourteen half-GPU runs are within 2 % of
  each other on every column; the five quarter runs within 5 %. No drift across the 7.5 h.
- **Half GPU meets the SLA with margin at defaults:** TTFT p95 ~440 ms server-side against 1500,
  ITL ~140 ms against 300, with a neighbour serving 128 users on the other half at the same
  latency. Score 47.37 GiB against the baseline's 94.97 (Akamas: −50.12 %).
- **Quarter GPU misses both constraints at defaults, by far:** TTFT p95 ~22 s (15× the limit) and
  ITL p95 ~320 ms (just above 300). The KV cache (73 392 tokens bf16 at gmu 0.92) holds ~89 of
  the 128 in-flight requests; the other ~38 queue (TTFT = queue time) and ~1 preemption/s
  recomputes evicted contexts. The three neighbours at 64 users each stay at ~510 ms.
- **Power:** the half-GPU layouts drew 570–580 W (customer + neighbour both busy) against
  529–537 W with the whole GPU serving the customer alone, and 526–534 W with the four quarters
  busy. A fully busy GPU draws more than a partly idle one, whatever the split.
- **GPU pack / DCGM under DRA:** the slice-roles join worked in every layout (customer
  framebuffer 97 249 / 48 512 / 24 192 MiB, SM active 0.84 / 0.91 / 0.88).
- **Akamas 4.1 ran 19 optimize experiments for `numberOfExperiments: 15`:** 15 VALID plus the 4
  CONSTRAINTS_VIOLATED quarter runs. Observed once; it reads as if only valid experiments count
  towards the budget. After experiment 12 the optimizer never proposed the quarter again.
- **CPU fairness guard:** `container_cpu_throttle_time` was 0 in every experiment.

## Conclusions

- **At vLLM's defaults, 128 closed-loop ShareGPT users of Qwen3-8B-FP8 fit in half an RTX PRO
  6000 (2g.48gb) within TTFT p95 1.5 s / ITL p95 300 ms, with 3.5× margin on TTFT and 2× on ITL,
  next to a busy neighbour.** The whole GPU is not needed for this load; the other half is free.
- **The quarter (1g.24gb) does not fit at defaults, and not marginally:** the KV cache holds
  ~89 of the 128 contexts, the rest queue for ~20 s, and ITL p95 already sits at the limit with
  1/4 of the SMs. Whether tuning recovers it is open: FP8 KV doubles the capacity (~147k tokens,
  enough for 128 × ~800 tokens) and would remove the queue, but ITL p95 at 128 concurrent
  sequences on a quarter of the compute may still bind. This is what the rerun has to resolve;
  the margin to the SLA cannot be quantified from this run.
- **A smoke check must read the effective configuration where it runs** (the pod's args,
  `vllm:cache_config_info`), not the rendered file: a correct `params.env`, a manifest with no
  token left and a healthy load hid 23 experiments at defaults. `RENDER_ONLY` tests that stop
  before the manifest is rendered do not cover the rendering.
- **Next (still pending; started and stopped on 2026-11-10, see Status):** the rerun runs as
  a new study with the same manifest (~19 experiments, ~7.5 h, ~35 USD of g7e plus the load
  node); the toolbox sync and smoke check 6 are done. `49-G7E-MIG-Smallest-Slice-Rerun` holds
  only four default presets, so the next attempt is a third study name (or a `finish` + `start`
  of the rerun study, which restarts it from zero). On 4.1 `finish` + `start` would have
  restarted this study from zero and deleted its 23 experiments on Akamas (they stay in
  `results/export.tar.gz`); the new study keeps this run visible next to the valid one. The
  whole-GPU and half-GPU numbers above stay as the reference for the defaults. Its results go
  in a "Rerun" section of this README.
