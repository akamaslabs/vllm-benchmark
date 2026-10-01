# 26-g7-4500-gpu-slice-right-sizing

**Status:** DONE — 7 presets, all FINISHED and VALID, 2026-09-30 09:01-18:00 UTC (~9 h,
~27 USD). Node group scaled to 0 afterwards (2026-10-01).
**Dates:** 2026-09-30

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

## Results

Raw data in `results/`: `export.tar.gz` (Akamas export), `trials.csv` (parameters, score,
scored window and its metrics per experiment), `analyze_levels.py` and its outputs
`levels.csv` (every load level of every experiment, rebuilt from Prometheus) and
`windows.csv` (the study's windowing emulated, within 0.3 % of Akamas' own scores where
compared), `aiperf-exp7-no-mig-repeat/` (AIPerf's per-level summaries, the only run whose
artifacts survive). Every trial collected every metric (`missingMetrics` empty, gpu0
36/36, cluster / cluster_loadtest 14/14).

### Akamas scores (scored window, 6 x 30 s ranked on total throughput)

| Exp. | Step | `mig_profile` | Replicas | `max_num_seqs` | Score (tok/s) | Per replica | Running | TTFT / ITL p95 (ms) | GPU °C / SM MHz |
|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline | none | 1 | 256 | 5926 | — | 127 | 201 / 48 | 87.5 / 1674 |
| 2 | MIG whole GPU seqs 256 | 2g.32gb | 1 | 256 | 5610 | — | 127 | 216 / 48 | 86.2 / 1799 |
| 3 | MIG whole GPU seqs 512 | 2g.32gb | 1 | 512 | 5588 | — | 126 | 219 / 48 | 87.8 / 1786 |
| 4 | MIG half GPU seqs 256 | 1g.16gb | 2 | 256 | 5962 | 2936 / 3026 | 191 | 244 / 72 | 79.2 / 1827 |
| 5 | MIG half GPU seqs 128 | 1g.16gb | 2 | 128 | 6018 | 2895 / 3123 | 191 | 245 / 72 | 78.3 / 1827 |
| 6 | MIG half GPU seqs 384 | 1g.16gb | 2 | 384 | 6037 | 3106 / 2931 | 190 | 357 / 72 | 75.7 / 1847 |
| 7 | no MIG repeat | none | 1 | 256 | **6202** | — | 127 | 170 / 48 | 76.0 / 1761 |

All at `gpu_memory_utilization` 0.90, `max_num_batched_tokens` 2048, `stream_interval` 1;
GPU at its 165 W cap in every scored window. No constraint was binding at any scored
window: the windowing picked 128 users (one replica) or 192 (two), well inside the SLA.

**Best configuration: no MIG — which is the baseline configuration.** Experiment 7 is the
baseline repeated, and its +4.7 % over experiment 1 is **thermal drift, not a result**:
the GPU ran at 76 °C instead of 87.5 °C, so the SM clock at 1761 instead of 1674 MHz at
the same 165 W (tokens/s per SM MHz at 128 users: 3.48 vs 3.54, the same within 2 % —
throughput followed the clock). The temperature fell through the day, so the step order and the thermal
state are confounded. Read naively against the baseline step the scores say "two halves
+1-2 %, whole-GPU MIG -5 %"; that is the temperature gradient. **Compare within thermal
pairs:**

| Pair (similar temperature) | MIG layout | vs no MIG |
|---|---|---|
| exp 1 (87.5 °C) <-> exp 2 / 3 (86-88 °C) | `2g.32gb` x1 | -5.3 % / -5.7 % |
| exp 7 (76 °C) <-> exp 4 / 5 / 6 (76-79 °C) | `1g.16gb` x2 | -3.9 % / -3.0 % / -2.7 % |

**MIG costs ~3-6 % of throughput, as one whole-GPU instance or as two halves**; the two
layouts are equivalent within that noise (per-clock 3.13 for `2g.32gb`, 3.19-3.22 for
the halves, levels 128 / 192). Measured, not explained: under MIG the GPU ran a
*higher* SM clock at the same 165 W (1764-1850 vs 1637-1753 MHz at the peak levels) but did ~10 % less work
per clock. The `2g.32gb` instance has the same KV cache as no MIG (159,024 vs 159,008
tokens), so it is not memory. Consistent with study 25 (MIG 2 x `1g.16gb` -4.6 % against
exclusive, re-scored).

**The two slices split the load evenly:** per-replica throughput within ±5 % of the two
replicas' mean at every level up to 320 users.

### Capacity per size (per load level, `results/levels.csv`)

Each AIPerf level is 300 s at a fixed number of closed-loop users (16..768, total across
replicas); the SLA is checked on the level's p95 from vLLM's histograms. "Users within
SLO" is the highest level that passed; the next level failed, so the true limit lies in
between.

| `mig_profile` / `max_num_seqs` | Peak within SLO (tok/s, users) | Per slice | Users within SLO | What breaks it next |
|---|---|---|---|---|
| none / 256 (exp 7, 77 °C) | 6093 @ 128 | — | **256** (fails at 320) | admission: 256 slots full, the queue takes TTFT p95 to 4.9 s; KV only 51 % used |
| none / 256 (exp 1, 90 °C) | 5794 @ 128 | — | 256 | same |
| 2g.32gb / 256 (exp 2) | 5589 @ 128 | — | 256 | same |
| 2g.32gb / 512 (exp 3) | 5527 @ 128 | — | **512** (fails at 640) | KV full (99 %), preemption; at 512 users ITL p95 199 ms, TTFT 652 ms, 4231 tok/s |
| 1g.16gb x2 / 256 (exp 4) | 5907 @ 192 | ~2950 @ 96 users | **320** = 160 per slice (fails at 384) | KV full (97 % at 320, 100 % at 384: 1313 preemptions, TTFT p95 2.3 s) |
| 1g.16gb x2 / 128 (exp 5) | 5873 @ 192 | ~2940 @ 96 users | 192 (256 at the threshold: TTFT p95 1674 ms) | admission: 128 slots per slice full at 256 users |
| 1g.16gb x2 / 384 (exp 6) | 5905 @ 192 | ~2950 @ 96 users | 320 = 160 per slice (fails at 384) | KV full, as with 256 |

- **Throughput halves linearly with the slice:** one `1g.16gb` slice sustains ~2900-3100
  tok/s within the SLO with its neighbour busy, 47-50 % of the whole GPU at the same
  temperature. In requests: the whole GPU served 18.1 req/s at 128 users (AIPerf, exp 7),
  so a slice is worth ~9 req/s of this ShareGPT mix.
- **Concurrent users do not:** a slice holds ~160 users within the SLO, the whole GPU 256
  (bounded by `max_num_seqs`) or ~512 (bounded by KV, exp 3). A slice has 56,016 tokens
  of KV, 35 % of the whole GPU's 159,008: each replica carries its own copy of the
  weights (~4.8 GiB) and its own runtime overhead (activations, CUDA graphs). When the KV cache is the limit, half a GPU holds about a third of the users.
- **`max_num_seqs` sets the user ceiling, not the throughput.** Whole GPU: 512 vs 256
  scored the same (5588 vs 5610) and peaked at the same level, but kept the SLO up to 512
  users instead of 256 — the extra users wait in the batch (ITL) instead of the queue
  (TTFT), at a lower throughput per user (4231 tok/s at 512 users vs 5527 at 128). Slice:
  256 and 384 behave identically because the KV cache binds first, at ~160 sequences;
  128 caps admission below that (192-256 users). In overload, 128 per slice is the one
  setting with no preemption (~6060 tok/s at 256-768 users, against 5540-5670 for 256
  / 384 at 512-768 users, with ~7000 preemptions per level) — but its users queue past the TTFT SLA.
- The Akamas goal (peak tokens/s at the best window) is blind to the user ceiling: it
  scored experiments 2 and 3 the same. The per-level table is where the right-sizing
  answer is.

## Conclusions

- **On this GPU and model, no MIG is the most efficient layout, and MIG's cost is small
  and layout-independent: ~3-6 %**, whether the GPU is one `2g.32gb` instance or two
  `1g.16gb` halves. Splitting into halves loses nothing beyond what MIG mode itself
  costs. Stack-specific: RTX PRO 4500 (165 W, power-capped in every configuration),
  Qwen3-4B FP8, vLLM 0.29.0.
- **Right-sizing answer for this GPU:** for up to ~3000 tok/s (~9 req/s, ~160 concurrent
  chat users at TTFT p95 <= 1.5 s / ITL p95 <= 300 ms) a `1g.16gb` slice is enough; above
  that, the whole GPU (~6000 tok/s, ~18 req/s, 256-512 users depending on
  `max_num_seqs`). The table is coarse because this GPU has only two MIG sizes.
- **Throughput scales linearly with the slice, concurrency does not**: KV per slice
  shrinks faster than the slice (each replica pays its own weights), so a
  concurrency-bound tenant needs a bigger slice than a throughput-bound one.
  Generalizable in shape; the ratio depends on model size vs slice memory.
- **`max_num_seqs` is the knob for how many users a slice holds within the SLO**, not
  for its tokens/s, and its useful range ends where the KV cache does (~160 here per
  slice). Untested follow-up: `max_num_seqs` ~160 per slice should keep both the SLO up to
  ~320 users and the no-preemption overload behaviour of 128.
- **Thermal drift moved the same configuration by ~5 %** (87.5 vs 76 °C, same 165 W) —
  as large as the MIG effect. On a power-capped cloud GPU, bracket a study with
  baseline repeats (as this one did), record `gpu_temp`, and compare configurations run
  at similar temperatures, not in step order.
- **Next:** the finer right-sizing table needs a GPU with more MIG sizes (H100/A100: 1g
  to 7g; RTX PRO 6000: 1g/2g/4g) — ROADMAP section D, study #4. On this GPU, a single
  follow-up would be `max_num_seqs` ~160 per slice.

## Data limitations

- Per-level numbers (`levels.csv`) are rebuilt from Prometheus, not from AIPerf: each
  experiment's Job deletes the previous run's artifacts, so AIPerf's own summaries exist
  only for experiment 7. TTFT / ITL p95 from vLLM's histogram buckets read high against
  AIPerf (exp 7 at 128 users: 182 vs 128 ms TTFT, 48 vs 29 ms ITL); pass/fail agrees
  wherever both exist, but experiment 5's failure at 256 users (1674 ms) is within that
  bucket error — its limit is "192-256 users".
- Level resolution: 192 / 256 / 320 / 384 / 512 users — "within SLO at 320" means the
  limit lies in [320, 384).
- Prometheus keeps ~10 days: re-run `results/analyze_levels.py` before ~2026-10-09 if
  needed; afterwards only the export (scored windows) remains.
- Thermal state was not controlled (cloud host); the pairing above is the mitigation.
