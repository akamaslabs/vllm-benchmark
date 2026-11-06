# 47-g7-kv-cache-offload-ttft

**Status:** DONE (Akamas 4.1, akamas-41). Started 2026-11-05 11:53 UTC, restarted from scratch at 12:26 UTC after the ITL constraint fix and again at 12:59 UTC after a contaminated baseline (see below). Finished 2026-11-06 01:15 UTC: 30 of 30 experiments valid. Node group scaled to 0 at ~05:00 UTC. **Design:**
`docs/superpowers/specs/2026-11-05-study-47-kv-cache-offload-design.md`. **Plan:**
`docs/superpowers/plans/2026-11-05-study-47-kv-cache-offload.md`. **Spike:**
`docs/spikes/2026-11-05-kv-offload/README.md`.

## Objective

A chat assistant whose users hold conversations of many turns and come back after reading the
answer. Each turn re-sends the whole history. When the live conversations do not fit in GPU
memory, vLLM either recomputes evicted history (slow prefill, high time to first token) or, with
CPU KV offload (`--kv-offloading-size`), reloads it from RAM.

**Goal:** minimize **TTFT p95** of a fixed load in steady state. The levers against KV memory
pressure are tuned together:

- CPU offload size (0 = off);
- FP8 KV cache;
- `gpu_memory_utilization`;
- batching;
- block size.

Presets measure each lever alone. The workload was fixed from the scenario before any offload
run (design, "Fairness"), so offload is not favoured by construction.

**Constraints:**

- ITL p95 ≤ 120 ms (Akamas constraint, server-side histogram; first set at 50 ms, see "ITL
  constraint fix").
- Errors < 1 % and ≥ 95 % of the sent requests completed. `run_load.sh` fails the trial
  otherwise (the load is open-loop).

## Stack & versions

- **Hardware:** one g7.4xlarge, node group `llm-serving-g7-kv`:
  - NVIDIA RTX PRO 4500 Blackwell 32 GB;
  - 16 vCPU, 64 GiB (57 GiB allocatable);
  - 3.04 USD/h.

  EKS `vllm-bench` 1.36.5, us-east-2. AlwaysOn tag while the study runs. See `infra/README.md`.
- **Server:**
  - `vllm/vllm-openai:v0.29.0`, model `Qwen/Qwen3-8B-FP8` served as `qwen3-8b`;
  - `--max-model-len 16384`, prefix caching on;
  - 10 CPU, memory = offload + 20 GiB, `/dev/shm` = offload + 1 GiB.
- **Load:**
  - AIPerf 0.11.0, synthetic multi-turn chat, `--user-centric-rate 1 --num-users 64`, i.e. each
    user sends one turn every 64 s;
  - 10 turns per conversation, 600-token user turns, 200-token answers (`ignore_eos`), history
    re-sent;
  - 1200 s + 60 s grace per experiment.

  Runs in a pod on the GPU node (4 CPU / 6 GiB). Frozen in `k8s/workload.env`.
- **Akamas:** 4.1.0 (`akamas-41`), packs vLLM **1.15.0** (offload parameters and metrics, MR
  https://gitlab.com/akamas/optimization-packs/vllm/-/merge_requests/12) and Kubernetes
  **1.9.0**. Resources and setup in `akamas/README.md`.
- **Telemetry:** Prometheus (kube-prometheus-stack). The ServiceMonitor `vllm-kv-offload`
  scrapes vLLM every 5 s.

## Calibration (2026-11-05, before freezing the workload)

The rule was fixed in the design before any data:

- the baseline keeps up if ≥ 95 % of the sent requests complete and the client TTFT p95 over
  minutes 6–20 is < 10 s;
- otherwise drop to 56 users, then 48.

Baseline (vLLM defaults, no offload) at 64 users, 11:19–11:40 UTC:

| Metric | Value |
|---|---|
| Requests completed | 839 of 839 |
| Errors | 0 |
| Client TTFT p50 / p95, minutes 6–20 | 408 / 884 ms |
| Server TTFT p95, Prometheus | 0.94 s |
| ITL p95 | 33.1 ms |
| GPU prefix-cache hits | 2.5 % of prompt tokens |

The GPU KV cache holds 123,440 tokens; the live context is ~250k, so the cache thrashes. The rule
passed and the workload stays at 64 users.

vLLM 0.29.0's defaults on this GPU, read from the log, `vllm:cache_config_info` and
`EngineArgs.get_batch_defaults` for < 70 GB:

- `gpu_memory_utilization` 0.92;
- `max_num_seqs` 256;
- `max_num_batched_tokens` 2048;
- block size 16;
- KV dtype auto.

The baseline step renders them as explicit flags, except block size, because Akamas 4.1 refuses
a parameter that is in both `values` and `doNotRenderParameters`.

**Offload smoke (11:41–11:50):** offload 24 GiB + FP8 KV, 5 min of load:

- `/dev/shm` mmap 25.77 GB, vLLM ready in 168 s;
- GPU KV cache 241,184 tokens with FP8;
- TTFT p95 96 ms over the first 5 min, before the contexts grow;
- all four offload PromQL queries return values.

## ITL constraint fix (2026-11-05 12:26 UTC)

The first threshold, `vllm.inter_token_latency_p95 <= 50`, flagged the baseline itself
(experiment 1: CONSTRAINTS_VIOLATED, TTFT p95 872 ms server-side / 885 ms client-side, 840/840
requests). The 50 ms came from AIPerf's client-side figure, a p95 of per-request average ITL
(33 ms). The server-side vLLM histogram instead measures every gap between two tokens,
including decode steps stalled by the prefill of recomputed histories: ~96 ms on average over
the window, 30 s values up to 164 ms.

The user chose **120 ms** (~baseline + 25 %). Akamas 4.1 refuses a constraint change on a
RUNNING study, so the study was finished, updated (experiment 1 re-scored VALID) and started
again. The start reran the study from experiment 1 and wiped the history: ~33 min lost.

## Contaminated baseline and second restart (2026-11-05 12:59 UTC)

The 12:26 restart's baseline scored TTFT p95 58.8 s server-side (client-side 73 s; minute 0
already 300 ms instead of 68 ms). The AIPerf process of the experiment aborted by the 12:26
`finish` had survived its killed `kubectl exec` and kept sending its 1 req/s until ~12:45, so the
baseline got twice the load.

The 12:59 `finish` left another 20 AIPerf processes alive (controller, workers, record
processors). `run_load.sh` now kills every process in the AIPerf pod except PID 1 before each
run and fails if any is left. The study was finished and started from scratch again with a
clean pod.

## Parameters tuned

| Parameter | Domain | Note |
|---|---|---|
| `vllm.kv_offloading_size` | 0–28 GiB | 0 = no flag. 28 + 20 (vLLM) + 6 (AIPerf) ≤ 57 GiB allocatable |
| `vllm.kv_cache_dtype` | auto, fp8 | |
| `vllm.gpu_memory_utilization` | 0.80–0.94 | |
| `vllm.max_num_seqs` | 16–256 | |
| `vllm.max_num_batched_tokens` | 1024–16384 | ≥ `max_num_seqs` (parameterConstraint) |
| `vllm.block_size` | 16, 32, 64, 128 | |

Steps:

1. `baseline` (defaults).
2. `Offload only` (24 GiB).
3. `FP8 KV only`.
4. `Offload and FP8`.
5. `Baseline repeat` (drift check).
6. `optimize`: AKAMAS, 4 init, 25 experiments, 10 failures max.

Each experiment takes ~27 min (vLLM restart + 21 min of load).

## Results

Raw data in `results/`:

- `export.tar.gz`: Akamas export with `study.json` and the per-metric timeseries.
- `experiments.csv`: one row per experiment. It holds the 6 tuned parameters, the score, and the
  window averages of the main metrics. The `client_*` columns are AIPerf's own RESULT line over
  minutes 6–20 of the load, taken from the Load task log.

All 30 experiments are VALID. The ITL constraint never bound: the highest value was 99 ms (the
baseline), and every configuration with FP8 KV stayed at ~25 ms. Every run completed at least 838
of its ~840 requests. Two runs (experiments 9 and 11) had 1 error each, below the 1 % limit.

**Best: experiment 18. TTFT p95 227 ms server-side (222 ms client-side), −74.1 % vs the baseline
(877 ms).**

| Parameter | Baseline (vLLM default) | Best (exp 18) |
|---|---|---|
| `kv_offloading_size` | 0 (off) | 22 GiB |
| `kv_cache_dtype` | auto (bf16) | fp8 |
| `gpu_memory_utilization` | 0.92 | 0.94 |
| `max_num_seqs` | 256 | 16 |
| `max_num_batched_tokens` | 2048 | 16384 |
| `block_size` | 16 | 32 |

Presets, one lever at a time. Hit rates are percentages of prompt tokens from the vLLM counters
over the whole run; client TTFT is the AIPerf RESULT line.

| Exp | Configuration | TTFT p95 server | TTFT p95 / p50 client | GPU prefix hits | CPU offload hits |
|---|---|---|---|---|---|
| 1 | Baseline (defaults) | 877 ms | 882 / 408 ms | 2.5 % | – |
| 2 | Offload 24 GiB only | 864 ms (−1.5 %) | 887 / 411 ms | 2.5 % | 5 % |
| 3 | FP8 KV only | 738 ms (−15.9 %) | 834 / 369 ms | 18.7 % | – |
| 4 | Offload 24 GiB + FP8 | 234 ms (−73.4 %) | 229 / 187 ms | 18.8 % | 74 % |
| 5 | Baseline repeat | 872 ms (−0.6 %) | 886 / 405 ms | 2.5 % | – |

The baseline repeat is within 0.6 % of the baseline, so the run-to-run noise is ~1 %.

The optimize step (experiments 6–30) found 15 configurations within 227–242 ms, plus preset 4.
Every one combines FP8 KV with an offload of 21–28 GiB:

| Exp | Offload | KV | gmu | mns | mnbt | block | TTFT p95 server | client p95 |
|---|---|---|---|---|---|---|---|---|
| 18 | 22 | fp8 | 0.94 | 16 | 16384 | 32 | **226.9** | 222 |
| 21 | 28 | fp8 | 0.94 | 16 | 1024 | 32 | 227.2 | 223 |
| 15 | 23 | fp8 | 0.94 | 256 | 1024 | 32 | 228.2 | 228 |
| 4 | 24 | fp8 | 0.92 | 256 | 2048 | 16 | 233.6 | 229 |
| 12 | 22 | fp8 | 0.83 | 71 | 13151 | 32 | 235.1 | 225 |
| 29 | 23 | fp8 | 0.94 | 256 | 16384 | 64 | 236.2 | 231 |

Each lever alone:

- **Offload size: a step, not a slope.**
  - With FP8, 21–28 GiB all land at 227–242 ms.
  - 18 GiB gave 605 ms, 16 GiB 601 ms, 14 GiB 758 ms and 7 GiB 605 ms.
  - Without FP8 (bf16), no size up to 24 GiB helped (exps 2, 7, 16, 30: 791–874 ms). The 28 GiB
    cap was never tried with bf16.
  - In the runs that miss the band, the offload hit-rate metric reads ~0 % over the scoring window.
    In those runs the CPU buffer thrashes: experiment 2 moved 574 GB from GPU to CPU and reused 5 %
    of the prompt tokens.
- **FP8 KV** is required for offload to pay. It doubles both the GPU KV cache (123k → 241k
  tokens) and the number of tokens that fit in the same offload size.
- **gpu_memory_utilization, max_num_seqs, max_num_batched_tokens, block size:** second order,
  ≤ 6 % inside the good band.
  - The top 3 share gmu 0.94 and block 32.
  - Blocks 64 and 128 in otherwise similar configurations gave 236–240 ms.
  - `max_num_seqs` 16 vs 256 made no consistent difference: about 7 requests run at once under
    this load.
  - The optimizer's gain over the preset (234 → 227 ms, −3 %) is a few times the run-to-run noise
    but small in absolute terms.

**Telemetry caveat.** Akamas's `prefix_cache_hit_rate` and `cpu_offload_prefix_hit_rate` are
averages of 5-s ratios over the scoring window. They do not match the whole-run counter ratios
above: for example, 0 % vs 2.5 % GPU hits on the baseline, and 67 % vs 74 % offload hits on
experiment 4. Use the counters from the Load task log for hit rates.

## Conclusions

- **CPU KV offload is worth it only when all the live conversations fit in GPU + CPU KV.** Here the
  live context was ~250k tokens:
  - offload alone (bf16, 24 GiB, ~175k tokens in RAM): −1.5 %;
  - FP8 alone (241k tokens on GPU, just under the live context): −16 %;
  - both: −74 %, with 74 % of the prompt tokens reloaded from RAM instead of recomputed.

  Below the band, the LRU buffer evicts exactly the conversations that come back next. It then
  costs PCIe traffic and RAM for nothing. This generalizes as a sizing rule, not a number:
  estimate the live context (users × average history), and size offload so that the GPU KV
  cache plus the RAM buffer holds it, in the KV dtype you run.
- **Tune the KV memory first and the batching flags second.** On this stack, gmu, batching and
  block size moved TTFT by a few percent once offload + FP8 fit the context.
- **The preset was nearly as good as the optimizer.** A 5-preset design, one lever at a time, found
  97 % of the gain in the first 2 hours. The optimizer added the gmu 0.94 / block 32 refinement.
  With hindsight, the 25-experiment optimize step could have been ~10.
- **Specific to this stack:**
  - Qwen3-8B-FP8 on one 32 GB RTX PRO 4500, vLLM 0.29.0 native offload backend;
  - 64 users, 600/200-token turns, 10 turns;
  - 57 GiB of allocatable RAM, so 28 GiB was the offload cap.

  A bigger model, a longer context, or the `lmcache` backend can move the band.
- **Next:**
  - repeat with load growing past the band (more users) to see how TTFT degrades once
    GPU + CPU no longer fits;
  - try the `lmcache` backend (in the pack, not tuned here);
  - try offload with long shared system prompts (RAG/agents), where reuse is across users and
    not only within a conversation.
