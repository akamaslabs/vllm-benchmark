# 48-g7-kv-offload-users-within-slo

**Status:** DONE (Akamas 4.1, akamas-41; 2026-11-06 → 2026-11-07, 15/15 experiments valid).

- **Design:** `docs/superpowers/specs/2026-11-06-study-48-kv-offload-users-within-slo-design.md`.
- **Plan:** `docs/superpowers/plans/2026-11-06-study-48-kv-offload-users-within-slo.md`.
- **Follows study 47:** `studies/47-g7-kv-cache-offload-ttft/README.md`.

## Objective

Study 47 fixed the load at 64 users and showed that CPU KV offload with FP8 KV cut TTFT p95 from
877 to 227 ms. At a fixed open-loop load every configuration serves the same tokens/s.

Study 48 asks how many **chat users** the same GPU serves within **client TTFT p95 ≤ 500 ms**.
The per-user load is study 47's. The configurations compared are:

- vLLM defaults;
- offload only;
- FP8 KV only;
- offload + FP8;
- a short optimization.

The 500 ms SLO sits between reloading a history from RAM (~230 ms p95 in study 47) and
recomputing it (600–900 ms). So a configuration's user ceiling is where its cache (GPU, or GPU +
RAM) stops holding the live conversations.

**Expected, from study 47's numbers (not a target):**

| Configuration | Cache | Users |
|---|---|---|
| Defaults (bf16) | 123k tokens on GPU | ~30 |
| FP8 | 241k tokens on GPU | ~60 |
| FP8 + 28 GiB offload | ~400k tokens | ~100 |

Each user holds ~3.9k tokens of live history.

## Method

- **Ramp task:** levels of 16, 24, 32, 48, 64, 80, 96, 112 and 128 users. Each level is a fresh
  AIPerf run of 960 s plus 60 s of grace. A level is judged on the requests started ≥ 600 s
  after its first request, once the conversations are at full size. It passes when:
  - client TTFT p95 ≤ 500 ms;
  - errors < 1 %;
  - ≥ 95 % of the sent requests completed;
  - ≥ 95 % of the expected requests sent.

  The ramp stops at the first failing level. The last passing level goes to `k8s/level.current`
  (16 if none passes).
- **Hold task:** the chosen level is run again from scratch.
  - Akamas windows on it with a trim of `[620s, 90s]`.
  - Score: `64 × completed requests/s − TTFT p95 / 1000`. This is the users held, with TTFT in
    seconds as a tie-break below one level step.
  - Server-side safety net: TTFT p95 ≤ 600 ms and ITL p95 ≤ 120 ms.
  - A Hold whose TTFT lands between 500 and 600 ms is VALID in Akamas but **flagged in the
    results**.
- **Granularity:** above 32 users the steps are 16 users. If 128 passes, the result is "≥ 128".
- **Read the score by level, not by decimals.** The rate is read over 30 s, so the "users" term
  is noisy by about ±0.5 user over the window, as large as the TTFT tie-break. Only differences
  across levels are findings; within a level, compare TTFT directly (final review, 2026-11-06).
- **Failures that must not score:**
  - AIPerf killed (exit ≥ 128, e.g. OOM) or crashed without a RESULT line fails the Ramp (exit
    6), instead of being taken as an SLO failure.
  - If no level passes, the Hold exits 7 instead of scoring the first level.
- **Every load prints** the RESULT line, the TTFT per minute, the vLLM prefix and offload
  counter deltas, and the AIPerf log tail. Before each load, stale AIPerf processes are killed:
  the study 47 reaper, now in `run_level`.

## Stack & versions

- **Hardware:** one g7.4xlarge, node group `llm-serving-g7-kv` (same as study 47), tagged
  AlwaysOn while the study runs. See `infra/README.md`.
  - NVIDIA RTX PRO 4500 Blackwell 32 GB;
  - 16 vCPU, 64 GiB (54.5 GiB allocatable: 57,165,240 KiB; study 47 wrote "57 GiB", which is 57 decimal GB);
  - 3.04 USD/h.

  EKS `vllm-bench` 1.36.5, us-east-2.
- **Server:**
  - `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as `qwen3-8b`;
  - `--max-model-len 16384`, prefix caching on;
  - 10 CPU, memory = offload + 20 GiB, `/dev/shm` = offload + 1 GiB;
  - namespace `kv-offload`.
- **Load:** AIPerf 0.11.0, synthetic multi-turn chat.
  - `--user-centric-rate USERS/64 --num-users USERS`: one turn per user every 64 s.
  - 10 turns, 600-token user turns, 200-token answers (`ignore_eos`), history re-sent.
  - AIPerf pod on the GPU node: 4 CPU, memory request 6 GiB and limit 8 GiB. Workload in `k8s/workload.env`.
- **Akamas:** 4.1.0, vLLM pack **1.15.0**, Kubernetes pack **1.9.0**. See `akamas/README.md`.
- **Telemetry:** Prometheus. The ServiceMonitor `vllm-kv-offload` scrapes vLLM every 5 s.

## Parameters tuned

Same as study 47:

| Parameter | Domain | Note |
|---|---|---|
| `vllm.kv_offloading_size` | 0–28 GiB | 0 = no flag. 28 + 20 (vLLM) + 6 (AIPerf request) = 54 GiB of the 54.5 allocatable |
| `vllm.kv_cache_dtype` | auto, fp8 | |
| `vllm.gpu_memory_utilization` | 0.80–0.94 | |
| `vllm.max_num_seqs` | 16–256 | |
| `vllm.max_num_batched_tokens` | 1024–16384 | ≥ `max_num_seqs` |
| `vllm.block_size` | 16, 32, 64, 128 | |

Steps:

1. `baseline` (vLLM 0.29 defaults: gmu 0.92, mns 256, mnbt 2048, kv auto, block 16).
2. `Offload only` (24 GiB).
3. `FP8 KV only`.
4. `Offload and FP8`.
5. `Baseline repeat`.
6. `optimize`: AKAMAS, 3 initial experiments, 10 experiments.

An experiment takes ~1 h if the configuration fails early and ~2–3 h if it climbs high.
Estimated total: ~30 h, ~90 USD.

## Results

Raw data in `results/`: `export.tar.gz` (Akamas export) and `experiments.csv` (parameters, score
and Hold aggregates per experiment). The ramp levels below come from the Ramp/Hold task logs, parsed into `results/ramp.json` (client TTFT p50/p95, ITL, sent/completed per level and experiment).

**Best: 80 users within client TTFT p95 ≤ 500 ms, vs 24 for the baseline (3.3×, +233 %).**
96 users fails in every configuration that reached it (client TTFT p95 837–4060 ms; 4060 ms is exp 11, whose max_num_seqs 16 queues the overflow).

| Exp | Source | Offload GiB | KV dtype | gmu | mns | mnbt | block | Users (first failing level) | Tokens/s at Hold |
|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 0 | auto | 0.92 | 256 | 2048 | 16 | 24 (32: 840 ms) | 1776 |
| 2 | Offload only | 24 | auto | 0.92 | 256 | 2048 | 16 | 32 (48: 854 ms) | 2283 |
| 3 | FP8 KV only | 0 | fp8 | 0.92 | 256 | 2048 | 16 | 48 (64: 847 ms) | 3455 |
| 4 | Offload and FP8 | 24 | fp8 | 0.92 | 256 | 2048 | 16 | 64 (80: 838 ms) | 4578 |
| 5 | Baseline repeat | 0 | auto | 0.92 | 256 | 2048 | 16 | 24 (32: 842 ms) | 1732 |
| 6 | optimize | 14 | fp8 | 0.87 | 136 | 8704 | 64 | 48 (64: 818 ms) | 3563 |
| 7 | optimize | 7 | auto | 0.905 | 196 | 4864 | 128 | 24 (32: 829 ms) | 1759 |
| 8 | optimize | 21 | fp8 | 0.835 | 76 | 12544 | 32 | 64 (80: 821 ms) | 4569 |
| 9 | optimize | 0 | fp8 | 0.80 | 17 | 1024 | 128 | 32 (48: 848 ms) | 2256 |
| **10** | optimize (Akamas best) | **28** | **fp8** | **0.94** | **256** | **16382** | **16** | **80 (96: 846 ms)** | **5722** |
| 11 | optimize | 28 | fp8 | 0.80 | 16 | 16384 | 64 | 80 (96: 4060 ms) | 5759 |
| 12 | optimize | 28 | fp8 | 0.94 | 256 | 1024 | 128 | 80 (96: 1013 ms) | 5811 |
| 13 | optimize | 28 | auto | 0.94 | 256 | 16384 | 64 | 32 (48: 829 ms) | 2415 |
| 14 | optimize | 28 | fp8 | 0.80 | 256 | 16384 | 64 | 80 (96: 842 ms) | 5745 |
| 15 | optimize | 28 | fp8 | 0.80 | 256 | 16384 | 128 | 80 (96: 837 ms) | 5803 |


- **Best configuration** (Akamas pick, exp 10): `--kv-offloading-size 28`, `--kv-cache-dtype fp8`,
  `--gpu-memory-utilization 0.94`, `--max-num-seqs 256`, `--max-num-batched-tokens 16382`,
  `--block-size 16`. Exp 14 (gmu **0.80**, block 64) scores the same 80.256: same users with 14 % of
  the VRAM left unused.
- **Score:** 23.66 → 80.26 (Akamas +239 %; read by level, not by decimals).
- **Binding:** the SLO (client TTFT p95 ≤ 500 ms) at the next level. The server-side constraints
  (TTFT ≤ 600 ms, ITL ≤ 120 ms) never bound: at the Hold, server TTFT p95 was 229–328 ms and ITL
  p95 24.3–24.8 ms in every experiment (offload costs no decode speed).
- **No drift:** the baseline repeat gave the same 24 users at the same TTFT.
- **All levels were ~20 % below the design's estimate** (~30 / ~60 / ~100 expected, 24 / 48 / 80
  measured), consistently across bf16, FP8 and FP8 + 28 GiB.
- **Telemetry note:** with offload on, `vllm.prefix_cache_hit_rate` (GPU) reads ~0 % and the hits
  show up in `vllm.cpu_offload_prefix_hit_rate` (~80 %); the combined hit rate stays ~80–86 % in every
  experiment. Compare the combined metric across configurations, not the GPU one.

## Conclusions

- **Users within a TTFT SLO scale with the total KV capacity (GPU + RAM, in the KV dtype), not
  with the scheduler flags.** All five FP8 + 28 GiB experiments reached 80 users whatever gmu
  (0.80–0.94), max_num_seqs (16–256), max_num_batched_tokens (1024–16384) or block size (16–128).
  The ceiling followed the offload size: 14 GiB → 48, 21–24 GiB → 64, 28 GiB → 80.
- **Offload without FP8 is worth one ramp step** (24 → 32, with 24 or 28 GiB). FP8 alone doubles the
  users (48); offload on top of FP8 adds two to three steps (64–80). The levers multiply, as in
  study 47.
- **At the same per-user load, 3.2× the tokens/s on the same GPU** (1776 → 5722), at the same ITL.
- **28 GiB is the node's limit** (54.5 GiB allocatable, 20 GiB for vLLM, 6 GiB for AIPerf): more
  users on this GPU need a node with more RAM, a smaller per-user history, or a load generator off
  the GPU node.
- **Specific to this stack:** an 8B FP8 model on a 32 GB GPU, ~3.9k tokens of live history per
  user, one turn per user every 64 s, vLLM 0.29 native offload. The step sizes (16 users above 32)
  make "80" mean "80 to 95".
- **Next:** a node with more RAM to find where offload stops paying (PCIe bandwidth or CPU); the
  `lmcache` backend; the SLO at a tighter value (e.g. 300 ms) to check that RAM reloads stay
  inside it.
