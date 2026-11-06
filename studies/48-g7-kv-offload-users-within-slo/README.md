# 48-g7-kv-offload-users-within-slo

**Status:** RUNNING (Akamas 4.1, akamas-41; started 2026-11-06).

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

TODO (study-recap at the end).

## Conclusions

TODO.
