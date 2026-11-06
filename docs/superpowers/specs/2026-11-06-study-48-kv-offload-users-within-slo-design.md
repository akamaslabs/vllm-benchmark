# Study 48 — Users within a TTFT SLO with CPU KV offload (design)

**Date:** 2026-11-06 · **Status:** draft, waiting for the user's approval · **Study folder (to
create):** `studies/48-g7-kv-offload-users-within-slo/`

## Intent and the why

Study 47 fixed the load (64 users) and showed that CPU KV offload + FP8 KV cut TTFT p95 from
877 to 227 ms. At a fixed open-loop load every configuration serves the same tokens/s: the
client sets the pace, and all 30 experiments completed ~840 of ~840 requests. So study 47
cannot show the other half of the story: **how many more users the same GPU serves when the
KV cache spills to RAM**.

Study 48 asks that question directly: **the highest number of chat users each configuration
serves with TTFT p95 ≤ 500 ms**, on the same node, model and per-user workload as study 47. The
sentence the README should let us write: *"one RTX PRO 4500 serves X users within a 500 ms
TTFT p95 with vLLM defaults, Y with FP8 KV, and Z with FP8 + N GiB of CPU offload."*

Expectation from study 47's numbers, not a target:

- each user holds ~3.9k tokens of live history (~250k / 64);
- the default bf16 GPU cache (123k tokens) covers ~30 users;
- FP8 on the GPU (241k tokens) covers ~60 users;
- FP8 + 28 GiB of offload (~400k tokens) covers ~100 users.

Past ~100 users the LRU buffer thrashes again (study 47, experiment 2), so the gain should be a
band with a ceiling. Finding that ceiling is part of the result.

## Why 500 ms (user's choice, 2026-11-06, on my recommendation)

The SLO sits between the two ways a turn's history gets back into the GPU, as measured in
study 47:

- reloaded from RAM: ~190 ms p50 / ~230 ms p95 at 64 users;
- recomputed: 600–900 ms p95.

At 500 ms, a configuration passes while its cache (GPU, or GPU + RAM) holds the live context,
and fails once it must recompute. The user ceiling then measures cache capacity.

The alternatives were rejected:

- 260 ms: too close to the reload cost itself, so queueing noise would decide the ceiling;
- 1 s: the recomputing baseline passes at 64 users, so the ceiling would measure prefill
  compute instead of the cache.

## Fairness

- **Per-user workload identical to study 47:** one turn every 64 s per user, 10 turns, 600-token
  user turns, 200-token answers (`ignore_eos`), history re-sent. Only the number of users
  changes, so the request rate is users / 64 s.
- **Ramp levels, step length and SLO are fixed in this design, before any data.** Every
  experiment walks the same levels.
- **Same five presets as study 47,** so each lever is measured alone.

## Prior results this builds on

- **Study 47** (`studies/47-g7-kv-cache-offload-ttft/README.md`):
  - offload alone −1.5 %, FP8 alone −16 %, both −74 % at 64 users;
  - offload is a step: with FP8, 21–28 GiB all gave 227–242 ms, while 18 GiB or less gave
    600–760 ms;
  - gmu, batching and block size were second order (≤ 6 %);
  - steady state needs ~11 min: TTFT p95 kept rising until minute ~11, because a conversation
    lives 10 turns × 64 s.
- **Study 46:** the ramp + hold pattern. A Ramp task finds the highest passing level with a
  client-side SLA check, then a Hold task holds that level while Akamas reads it from
  Prometheus. Study 48 reuses it.

## Goals

1. Users served within TTFT p95 ≤ 500 ms for the baseline, offload only, FP8 only, and offload
   + FP8, at study 47's per-user load.
2. Find the best configuration's user ceiling with a short optimize step.
3. Report tokens/s at each configuration's ceiling (it scales with users) as the "more
   throughput on the same GPU" figure.

## Non-goals

- A different model, GPU, context length or offload backend (`lmcache` stays a follow-up).
- Re-running study 47's fixed-load TTFT minimization.
- Changing the per-user workload to make the ramp faster.

## Anti-goals (a failure even if it runs)

- A ceiling set by the **load generator**, by **AIPerf CPU**, or by node RAM (an OOM or an
  evicted pod), presented as a cache effect.
- A ceiling set by the **ramp's top level** rather than by the SLO. If the top level passes,
  the result is "≥ top" and must be reported as such.
- Scoring a level before its conversations have reached steady size (study 47: ~11 min).

## Decided (with the user)

- Same stack as study 47:
  - node group `llm-serving-g7-kv` (g7.4xlarge, RTX PRO 4500 32 GB, 57 GiB allocatable);
  - Qwen3-8B-FP8 on vLLM 0.29.0, prefix caching on, `max-model-len` 16384;
  - AIPerf 0.11.0 user-centric multi-turn;
  - native offload backend;
  - vLLM pack 1.15.0.
- **SLO: client TTFT p95 ≤ 500 ms.**
- **Goal: maximize users within the SLO.** Tokens/s is a KPI.
- A **new study** (48), in its own folder copied from study 47. Not a re-import of study 47:
  Akamas 4.1 cannot change the windowing, the workflow or the goal type of a study that has
  already run.

## My calls (the user can override)

- **Ramp levels: 16, 24, 32, 48, 64, 80, 96, 112, 128 users** (my call after the builder
  check).
  - The low levels 16 and 24 exist because the baseline and offload-only are expected near ~30
    users. Starting at 32 could leave them at "fewer than 32", with no number for the README.
  - The ramp stops at the first failing level, so poor configurations are cheap.
  - Above 32 the step is 16 users, ~15–25 % of the expected ceilings. That is the result's
    granularity, stated in the README.
- **Each level is one fresh AIPerf run: 10 min warm-up, then 6 min measured.** The level passes
  when, over the measured 6 minutes:
  - client TTFT p95 ≤ 500 ms;
  - ≥ 95 % of the sent requests complete;
  - errors < 1 %.

  The 10 min warm-up comes from study 47's minute table, so contexts are at full size.
- **Hold task:** the last passing level, run again fresh for 17 min (10 warm-up + 6 scored + 1 of
  grace). The Akamas trim keeps only the scored minutes.
  - If no level passes, Hold runs at 16 and the safety-net constraint marks the experiment.
  - If the top level passes, the README notes the cap.
- **Score, read from Prometheus on the Hold window** (tie-break added after the builder check):
  `users = 64 × vllm.request_success_rate − vllm.time_to_first_token_p95 / 1000`.
  - The TTFT term is at most 0.5 under the SLO, smaller than one level step, so it only breaks
    ties between configurations that reach the same level. Study 47 makes such ties likely:
    every FP8 setup with ≥ 21 GiB of offload was within 7 %.
  - The README reports the users and TTFT separately.
  - At one turn per 64 s per user, the completed-request rate equals users / 64 s whenever the
    server keeps up, which a passing level guarantees.
  - This is the analogue of study 46's requests-in-flight trick, for an open-loop load with
    think time.
- **Constraints (absolute, on the Hold window, as a safety net):**
  - server TTFT p95 ≤ 600 ms (my call after the builder check). The SLO itself is judged by
    the Ramp, client-side, at 500 ms. The Hold re-runs the level fresh, and a configuration at
    its cache edge can land a little above 500 ms through run-to-run noise. A 500 ms safety
    net would then mark exactly the best configurations as infeasible. 600 ms still catches a
    Hold that does not reproduce the Ramp. The README reports every Hold that lands between
    500 and 600 ms.
  - ITL p95 ≤ 120 ms (study 47's threshold, server-side histogram).
- **Parameters and domains: same as study 47.**

  | Parameter | Domain |
  |---|---|
  | `kv_offloading_size` | 0–28 GiB |
  | `kv_cache_dtype` | auto, fp8 |
  | `gpu_memory_utilization` | 0.80–0.94 |
  | `max_num_seqs` | 16–256 |
  | `max_num_batched_tokens` | 1024–16384 |
  | `block_size` | 16, 32, 64, 128 |

  Keep mnbt ≥ mns. At more users, `max_num_seqs` 16 may cap concurrency: study 47 ran ~7
  sequences at once at 64 users, so ~14 at 128. Keep the domain and let the data show it.
- **Steps:**
  - baseline (vLLM defaults, rendered explicitly as in study 47);
  - presets: offload 24 GiB only, FP8 only, offload 24 GiB + FP8;
  - baseline repeat;
  - optimize: **10 experiments**. Study 47's optimizer added 3 % over the preset, so a short
    optimize step is enough.
- **AIPerf pod:** same node, 4 CPU. Raise memory from 6 to 8 GiB for 128 users. vLLM's memory
  limit (offload + 20 GiB) still fits in 57 GiB. If AIPerf CPU saturates at the top levels,
  the builder moves it to a non-GPU node and notes it.
- **Cost estimate, per experiment:**
  - a configuration that passes up to ~96 users: 6 levels × 16 min + Hold 17 min + restart
    ~5 min ≈ **2 h**;
  - one that fails early: ~1 h.

  - the levels 16 and 24 add ~30 min to every experiment that passes them.

  For ~15 experiments, about half of them good, that is **≈ 30 h ≈ 90 USD** of g7.4xlarge.
  This is above the ~60 USD I quoted in chat: the 10 min warm-up per level costs more than I
  first estimated. Tag the node AlwaysOn and scale it to 0 at the end.

## Prerequisites

1. Nothing new in the pack: `request_success_rate`, the TTFT/ITL histograms and the offload
   metrics already exist (vLLM pack 1.15.0, study 47 telemetry).
2. The node group `llm-serving-g7-kv` scaled back up (`gpu-nodegroup.sh --up --always-on`).
3. Study 47's stale-AIPerf reaper and fresh-pod-per-experiment logic carried over. The ramp
   runs several AIPerf processes in sequence, so each level must reap the previous one.

## Left to the builder

- Ramp and Hold scripts, adapted from study 46's `run_ramp.sh` and `run_hold.sh` and study 47's
  `run_load.sh` (reaper, restart check, per-minute TTFT table, counters).
- Telemetry and the KPI list (≤ 8): users, tokens/s, TTFT p95, ITL p95, GPU prefix hit rate,
  offload hit rate, KV usage, container memory.
- The exact Akamas trim on Hold (single unit, e.g. `600s` start).
- Whether to log each level's per-minute TTFT table, to show where a failing level crosses
  500 ms.
