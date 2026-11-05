# Study 47 — Multi-turn chat TTFT under KV memory pressure, with CPU KV offload (design)

**Date:** 2026-11-05 · **Status:** draft, waiting for the user's approval · **Study folder (to
create):** `studies/47-g7-kv-cache-offload-ttft/`

## Intent and the why

A customer runs a chat assistant: users hold conversations of many turns and come back after
reading the answer. Every turn re-sends the whole history, so the prompt grows and the server
either still has the conversation's KV cache (prefix cache hit, cheap) or recomputes it (slow
prefill, high time to first token). When the live conversations do not fit in GPU memory, vLLM
0.29 can keep evicted KV blocks in CPU RAM (`--kv-offloading-size`) and reload them instead of
recomputing.

The study answers: **for a fixed, realistic multi-turn chat load that overflows the GPU's KV
cache, which vLLM configuration gives the lowest TTFT p95, and is CPU offload part of it?** The
sentence the README should let us write is of the form: *"on one RTX PRO 4500, 64 chat users
get TTFT p95 X ms with vLLM defaults and Y ms tuned; the best configuration uses / does not
use N GiB of CPU offload, together with …"*.

Why now: vLLM 0.29 ships native CPU offload; the 2026-11-05 spike
(`docs/spikes/2026-11-05-kv-offload/README.md`) showed it halves TTFT p50 and cuts p95 3× in
the right load band, and does nothing outside it. Whether it beats the cheaper alternatives
(FP8 KV cache, a larger `gpu_memory_utilization`) has not been measured.

## Fairness: the study must not be built so that offload wins

The user asked (2026-11-05) whether we would set the study up so that TTFT is higher without
offload. The answer that drives this design:

- **The workload is fixed from the customer scenario, not tuned to the offload.** We pick it so
  that the live context overflows the GPU KV cache **with vLLM's default settings** — that is the
  situation in which a customer would consider offload at all, and it is a precondition, not a
  result. It is fixed before the first experiment and never changed during the study.
- **Offload is one option among several that fight the same memory pressure**, all tuned
  together: `kv_cache_dtype` fp8 doubles the tokens that fit in the GPU, a higher
  `gpu_memory_utilization` adds a few, `max_num_seqs` / `max_num_batched_tokens` change how much
  prefill competes with decode. Offload size includes **0 (off)** in its domain.
- **Presets measure each lever alone** (offload only, FP8 KV only) next to the baseline, so the
  README can say which lever does the work. If FP8 KV alone matches offload, that is the finding.
- Expected, from the spike's numbers: 64 users ≈ 250k live tokens; GPU KV holds 120k (bf16) or
  ~240k (fp8); 24 GiB of offload holds ~175k (bf16) / ~350k (fp8). So fp8 alone is borderline,
  offload alone is borderline, both together have margin. This is a guess to be checked, not a
  premise.

## Prior results this builds on

- Spike 2026-11-05 (`docs/spikes/2026-11-05-kv-offload/README.md`): setup, numbers, gotchas
  (`/dev/shm` must hold the buffer; LRU thrash above GPU+CPU capacity; Qwen3 template never
  reuses the latest assistant answer; AIPerf multi-turn reuse verified at 66 % on one user).
- Study 28: Qwen3-8B-FP8 footprint, AIPerf open-loop setup. Study 46: AIPerf 0.11 pitfalls
  (dataset validation time, warmup dip), Akamas 4.1 pitfalls (trim single unit, tiny reals
  rendered without exponent, start-study timeout).

## Goals

1. Minimize **TTFT p95** of a fixed multi-turn chat load on one g7.4xlarge, measured over the
   steady state only.
2. Find out whether CPU KV offload is part of the best configuration, and how much each lever
   (offload, FP8 KV, memory utilization, batching) contributes, via presets.
3. Record the offload metrics (CPU hits, bytes moved) per experiment so the README can explain
   *why* a configuration wins.

## Non-goals

- The `lmcache` backend (extra dependency; native only). Disk or remote KV tiers.
- Changing the workload during the study, or several workloads in one study (a load-level
  sensitivity check is a possible follow-up study, not this one).
- Throughput maximization: the load is open-loop at a fixed rate, so output throughput is the
  same for every configuration that keeps up; it is a constraint, not the goal.
- Other models, other GPUs, multiple replicas.

## Anti-goals (a failure even if it runs)

- A workload chosen or changed after seeing results so that offload looks good.
- A TTFT gain bought with worse inter-token latency or dropped/failed requests.
- Measuring the warm-up (contexts still growing) instead of the steady state.
- vLLM OOM-killed or failing at startup because `/dev/shm` or the pod memory limit does not
  follow the offload size (the spike's crash).
- The load generator competing with vLLM for CPU and being the bottleneck.

## Decided (with the user or by the spike)

- Hardware: one **g7.4xlarge** (RTX PRO 4500 Blackwell 32 GB, 16 vCPU, 64 GiB, 3.04 USD/h),
  node group **`llm-serving-g7-kv`** (created 2026-11-05, label `node-role=llm-serving-g7-kv`,
  kube-system device plugin; `llm-serving-g7-4500` is not usable, see the spike).
- Model **Qwen/Qwen3-8B-FP8**, vLLM **0.29.0**, prefix caching **on** in every experiment,
  `max-model-len` 16384.
- Load generator **AIPerf 0.11.0**, synthetic multi-turn, `--user-centric-rate`.
- Goal: **minimize TTFT p95** (user's choice, 2026-11-05).
- Offload backend **native**.

## My calls (the user can override)

- **Workload:** 64 users, one turn per user every 64 s (`--user-centric-rate 1 --num-users 64`),
  10 turns per conversation, 600-token user turns, 200-token answers (`ignore_eos`), history
  re-sent. In the spike this load did not collapse with offload (TTFT p95 0.8 s) and its live
  context (~250k tokens) is about twice the default GPU KV cache.
- **Calibration before freezing (rule fixed now, before seeing data):** the spike never ran this
  load without offload. One calibration run with the baseline configuration comes first. If the
  baseline keeps up (completed requests ≥ 95 % of sent, TTFT p95 over minutes 6–15 below 10 s),
  the workload is frozen as above. If it collapses (as A64 did at 2 req/s), the user count steps
  down 64 → 56 → 48 at the same 64 s per user until the baseline keeps up, and that is frozen.
  The rule only looks at the baseline, never at an offload run, so it cannot favour offload.
- **Steady state:** AIPerf's user-centric mode already staggers users (spike B64r1: 57 active
  conversations constant from minute 2 to 12, gaps exactly 64 s; finished conversations are
  replaced one at a time, no synchronized restart). Run 20 min per experiment and score minutes
  6–20, so the window holds ~14 min of steady state.
- **Steady-state window:** 20 min of load per experiment; scoring window from minute 6 to the
  end (contexts reach their steady size around minute 5–6 in the spike). Trim written in one
  unit (`360s`), per the study 46 lesson.
- **TTFT source:** client-side from AIPerf if it can be exposed to Akamas for the window;
  otherwise vLLM's server-side `vllm:time_to_first_token_seconds` histogram via Prometheus, with
  the Akamas trim doing the windowing. Builder decides; the two must not be mixed across
  experiments.
- **Constraints (absolute, so they stay meaningful whatever the baseline does):** ITL p95
  ≤ 50 ms (spike: 21–33 ms at this load); error rate < 1 %; completed requests ≥ 95 % of those
  sent (the load is open-loop, so a configuration that falls behind shows up here).
- **Baseline = vLLM 0.29's own defaults:** only the fixed flags (model, served name, max-model-len
  16384, prefix caching on) and no tuned flag at all, so the README's "with defaults" sentence is
  literal. The spike's gmu 0.90 / max_num_seqs 128 are not the baseline.
- **Parameters** (domains to be checked against the pack):
  - `kv_offloading_size` 0–32 GiB (0 = off; node RAM 57 GiB allocatable caps it).
  - `kv_cache_dtype` {auto, fp8}.
  - `gpu_memory_utilization` [0.80, 0.94].
  - `max_num_seqs` [16, 256], `max_num_batched_tokens` [1024, 16384] with mnbt ≥ mns.
  - `block_size` {16, 32, 64, 128} (offload moves blocks; size may matter).
  - Optional, if the pack has them and budget allows: `async_scheduling`,
    `scheduler_reserve_full_isl`.
- **Derived, not tuned:** the apply script sets `/dev/shm` = offload + 1 GiB and the pod memory
  limit = offload + 20 GiB.
- **Steps:** baseline (vLLM defaults, offload 0, kv auto); preset offload 24 GiB only; preset
  FP8 KV only; preset both; baseline repeat (drift); optimize ~25 experiments.
- **Cost estimate:** ~27 min per experiment (≈5 min vLLM restart + 20 min load + teardown),
  ~30 experiments ≈ 14 h ≈ 43 USD of g7.4xlarge, plus Akamas. The AIPerf client runs on the same
  node with a CPU limit (4 cores) and vLLM keeps 10 cores, as in the spike — or on a separate
  load node if the builder finds CPU contention.

## Prerequisites (block the study)

1. **vLLM optimization pack — done 2026-11-05:** pack 1.15.0 adds `kv_offloading_size` (GiB,
   0-512, default 0 = off), `kv_offloading_backend` (native, lmcache) and the metrics
   `cpu_offload_prefix_hit_rate`, `prefix_cache_hit_rate_with_offload`,
   `kv_offload_gpu_to_cpu_throughput`, `kv_offload_cpu_to_gpu_throughput` (the CPU buffer usage
   gauge read 0 with the buffer full and was left out). Installed on akamas-41; MR
   https://gitlab.com/akamas/optimization-packs/vllm/-/merge_requests/12 to `develop`, merged by
   the user (who also manages the packs).
2. Prometheus scraping the vLLM pod in the study namespace (ServiceMonitor), as in study 46.
3. Pod memory and shm sized from the offload parameter (see "Derived").

## Left to the builder

- Exact Akamas telemetry wiring and KPI list (≤ 8 KPIs).
- How AIPerf results reach Akamas (if client-side TTFT is used).
- Whether to keep the load generator on the GPU node.
- Workflow details: apply script, wait for vLLM ready, run load, dump logs to stdout.
