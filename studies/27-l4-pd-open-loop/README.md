# 27-L4-PD-Open-Loop

**Status:** FINISHED. Second run `27-L4-PD-Open-Loop-Gamma` (gamma arrivals, 2.4 req/s in 120 min), started 2026-09-30, stopped 2026-10-01 after 31 experiments (see "Results"). The first run `27-L4-PD-Open-Loop` (Poisson, 3.0 req/s in 60 min; study id a1e35c1b-9d3e-40b7-ada2-71e850445b86) was finished after its two baselines: see "First run: Poisson".
**Needs:** vLLM optimization pack **1.12.0** (as study 24).

> Study 24 with an **open-loop load**. Same model, node, parameters, domains, presets and
> goal. The load is an AIPerf Poisson rate ramp, and a watchdog ends it past the SLA. Own
> system, telemetry instance and workflow. The workflow runs this folder's `k8s/` on the
> toolbox.

## Why this study exists

Studies 18-24 ran a closed loop: `--concurrency` only, 6 levels × 600 s (2, 4, 8, 16, 24, 32).
Study 24 showed two problems (details in the research notes, `load_generation.md`):

- **The score jumps by one level.** The score is the throughput of the most loaded valid
  window. With levels that double, it can only take the value of one level or the next.
  Study 22 exp 2 and study 24 exp 4 have the same config and the same metrics at every
  level. They scored 1382.61 and 985.84 (−29%): at concurrency 16 the TTFT p95 moved around
  the 10 s limit, and it stayed below 10 s for 4 minutes in study 22 and for 2 in study 24.
- **The users are synchronized.** AIPerf sends the next request 0 ms after the previous
  one ends, and all requests have the same length. The users end together and start again
  together, so the prefill receives the requests in bursts. At concurrency 8 the prefill
  is at ~45% of its capacity and the TTFT p95 is still ~9.6 s.

## What changes against study 24

| Item | Study 24 | Study 27 |
|---|---|---|
| Load | closed loop, `--concurrency 2,4,8,16,24,32`, 600 s each | open loop, `--request-rate 2.4 --request-rate-ramp-duration 7200`, gamma arrivals (smoothness 4), seed 18 |
| End of the test | after 60 min | watchdog: TTFT p95 (150 s) > 20 s or ITL p95 (150 s) > 225 ms for 120 s, or 120 min |
| Queue constraints | `num_requests_waiting <= 1` on each role | removed |
| Steps | baseline, 11 presets, 60 AKAMAS | baseline twice, the same 11 presets, 2 scheduler presets, 60 AKAMAS |
| Experiment length | ~65 min | ~40-85 min (setup ~8 min, then the ramp until the watchdog) |

Unchanged: parameters, domains, parameter constraints, goal formula, latency constraints
(TTFT p95 150 s `:max` <= 10 s, ITL p95 150 s `:max` <= 75 ms), windowing, KPIs, telemetry
queries, serving template, router, launcher, tuned configs.

## The load

- **Verified (AIPerf 0.11.0 source and a local dry run, 2026-09-30):**
  - `--request-rate R` without `--concurrency` is an open loop. With `--concurrency` the
    concurrency is a cap, and at saturation the loop closes again.
  - `--request-rate-ramp-duration D` starts at R × 0.1 / D and raises the rate linearly to R.
    The rate changes every 0.1 s. The dry run printed
    `Starting request rate ramp: 0.00 → 1.5 QPS over 600.0s`.
  - Only a linear ramp is available for the rate from the CLI (`timing/phase/runner.py`
    sets `RampType.LINEAR`).
  - The Poisson intervals come from a generator derived from the global seed, so every
    trial gets the same arrival sequence.
- **R = 2.4 req/s in D = 7200 s** (0.02 req/s per minute). R is above every config in the
  search space (2P2D with fp8 KV is estimated at ~1.8 req/s). The watchdog ends the ramp,
  so the true peak is always measured. The 180 s scoring window spans 0.06 req/s: 11% of
  the rate at 0.55 req/s, 4% at 1.5 req/s.
- **Gamma arrivals, smoothness 4:** the same mean rate as Poisson, a quarter of the
  variance of the intervals, so about half the noise on the number of arrivals in a window.
  Checked locally: `pattern=gamma, rate=2.4, smoothness=4.0`, 51 requests sent in 120 s
  (~58 expected).
- **Watchdog** (`k8s/run_test_tps.sh`): every 15 s it reads the router's TTFT and ITL p95
  over 150 s from Prometheus. With an open loop the queue past the capacity grows for the
  rest of the ramp, so every later window is invalid. The watchdog ends the test with
  success. The fail-fast checks of study 24 (job failed, restart, stall, deadline) stay.
- **Windowing:** `maxStdDev: 300000000` disables the stability filter, as in study 24.
  Akamas takes the valid window with the highest throughput. The ramp only goes up, so the
  150 s p95 with `:max` works as in study 24.

## Steps and expected results (written before the start)

All values are **hypotheses**. They come from the capacities measured in study 24 and a
simple queue model: with Poisson arrivals the TTFT p95 reaches 10 s at ~70-85% of the
prefill capacity.

| Step | Expected score (tok/s/GPU) | Why |
|---|---|---|
| baseline aggregated | ~950-1100 | study 24: 1029.05, limited by ITL |
| baseline aggregated repeat | within ~5% of the first | first measure of the repeatability |
| P1D1 prefill marlin | ~800-950 | prefill capacity 0.54 req/s, SLA at ~0.37-0.45 req/s |
| P1D1 prefill humming | ~1150-1350 | prefill ~0.75 req/s, the bf16 decode (~0.69) saturates first |
| P1D1 prefill triton default | ~1250-1400 | bf16 decode-bound (~0.69 req/s), ITL reaches 75 ms at ~0.6 req/s |
| P1D1 prefill triton tuned | ≈ triton default (+0-5%) | same decode-bound capacity |
| P1D1 decode humming | ≈ triton tuned | same decode capacity |
| P1D1 kv fp8 | ~1500-1800 | decode ~1.5 req/s, prefill-bound at ~0.9 req/s |
| P1D1 flash attn | ≈ triton tuned | same prefill step |
| P2D1 / P1D2 / P2D2 / P3D1 | below P1D1 per GPU | bf16 decode or prefill idle, as in study 24's table |
| P1D1 prefill batched 4160 | triton tuned +0-10% | shorter prefill steps, lower TTFT; decode-bound capacity unchanged |
| P1D1 kv fp8 decode seqs 32 | ≈ P1D1 kv fp8 | ~29-31 sequences is the ITL and KV limit, 32 does not bind |

- If the two baselines differ by more than ~5%, the noise of the new load is too high.
  Then try `--arrival-pattern gamma --arrival-smoothness 4` (half the variance of Poisson).
- In study 24 a closed loop at concurrency 16 scored the Triton P1D1 at 1519.91 and 985.84.
  If the open-loop scores of the P1D1 presets are close to each other, as expected, the
  level jump is gone.

## Smoke test (before the study)

`akamas/27-L4-PD-Open-Loop-Smoke.yaml` with `akamas/27-L4-PD-Open-Loop-Smoke-Workflow.yaml`:
one trial of the P1D1 (prefill triton tuned) with a 10-minute ramp to 1.5 req/s
(`RT_RATE_MAX=1.5 RT_RAMP_S=600`). The P1D1 saturates at ~0.69 req/s, so the ramp crosses
it after ~4.6 min, and the watchdog should end it after ~6-8 min. About 20-25 min with the
pod start. Check:

1. The AIPerf log shows `Starting request rate ramp: 0.00 → 1.5 QPS over 600.0s`.
2. The RunTest log shows the watchdog lines and ends with success.
3. The trial is VALID in Akamas, with a score and all the KPIs.
4. The windows before the watchdog meet the constraints, and the later ones do not.

Then delete the smoke study.

### Smoke test results (2026-09-30)

- **Run 1 failed: the ramp sent no request** (`Phase profiling sending complete | sent=0`
  in 600 s, then AIPerf exited with "No profile results to export"). Cause, in AIPerf
  0.11.0: the ramp starts at R × 0.1 / D (1.5 × 0.1 / 600 = 0.00025 req/s). The first
  Poisson interval is drawn at that rate (mean ~4000 s). A later rate change does not
  reschedule the pending wait (`set_request_rate` only updates the interval generator).
  Fix: `AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10` (the maximum) in `k8s/05-job.yaml`. The
  ramp then starts at R × 10 / D (0.0083 req/s in the study, first request after ~2 min on
  average). Reproduced and fixed locally: sent=0 → sent=21 in 120 s.
- **Run 2 passed: VALID, score 1068.69**, 12 min 26 s.
  - The ramp started (`0.03 → 1.5 QPS over 600.0s`).
  - The watchdog fired at TTFT p95 23.7 s and ended the test after 577 s ("TTFT p95
    57421 ms, ITL p95 207 ms for 121 s"). The trial did not fail.
  - The scored window ends where the SLA breaks: TTFT p95 9.8 s, ITL p95 74.7 ms.
  - The bf16 decode saturated at ~0.64-0.71 req/s (15 running sequences, decode queue
    growing), as in study 24 (~0.69 req/s).
  - The score is low for this config because the smoke ramp is 6× steeper: the 180 s
    window spans ~0.45 req/s and its average (~0.49 req/s) is far below the SLA point
    (~0.63 req/s). In the study the window spans ~0.15 req/s.
  - **Noise:** the 30 s throughput samples moved between ~900 and ~3300 tok/s. 94% of the
    tokens are prompt tokens, and the router counts each prompt (4096 tokens) at its first
    token. So a window's throughput is the number of arrivals in 180 s: ~90 requests at
    0.5 req/s, about ±10%. The repeated baseline measures it.
- The first `akamas start` of each new study again left it RUNNING with no experiment
  (study 24's Airflow DAG issue). A 2 min pause between `create` and `start` worked.

## First run: Poisson, 3.0 req/s in 60 min (2026-09-30)

Study `27-L4-PD-Open-Loop`, finished after its two baselines (exp 3 aborted).

| Exp | Score | Deciding constraint | ITL breaks at |
|---|---|---|---|
| baseline aggregated | 1158.49 | ITL p95 74.7 ms (TTFT p95 3.7 s) | ~0.55 req/s |
| baseline aggregated repeat | 1089.98 (−5.9%) | ITL p95 74.7 ms | ~0.55 req/s |

- The system behaved the same in both runs: the ITL broke at the same rate. A
  recalculation on a 30 s grid gives 1110 and 1121 (1% apart).
- The 5.9% is measurement noise, from two causes:
  - **Arrival count.** The score is the throughput realized in the 180 s window, that is
    the number of arrivals in it. With Poisson at ~0.5 req/s that is ~90 requests ± ~10%,
    and the 30 s samples moved by ±30%. The max over the valid windows picks a lucky
    window, and a small shift of the Akamas samples changes which one.
  - **Ramp slope.** At 0.05 req/s per minute the window spanned 0.15 req/s, 27% of the
    baseline's SLA rate. A 30 s shift of the crossing moved the score by ~5%.
- Changes for the second run: gamma arrivals with smoothness 4, and 2.4 req/s in 120 min.

## Results (second run, 2026-09-30 → 2026-10-01)

Study `27-L4-PD-Open-Loop-Gamma` (id 137fd899-7487-4150-a134-02c6c8f59ac2), finished with
`akamas finish study` after 31 experiments: 15 presets and 16 AKAMAS experiments of 60 (exp 32
aborted at its start). Score = total tokens/s per active GPU. The best configuration was on a
plateau since exp 20, and the remaining budget was ~34 h of node. Afterwards `vllm-pd` was
scaled to 0 replicas.

| Exp | Configuration | Score |
|---|---|---|
| 1 / 2 | baseline aggregated P0D2, bf16, Marlin, vLLM defaults / repeat | 1136.92 / 1191.99 |
| 3 / 4 | P1D1 bf16, prefill Marlin / Humming | 1029.94 / 1306.22 |
| 5 / 6 | P1D1 bf16, prefill Triton default / tuned | 1422.01 / 1436.31 |
| 7 | P1D1 bf16, decode Humming | 1449.93 |
| 8 | **P1D1 fp8** | 1754.57 |
| 9 | P1D1 bf16, FLASH_ATTN | 1290.59 |
| 10 / 11 / 12 / 13 | P2D1 / P1D2 / P2D2 / P3D1, bf16 | 902.06 / 1203.20 / 1401.14 / 674.69 |
| 14 / 15 | P1D1 bf16 batched 4160 / P1D1 fp8 decode seqs 32 | 1394.22 / 1659.32 |
| 16-19 | AKAMAS, P1D1 fp8 | 1671.22 / 1712.48 / 1686.27 / 1739.12 |
| 20 / 21 / 28 | AKAMAS, **P0D1 fp8**, Humming, batched 4655 / 14476 / 16348 | **1974.69** / 1944.15 / 1941.11 |
| 22 / 25 / 26 | AKAMAS, **P0D2 fp8**, Humming, batched 6983 / 14764 / 8659 | **1980.22** / 1855.65 / 1968.08 |
| 23 / 30 | AKAMAS, P0D1 / P0D2 bf16, Marlin | 1014.33 / 1262.25 |
| 24 / 27 | AKAMAS, P1D2 fp8 (decode Marlin / prefill batched 512) | 1075.74 / 1114.11 |
| 29 | AKAMAS, **P2D1 fp8**, prefill Triton, decode Humming | 1132.53 |
| 31 | AKAMAS, P0D2 fp8, Humming, batched 3180 (2 prefill steps per prompt) | 1509.54 |

### Learnings

Valid for this setup only: Qwen3-8B-FP8, vLLM 0.29.0, one g6.12xlarge (4× L4 at 72 W, no
GPU P2P), KV through host memory, 4096 in / 256 out, TTFT p95 ≤ 10 s and ITL p95 ≤ 75 ms.

1. **Verified: the open loop removes the level jump.** Triton tuned against Triton default
   was +54% in study 24 (closed loop) and +1% here. The three P1D1 bf16 presets with Triton
   or Humming on the decode scored 1422-1450.
2. **Verified: noise ~±4%.** The two baselines differ by 4.8%. Five near-identical aggregated
   fp8 configurations scored 1941-1980. The residual noise comes from the window position at
   the ITL limit, not from the system (see "First run").
3. **Verified: the best configuration is aggregated with fp8 KV and Humming,** ~1940-1980
   (+70% against the baseline, +13% against P1D1 fp8). P0D1 and P0D2 give the same score per
   GPU. Without fp8 the aggregated configuration stays at the baseline level (exps 23, 30).
4. **Hypothesis (agrees with exps 1, 2, 20-31): the aggregated score is set by the prefill
   stalls in the ITL.**
   - A prompt that enters a step stops the decode of the other sequences for the full step
     (~1.35 s with Humming).
   - The ITL p95 stays below 75 ms while fewer than ~5% of the steps contain a prefill.
   - `max_num_batched_tokens` ≥ 4096 puts one prompt in one step. A smaller value doubles the
     long steps. A much larger value (~15k) puts several queued prompts in one step (exp 25,
     −6%).
   - Model: R prefill steps/s against (1 − 1.35 R) / 0.037 decode steps/s gives R ≤ ~0.49
     req/s per GPU, measured ~0.45. The same model gives ~1230 for the bf16 baseline,
     measured 1137-1192.
   - Test written before the result: exp 31 (batched 3180, 2 prefill steps per prompt) was
     predicted at 1450-1550 and scored 1509.54.
5. **Verified: P/D capacity per instance** (where the ITL or TTFT limit is reached):
   - prefill with Triton: ~0.9 req/s
   - decode with fp8 KV: **~0.8 req/s** (exp 29: decode ITL p95 at 75 ms with 11-15
     sequences, KV at 50-70%, no preemption)
   - **Retracted:** "decode with fp8 KV ~1.5 req/s". That is where the ITL p50, not the
     p95, reaches 75 ms.
   - The decode is memory-bound. A step reads the weights (~8.8 GB, 37 ms at 1 sequence) and
     the KV of each sequence (~320 MB at 4.3k context, +1.35 ms per sequence). The p95 is
     ~1.4× the p50 because the batch size moves with the arrivals.
   - With these capacities: P1D1 0.40 req/s per GPU (measured 0.40), P2D1 0.27 (0.26), P1D2
     0.30 (~0.25). No P/D topology can reach the aggregated 0.45. The aggregated GPU uses its
     compute (prefill) and its memory bandwidth (decode) at the same time. A P/D GPU uses
     only one of the two.
6. **Verified: P/D gives a better ITL tail, the aggregated gives a better TTFT.** At the same
   rate per GPU on 2 GPUs (exp 22 against exp 8):
   - ITL p99: aggregated 1.4-2.2 s from 0.25 req/s per GPU (the prefill stalls), P1D1
     0.3-0.47 s.
   - TTFT p95: aggregated 1.5 s up to saturation, P1D1 3-15 s, P2D1 2 s.
   - **Hypothesis:** the P/D ITL p99 is one gap per request between the first token (from
     the prefill) and the second (from the decode, after the NIXL transfer and the decode
     queue, ~0.5 s). One gap in 256 tokens is in the p99 and not in the p95.
7. **Verified: FLASH_ATTN is 10% below FLASHINFER** on the P1D1 with bf16 decode (exp 9).
   The kernel microbenchmark did not show it because its decode test used ~256-token prompts.
8. **Verified: prefill kernels on P1D1 bf16:** Marlin 1030, Humming 1306, Triton 1422-1436.
   The tuned Triton configs give +1%.
9. **Verified: the two scheduler presets do not help.** Batched 4160 on the prefill: −3%
   (1394). Decode `max_num_seqs` 32 with fp8: −5% (1659), inside the noise.
10. **Design error, carried over from study 24:** `vllm_decode.linear_backend` excludes
    Triton because Triton is slower on a pure decode. In P0 the decode instance is aggregated
    and also does the prefill, so the aggregated configuration could not use Triton. See the
    notes for the next study.
11. **Optimizer:** the AKAMAS step found the aggregated fp8 configuration at its fifth
    experiment (exp 20). Of the next 10 experiments, 3 went to dominated P/D topologies
    (P1D2 twice, P2D1) and 2 to the aggregated configuration with bf16 KV, because the GP
    does not transfer the effect of fp8 and of the kernel across topologies. The GP analysis is in
    `scikit-optimize/benchmarks/cocabo/README.md` (branch `eval/hyperparameter-fitting`).

## Create and start (from the toolbox)

```bash
cd /work/vllm-benchmark && git pull
cd studies/27-l4-pd-open-loop/akamas
akamas create system system.yaml
akamas create component components/ vLLM_Benchmark_27_L4_PD_Open_Loop
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_27_L4_PD_Open_Loop
akamas create workflow 27-L4-PD-Open-Loop-Smoke-Workflow.yaml
akamas create workflow 27-L4-PD-Open-Loop-Workflow.yaml
akamas create study 27-L4-PD-Open-Loop-Smoke.yaml
akamas start study 27-L4-PD-Open-Loop-Smoke
# after the check:
akamas delete study 27-L4-PD-Open-Loop-Smoke
akamas create study 27-L4-PD-Open-Loop.yaml       # study name: 27-L4-PD-Open-Loop-Gamma
sleep 120
akamas start study 27-L4-PD-Open-Loop-Gamma
```

After a change of the workflow file (for example the RunTest timeout), run
`akamas update workflow 27-L4-PD-Open-Loop-Workflow.yaml` before the next study.

If `akamas start` leaves the study RUNNING with no experiment (Airflow DAG timeout, study
24), delete the study, create it again, wait a minute, and start it again.

## Other users of the cluster

As study 24: the pods pin `node-role: llm-serving-l4`, and the telemetry scopes the vLLM,
container and GPU series to this study's pod names. The second GPU node
(`llm-serving-g7-4500`) runs studies 25-26 in namespace `gpu-sharing`.

## Notes for the next study (2026-10-01)

From the optimizer engine run locally on this study's inputs (engine 1.9.6 = deployed 1.9.7
code, skopt 0.9.2rc39, one thread).

- **Baseline:** P0D1 (one GPU, vLLM defaults), with every parameter rendered. The baselines of
  this study have `doNotRenderParameters`, so the engine never receives them: it has no point
  with P = 0, and it loses the only repeated configuration (1136.92 / 1191.99), which is the
  data that shows the noise.
- **Locked parameters:** `pd_kv_connector` and `pd_kv_buffer_device` have two categories (Akamas
  rejects a one-value domain) and are fixed by constraints. The engine still receives them: 2 of
  the 17 GP dimensions are constant columns, their length scales are arbitrary, their ARD values
  mean nothing, and their constraints reject candidates in the acquisition step. Next study:
  remove them from `parametersSelection` and set the values in the baseline only (to check: the
  value Akamas renders for a non-selected parameter in the other steps), or write them as
  literals in the template.
- **FLASH_ATTN:** remove it. It scored −10% against FLASHINFER on the bf16-decode P1D1, and on
  sm89 it has no tunable knob (FA2 only, `flash_attn_max_num_splits_for_cuda_graph` acts only on
  FA3).
- **~~One preset: P2D1 with fp8 KV.~~ Done by the optimizer (exp 29): 1132.53.** The
  estimate of ~0.50 req/s per GPU used a decode capacity of ~1.5 req/s, which is wrong (see
  "Results", learning 5). The GP does not transfer the topology effect from bf16 to fp8
  (`kv_cache_dtype` length scale 0.01). P0 and P2 are symmetric around P1 for the GP.
- **Triton for the aggregated configuration (from learning 10).**
  - Add `triton` to `vllm_decode.linear_backend`.
  - Change the tuned-config constraint to
    `vllm_decode.tuned_kernel_configs == "false" || vllm_prefill.linear_backend == "triton" || (pd_topology.pd_prefill_instances == 0 && vllm_decode.linear_backend == "triton")`.
  - Add one preset: P0D1 fp8, `vllm_decode.linear_backend` triton, tuned configs on, batched
    ~6000-8000, the other values as exp 20.
  - Expected: Triton has a 20% shorter prefill step (1.07 s against 1.33 s) and a ~10% slower
    decode step. With the model of learning 4 that gives ≤ ~0.55 req/s per GPU against ≤ ~0.49
    for Humming, so between +0% and +10%.
- **A study with an ITL tail constraint** (ITL p99, or the maximum gap) in place of the ITL
  p95. It asks the latency question of learning 6: P/D against an aggregated configuration
  with small chunks. The aggregated configuration then needs a small
  `max_num_batched_tokens` (many short stalls in place of a few long ones), and the score
  goes down.
- **`--no-async-scheduling` on the prefill (one preset, P/D only).** With async scheduling
  the prefill engine starts step N+1 before it returns the output of step N. The KV copy to
  the host buffer (`save_kv_to_host`, synchronous) then blocks the engine thread until step
  N+1 ends, so a request waits one full extra prefill step when a second request is in the
  queue (`research/vllm/vllm_learnings.md` 6.4-6.5). Sync scheduling on the prefill should
  remove that step from the TTFT. The cost is the lost CPU/GPU overlap, small with ~1 s
  steps. The template must render the flag: `vllm_prefill.async_scheduling` has
  `render: false`, and vLLM accepts `--async-scheduling` or `--no-async-scheduling`, not
  `=false`. Not tested. It changes the TTFT, not the prefill capacity, so it matters for a
  latency study more than for this goal.
- **Engine findings, not changed:**
  - Noise: the fitted noise is `gp.noise_`. skopt sets the WhiteKernel to 0 in `kernel_` after
    the fit on purpose (predictions exclude the noise), so the printed kernel always shows
    `WhiteKernel(noise_level=0)`. **Retracted (2026-10-01):** "the nugget stays at 0". Measured
    noise SD: ~11 tok/s (exp 19 input), ~28 (exp 20), ~31 with the two baselines added, ~36 with
    the baselines and the nugget multistart of `eval/hyperparameter-fitting`. The repeated
    baselines alone give ~39. So the rendered baselines bring the noise to the right value.
  - The topology length scales are not identifiable with ~19 points in 17 dimensions (P 0.39
    → 0.018, D 0.105 → 0.012 between fits).
  - Tried locally, no effect on the fp8 transfer: a lower bound of 0.5 or 1.0 on the
    categorical length scales, and a linear (DotProduct) term added to the Matern (the fit
    shrinks it to ~0). In the data fp8 exists only around P1D1, so no kernel change gives the
    GP evidence for "fp8 adds +X on every topology". The fix is data: fp8 points on other
    topologies.
- **Profiler run before the next study (~half a day of node, outside Akamas, manual deployment
  as `../24-l4-pd-kernels/kernel-bench/`).** `kernel-bench` measured client-side wall time of
  HTTP requests (`time.perf_counter()` around the OpenAI API), so a "prefill step" includes
  HTTP, tokenization, scheduling and the first-token sampling, not only the forward pass. Its
  decode test used ~256-token prompts, so it did not cover long-context decode (this is why
  the FLASH_ATTN −10% was not predicted). A profiler run (vLLM torch profiler:
  `--profiler-config` with `/start_profile` and `/stop_profile`, a few steps only; or `nsys` for
  the CPU timeline too) answers two questions:
  1. **Prefill, now the bottleneck of the best config:** the step time per kernel (linear
     GEMMs, attention, norms, RoPE, sampling, KV copies) and the TFLOP/s each GEMM reaches
     against the L4 roofline at 72 W. The estimate is ~65 TFLOP/s per prompt for triton
     tuned, about half of the capped FP8 peak. It tells if better kernels or more Triton
     tuning can still help.
  2. **Decode at ~4350-token context and ~15 sequences:** attention against weight reads in
     the step (what fp8 or a 4-bit KV can gain), FLASH_ATTN against FLASHINFER, and whether
     the NIXL transfer on the decode side overlaps the compute or blocks it.
  Also visible: GPU idle gaps between steps (CPU scheduling, CUDA graphs not used for some
  batch sizes) and the kernel that really runs. The profiler slows the run: use the ratios
  between kernels, not the absolute times.
