# 27-L4-PD-Open-Loop

**Status:** second run `27-L4-PD-Open-Loop-Gamma` (gamma arrivals, 2.4 req/s in 120 min), started 2026-09-30. The first run `27-L4-PD-Open-Loop` (Poisson, 3.0 req/s in 60 min; study id a1e35c1b-9d3e-40b7-ada2-71e850445b86) was finished after its two baselines: see "First run: Poisson".
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
- **One preset: P2D1 with fp8 KV.** Expected ~0.50 req/s per GPU against 0.405 for the best
  (P1D1 fp8). The GP does not transfer the topology effect from bf16 to fp8 (`kv_cache_dtype`
  length scale 0.01), so without an fp8 point outside P1D1 it predicts the prior mean (~1350 ±
  290) for every other fp8 topology. P0 and P2 are symmetric around P1 for the GP: P0D1 fp8
  and P2D1 fp8 get the same prediction and the same EI.
- **Engine findings, not changed:** the fitted nugget stays at its lower bound (noise 0) also
  with the two baselines added and with the nugget multistart of `eval/hyperparameter-fitting`.
  The topology length scales are not identifiable with ~19 points in 17 dimensions (P 0.39 →
  0.018, D 0.105 → 0.012 between fits). Candidates: a nugget floor (~2-4% of the normalized
  range), fewer dimensions, a kernel that separates topology and KV dtype.
