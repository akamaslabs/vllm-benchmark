# 27-L4-PD-Open-Loop

**Status:** PREPARED (2026-09-30). Not created on Akamas yet. Run the smoke study first.
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
| Load | closed loop, `--concurrency 2,4,8,16,24,32`, 600 s each | open loop, `--request-rate 3.0 --request-rate-ramp-duration 3600`, Poisson, seed 18 |
| End of the test | after 60 min | watchdog: TTFT p95 (150 s) > 20 s or ITL p95 (150 s) > 225 ms for 120 s, or 60 min |
| Queue constraints | `num_requests_waiting <= 1` on each role | removed |
| Steps | baseline, 11 presets, 60 AKAMAS | baseline twice, the same 11 presets, 2 scheduler presets, 60 AKAMAS |
| Experiment length | ~65 min | ~25-55 min (setup ~10 min, then the ramp until the watchdog) |

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
- **R = 3.0 req/s** is above every config in the search space (2P2D with fp8 KV is
  estimated at ~1.8 req/s). The watchdog ends the ramp, so the true peak is always measured.
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
akamas create study 27-L4-PD-Open-Loop.yaml
akamas start study 27-L4-PD-Open-Loop
```

If `akamas start` leaves the study RUNNING with no experiment (Airflow DAG timeout, study
24), delete the study, create it again, wait a minute, and start it again.

## Other users of the cluster

As study 24: the pods pin `node-role: llm-serving-l4`, and the telemetry scopes the vLLM,
container and GPU series to this study's pod names. The second GPU node
(`llm-serving-g7-4500`) runs studies 25-26 in namespace `gpu-sharing`.
