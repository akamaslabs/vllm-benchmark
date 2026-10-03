# 29-L4-PD-Open-Loop-Engine-1.9.8

**Status:** FINISHED (2026-10-03, 31 experiments: 15 imported + 16 AKAMAS). See "Results".
**Needs:** vLLM optimization pack **1.12.0** (as study 27), optimizer engine **1.9.8** on the
optimizer service (see "Engine version on the lab").

> An exact clone of study 27 (`27-L4-PD-Open-Loop-Gamma`), run with a new optimizer engine.
> Same system, telemetry instance, workflow, parameters, constraints, presets and load. Only
> the engine version changes. The question is: do we prefer engine 1.9.8 to 1.9.7?

## What changes against study 27

| Item | Study 27 | Study 29 |
|---|---|---|
| Optimizer engine | 1.9.7 (skopt 0.9.2rc39) | **1.9.8** (skopt 0.9.2rc40, mixed kernel: an overlap kernel on the one-hot categoricals) |
| Study file | `../27-l4-pd-open-loop/akamas/27-L4-PD-Open-Loop.yaml` | `akamas/29-L4-PD-Open-Loop-Engine-1.9.8.yaml`, a copy with a new `name`, `description` and first steps |
| Experiments 1-15 | baseline + 14 presets, run | **imported from study 27**: `baseline` with `from` (exp 1), `bootstrap` (exps 2-15) |
| Budget | stopped after 31 experiments | stopped after 31 experiments (15 imported + 16 AKAMAS) |

Unchanged: system `vLLM_Benchmark_27_L4_PD_Open_Loop`, its telemetry instance, workflow
`27-L4-PD-Open-Loop-Workflow` (it runs `../27-l4-pd-open-loop/k8s/`), the load (AIPerf rate
ramp 0 → 2.4 req/s in 7200 s, gamma arrivals with smoothness 4, seed 18), the watchdog, the
goal, the constraints and the AKAMAS step.

Why the import: running the 15 presets again costs ~13 h of node and changes their scores by
the noise (~±4%), so the two engines would start from different data. With the import, the
AKAMAS step of engine 1.9.8 starts from the data that engine 1.9.7 had at study 27's
experiment 16.

**Check after the first AKAMAS experiment:** the engine input of study 29's experiment 16
(ConfigMap `opt-engine-<study 29 id>-16-*`) must equal the input of study 27's experiment 16
(`opt-engine-137fd899-...-16-*`), except `writeBackApi`. One open point: study 27's two
baselines have `doNotRenderParameters`, and the engine did not receive them. The check shows
if the imported baseline and the bootstrapped baseline repeat behave the same.

History of the study on Akamas (2026-10-02):

1. A first study 29 with the 15 presets run again was created and started at 09:33 UTC, then
   deleted during its first experiment, in favour of the import.
2. A second study 29 (id 5d1df06f-8470-489e-b20e-eb5f81390fd9) imported experiments 2-15 in
   one bootstrap step. Every AKAMAS experiment failed at once with "An optimization of type
   AKAMAS requires at least one assignments" (19 in a few minutes), and the study was
   finished and deleted. Cause (campaign service code and log): a step is tainted when one of
   its parameters is not rendered (`StepLogicHelpers.isStepToBeTainted`), and the optimizer
   ignores every experiment of a tainted step. Experiment 2 has `doNotRenderParameters`, so
   the whole bootstrap step was tainted. The imported baseline is not sent either (as in
   study 23). So the AKAMAS step had no experiments.
3. A third study 29 (id 63aa55c4-518b-4ebd-9ba5-1fc468a324c2) put experiment 2 in its own
   bootstrap step and experiments 3-15 in a second one. Its engine input at experiment 16
   was identical to study 27's (only the order of the constraint list differed). But its
   experiment 16 was not valid: 6 min, CONSTRAINTS_VIOLATED, 753.09. When the first study 29
   was deleted, Akamas stopped the workflow but not the AIPerf job `aiperf-benchmark`
   (namespace `llm-benchmark`). The job ran its ramp on into overload (router TTFT p95 up to
   120 s) until the RunTest of experiment 16 replaced it. The watchdog of experiment 16 reads
   the router p95 over the last 150 s, saw the leftover TTFT above 20 s and ended the test at
   once. The bad point was in the engine data from experiment 17, so the study was finished
   and deleted after experiment 17 (1616.16).
4. The fourth study 29 has the same file as the third. Before its creation the job was
   deleted and the router was idle for 4 minutes.

**After deleting or finishing a study, delete the load job by hand** and wait until the
router is idle for more than 150 s before the next start:
`kubectl --context lab-vllm-bench -n llm-benchmark delete job aiperf-benchmark --ignore-not-found`.

## Engine version on the lab

The optimizer service reads the engine image from the environment variable
`OPTIMIZER_ENGINE_DOCKER_IMAGE_FULL` of the deployment `optimizer` (namespace `akamas`,
context `lab-vllm-bench`). The Helm release `akamas` (chart 1.7.1) sets it from the chart
default `optimizer.engine.image.tag: 1.9.7`. The lab has no repository for its Helm values,
so the change is manual:

```bash
# 2026-10-02: set to 1.9.8 for this study
kubectl --context lab-vllm-bench -n akamas set env deploy/optimizer \
  OPTIMIZER_ENGINE_DOCKER_IMAGE_FULL=485790562880.dkr.ecr.us-east-2.amazonaws.com/akamas/optimizer_engine:1.9.8
# back to the chart default
kubectl --context lab-vllm-bench -n akamas set env deploy/optimizer \
  OPTIMIZER_ENGINE_DOCKER_IMAGE_FULL=485790562880.dkr.ecr.us-east-2.amazonaws.com/akamas/optimizer_engine:1.9.7
```

The change applies to every study on this Akamas instance. A `helm upgrade` of the release
sets the chart value again.

## Comparison criteria (written before the start)

Both studies have the same 15 experiments before the AKAMAS step (study 29 imports them), so
the comparison uses the 16 AKAMAS experiments (exps 16-31) of each study.

| Criterion | Study 27 (engine 1.9.7) |
|---|---|
| Best score | 1980.22 (exp 22) |
| AKAMAS experiments to reach ≥ 1900 | 5 (exp 20) |
| Wasted AKAMAS experiments (score < 1300) | 5 of 16 (exps 23, 24, 27, 29, 30) |
| Failed experiments | 0 |

We prefer 1.9.8 if:

1. its best score is ≥ ~1900 (study 27's best minus the ~±4% noise), **and**
2. it reaches ≥ 1900 in fewer AKAMAS experiments, **or** it wastes fewer AKAMAS experiments,
   **and**
3. it has no more failed experiments.

If 1 or 3 fails, we do not prefer 1.9.8. If 1 and 3 hold and 2 does not, the result is a tie.

Limits: one run for each engine, with ~±4% noise on each score. This is evidence from one
real case, not a statistical certification.

## Create, start and stop (from the toolbox)

```bash
cd /work/vllm-benchmark && git pull
cd studies/29-l4-pd-open-loop-engine-1.9.8/akamas
akamas create study 29-L4-PD-Open-Loop-Engine-1.9.8.yaml
sleep 120
akamas start study 29-L4-PD-Open-Loop-Engine-1.9.8
# after experiment 31:
akamas finish study 29-L4-PD-Open-Loop-Engine-1.9.8
```

Node: the node group `llm-serving-l4` (cluster `vllm-bench`, AWS profile `lab`) goes to
`desiredSize=1` before the start and to 0 after the stop. Then scale `vllm-pd` (namespace
`llm-serving`) to 0 replicas.

## Results (2026-10-02 → 2026-10-03)

Study `29-L4-PD-Open-Loop-Engine-1.9.8` (id 5f8c60d4-fa78-416d-9979-170360049c62), started
2026-10-02 14:44 UTC, finished 2026-10-03 03:39 UTC by the stop script after experiment 31
(experiment 32 aborted). Then the load job was deleted, `vllm-pd` scaled to 0 and the node
group `llm-serving-l4` set to 0. The engine input at experiment 16 was identical to study
27's (checked on the ConfigMaps, saved in `optimizer-engine/study_inputs/`).

| Exp | Study 29 (engine 1.9.8) | Score | Study 27 (engine 1.9.7) | Score |
|---|---|---|---|---|
| 16 | P1D1 fp8, decode Humming | 1711.26 | P1D1 fp8, decode Marlin | 1671.22 |
| 17 | P1D1 fp8, decode Humming | 1670.98 | P1D1 fp8 | 1712.48 |
| 18 | P1D1 fp8, decode Marlin | 1589.49 | P1D1 fp8 | 1686.27 |
| 19 | P1D1 fp8, decode Humming | 1711.08 | P1D1 fp8 | 1739.12 |
| 20 | P0D1 fp8, Humming, batched 16265 | 1738.18 | P0D1 fp8, Humming, batched 4655 | 1974.69 |
| 21 | P0D1 fp8, Marlin, batched 4462 | 1504.69 | P0D1 fp8, Humming, batched 14476 | 1944.15 |
| 22 | P2D2 fp8 | 1550.31 | P0D2 fp8, Humming, batched 6983 | 1980.22 |
| 23 | P2D2 fp8 | 1613.26 | P0D1 bf16, Marlin | 1014.33 |
| 24 | P0D2 fp8, Marlin | ERROR (telemetry) | P1D2 fp8 | 1075.74 |
| 25 | P0D1 fp8, Humming, batched 13677 | 1801.99 | P0D2 fp8, batched 14764 | 1855.65 |
| 26 | **the same configuration as exp 25** | 1801.73 | P0D2 fp8, batched 8659 | 1968.08 |
| 27 | P0D1 fp8, Humming, batched 16315 | **1864.02** | P1D2 fp8 | 1114.11 |
| 28 | P1D2 fp8 | 1122.78 | P0D1 fp8, batched 16348 | 1941.11 |
| 29 | P0D1 bf16, FLASH_ATTN, batched 16342 | 1483.58 | P2D1 fp8 | 1132.53 |
| 30 | P3D1 fp8 | 842.91 | P0D2 bf16, Marlin | 1262.25 |
| 31 | P0D2 fp8, Marlin, batched 5740 | 1509.75 | P0D2 fp8, batched 3180 | 1509.54 |

Experiment 24 failed outside the engine: the Akamas `telemetry` pod was OOMKilled at
2026-10-02 22:59:19 UTC (10 restarts in its life), and the metric collection got
"Connection refused".

### Criteria (fixed before the start)

| Criterion | Study 27 (1.9.7) | Study 29 (1.9.8) |
|---|---|---|
| Best score | 1980.22 | 1864.02 (−5.9%) |
| AKAMAS experiments to reach ≥ 1900 | 5 | never |
| Wasted AKAMAS experiments (< 1300) | 5 of 16 | 2 of 16 |
| Failed experiments | 0 | 1 (telemetry OOM, not the engine) |
| Mean / median of the 16 scores | 1599 / 1699 | 1568 / 1613 |
| Experiments ≥ 1800 | 6 | 3 |

**Verdict by the rule written before the start: we do not prefer 1.9.8.** Criterion 1 fails
(1864 < ~1900). Criterion 2 holds on wasted experiments (2 against 5). Criterion 3 fails
on the count, but the failure is an infrastructure fault, not an engine fault.

### What the engines did differently

- **The same categorical path.** Both engines ran four P1D1 fp8 experiments, then found the
  aggregated fp8 configuration at experiment 20.
- **1.9.8 kept the numeric parameters at the upper bounds.** In its 4 aggregated fp8
  experiments with Humming, `max_num_batched_tokens` was 13677-16315, `max_num_seqs`
  511-512 and decode `gpu_memory_utilization` 0.92. Engine 1.9.7 spread its aggregated
  experiments over batched 4655-16348 and seqs 70-425.
- **The corner does not explain the gap by itself.** Study 27's experiment 28 was near the
  same corner (batched 16348, seqs 425, gmu 0.92) and scored 1941.11, against 1801-1864 here.
  **Hypothesis:** part of the gap is a difference between runs, not between engines. Study
  29 ran on another EC2 instance (launched 2026-10-02), and study 26 measured a ~5% drift of
  the same baseline with the GPU temperature. The imported presets cannot show this offset,
  because they were not run again. A re-run of one study 27 configuration on the study 29
  node would have measured it.
- **1.9.8 proposed an exact duplicate.** Experiments 25 and 26 have the same values in all
  15 parameters (the corner of the domain). They scored 1801.99 and 1801.73: a repeat within
  0.01%, but one experiment of the budget spent on a known point.
- **1.9.8 explored the categorical space more evenly.** It tried P2D2 fp8 twice (never tried
  by 1.9.7, 1550-1613, as the capacity model predicts), P3D1 fp8, P1D2 fp8 once, and
  FLASH_ATTN with bf16. It wasted fewer experiments below 1300.

Limits: one run for each engine, ~±4% noise on a single score, and a possible offset
between the two nodes (see above). The best-score gap (5.9%) is just above the noise. The
pattern at the bounds and the duplicate are structural and do not depend on the noise.
Next time: run one known configuration (for example study 27's experiment 20) again as the
first experiment, to measure the node offset.
