# 29-L4-PD-Open-Loop-Engine-1.9.8

**Status:** RUNNING (created 2026-10-02).
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
3. The third study 29 puts experiment 2 in its own bootstrap step, and experiments 3-15 in a
   second one.

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

## Results

<Filled in when the study stops.>
