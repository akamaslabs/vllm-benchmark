# akamas/ — 31-L40S-Gemma4-TPS-Thinking

**Created:** 2026-10-08, with the `akamas-study-manager` plugin (0.3.0, modify mode), from
study 30's resources (`../../30-l40s-gemma4-26b-tps/akamas/`) as they were accepted by the
Akamas 4.1 server on 2026-10-06. **Target: Akamas Studio 4.1** (`akamas41.lab.akamas.io`,
namespace `akamas-41`). Not created on the server yet.

## What it optimizes

**How many tokens per second can one L40S serve with Gemma 4 26B-A4B in thinking mode within
a chat SLA, and does the best vLLM configuration change once the model reasons?** The same
study as study 30: vLLM's total token throughput (prompt + generation tokens/s, reasoning
tokens included) of `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` on one NVIDIA L40S, over the
same 13 vLLM settings, subject to TTFT p95 <= 1500 ms and ITL p95 <= 300 ms (150 s p95,
`:max` over the scored window), with study 27's open-loop AIPerf rate ramp on ShareGPT, a
watchdog past the SLA and the best valid 3-minute window as the score. What differs is outside
this folder: the server runs with `enable_thinking` true, `--reasoning-parser=gemma4` and
`--max-model-len=16384` (`../k8s/01-statefulset_template.yaml`), and the ShareGPT requests
carry no `max_tokens` (`../k8s/05-job_template.yaml`). TTFT is now the time to the first
*reasoning* token. Design: `../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | Studio 4.1.0 (chart 1.9.0-rc14) |
| Optimization pack **vLLM** | 1.12.0 (component type `vLLM`; `total_token_throughput`, `performance_mode`, `optimization_level`, ordinal `block_size`, `spec_method` / `spec_tokens`, `linear_backend`, `spec_decode_*`), as on the 4.1 server for study 30 |
| Optimization pack **GPU** | 1.4.0 (component type `GPU`; only metrics used) |
| Kubernetes pack | 1.9.0 on the 4.1 server (`Kubernetes Container`, `Kubernetes Cluster`, only metrics used) |
| Serving | `vllm/vllm-openai:v0.29.0`, `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` served as `gemma4-26b-l40s-think`, thinking on, reasoning parser `gemma4` |
| Load generator | AIPerf 0.11.0, ShareGPT replay without `max_tokens`, open-loop linear rate ramp (gamma arrivals, smoothness 4, seed 30) |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter on the L40S node |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_31_L40S_Gemma4_TPS_Thinking` |
| `components/vllm.yaml` | `vllm` (vLLM, pod `vllm-0`, model `gemma4-26b-l40s-think`): every tuned parameter, goal, constraints, windowing |
| `components/gpu0.yaml` | `gpu0` (GPU, `gpu="0"`, model `.*L40S.*`) |
| `components/container.yaml` | `container` (Kubernetes Container, pod `vllm-0`) |
| `components/cluster.yaml`, `cluster_loadtest.yaml`, `container_loadtest.yaml` | node and load-generator views (Kubernetes pack) |
| `telemetry/prometheus.yaml` | `Prometheus_31_L40S_Gemma4_TPS_Thinking`: study 30's metrics unchanged (the model filter comes from the `vllm` component) |
| `31-L40S-Gemma4-TPS-Thinking-Workflow.yaml` | `Write config` -> `Apply config` (45 m) -> `RunTest` (130 m, `RT_RATE=4 RT_RAMP_S=6000`, from the smoke run), scripts in `../k8s/` |
| `31-L40S-Gemma4-TPS-Thinking.yaml` | study `31-L40S-Gemma4-TPS-Thinking` |
| `check_offline.py` | offline checks against the repo rules and the local pack checkouts |

**Identical to study 30** (checked by loading both manifests: `goal`, `windowing`,
`parametersSelection`, `parameterConstraints`, `kpis`, `numberOfTrials` are equal, and the
steps but one; the telemetry metrics are equal line by line; the workflow differs only in the
study folder and `RT_RATE`). **Steps:** `baseline` (vLLM 0.29.0 defaults, every parameter
written out), `baseline repeat`, `kv fp8`, `kv fp8 large batch`, `kv fp8 mtp2`, **`study 30
best`** (the one step study 30 does not have, added 2026-10-08: study 30's best configuration,
experiment 28, as `akamas describe study 30-L40S-Gemma4-TPS` reports it), `optimize`
(AKAMAS, 0 init, 60 experiments, `maxFailedExperiments` 20). **parameterConstraints:** `max_num_batched_tokens
>= max_num_seqs`; `spec_method != "none" || spec_tokens == 0` and `spec_method == "none" ||
spec_tokens > 0`. **No Akamas smoke study:** the smoke run is manual
(`../smoke/smoke_manual.sh`), as study 30's turned out to be. **Placeholders left:** none
(host `toolbox`, user `akamas`, key `/home/akamas/.ssh/id_rsa`, Prometheus address, as study
30); `RT_RATE` / `RT_RAMP_S` set from the manual smoke run of 2026-10-08.

## Validation

Offline, 2026-10-08: `python3 check_offline.py` -> 0 failures. Study 30's checks (component
names, template tokens <-> `parametersSelection`, domains inside the local packs vLLM 1.12.0 /
GPU 1.4.0 / Kubernetes 1.9.0-dev, step names, every baseline/preset rendering through the real
`render_statefulset.sh`, <= 8 KPIs, every referenced metric bound and produced, no underscore
in placeholders) plus three for the copy: every component, the telemetry instance and the
study name the same system; every workflow path points at this study's folder; the `vllm`
component's `model` equals the StatefulSet's `--served-model-name`. Mutation-tested (a
component left on study 30's system, a workflow path to study 30's folder, the old model name:
all reported). **Not validated on the Akamas server yet** (pending the user's go-ahead).

## Setup & run

From the 4.1 toolbox (`toolbox-ssh vllm-bench akamas-41`, or prefix each command with
`kubectl -n akamas-41 exec deploy/toolbox -c toolbox --`), in `/work/vllm-benchmark`, after the
user's go-ahead and a `git pull`; study 30 finished first (same GPU node); the workflow's
`RT_RATE` / `RT_RAMP_S` come from the manual smoke run (`../README.md`, "Smoke run").

```bash
A=studies/31-l40s-gemma4-26b-tps-thinking/akamas

# Every file carries kind: (and system: where system-scoped), so the repo rule
# (.claude/rules/akamas-yaml.md) is the -f form, one file at a time in dependency order.
akamas create -f $A/system.yaml
for c in vllm gpu0 container cluster cluster_loadtest container_loadtest; do
  akamas create -f $A/components/$c.yaml
done
akamas create -f $A/telemetry/prometheus.yaml
akamas create -f $A/31-L40S-Gemma4-TPS-Thinking-Workflow.yaml
akamas create -f $A/31-L40S-Gemma4-TPS-Thinking.yaml
sleep 120   # studies 24/27/28: a start right after create can leave the study RUNNING with no experiment
akamas start study "31-L40S-Gemma4-TPS-Thinking"
```

Typed equivalent (one file per call; the typed form never takes a folder): `akamas create
system $A/system.yaml`, then `akamas create component $A/components/<c>.yaml
vLLM_Benchmark_31_L40S_Gemma4_TPS_Thinking` for each of the six components, `akamas create
telemetry-instance $A/telemetry/prometheus.yaml vLLM_Benchmark_31_L40S_Gemma4_TPS_Thinking`,
`akamas create workflow $A/31-L40S-Gemma4-TPS-Thinking-Workflow.yaml`, `akamas create study
$A/31-L40S-Gemma4-TPS-Thinking.yaml`. `akamas create -f $A/` on the whole folder would also
work (one study, one workflow here), but the per-file order above is the one the repo uses.

If RT_RATE / RT_RAMP_S change after the workflow was created: edit
`31-L40S-Gemma4-TPS-Thinking-Workflow.yaml` (command and, if needed, the RunTest `timeout`
above RT_RAMP_S + 1500 s), push, pull, then `akamas delete workflow
31-L40S-Gemma4-TPS-Thinking-Workflow` and create it again before creating the study (no
`update workflow` verb). If a start leaves the study RUNNING with no experiment after ~5 min,
delete it, create it again, wait a minute and start it again.

Monitoring: `akamas list experiment "31-L40S-Gemma4-TPS-Thinking"`,
`akamas log --dump -s "31-L40S-Gemma4-TPS-Thinking" -e <n> -l ERROR,WARN` (never `-d`: it
prints the token). After an `akamas finish study`, delete the load Job as well
(`kubectl -n llm-l40s delete job -l app=aiperf-l40s`, study 29). At the end:
`akamas export study "31-L40S-Gemma4-TPS-Thinking"
studies/31-l40s-gemma4-26b-tps-thinking/results/export.tar.gz`, then the `study-recap` skill.
