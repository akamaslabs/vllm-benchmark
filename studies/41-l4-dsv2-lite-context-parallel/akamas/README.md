# Study 41 — Akamas resources

Created 2026-10-22 with the `akamas-study-manager` plugin, modeled on study 39's resources
(structure, workflow, scoring) with study 16's per-GPU components (`gpu0`..`gpu3`). Design
and reasons: [`../README.md`](../README.md).

Layout note: the plugin's default layout (`system/`, `scripts/`, `templates/`) is not used;
this repo's study layout is (`akamas/{system.yaml,components/,telemetry/}`, scripts and
templates under `../k8s/`), as in every other study here.

## What it optimizes

Generation tokens/s (from vLLM's own Prometheus counters, summed over DP engines; with a fixed
256-token reply it is completed turns/s x 256 — see ../README.md for why not total tokens/s) of `deepseek-ai/DeepSeek-V2-Lite-Chat` (MLA + MoE, bf16, served as `dsv2-lite`,
`vllm/vllm-openai:v0.32.0`) on the four L4s of one g6.12xlarge, under TTFT p95 <= 10000 ms and
TPOT p95 <= 50 ms (150 s p95s, read with `:max` over the 6-sample scored window). The search is
over the parallel layout (TP / DP / DCP / PCP / EP, always on 4 GPUs) and four engine settings.

## Versions

| Item | Version |
|---|---|
| Akamas | 4.1.x (namespace `akamas-41`) |
| vLLM optimization pack | 1.14.0, INSTALLED (reference engine vLLM 0.29.0; every flag re-checked against 0.32.0) |
| Kubernetes optimization pack | 1.9.0, INSTALLED |
| GPU optimization pack | 1.4.0, INSTALLED (component type `GPU`) |
| vLLM | 0.32.0, image `vllm/vllm-openai:v0.32.0` |
| Model | `deepseek-ai/DeepSeek-V2-Lite-Chat`, bf16 |
| Load generator | AIPerf 0.11.0 (Job on the system node) |
| Telemetry | Prometheus (kube-prometheus-stack, in-cluster) + dcgm-exporter on the L4 node |

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_41_L4_DSV2_Lite_CP` |
| `components/vllm.yaml` | component (`vLLM`) | `vllm` — pod `^vllm-dsv2-0$`, model `dsv2-lite`; every tuned parameter |
| `components/container.yaml` | component (`Kubernetes Container`) | `container` — the vLLM container (4 GPUs, fixed CPU/RAM) |
| `components/container_loadtest.yaml` | component (`Kubernetes Container`) | `container_loadtest` — AIPerf Job `aiperf-dsv2` |
| `components/cluster.yaml` | component (`Kubernetes Cluster`) | `cluster` — node role `llm-serving-l4` |
| `components/cluster_loadtest.yaml` | component (`Kubernetes Cluster`) | `cluster_loadtest` — node role `system-m8a` |
| `components/gpu0.yaml` .. `gpu3.yaml` | component (`GPU`) | `gpu0`..`gpu3` — one per physical L4 (`gpu: "0".."3"`, `noderole: llm-serving-l4`) |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_41_L4_DSV2_Lite_CP` (107 metrics) |
| `41-L4-DSV2-Lite-Context-Parallel-Workflow.yaml` | workflow | Write config → Apply config → RunTest (users 1 → 96 over 3600 s, TODO calibrate) |
| `41-L4-DSV2-Lite-Context-Parallel-Smoke-Workflow.yaml` | workflow | same, steeper ramp (1 → 96 over 1800 s) for the calibration run |
| `41-L4-DSV2-Lite-Context-Parallel.yaml` | study | main study |
| `41-L4-DSV2-Lite-Context-Parallel-Smoke.yaml` | study | calibration: baseline + `tp4 dcp4`, one trial each |
| `gen_study.py` | — | generates both study manifests (keeps them in sync) |
| `check_offline.py` | — | offline checks against the repo rules, the pack sources and the real renderer |

Steps of the main study: `baseline` (TP4, vLLM 0.32.0 defaults, gmu 0.90; 2 trials), presets
`tp4 dcp2`, `tp4 dcp4`, `tp4 dcp4 kv fp8`, `tp1 pcp4 ep`, `tp2 pcp2 ep`, `dp4 ep`, then `optimize`
(AKAMAS, 0 init experiments, 60 experiments, stop after 20 failures). 13 layouts are feasible
under the parameterConstraints (`check_offline.py` enumerates them and checks that
`../k8s/render_statefulset.sh` accepts exactly the same ones).

**KPIs (8):** Generation throughput, TTFT P95 150s, TPOT P95 150s, Completed requests, KV cache
usage (max), Preemptions, Waiting requests (max), Prefix cache hit rate.

No placeholders are left in these files: hosts, key path and Prometheus address are the
toolbox / in-cluster values every 4.1 study here uses. The open calibration values
(`RT_USERS` / `RT_RAMP_S` in the two workflows) are marked `TODO(calibrate)`.

## Checks

```bash
python3 gen_study.py                    # regenerate both study manifests
python3 check_offline.py --packs <dir with vllm/ (1.14.0 = origin/develop), nvidia-gpu/ and kubernetes/>
```

## Setup & run (from the toolbox, after `akamas login`) — only after the user confirms

Sync the repo to `/work/vllm-benchmark` in the toolbox first, scale the node group up in
us-east-2a and apply the Kubernetes layer (`../infra/`). Then, from
`/work/vllm-benchmark/studies/41-l4-dsv2-lite-context-parallel/akamas`:

```bash
akamas list optimization-pack           # confirm vLLM 1.14.0, Kubernetes 1.9.0 and GPU 1.4.0 are installed

S=vLLM_Benchmark_41_L4_DSV2_Lite_CP
akamas create system             system.yaml
akamas create component          components/vllm.yaml               "$S"
akamas create component          components/container.yaml          "$S"
akamas create component          components/container_loadtest.yaml "$S"
akamas create component          components/cluster.yaml            "$S"
akamas create component          components/cluster_loadtest.yaml   "$S"
akamas create component          components/gpu0.yaml               "$S"
akamas create component          components/gpu1.yaml               "$S"
akamas create component          components/gpu2.yaml               "$S"
akamas create component          components/gpu3.yaml               "$S"
akamas create telemetry-instance telemetry/prometheus.yaml          "$S"

# 1. Calibration (Smoke study: baseline + tp4 dcp4 on the steeper ramp).
akamas create workflow 41-L4-DSV2-Lite-Context-Parallel-Smoke-Workflow.yaml
akamas create study    41-L4-DSV2-Lite-Context-Parallel-Smoke.yaml
akamas start study "41-L4-DSV2-Lite-Context-Parallel-Smoke"
# ... read both knees, set RT_USERS / RT_RAMP_S in the main workflow file, then:
akamas delete study "41-L4-DSV2-Lite-Context-Parallel-Smoke"

# 2. Main study
akamas create workflow 41-L4-DSV2-Lite-Context-Parallel-Workflow.yaml
akamas create study    41-L4-DSV2-Lite-Context-Parallel.yaml
akamas start study "41-L4-DSV2-Lite-Context-Parallel"
```

Bulk alternative (every file carries `kind:` and `system:`): `akamas create -f system.yaml`,
then `akamas create -f components/`, `akamas create -f telemetry/`, then the workflow and the
study files one by one, in that order. Akamas resolves names at creation time.

A workflow change on 4.1 means `akamas delete workflow <name>` + `akamas create workflow
<file>` (no update verb; delete the studies that use it first). On a study that has already
run, only the `goal` can be changed (`akamas update study`); a change to the parameters,
steps or windowing needs a new study.
