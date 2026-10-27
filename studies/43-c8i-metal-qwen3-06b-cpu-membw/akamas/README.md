# Study 39 — Akamas resources

Created 2026-10-16 with the `akamas-study-manager` plugin, modeled on study 30's resources
minus every GPU part. Design and reasons: [`../README.md`](../README.md).

## What it optimizes

Total tokens/s (prompt + generation, from vLLM's own Prometheus counters) of
`Qwen/Qwen3-0.6B` served by the **vLLM CPU backend** (`vllm/vllm-openai-cpu:v0.31.0`, bf16,
thinking off, served as `qwen3-06b`) on one c8i.12xlarge, under TTFT p95 <= 3000 ms and
TPOT p95 <= 100 ms (150 s p95s, read with `:max` over the 6-sample scored window).

## Versions

| Item | Version |
|---|---|
| Akamas | 4.1.x (namespace `akamas-41`) |
| vLLM optimization pack | 1.14.0, INSTALLED (reference engine vLLM 0.29.0; flags re-checked against 0.31.0) |
| Kubernetes optimization pack | 1.9.0, INSTALLED |
| vLLM | 0.31.0, image `vllm/vllm-openai-cpu:v0.31.0` |
| Model | `Qwen/Qwen3-0.6B`, bf16 |
| Telemetry | Prometheus (kube-prometheus-stack, in-cluster) |

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_43_C8i_Metal_Qwen3_06B_CPU_Membw` |
| `components/vllm.yaml` | component (`vLLM`) | `vllm` — pod `^vllm-0$`, model `qwen3-06b`; every tuned parameter |
| `components/container.yaml` | component (`Kubernetes Container`) | `container` — vLLM container, fixed 44 vCPU / 48 GiB |
| `components/container_loadtest.yaml` | component (`Kubernetes Container`) | `container_loadtest` — AIPerf Job `aiperf-cpu` |
| `components/cluster.yaml` | component (`Kubernetes Cluster`) | `cluster` — node role `llm-serving-cpu` |
| `components/cluster_loadtest.yaml` | component (`Kubernetes Cluster`) | `cluster_loadtest` — node role `system-m8a` |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_43_C8i_Metal_Qwen3_06B_CPU_Membw` |
| `43-C8i-Metal-Qwen3-06B-CPU-Membw-Workflow.yaml` | workflow | Write config → Apply config → RunTest |
| `43-C8i-Metal-Qwen3-06B-CPU-Membw.yaml` | study | main study |
| `gen_study.py` | — | generates both study manifests (keeps them in sync) |
| `check_offline.py` | — | offline checks against the repo rules and the pack sources |

Steps of the main study: `baseline` (vLLM 0.31.0 CPU defaults), `baseline repeat` (noise),
presets `kv fp8`, `no prefix cache`, `small kv budget` (gpu_memory_utilization 0.15), then
`optimize` (AKAMAS, 0 init experiments, 60 experiments, stop after 20 failures).

## Calibration (done 2026-10-16)

- Main workflow: `RT_RATE=4 RT_RAMP_S=4000`.
  From the probe (`../probe/results/summary.txt`): baseline knee 1-2 req/s (TPOT binds),
  saturation ~2.5 req/s.
- `vllm.gpu_memory_utilization` `[0.15, 0.90]`, baseline 0.90: every edge starts (probe phase A).

## Checks

```bash
python3 gen_study.py                    # regenerate both study manifests
python3 check_offline.py --packs <dir with vllm/ and kubernetes/ packs at the installed versions>
```

## Setup & run (from the toolbox, after `akamas login`)

Sync the repo to `/work/vllm-benchmark` in the toolbox first (only after confirmation), and
provision the node group (`../infra/`). Then, from `/work/vllm-benchmark/studies/43-c8i-metal-qwen3-06b-cpu-membw/akamas`:

```bash
akamas list optimization-pack           # confirm vLLM 1.14.0 and Kubernetes are installed

S=vLLM_Benchmark_43_C8i_Metal_Qwen3_06B_CPU_Membw
akamas create system             system.yaml
akamas create component          components/vllm.yaml               "$S"
akamas create component          components/container.yaml          "$S"
akamas create component          components/container_loadtest.yaml "$S"
akamas create component          components/cluster.yaml            "$S"
akamas create component          components/cluster_loadtest.yaml   "$S"
akamas create telemetry-instance telemetry/prometheus.yaml          "$S"

# Main study (no Smoke study: study 39's calibration, 0 -> 4 req/s over 4000 s, holds)
akamas create workflow 43-C8i-Metal-Qwen3-06B-CPU-Membw-Workflow.yaml
akamas create study    43-C8i-Metal-Qwen3-06B-CPU-Membw.yaml
akamas start study "43-C8i-Metal-Qwen3-06B-CPU-Membw"
```

Bulk alternative (every file carries `kind:` and `system:`): `akamas create -f system.yaml`,
then `akamas create -f components/`, `akamas create -f telemetry/`, then the workflow and
the study files one by one, in that order. Akamas resolves names at creation time.

A workflow change on 4.1 means `akamas delete workflow <name>` + `akamas create workflow
<file>` (no update verb; delete the studies that use it first). On a study that has
already run, only the `goal` can be changed (`akamas update study`); a change to the
parameters, steps or windowing needs a new study.
