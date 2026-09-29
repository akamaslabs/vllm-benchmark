# akamas/ — 25-G7-4500-GPU-Sharing-Goodput

**Created:** 2026-09-29, with the `akamas-study-manager` plugin (create mode), from study
17's resources. **Not created on the Akamas server yet** — validated offline only (see
"Validation").

## What it optimizes

Maximize aggregate `vllm.prefill_token_throughput + vllm.decode_token_throughput`
(both replicas summed) under TTFT p95 <= 1500 ms and ITL p95 <= 300 ms, `stability`
windowing, by choosing how one NVIDIA RTX PRO 4500 Blackwell is shared
(`gpu0.sharing_mode`: exclusive / mig / time_slicing / mps) together with four vLLM
parameters. Full design and phase 0 results: `../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | 3.7.x |
| Optimization pack **GPU** | **1.3.0** — required for `sharing_mode`; built locally (`nvidia-gpu` repo, branch `feature/gpu-sharing-mode`, commit `74d0fd2`), **not installed** (server has 1.2.0) |
| Optimization pack **vLLM** | 1.12.0 (installed) |
| Kubernetes pack | the installed build (component types `Kubernetes Container`, `Kubernetes Cluster`) |
| Serving | `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-4B-Instruct-2507-FP8` served as `qwen3-4b` |
| Load generator | AIPerf 0.11.0, ShareGPT replay |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter 4.6.0-4.8.3 |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_25_G7_4500_GPU_Sharing` |
| `components/vllm.yaml` | `vllm` (vLLM) — both replicas, carries the vLLM parameters; the only component goal/constraints/windowing/KPIs read |
| `components/vllm_r0.yaml`, `vllm_r1.yaml` | per-replica views (`vllm_r1` is empty in exclusive experiments) |
| `components/gpu0.yaml` | `gpu0` (GPU) — carries `sharing_mode`; GPU queries filter on `modelName`, never on pod |
| `components/cluster.yaml`, `container.yaml` | GPU node and vLLM containers (Kubernetes pack) |
| `components/cluster_loadtest.yaml`, `container_loadtest.yaml` | load-generator node and AIPerf container |
| `telemetry/prometheus.yaml` | telemetry instance `Prometheus_25_G7_4500_GPU_Sharing`, 117 metrics (study 17's catalog, GPU and cAdvisor queries rewritten after the 2026-09-29 audit; rules in the file header) |
| `25-G7-4500-GPU-Sharing-Goodput-Workflow.yaml` | `Write config` (FileConfigurator: `k8s/params.env.template` -> `k8s/params.env`), `Apply config` (`k8s/apply_config.sh`, 60 m), `RunTest` (`k8s/run_test_goodput.sh`, 105 m) |
| `25-G7-4500-GPU-Sharing-Goodput.yaml` | study `25-G7-4500-GPU-Sharing-Goodput` |

Layout note: this repo keeps the Akamas files flat under `akamas/` and the scripts and
templates under `../k8s/` (as every other study here), not the plugin's default
`system/`, `scripts/`, `templates/` tree. The workflow's paths point at the toolbox
checkout `/work/vllm-benchmark/studies/25-g7-4500-gpu-sharing-goodput/k8s/`.

**Steps:** `baseline` (exclusive) + `Preset MIG` + `Preset time slicing` + `Preset MPS`,
all at vLLM 0.90 / 256 / 2048 / 1 and every parameter rendered; then `optimize`
(AKAMAS optimizer, 30 experiments, no extra init experiments, abort after 8 failures).
**parameterConstraints:** none — the domains were chosen so every combination is valid
(`max_num_batched_tokens` >= 1024 > 768 >= `max_num_seqs`; the memory bounds are in the
domains themselves).

**Placeholders left:** none. Host `toolbox`, user `akamas` and key path
`/home/akamas/.ssh/id_rsa` are the ones every other study's workflow uses (a path inside
the toolbox pod, not a key in this repo).

## Validation

Offline, 2026-09-29: every component type, parameter (and domain subset), metric
referenced by goal/constraints/windowing/KPIs, telemetry metric, `$KEY$` placeholder
and `${component.param}` template token resolved against vLLM 1.12.0 and GPU 1.3.0
sources and the server's Kubernetes component types; step names match
`^[a-zA-Z\s][a-zA-Z0-9_\s]*$`; 8 KPIs (the 3.7 limit). Re-run after the audit against the
installed Kubernetes pack build (`/work/Kubernetes_1-8-0-dev.json` in the toolbox), vLLM
1.12.0 and GPU 1.3.0: no errors. The goal/constraint/KPI queries were also evaluated on
the real `vllm-0`/`vllm-1` series of a toolbox test run (study README, phase 0). **Not
yet** validated by `akamas create` on the server — that waits for the GPU 1.3.0 install
(Administrator login) and the toolbox sync.

## Setup & run

Prerequisites, in order (study README, "Before starting"): GPU pack 1.3.0 installed (Administrator login);
`infra/eks/provision.sh` run (sharing layer, PVCs, Services, ServiceMonitor);
dcgm-exporter re-pointed to `llm-serving-g7-4500`; toolbox checkout updated.

From the toolbox (`kubectl -n akamas exec -it deploy/toolbox -c toolbox -- bash`), in
`/work/vllm-benchmark`:

```bash
A=studies/25-g7-4500-gpu-sharing-goodput/akamas
SYS=vLLM_Benchmark_25_G7_4500_GPU_Sharing

akamas create system             $A/system.yaml
akamas create component          $A/components/vllm.yaml               $SYS
akamas create component          $A/components/vllm_r0.yaml            $SYS
akamas create component          $A/components/vllm_r1.yaml            $SYS
akamas create component          $A/components/gpu0.yaml               $SYS
akamas create component          $A/components/cluster.yaml            $SYS
akamas create component          $A/components/container.yaml          $SYS
akamas create component          $A/components/container_loadtest.yaml $SYS
akamas create component          $A/components/cluster_loadtest.yaml   $SYS
akamas create telemetry-instance $A/telemetry/prometheus.yaml          $SYS
akamas create workflow           $A/25-G7-4500-GPU-Sharing-Goodput-Workflow.yaml
akamas create study              $A/25-G7-4500-GPU-Sharing-Goodput.yaml

akamas start study "25-G7-4500-GPU-Sharing-Goodput"
```

Bulk alternative (every file carries `kind:` and `system:`; same dependency order is
resolved by name at creation time):

```bash
akamas create -f studies/25-g7-4500-gpu-sharing-goodput/akamas/
akamas start study "25-G7-4500-GPU-Sharing-Goodput"
```

Monitoring: `akamas list experiment "25-G7-4500-GPU-Sharing-Goodput"`,
`akamas log --study "25-G7-4500-GPU-Sharing-Goodput" --log-level INFO` (never `-d`), and
`akamas export study "25-G7-4500-GPU-Sharing-Goodput" studies/25-g7-4500-gpu-sharing-goodput/results/export.tar.gz`
at the end. Only the `goal` can be edited on a running study (`akamas update study`);
changing parameters, windowing or steps needs a new study.
