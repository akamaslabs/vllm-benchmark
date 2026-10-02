# akamas/ — 28-G7-4500-MIG-Min-Cost

**Created:** 2026-10-02, with the `akamas-study-manager` plugin (0.3.0) conventions, from
study 26's resources (same system shape, components, telemetry, workflow). **On the Akamas
server since 2026-10-02:** system, components, telemetry instance, both workflows and the
calibration study (finished; its first start stayed RUNNING with no experiment, the known
Airflow issue: finished, `akamas delete --force study ...` (`--force` before the
subcommand), recreated, started). The main study is not created yet (plan Task 11).

## What it optimizes

**Does Qwen3-8B-FP8 fit in half a GPU?** The main study minimizes the tenant's estimated
hourly cost (USD/h, AWS on-demand us-east-2: GPU share + CPU + memory of the pod) over the
MIG slice (`none` = whole GPU, or one `1g.16gb` half with a busy neighbour on the other),
the pod's CPU / memory, and six vLLM settings, subject to a fixed open-loop ShareGPT load of
3.3 req/s holding TTFT p95 <= 1500 ms and ITL p95 <= 300 ms (150 s p95, `:max`) with >= 95 %
of the requests completed. A calibration study runs first, with study 27's rate ramp, to
measure each layout's capacity and confirm the target rate. Design: `../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | 3.7.x |
| Optimization pack **GPU** | 1.4.0 (`mig_profile`), installed 2026-09-30 |
| Optimization pack **vLLM** | 1.12.0 (`linear_backend`, `attention_backend`, `kv_cache_dtype`) |
| Kubernetes pack | 1.9.0-dev (installed on the server, checked 2026-10-02; `Kubernetes Container` has `cpu_limit` / `memory_limit`, FileConfigurator confTemplates `${value}m` / `${value}M`) |
| Serving | `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as `qwen3-8b-mig` |
| Load generator | AIPerf 0.11.0, ShareGPT replay, open loop (gamma arrivals, smoothness 4) |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter on the g7 node |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_28_G7_4500_MIG_Min_Cost` |
| `components/vllm.yaml` | `vllm` (vLLM) — both replicas; carries every vLLM parameter |
| `components/vllm_r0.yaml` | `vllm_r0` (vLLM) — the tenant `vllm-0`: goal (`active_gpus`), constraints, windowing |
| `components/vllm_r1.yaml` | `vllm_r1` (vLLM) — the neighbour `vllm-1` (empty with `none`), KPI only |
| `components/gpu0.yaml` | `gpu0` (GPU) — carries `mig_profile` |
| `components/container.yaml` | `container` (Kubernetes Container, pod `vllm-0`) — carries `cpu_limit`, `memory_limit`; cost metrics |
| `components/cluster.yaml`, `cluster_loadtest.yaml`, `container_loadtest.yaml` | node and load-generator views (Kubernetes pack) |
| `telemetry/prometheus.yaml` | `Prometheus_28_G7_4500_MIG_Min_Cost`: study 26's 117 metrics; `time_to_first_token_p95` / `inter_token_latency_p95` over `[150s]` (see its header) |
| `28-G7-4500-MIG-Min-Cost-Workflow.yaml` | `Write config` -> `Apply config` (60 m) -> `RunTest` (45 m, fixed load), scripts in `../k8s/` |
| `28-G7-4500-MIG-Min-Cost-Calibration-Workflow.yaml` | the same, `RunTest` with `RT_MODE=ramp` (65 m) |
| `28-G7-4500-MIG-Min-Cost.yaml` | study `28-G7-4500-MIG-Min-Cost` |
| `28-G7-4500-MIG-Min-Cost-Calibration.yaml` | study `28-G7-4500-MIG-Min-Cost-Calibration` |
| `check_offline.py` | offline checks against the repo rules and the local pack checkouts |

**Steps, main study:** `baseline` (none, 7000 m / 28000 MB, bf16), `half GPU bf16`,
`half GPU fp8`, `half GPU fp8 lean` (2000 m / 8500 MB), `optimize` (40 AKAMAS). Every preset
renders every parameter (gpu_memory_utilization 0.90, max_num_seqs 256,
max_num_batched_tokens 2048, linear_backend auto, attention_backend FLASHINFER).
**Calibration:** `baseline` and `half GPU bf16`, no optimize step.
**parameterConstraints:** FLASH_ATTN only with `kv_cache_dtype` auto (both studies).
**Placeholders left:** none (host `toolbox`, user `akamas`, key `/home/akamas/.ssh/id_rsa`,
as every study here). **From the kernel probe (2026-10-02):** `linear_backend`
[auto, cutlass], `attention_backend` [FLASHINFER, FLASH_ATTN, TRITON_ATTN], `memory_limit`
>= 8500 MB (`../README.md`, "Kernel probe").

**To check at `akamas create study`:** numeric constants in the goal formula. The 3.7 docs
neither show nor exclude them (they are accepted in `parameterConstraints`, studies
20-27). If the goal is rejected, fall back to telemetry-side scaling: three cost metrics
whose queries carry the prices, and a goal that sums them.

## Validation

Offline, 2026-10-02: `python3 check_offline.py` (component names, every
`params.env.template` token in `parametersSelection`, domains and categories inside the
local packs' (vLLM 1.12.0, GPU 1.4.0, Kubernetes 1.9.0-dev), step names, every preset
rendering every parameter, <= 8 KPIs, FLASH_ATTN constraint present iff FLASH_ATTN is in
the domain, no underscore in telemetry placeholders) -> 0 failures. Metrics referenced by
goal, constraints, windowing and KPIs: all bound to the components' types and produced by
the telemetry instance. Not yet validated on the Akamas server.

## Setup & run

From the toolbox, in `/work/vllm-benchmark` (after the user's go-ahead and a `git pull`).
Calibration first; the main study only after the target rate is decided (plan Task 10).

```bash
A=studies/28-g7-4500-mig-min-cost-fixed-load/akamas

# Every file carries kind: (and system: where system-scoped), so the repo rule
# (.claude/rules/akamas-yaml.md) is the -f form, one file at a time in dependency order.
akamas create -f $A/system.yaml
for c in vllm vllm_r0 vllm_r1 gpu0 container cluster cluster_loadtest container_loadtest; do
  akamas create -f $A/components/$c.yaml
done
akamas create -f $A/telemetry/prometheus.yaml
akamas create -f $A/28-G7-4500-MIG-Min-Cost-Calibration-Workflow.yaml
akamas create -f $A/28-G7-4500-MIG-Min-Cost-Workflow.yaml
akamas create -f $A/28-G7-4500-MIG-Min-Cost-Calibration.yaml
sleep 120   # studies 24/27: a start right after create can leave the study RUNNING with no experiment
akamas start study "28-G7-4500-MIG-Min-Cost-Calibration"

# After the calibration (export it first):
akamas export study "28-G7-4500-MIG-Min-Cost-Calibration" /tmp/28-calibration-export.tar.gz
akamas create -f $A/28-G7-4500-MIG-Min-Cost.yaml
sleep 120
akamas start study "28-G7-4500-MIG-Min-Cost"
```

`akamas create -f $A/components/` creates the eight components in one call. Do not run
`-f` on the whole folder: it would create both studies at once, before the calibration has
decided the target rate.

If a start leaves a study RUNNING with no experiment after ~5 min, delete it, create it
again, wait a minute and start it again. After a change of a workflow file, delete and
recreate the workflow before the next study (`akamas delete workflow <name>`, then `akamas
create workflow <file>`): the 3.7 CLI has no `update workflow` (checked with `akamas update
--help` on the toolbox, 2026-10-02: experiment, password, study, trial, user, workspace).
Monitoring: `akamas list experiment "<study>"`,
`akamas log --dump -s "<study>" -e <n> -l ERROR,WARN` (never `-d`: it prints the token).
At the end: `akamas export study "28-G7-4500-MIG-Min-Cost"
studies/28-g7-4500-mig-min-cost-fixed-load/results/export.tar.gz`, then the `study-recap`
skill.
