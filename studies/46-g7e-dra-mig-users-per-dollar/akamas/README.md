# akamas/ — study 46's Akamas resources

Created 2026-11-03 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.0**
(`akamas-41`). The goal, design and Phase 0 results are in `../README.md`.

## Versions

- **Optimization packs installed on akamas-41 (2026-11-03):** GPU **1.4.0**
  (`mig_profile`), vLLM **1.14.0**, Kubernetes **1.9.0**. Every component type,
  parameter and metric used here was checked against `akamas describe optimization-pack`
  output on that date. All 117 telemetry metrics and all study references resolve.
- **Workload:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as `qwen3-8b-dra`.
- **Load generator:** AIPerf 0.11.0 (`python:3.12-slim` client pods).
- **Telemetry:** Prometheus (kube-prometheus-stack, in-cluster
  `kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090`) and dcgm-exporter
  4.6.0-4.8.3 with `KUBERNETES_ENABLE_DRA=true`.

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_46_G7E_DRA_MIG` |
| `components/gpu0.yaml` | component (GPU) | `gpu0`, carries `mig_profile` |
| `components/vllm.yaml` | component (vLLM) | `vllm`, all replicas; goal, constraints, KPIs, tuned vLLM params |
| `components/vllm_r0..3.yaml` | component (vLLM) | one per replica, for reading only (`vllm_r1..3` are empty when fewer replicas run) |
| `components/container.yaml` | component (Kubernetes Container) | vLLM containers (CPU throttling guard) |
| `components/container_loadtest.yaml` | component (Kubernetes Container) | AIPerf client pods |
| `components/cluster.yaml` / `cluster_loadtest.yaml` | component (Kubernetes Cluster) | GPU node / load node |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_46_G7E_DRA_MIG` (117 metrics, study 26's catalog adapted) |
| `46-G7E-DRA-MIG-Users-Workflow.yaml` | workflow | Write config → Apply config → Ramp → Hold |
| `46-G7E-DRA-MIG-Users-per-Dollar.yaml` | study | `46-G7E-DRA-MIG-Users-per-Dollar` |

### How the goal "users within the SLA" is measured

No pack metric holds "the highest concurrency within the SLA", so the measurement has
two parts.

1. **Ramp** (`../k8s/run_ramp.sh`):
   - Every replica is driven by its own AIPerf client at the same closed-loop
     concurrency, and all replicas step up together.
   - Levels per replica: 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512,
     640, 768, 1024. Each level runs a 20 s warm-up, then 120 s of measurement.
   - At each level the script checks the SLA per replica, client-side: TTFT p95 ≤ 1500 ms,
     ITL p95 ≤ 300 ms, at least 95 % of requests completed.
   - The ramp stops at the first level where any replica fails. The last level that
     passed is written to `k8s/level.current`.
2. **Hold** (`../k8s/run_hold.sh`) keeps every replica at that level for 240 s. The study
   windows on this task only (`trim [1m, 45s]`, `task: Hold`).

Akamas then scores `vllm.num_requests_running + vllm.num_requests_waiting`, the requests
in flight. In a closed loop with no think time, each user always has exactly one request
in flight, so this score equals **N × level**. That is the approved formula
N × min-per-replica passing level: the ramp stops at the first replica that fails, so the
last passing level is the minimum over the replicas.

The SLA constraints in the study are checked on the merged histograms of all replicas,
as a safety net. The per-replica SLA is the one `run_ramp.sh` enforces.

### Parameters, constraints, steps

- `gpu0.mig_profile` ∈ {none, 2g.48gb, 1g.24gb}. The replica count follows from it: 1, 2
  or 4.
- vLLM, the same on every replica:
  - `max_num_seqs` [16, 1024]
  - `max_num_batched_tokens` [1024, 16384]
  - `gpu_memory_utilization` [0.85, 0.95]
  - `kv_cache_dtype` {auto, fp8}
- **`parameterConstraints`: none (to confirm with the user).** vLLM 0.29 runs chunked
  prefill, so `max_num_batched_tokens` may be below `--max-model-len 4096`.
- **`doNotRenderParameters`: none.** Every step renders all five parameters, and
  `apply_config.sh` refuses to run if any is empty.
- **Steps:**
  1. `baseline`: whole GPU, `max_num_seqs` 256, `max_num_batched_tokens` 2048, gmu 0.90,
     KV `auto`.
  2. `MIG half GPU`: same settings on `2g.48gb`.
  3. `MIG quarter GPU`: same settings on `1g.24gb`.
  4. `whole GPU repeat`: drift check.
  5. `optimize`: 25 experiments, 4 init, at most 10 failed.
- **KPIs (8):** TTFT p95, ITL p95, token throughput, KV cache used, preemptions (peak),
  GPU power, vLLM CPU throttled, AIPerf CPU.

## Placeholders

None. Hosts, key path and Prometheus address are the akamas-41 toolbox values already
used by studies 26 and 45 (`toolbox`, user `akamas`, `/home/akamas/.ssh/id_rsa`, repo at
`/work/vllm-benchmark`).

## Setup & run

Before you run anything:

- **Sync the toolbox checkout**: put branch `study-46-dra-mig` (or `master` after the
  merge) under `/work/vllm-benchmark` on the akamas-41 toolbox. GitLab is not reachable
  from there: use a git bundle.
- **Scale up the GPU node and the load node**, as `../README.md` describes:
  `infra/eks/gpu-nodegroup.sh --up --always-on` and `infra/eks/load-node.sh up`.
- **Check the DRA driver**: `infra/dra/install.sh` must have run.

CLI through the toolbox:
`kubectl -n akamas-41 exec deploy/toolbox -c toolbox -- akamas ...`. Paths below are
relative to `/work/vllm-benchmark/studies/46-g7e-dra-mig-users-per-dollar/akamas`.

```bash
akamas create system             system.yaml
akamas create component          components/gpu0.yaml               vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/vllm.yaml               vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/vllm_r0.yaml            vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/vllm_r1.yaml            vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/vllm_r2.yaml            vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/vllm_r3.yaml            vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/container.yaml          vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/container_loadtest.yaml vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/cluster.yaml            vLLM_Benchmark_46_G7E_DRA_MIG
akamas create component          components/cluster_loadtest.yaml   vLLM_Benchmark_46_G7E_DRA_MIG
akamas create telemetry-instance telemetry/prometheus.yaml          vLLM_Benchmark_46_G7E_DRA_MIG
akamas create workflow           46-G7E-DRA-MIG-Users-Workflow.yaml
akamas create study              46-G7E-DRA-MIG-Users-per-Dollar.yaml

akamas start study "46-G7E-DRA-MIG-Users-per-Dollar"
```

Bulk alternative: every file declares its own `kind:` and `system:`. Keep the same
dependency order:

```bash
akamas create -f system.yaml
akamas create -f components/
akamas create -f telemetry/
akamas create -f 46-G7E-DRA-MIG-Users-Workflow.yaml
akamas create -f 46-G7E-DRA-MIG-Users-per-Dollar.yaml
```

If the study shows "Could not start the study" on 4.1 (late Airflow DAG): it stays
RUNNING with no experiments. Run `akamas finish study "46-G7E-DRA-MIG-Users-per-Dollar"`,
then `echo y | akamas start study "46-G7E-DRA-MIG-Users-per-Dollar"`.

Monitor and stop:

```bash
akamas describe study "46-G7E-DRA-MIG-Users-per-Dollar"
akamas list experiments "46-G7E-DRA-MIG-Users-per-Dollar"
akamas finish study "46-G7E-DRA-MIG-Users-per-Dollar"
akamas export study "46-G7E-DRA-MIG-Users-per-Dollar" ../results/export.tar.gz
```
