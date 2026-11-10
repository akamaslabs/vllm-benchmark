# akamas/ — study 49's Akamas resources

Created on 2026-11-06 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.0**
(`akamas-41`). Adapted from study 46. Goal and design: `../README.md` and
`docs/superpowers/specs/2026-11-06-study-49-g7e-mig-smallest-slice-design.md`.

**Status:** created on akamas-41 and run on 2026-11-07 (`49-G7E-MIG-Smallest-Slice`, FINISHED,
invalid: see `../README.md`). **Rerun:** `49-G7E-MIG-Smallest-Slice-Rerun.yaml` (same file with a
new name and a description note), created and started on 2026-11-10 against the same system,
components, telemetry instance and workflow.

## Versions

- **Optimization packs on akamas-41:** GPU **1.4.0** (`mig_profile`), vLLM **1.15.0**,
  Kubernetes **1.9.0**.
- **Workload:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` served as `qwen3-8b-dra`,
  `--max-model-len 4096`, prefix caching off (as study 46).
- **Load generator:** AIPerf 0.11.0, closed loop on ShareGPT (10k-conversation subset). One
  client pod per vLLM pod, on the `m8a-2xlarge` node.
- **Telemetry:** Prometheus (kube-prometheus-stack). Scrapes:
  - vLLM, via ServiceMonitor `vllm-dra-mig-49`, every 5 s;
  - dcgm-exporter;
  - the slice-roles exporter, via ServiceMonitor `slice-roles-49`.

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_49_G7E_MIG_Smallest_Slice` |
| `components/vllm.yaml` | vLLM | `vllm`: the customer, every tuned parameter, constraints |
| `components/vllm_neighbours.yaml` | vLLM | `vllm_neighbours`: the other tenants (read only) |
| `components/container.yaml` | Kubernetes Container | the customer's container (CPU-throttle guard) |
| `components/container_loadtest.yaml` | Kubernetes Container | the AIPerf clients |
| `components/cluster.yaml` | Kubernetes Cluster | the GPU node |
| `components/cluster_loadtest.yaml` | Kubernetes Cluster | the load node |
| `components/gpu0.yaml` | GPU | the physical GPU: `mig_profile`, all DCGM series (`slice: .+`) |
| `components/gpu_customer.yaml` | GPU | the customer's slice (`slice: customer`): the goal |
| `components/gpu_neighbours.yaml` | GPU | the neighbours' slices (`slice: neighbour`) |
| `telemetry/prometheus.yaml` | telemetry-instance | `Prometheus_49_G7E_MIG_Smallest_Slice` |
| `49-G7E-MIG-Smallest-Slice-Workflow.yaml` | workflow | Write config → Apply config → Load |
| `49-G7E-MIG-Smallest-Slice.yaml` | study | `49-G7E-MIG-Smallest-Slice` |

The telemetry uses study 46's catalog. Every DCGM selector is joined with
`study49_slice_role{role=~"$SLICE$"}` on `GPU_I_ID`, so each GPU component sees its own
slices; the header of `telemetry/prometheus.yaml` explains why.

## Study

- **Goal:** minimize
  `gpu_customer.gpu_fb_total / 1024 + vllm.time_to_first_token_p95 / 1000000`.
  - The first term is the GiB in the customer's slice: ~24, ~48, or ~96 for the whole GPU.
  - The second term is the TTFT p95 tie-break.
  - The window is the `Load` task, trimmed `[90s, 60s]`.
- **Constraints, on the customer:** TTFT p95 ≤ 1500 ms and ITL p95 ≤ 300 ms. `run_load.sh`
  enforces two more things client-side (exit 6): the customer completes ≥ 95 % of its requests,
  and the neighbours are really busy.
- **Parameters:** `gpu0.mig_profile` {none, 2g.48gb, 1g.24gb}, plus study 46's 12 vLLM
  parameters on the customer. `mnbt ≥ mns`.
- **Steps:**
  1. `baseline`: whole GPU with defaults.
  2. `MIG half GPU` and `MIG quarter GPU`, with the same defaults.
  3. `whole GPU repeat`.
  4. `optimize`: AKAMAS, 4 init, 15 experiments, 6 failures max.
- **8 KPIs:**
  - customer slice MiB;
  - customer TTFT p95 and ITL p95;
  - customer tokens/s;
  - neighbours TTFT p95;
  - customer slice SM active;
  - GPU power;
  - vLLM CPU throttle.

  Every other DCGM metric stays as telemetry for analysis.

## Setup & run

From the toolbox (`kubectl -n akamas-41 exec -it deploy/toolbox -c toolbox -- bash`), with the
study copied to `/work/study49/49-g7e-dra-mig-smallest-slice`:

```
cd /work/study49/49-g7e-dra-mig-smallest-slice/akamas
SYS=vLLM_Benchmark_49_G7E_MIG_Smallest_Slice
akamas create system system.yaml
for c in vllm vllm_neighbours container container_loadtest cluster cluster_loadtest gpu0 gpu_customer gpu_neighbours; do
  akamas create component components/$c.yaml $SYS
done
akamas create telemetry-instance telemetry/prometheus.yaml $SYS
akamas create workflow 49-G7E-MIG-Smallest-Slice-Workflow.yaml
akamas create study 49-G7E-MIG-Smallest-Slice.yaml

akamas start study "49-G7E-MIG-Smallest-Slice"
```

Bulk alternative, in the same dependency order:

1. `akamas create -f system.yaml`
2. `akamas create -f components/`
3. `akamas create -f telemetry/`
4. `akamas create -f 49-G7E-MIG-Smallest-Slice-Workflow.yaml`
5. `akamas create -f 49-G7E-MIG-Smallest-Slice.yaml`

If the start fails with "Could not start the study", run `akamas finish study`, then
`yes y | akamas start study`.
