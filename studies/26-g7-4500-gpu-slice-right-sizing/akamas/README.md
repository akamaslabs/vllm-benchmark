# akamas/ — 26-G7-4500-GPU-Slice-Right-Sizing

**Created:** 2026-09-29, with the `akamas-study-manager` plugin conventions, from study 25's
resources (same system shape, workflow and telemetry). **Not created on the Akamas server
yet** — validated offline only (see "Validation").

## What it optimizes

A MIG right-sizing sweep, not an optimization: aggregate goodput
(`vllm.prefill_token_throughput + vllm.decode_token_throughput`) under TTFT p95 <= 1500 ms
and ITL p95 <= 300 ms, `stability` windowing on `vllm.total_token_throughput`, for each MIG
partition size of the RTX PRO 4500 (`none`, `2g.32gb`, `1g.16gb`), the GPU always fully
partitioned with one replica per instance. Design: `../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | 3.7.x |
| Optimization pack **GPU** | **1.4.0** (`mig_profile`; also `sharing_mode`), branch `feature/gpu-sharing-mode` of the `nvidia-gpu` repo (commit `709b634`), installed 2026-09-30 |
| Optimization pack **vLLM** | 1.12.0 (installed) |
| Kubernetes pack | the installed build 1.8.0-dev (`Kubernetes Container`, `Kubernetes Cluster`) |
| Serving | `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-4B-Instruct-2507-FP8` served as `qwen3-4b` |
| Load generator | AIPerf 0.11.0, ShareGPT replay |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter 4.6.0-4.8.3 |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_26_G7_4500_GPU_Slice` |
| `components/vllm.yaml` | `vllm` (vLLM) — both replicas; the only component goal/constraints/windowing read |
| `components/vllm_r0.yaml`, `vllm_r1.yaml` | per-replica views (`vllm_r1` empty with `none` / `2g.32gb`) |
| `components/gpu0.yaml` | `gpu0` (GPU) — carries `mig_profile` |
| `components/cluster.yaml`, `container.yaml`, `cluster_loadtest.yaml`, `container_loadtest.yaml` | node and container views (Kubernetes pack) |
| `telemetry/prometheus.yaml` | `Prometheus_26_G7_4500_GPU_Slice`, study 25's 117 metrics, placeholder keys without underscore (`$GPUMODEL$`, `$NODEROLE$`) |
| `26-G7-4500-GPU-Slice-Right-Sizing-Workflow.yaml` | `Write config` -> `Apply config` (60 m) -> `RunTest` (105 m), scripts in `../k8s/` |
| `26-G7-4500-GPU-Slice-Right-Sizing.yaml` | study `26-G7-4500-GPU-Slice-Right-Sizing` |

**Steps:** `baseline` (none, 256), `MIG whole GPU seqs 256`, `MIG whole GPU seqs 512`,
`MIG half GPU seqs 256`, `MIG half GPU seqs 128`, `MIG half GPU seqs 384`, `no MIG repeat`
— all at `gpu_memory_utilization` 0.90 / `max_num_batched_tokens` 2048 / `stream_interval`
1, every parameter rendered; no optimize step. **parameterConstraints:** none (every
combination is valid; the replica count follows from the profile). **Placeholders left:**
none (host `toolbox`, user `akamas`, key `/home/akamas/.ssh/id_rsa`, as every study here).

## Validation

Offline, 2026-09-30: component types, parameter names and domain subsets (`mig_profile`
categories inside GPU 1.4.0's), metrics referenced vs produced, `$KEY$` placeholders (all
backed by a component key, none with an underscore), `${component.param}` tokens (5, all
in `parametersSelection`), step names, 8 KPIs — against vLLM 1.12.0, GPU 1.4.0 and the
installed Kubernetes build. `apply_config.sh` tested in the toolbox on all three profiles,
2026-09-30 08:39-08:57 UTC, in the order `2g.32gb` -> `1g.16gb` -> `none` (every transition
the sweep makes): all exit 0; 1 / 2 / 1 replicas; KV cache 159,024 / 56,016 per slice /
159,008 tokens; rollout 244 / 486 / 243 s (the two `1g.16gb` replicas start in order);
node left with MIG off, device-plugin config `exclusive`, vLLM at 0.

## Setup & run

Prerequisites: GPU pack 1.4.0, dcgm-exporter covering `llm-serving-g7-4500`,
`infra/eks/provision.sh`, toolbox sync, and study 25 not running (all done 2026-09-30).

From the toolbox, in `/work/vllm-benchmark`:

```bash
A=studies/26-g7-4500-gpu-slice-right-sizing/akamas
SYS=vLLM_Benchmark_26_G7_4500_GPU_Slice

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
akamas create workflow           $A/26-G7-4500-GPU-Slice-Right-Sizing-Workflow.yaml
akamas create study              $A/26-G7-4500-GPU-Slice-Right-Sizing.yaml

akamas start study "26-G7-4500-GPU-Slice-Right-Sizing"
```

Bulk alternative (every file carries `kind:` and `system:`):

```bash
akamas create -f studies/26-g7-4500-gpu-slice-right-sizing/akamas/
akamas start study "26-G7-4500-GPU-Slice-Right-Sizing"
```

Monitoring: `akamas list experiment "26-G7-4500-GPU-Slice-Right-Sizing"`, `akamas log
--study "26-G7-4500-GPU-Slice-Right-Sizing" --log-level INFO` (never `-d`), and at the end
`akamas export study "26-G7-4500-GPU-Slice-Right-Sizing"
studies/26-g7-4500-gpu-slice-right-sizing/results/export.tar.gz`.
