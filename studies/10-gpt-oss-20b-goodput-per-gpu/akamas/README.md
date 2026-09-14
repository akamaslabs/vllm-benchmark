# akamas/ — Akamas resources of 10-gpt-oss-20b-goodput-per-gpu

Generated on 2026-09-14 following the `akamas-study-manager` build skill, by cloning
`9-goodput-per-gpu`'s validated resources (system, 9 components, 108-metric Prometheus
telemetry instance, 3-task workflow) and rewriting only what the model change requires:

| File | What changed vs. study 9 |
|---|---|
| `system.yaml` | new system `vLLM_Benchmark_10_GPT_OSS_20B` |
| `components/*.yaml` | same 9 components (vLLM, container, cluster, gpu0-3, container_loadtest, cluster_loadtest), `system:` swapped |
| `telemetry/prometheus.yaml` | identical 108 metrics, `system:`/`name:` swapped (`Prometheus_10_GPT_OSS_20B`) — every metric the goal/constraints reference (`prefill_token_throughput`, `decode_token_throughput`, `active_gpus`, `time_to_first_token_p95`, `inter_token_latency_p95`) is produced here |
| `10-GPT-OSS-20B-Goodput-Per-GPU-Workflow.yaml` | paths under `studies/10-gpt-oss-20b-goodput-per-gpu/`, same operators/timeouts |
| `10-GPT-OSS-20B-Goodput-Per-GPU.yaml` | 14 tuned parameters, 5 `parameterConstraints`, baseline pinned at `gpu_memory_utilization 0.85`, optimize 200/40 — rationale inline |

Pack vocabulary used: vLLM pack **1.8.0** (`active_gpus`, `active_dp_engines`,
`gpu_memory_allocated_gb`, `kv_cache_*`), GPU pack **1.2.0**, Kubernetes pack **1.8.0-dev** —
the same versions study 9 requires. The vLLM pack **1.9.0** knobs (`stream_interval`, ...)
are referenced only in commented-out blocks; `akamas create` would fail on them until that
pack version (MR !5 on `gitlab.com/akamas/optimization-packs/vllm`) is installed.

`id_rsa` (the toolbox SSH key the workflow uses) must be placed at
`/work/vllm-benchmark/studies/10-gpt-oss-20b-goodput-per-gpu/akamas/id_rsa` on the toolbox
host by hand — it is git-ignored and must never be committed (five earlier studies did
commit theirs; see the repo's `ROADMAP.md` security debt).

Setup and run commands: see `../README.md`, "Setup & run".
