# akamas/ — Akamas resources of 12-gpt-oss-20b-steady-throughput

Generated on 2026-09-15 following the `akamas-study-manager` build skill, by cloning
`10-gpt-oss-20b-goodput-per-gpu`'s resources — which had themselves been validated on the
live instance (the study ran a full baseline plus a first optimize experiment on
2026-09-15) — and changing only the load profile and what follows from it. Same model,
same hardware, same 14 parameters, same goal and SLA: study 11 exists to read the TPS
peak in ~6 minutes of load instead of ~67.

| File | What changed vs. study 10 |
|---|---|
| `system.yaml` | new system `vLLM_Benchmark_12_GPT_OSS_20B_Steady` |
| `components/*.yaml` | same 9 components (vLLM, container, cluster, gpu0-3, container_loadtest, cluster_loadtest), `system:` swapped |
| `telemetry/prometheus.yaml` | identical 108 metrics, `system:`/`name:` swapped (`Prometheus_12_GPT_OSS_20B_Steady`) — every metric the goal/constraints reference (`prefill_token_throughput`, `decode_token_throughput`, `active_gpus`, `time_to_first_token_p95`, `inter_token_latency_p95`) is produced here |
| `12-GPT-OSS-20B-Steady-Throughput-Workflow.yaml` | paths under `studies/12-gpt-oss-20b-steady-throughput/`, same operators/timeouts |
| `12-GPT-OSS-20B-Steady-Throughput.yaml` | **two edits only**: `windowing.stability.resolution: 15s` (window = 90s = exactly one concurrency level, since each level now lasts 90s instead of 300s) and optimize **60/12** instead of 200/40 (~11 h instead of ~10 days). Goal, constraints, 14 parameters, domains, 5 `parameterConstraints` and the baseline are byte-identical to study 10 |
| `../k8s/05-job.yaml` | the actual point of this study: `CONCURRENCY_LIST="128,256,512,1024"` (4 log-spaced levels, ratio 2x, ceiling matching `max_num_seqs`'s domain) and `--benchmark-duration 90` |

Pack vocabulary used: vLLM pack **>= 1.9.1** (the union of 1.8.0's `active_gpus`,
`active_dp_engines`, `gpu_memory_allocated_gb`, `kv_cache_*` and 1.9.0's agentic knobs —
1.9.0 alone lacks the seven metrics and the telemetry instance fails to create), GPU pack
**1.2.0**, Kubernetes pack **1.8.0-dev**. The 1.9.x-only knob `stream_interval` is referenced
only in commented-out blocks.

`id_rsa` (the toolbox SSH key the workflow uses) must be placed at
`/work/vllm-benchmark/studies/12-gpt-oss-20b-steady-throughput/akamas/id_rsa` on the toolbox
host by hand — it is git-ignored and must never be committed (five earlier studies did
commit theirs; see the repo's `ROADMAP.md` security debt).

Setup and run commands: see `../README.md`, "Setup & run".
