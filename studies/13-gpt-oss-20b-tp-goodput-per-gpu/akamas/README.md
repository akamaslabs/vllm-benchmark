# akamas/ — Akamas resources of 13-gpt-oss-20b-tp-goodput-per-gpu

Generated on 2026-09-15 following the `akamas-study-manager` build skill (Modify mode), by
cloning `12-gpt-oss-20b-steady-throughput`'s resources — validated on the live instance,
where study 12 had run 13 experiments by that date — and changing the parallelism domains,
the load profile (back to study 10's ramp) and what follows from them. Same model, same
hardware, same 14 parameters, same goal and SLA.

| File | What changed vs. study 12 |
|---|---|
| `system.yaml` | new system `vLLM_Benchmark_13_GPT_OSS_20B_TP`, description updated (TP in [2,4], ramp + stability, pack 1.9.1) |
| `components/*.yaml` | same 9 components (vLLM, container, cluster, gpu0-3, container_loadtest, cluster_loadtest), `system:` swapped |
| `telemetry/prometheus.yaml` | identical 108 metrics, `system:`/`name:` swapped (`Prometheus_13_GPT_OSS_20B_TP`) — every metric the goal/constraints reference (`prefill_token_throughput`, `decode_token_throughput`, `active_gpus`, `time_to_first_token_p95`, `inter_token_latency_p95`) is produced here |
| `13-GPT-OSS-20B-TP-Goodput-Per-GPU-Workflow.yaml` | paths under `studies/13-gpt-oss-20b-tp-goodput-per-gpu/`, same operators and timeouts (Apply config 75 m, RunTest 105 m — already sized for the 12 x 300 s ramp, which study 12 had inherited unchanged) |
| `13-GPT-OSS-20B-TP-Goodput-Per-GPU.yaml` | **`tensor_parallel_size` domain [2, 4]** and **`data_parallel_size` [1, 2]** (topologies TP2/DP1, TP2/DP2, TP4/DP1); **new `parameterConstraint`** `gpu_memory_utilization * 22.03 + max_num_seqs * 0.00375 <= 21.63` (sampler-warmup OOM guard, derived from study 12's experiment 12 — see the manifest's comment); the "expert parallelism needs more than one GPU" constraint dropped (always true with TP >= 2), so still **5 constraints**; **windowing back to study 10's `stability` block** (prefill_token_throughput, width 6, `is: max`) instead of `trim`; baseline renders `tensor_parallel_size: 2` (the pack default 1 is outside the domain) and `gpu_memory_utilization: 0.90`, `kv_cache_dtype` back to unrendered (study 12's fp8 pin was specific to its constant 512 load); optimize **100/20** (~5 days at ~75 min); **explicit `kpis` block** (added 2026-09-16 — the five metrics goal/constraints already reference, written out to pin `active_gpus: minimize`; study 12 had none) |
| `../k8s/05-job.yaml` | study 10's `CONCURRENCY_LIST="150,...,1024"` x `--benchmark-duration 300`, no `--concurrency-ramp-duration` |
| `../k8s/apply_config.sh` | crash-loop fail-fast: the rollout wait polls in 30 s slices and fails the trial as soon as the new pod's vllm container has restarted twice |
| `../k8s/smoke_test.sh` | six TP >= 2 configurations, two of them on the guard's boundary |

Pack vocabulary used: vLLM pack **1.9.1** (installed and confirmed with `akamas list
optimization-pack` on 2026-09-15; the local checkout at
`~/akamas/offline/optimization-packs/vllm`, branch `feature/agentic-scheduling-knobs-1.9.1`,
is what the domains were re-checked against), GPU pack **1.2.0**, Kubernetes pack
**1.8.0-dev**. The 1.9.x-only knob `stream_interval` is referenced only in commented-out
blocks.

There is no `akamas update` verb for the fields changed here (domains, constraints, KPIs,
windowing, steps): a study with these settings has to be created as a new study, which is
what this folder is. Only `goal` can be updated on a running study (`akamas update study`).

**2026-09-16:** `kpis` added to `13-GPT-OSS-20B-TP-Goodput-Per-GPU.yaml`. The study resource
created on the live instance on 2026-09-15 predates it and has run no experiment, so it must
be deleted and re-created (study resource only — system, components, telemetry instance and
workflow are unchanged). Commands in `../README.md`, "Setup & run", step 1b.

`id_rsa` (the toolbox SSH key the workflow uses) must be placed at
`/work/vllm-benchmark/studies/13-gpt-oss-20b-tp-goodput-per-gpu/akamas/id_rsa` on the toolbox
host by hand — it is git-ignored and must never be committed (five earlier studies did
commit theirs; see the repo's `ROADMAP.md` security debt).

Setup and run commands: see `../README.md`, "Setup & run".
