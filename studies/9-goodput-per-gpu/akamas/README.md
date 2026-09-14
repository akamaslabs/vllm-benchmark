# 9-Goodput-Per-GPU — Akamas resources

**Created:** 2026-09-10 (everything copied from `8-parallelism-tuning`; the goal formula and
two telemetry metrics are the only differences)

**Last modified:** 2026-09-11 — `9-Goodput-Per-GPU.yaml` now defines the study's **third
instance, `9-Goodput-Per-GPU-v3`**: same system/workflow/goal/parameters as the first, plus
the `vLLM.max_num_batched_tokens >= vLLM.max_num_seqs` `parameterConstraint` (experiments
16, 17 and 22 of the first instance failed vLLM's `SchedulerConfig` validation on it), with
the baseline taken `from` the first instance's experiment 1 and a `bootstrap` step importing
its FINISHED experiments (2..23 minus 16, 17, 22). The second instance, `v2`, imported the
three ERROR experiments too and never got past the bootstrap step (Akamas bug, see "Setup &
run"). Neither the first instance (`9-Goodput-Per-GPU`, the import source) nor `v2` is
deleted.

Goodput-per-GPU study — maximize `(vLLM.prefill_token_throughput +
vLLM.decode_token_throughput) / vLLM.active_gpus` subject to P95 TTFT ≤ 1500 ms / P95 ITL ≤ 300 ms — on 4x NVIDIA L4 (`g6.12xlarge`, node
group `llm-serving-l4`), with the optimizer searching **16 parameters**: the 14 of
`2-larger-model-g7e` plus `vLLM.tensor_parallel_size` and `vLLM.data_parallel_size`
in [1, 4], constrained to `TP × DP ≤ 4` and `TP ≠ 3`. See the study's top-level README
for the design rationale (ROADMAP H5/H6) and caveats.

## Versions

- **vLLM optimization pack**: **1.8.0** — adds `active_gpus`, `active_dp_engines` and
  `gpu_memory_allocated_gb` to the `vLLM` component type on top of 1.7.0; **not yet built/installed** as of 2026-09-10 (the
  installed version is 1.7.0). `parametersSelection` domains unchanged from 1.7.0.
- **GPU optimization pack**: **1.2.0** — adds `gpu_nvlink_bandwidth_l0`, `gpu_nvlink_tx_bytes`,
  `gpu_nvlink_rx_bytes` (branch `feature/nvlink-profiling-counters`); **not yet built/installed**
  (installed: 1.1.0). **Kubernetes optimization pack**: **1.8.0-dev** (verified 2026-09-08).
- **Target workload**: `vllm/vllm-openai:v0.22.0`, `Qwen/Qwen2.5-7B-Instruct`.
- **Telemetry provider**: Prometheus (`kube-prometheus-stack`), Akamas platform 3.7.1.

## System — `vLLM_Benchmark_9_Goodput_Per_GPU`

9 components, identical to `7-tensor-parallelism` except the `system:` field:
- `vLLM` → `vLLM` (`prometheus.{pod,model}: .*`)
- `gpu0`..`gpu3` → `GPU` (`prometheus.pod: .*`, `prometheus.gpu: "0".."3"` — one
  component per physical L4, so every DCGM metric is one series per GPU; DCGM's `gpu`
  label verified as `0`–`3` on this node 2026-09-08)
- `container` → `Kubernetes Container` (`prometheus.pod: ^vllm-.*`)
- `container_loadtest` → `Kubernetes Container` (`prometheus.pod: ^aiperf-benchmark-.*`)
- `cluster` → `Kubernetes Cluster` (`prometheus.node_role: llm-serving-l4`)
- `cluster_loadtest` → `Kubernetes Cluster` (`prometheus.node_role: system-m8a`)

## Telemetry — `Prometheus_9_Goodput_Per_GPU`

`telemetry/prometheus.yaml`: 108 metrics — `8-parallelism-tuning`'s 102 (i.e. `7-tensor-parallelism`'s catalog (98 + the 4
absolute KV-cache metrics of pack 1.7.0) with **two query fixes applied only in this
study**, both found by running every query against the live Prometheus from the
`toolbox` pod on 2026-09-08 (306 runs: one per component, ×2 window values for the 36
`$DURATION$` queries; 0 queries returned more than one series) — plus the three vLLM-pack 1.8.0 metrics
described after the list and the three GPU-pack 1.2.0 NVLink metrics (item 4):

1. `fleet_e2e_latency_percentile`, `fleet_inter_token_latency_percentile`:
   `histogram_quantile(${percentile}, …)` → `histogram_quantile(0.95, …)`. `${…}` is
   Grafana variable syntax, not an Akamas placeholder — Prometheus rejected both with
   `parse error: unexpected character: '$'`, and neither metric appears in the export
   of `1-goodput-realistic-load` (DONE), i.e. they never collected in any study. 0.95
   matches the study's own P95 SLA. The pack's intended semantic for a "percentile"
   metric with no percentile parameter is unclear — raised as a note for the pack owner.
2. The 5 `k8s_cluster_{cpu,memory_pressure_some,memory_pressure_full,io_pressure_some,
   io_pressure_full}` queries: `[$DURATION$]` → `[2m]`. They are the only queries
   reading the `node-exporter` job, scraped every **30 s**; with `config.duration: 30`
   the 30 s window holds one sample and `rate()` returns nothing (measured: empty at
   15/30/45 s, one series from 60 s). `[2m]` ≥ 4× the scrape interval and is independent
   of how `$DURATION$` expands. All other `$DURATION$` queries read 5 s-scraped jobs
   (`vllm`, `dcgm-exporter`) and are unchanged.

3. **NEW (2026-09-10, pack 1.8.0)** `active_gpus` =
   `count(DCGM_FI_DEV_FB_USED{exported_pod=~"$POD$"} > 1024)` — GPUs holding model
   weights (TP × DP); and `active_dp_engines` =
   `count(count by (engine) (vllm:cache_config_info{pod=~"$POD$"}))` — data-parallel
   engines (DP); and `gpu_memory_allocated_gb` =
   `sum(DCGM_FI_DEV_FB_USED{exported_pod=~"$POD$"} > 1024) * 1048576 / 1e9` — GPU memory
   vLLM occupies, decimal GB. Verified live at TP=1/DP=3: 3, 3 and 64.06 GB. `active_gpus`
   is the goal's denominator; it filters on `exported_pod` because on DCGM series `pod` is the
   exporter's own pod (Operator relabeling) — see the study README.

4. **NEW (2026-09-10, GPU pack 1.2.0)** `gpu_nvlink_bandwidth_l0`, `gpu_nvlink_tx_bytes`,
   `gpu_nvlink_rx_bytes` on `gpu0`–`gpu3` — `sum(<DCGM field>{pod=~"$POD$", gpu=~"$GPU$"})` over
   `DCGM_FI_DEV_NVLINK_BANDWIDTH_L0` / `DCGM_FI_PROF_NVLINK_TX_BYTES` / `DCGM_FI_PROF_NVLINK_RX_BYTES`,
   completing NVIDIA's GPU-profiling counter set (see the study README). Empty on L4 (no
   NVLink), by nature.

Known-empty by nature on this node, not fixed: `gpu_fan_speed` (passively cooled),
`gpu_nvlink_bandwidth_total` (no NVLink), `gpu_fp64_pipe_active` (dcgm-exporter: "metric
not enabled"), `gpu_xid_errors`; and `container_cpu_limit`/`_util`/`_util_max`/
`_throttle_time`/`_throttled_millicores` on `container_loadtest` (`k8s/05-job.yaml` sets
no `limits.cpu` on the AIPerf container — deliberate).

**Percent scale fix (2026-09-10)**: 16 Kubernetes-pack percent queries lost their `100 *`
(Akamas' `percent` = 0-1 ratio; an 8.5% PSI showed as 850%), 5 DCGM `DEV_*` utilization queries
gained `/ 100` (0-100 natively), `container_cpu_throttled_millicores` re-pointed at
`1e3 * rate(container_cpu_cfs_throttled_seconds_total)` per Akamas' reference mapping. Details and
verification in the study README, "Percent metrics".

**Sidecar fix (2026-09-10)**: `kv_cache_capacity_*` used to appear only once
`vllm:kv_cache_usage_perc` had samples — never during the idle phase at DP > 1. The
sidecar (`k8s/04-kv-cache-exporter-configmap.yaml`) now publishes capacity as soon as
`vllm:cache_config_info` exists and usage/used only when usage samples exist. Live
ConfigMap re-applied; effective at the next rollout.

**Caveat inherited from the sidecar**: `kv_cache_capacity_gb`/`kv_cache_used_gb` use a
fixed bf16 bytes-per-token constant; this study tunes `kv_cache_dtype`, so on fp8
trials those two read 2× too high. The `*_tokens` variants are exact.

## Workflow — `9-Goodput-Per-GPU-Workflow`

Three tasks, as `7-tensor-parallelism` (paths re-pointed to this study's folder on the
`toolbox` host): `Write config` (FileConfigurator, `ignoreUnsubstitutedTokens: true`,
renders `k8s/01-deployment_template.yaml` → `k8s/01-deployment.yaml`), `Apply config`
(Executor, `k8s/apply_config.sh`, 75 m — strips unrendered `${vLLM.*}` lines, applies,
waits for rollout, dumps all container logs), `RunTest` (Executor,
`k8s/run_test_goodput.sh`, 105 m — deletes and re-creates the AIPerf Job, waits 5700 s,
dumps its logs).

## Study — `9-Goodput-Per-GPU-v3` (file: `9-Goodput-Per-GPU.yaml`; first instance: `9-Goodput-Per-GPU`)

- **Goal**: `(vLLM.prefill_token_throughput + vLLM.decode_token_throughput) / vLLM.active_gpus`
  (goodput per GPU actually used — the only change vs `8-parallelism-tuning`); SLA
  constraints and `stability` windowing (`vLLM.prefill_token_throughput`, width 6) verbatim.
- **`parametersSelection`**: 14 from `2-larger-model-g7e` + `vLLM.tensor_parallel_size`
  [1, 4] + `vLLM.data_parallel_size` [1, 4] (pack: integer, [1, 16] and [1, 8]).
- **`parameterConstraints`**: 4 vLLM-wide validation rules (FLASH_ATTN ⇒ `kv_cache_dtype
  == auto`; TRITON_ATTN and FLASHINFER ⇒ `kv_cache_dtype != fp8_e5m2`; and, re-added
  2026-09-11 after experiment 22, `vLLM.max_num_batched_tokens >= vLLM.max_num_seqs` —
  vLLM 0.22.0's `SchedulerConfig` pydantic check) + 2 topology rules —
  `vLLM.tensor_parallel_size * vLLM.data_parallel_size <= 4` and
  `vLLM.tensor_parallel_size != 3`. 7 valid (TP, DP) pairs out of 16.
- **`baseline`**: `from: [{study: 9-Goodput-Per-GPU, experiments: [1]}]` — imported, not
  re-run (with `from` no experiment executes, and `values`/`doNotRenderParameters` are
  not allowed). Experiment 1 of the first instance is the `gpu_memory_utilization: 0.90`
  pinned / everything-else-unrendered baseline (TP/DP → vLLM defaults 1/1).
- **`bootstrap`** (new 2026-09-11): `from: [{study: 9-Goodput-Per-GPU, experiments:
  [2..15, 18..21, 23]}]` — explicit list: omitting `experiments` would import experiment
  1 a second time, and the ERROR experiments 16, 17, 22 must stay out because importing
  a failed experiment crashes the campaign service (see "Setup & run" for the v2
  incident). CONSTRAINTS_VIOLATED experiments (18, 21) import fine.
- **`optimize`**: `numberOfExperiments: 1000`, `maxFailedExperiments: 200` — unchanged;
  its default 10 `numberOfInitExperiments` account for bootstrapped experiments.

## Validation performed

- The three new queries executed against the live Prometheus on 2026-09-10 with the DP=3 pod
  of `8-parallelism-tuning` experiment 2 running: `active_gpus` = 3, `active_dp_engines`
  = 3, `gpu_memory_allocated_gb` = 64.06; idle GPU at 2 MiB (below the 1024 MiB threshold). The other 102 queries are the
  ones verified for study 8 on 2026-09-08.
- Offline: manifest/telemetry/system/workflow cross-references consistent; every
  parameter and metric name resolves on the local pack clone at 1.8.0.
- **Live, 2026-09-11 12:35-12:37 UTC — `9-Goodput-Per-GPU-v3`**: `akamas create -f` of this
  manifest accepted (6 `parameterConstraints`, 3 steps); after `akamas start` the baseline
  was imported at once, the bootstrap step went RUNNING → FINISHED in 1 m 44 s with the 19
  listed experiments (v3 experiments 2..20, scores identical to the source's), and the
  `optimize` step started experiment 21 — the first new configuration under the
  `max_num_batched_tokens >= max_num_seqs` constraint — at 12:37 UTC. The two imported
  CONSTRAINTS_VIOLATED experiments (source 18 and 21) count as "# exp with errors: 2".

## Placeholders / secrets

- **`akamas/id_rsa`** — not in this repo (`.gitignore` covers `id_rsa*`). The workflow
  expects it on the `toolbox` host at
  `/work/vllm-benchmark/studies/9-goodput-per-gpu/akamas/id_rsa`; copy it there from
  an existing study's folder **on the host**, never through git.

## Setup & run

Run from the `toolbox` pod (namespace `akamas`, EKS `vllm-bench`), repo checked out at
`/work/vllm-benchmark`. GPU 1.1.0 / Kubernetes 1.8.0-dev packs and the `kv-cache-exporter` Kubernetes objects are
already in place from `7-tensor-parallelism`; **vLLM pack 1.8.0 must be built and installed
first** — staged in the `toolbox` pod at `/work/vllm-180` (1.7.0 KV-cache metrics + 1.8.0
additions; do NOT build from the pack's `develop`, which is 1.6.1):

```bash
akamas build optimization-pack /work/vllm-180                 # writes vLLM_1-8-0.json
akamas install -f optimization-pack vLLM_1-8-0.json
akamas describe optimization-pack vLLM | grep -E 'version|active_gpus|active_dp_engines|gpu_memory_allocated_gb'
akamas build optimization-pack /work/nvidia-gpu-120           # writes GPU_1-2-0.json
akamas install -f optimization-pack GPU_1-2-0.json
akamas describe optimization-pack GPU  | grep -E 'version|gpu_nvlink'
```

```bash
S="vLLM_Benchmark_9_Goodput_Per_GPU"
D="studies/9-goodput-per-gpu/akamas"

akamas create system             $D/system.yaml
akamas create component          $D/components/vllm.yaml               "$S"
akamas create component          $D/components/gpu0.yaml               "$S"
akamas create component          $D/components/gpu1.yaml               "$S"
akamas create component          $D/components/gpu2.yaml               "$S"
akamas create component          $D/components/gpu3.yaml               "$S"
akamas create component          $D/components/container.yaml          "$S"
akamas create component          $D/components/container_loadtest.yaml "$S"
akamas create component          $D/components/cluster.yaml            "$S"
akamas create component          $D/components/cluster_loadtest.yaml   "$S"
akamas create telemetry-instance $D/telemetry/prometheus.yaml          "$S"
akamas create workflow           $D/9-Goodput-Per-GPU-Workflow.yaml
akamas create study              $D/9-Goodput-Per-GPU.yaml   # first instance: the manifest at commit a07c58b

akamas describe study "9-Goodput-Per-GPU"
akamas start study    "9-Goodput-Per-GPU"
```

Bulk alternative (every file self-describes `kind:`/`system:`; same dependency order
applies internally): `akamas create -f studies/9-goodput-per-gpu/akamas/`.

To tear down (e.g. to recreate after a change to `parametersSelection`/
`parameterConstraints`/`steps`, which have no update verb — only `goal` does, via
`akamas update study`): `akamas delete study "9-Goodput-Per-GPU-v3"`, then the workflow,
telemetry instance, components and system in reverse order — or
`akamas delete -f studies/9-goodput-per-gpu/akamas/`. **Never delete the first instance
`9-Goodput-Per-GPU`**: it is the record of what ran on 2026-09-10/11 and the source the
`baseline`/`bootstrap` steps import from. **Do not delete `v2` either** without checking
with Akamas first: its id is written onto the first instance's datapoints (below), and how
Akamas deletes shared datapoints is unknown.

### Restart as `9-Goodput-Per-GPU-v3` (2026-09-11)

The manifest carries the `max_num_batched_tokens >= max_num_seqs` constraint and the
baseline-`from` + `bootstrap` steps. Everything but the study already exists and is reused;
the first instance stays in place as the import source:

```bash
akamas describe study "9-Goodput-Per-GPU"               # must be FINISHED (it is, 23 experiments)
akamas create -f studies/9-goodput-per-gpu/akamas/9-Goodput-Per-GPU.yaml     # single file, NOT the folder: system/workflow already exist
akamas describe study "9-Goodput-Per-GPU-v3"            # 6 parameterConstraints, steps
                                                        # baseline / bootstrap / optimize
akamas start study    "9-Goodput-Per-GPU-v3"
```

Then watch the bootstrap step: within a couple of minutes `akamas list experiment
"9-Goodput-Per-GPU-v3"` must show experiments 1..20 (1 = baseline, 2..20 = the 19
imported ones) and the `optimize` step must be RUNNING. If the bootstrap step stays
CREATED, look at `kubectl -n akamas logs deploy/campaign` for the NPE below.

#### The v2 incident (2026-09-11 11:09-12:27 UTC) — why failed experiments are excluded

`9-Goodput-Per-GPU-v2` was this manifest with the bootstrap list 2..23 *including* the
ERROR experiments 16, 17 and 22 (the 3.7 docs do not say how imported failures are
treated). It never got past the bootstrap step, and the UI sat at 0 %. From the cluster:

- The baseline step imported experiment 1 instantly (score 1738.5, trial duration copied).
- Airflow's `StepOperator` asked the `campaign` service to move `step_bootstrap` from
  CREATED to RUNNING; the campaign service imported experiments 2..22, then crashed in
  `TrialServiceImpl.refreshNormalization` (called from
  `BootstrapStepLogic.processExperimentsToBootstrap`) with
  `NullPointerException: Trial.getScore() is null` — the trials of ERROR experiments
  have no score. CONSTRAINTS_VIOLATED experiments (18, 21) have one and passed.
- The 500 made Airflow mark the task UP_FOR_RETRY and retry every ~90 s, with a retry
  budget of 1073741825 — i.e. forever (62 attempts before the study was finished).
- The campaign-side import is rolled back on each crash (v2 kept showing only experiment
  1), but the metrics-service side is not: each attempt appends the v2 study id once more
  to the `studyExperimentTrialIds` array of every datapoint of the source's experiments
  2..21 (~30 k datapoints each; 62 copies of the id per datapoint when stopped). The
  source datapoints themselves are intact (counts unchanged), just carrying duplicate ids.
  `data_point` index size at the time: 251 MB, 2.3 M docs.

Stopped with `akamas finish study "9-Goodput-Per-GPU-v2"` at 12:27:41 UTC. v2 is left in
place, not deleted (see the tear-down note above). To report to Akamas: null-score trials
crash the bootstrap import; the StepOperator retries a 500 forever; the datapoint id update
is not idempotent.
