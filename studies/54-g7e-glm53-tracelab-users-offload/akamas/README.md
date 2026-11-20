# akamas/ — study 54's Akamas resources

Created on 2026-11-20 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.3**
(`akamas-41`), from study 53's resources (system, components and Prometheus telemetry copied and
renamed) and study 38's AIPerf CSV telemetry. Goal and method: `../README.md`. Design:
`docs/superpowers/specs/2026-11-20-study-54-glm53-tracelab-users-offload-design.md`.

## Versions

- **Packs:** vLLM 1.15.0 (the installed `vLLM` component type: `kv_offloading_size` [0, 512],
  `kv_offloading_backend` {native, lmcache} default native, `max_num_seqs` [1, 4096],
  `max_num_batched_tokens` [1, 65536], `spec_tokens` [0, 16]; checked 2026-11-20), AIPerf 1.0.0 (its
  31 metrics, the converter's columns), GPU 1.4.0, Kubernetes 1.9.0.
- **Target:** `vllm/vllm-openai:v0.34.0`, `canada-quant/GLM-5.3-Flash-W4A16-MTP` (revision 5681f7c2)
  served as `zai-org/GLM-5.3-Flash`, TP4; LMCache 0.5.8rc1 for the LMCache preset; AIPerf 0.13.0.
- **Telemetry:** Prometheus (kube-prometheus-stack on `vllm-bench`) and CSV File (the toolbox).

## Resources

- **System** `vLLM_Benchmark_54_G7E_GLM53_TraceLab`, 13 components (`components/`): study 53's 12 —
  `vllm` (vLLM), `vllm_workload` (Kubernetes Workload), `gpu0`-`gpu3` (GPU, one per physical GPU),
  `node_gpu` and `node_load` (Kubernetes Node), `container` and `container_loadtest` (Kubernetes
  Container), `cluster` (Kubernetes Cluster) — plus `aiperf` (AIPerf: the client side).
- **Telemetry** `Prometheus_54_G7E_GLM53_TraceLab` (`telemetry/prometheus.yaml`): study 53's full
  catalog (CLAUDE.md rule 9) with `duration: 30` (study 52's; requests arrive every second or so).
  `AIPerf_CSV_54_G7E_GLM53_TraceLab` (`telemetry/aiperf-csv.yaml`): the 31 AIPerf metrics from
  `/work/aiperf-csv/study54/*.csv` on the toolbox over SCP.
- **Workflow** `54-G7E-GLM53-TraceLab-Workflow`: `Write config` (FileConfigurator,
  `k8s/params.env.template`), `Apply config` (`k8s/apply_config.sh`, 140 min), `RunTest`
  (`k8s/run_test.sh`, 100 min: above its deadline RAMP_S + 900 s), `Stop AIPerf`
  (`k8s/stop_aiperf.sh`, alwaysRun, not critical), `AIPerf to CSV` (`k8s/aiperf_csv.sh`).
- **Study** `54-G7E-GLM53-TraceLab-Users-Offload`:
  - goal: maximize `aiperf.aiperf_active_sessions` (users);
  - constraints: `aiperf.aiperf_time_to_first_token_p95_150s <= 5000` and
    `aiperf.aiperf_inter_token_latency_p95_150s <= 50` (= `../k8s/workload.env`'s `SLO_TTFT_MS`,
    `SLO_ITL_MS`; `../k8s/test_akamas.sh` checks they agree);
  - windowing: `stability` on users, width 6 (3 min), `when` max (study 38); the CSV ends at the knee;
  - parameters: `kv_offloading_size` [0, 512], `gpu_memory_utilization` [0.85, 0.97],
    `max_num_batched_tokens` [2048, 32768], `max_num_seqs` [16, 1024], `spec_tokens` [0, 5];
    constraint `max_num_batched_tokens >= max_num_seqs`;
  - `kv_offloading_backend` outside the selection: `native` in the Native presets, `lmcache` in
    `LMCache 512`, the component's default (native) in `baseline` and `optimize` (`renderParameters`);
  - steps: `baseline` (gmu 0.95, MTP 5; offload, `max_num_seqs`, `max_num_batched_tokens` not
    rendered), `Native 128`, `Native 256`, `Native 512`, `LMCache 512` (`runOnFailure: true`),
    `optimize` (AKAMAS, 30 experiments, 10 failures allowed);
  - 8 KPIs: users, TTFT and ITL p95 150 s, completed requests/s (client), prefix hit incl. offload,
    CPU offload hit, offload read-back GB/s, GPU KV usage (server).

## Setup & run

Only after the user's go (the gate of the plan, Task 7). From the toolbox
(`kubectl -n akamas-41 exec -it deploy/toolbox -c toolbox -- bash`), with the study copied to
`/work/study54/54-g7e-glm53-tracelab-users-offload` (tar, as studies 45-53) and the load staged:

```bash
bash /work/study54/54-g7e-glm53-tracelab-users-offload/load/prepare_load.sh   # -> /work/study54-load
cd /work/study54/54-g7e-glm53-tracelab-users-offload/akamas
akamas create system             system.yaml
for f in components/*.yaml; do akamas create component "$f" vLLM_Benchmark_54_G7E_GLM53_TraceLab; done
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_54_G7E_GLM53_TraceLab
akamas create telemetry-instance telemetry/aiperf-csv.yaml vLLM_Benchmark_54_G7E_GLM53_TraceLab
akamas create workflow           54-G7E-GLM53-TraceLab-Workflow.yaml
akamas create study              54-G7E-GLM53-TraceLab-Users-Offload.yaml
akamas describe study 54-G7E-GLM53-TraceLab-Users-Offload

akamas start study 54-G7E-GLM53-TraceLab-Users-Offload
```

Bulk alternative (every file has `kind:` and `system:`), same dependency order:
`akamas create -f system.yaml`, then `akamas create -f components/`, `akamas create -f telemetry/`,
`akamas create -f 54-G7E-GLM53-TraceLab-Workflow.yaml`,
`akamas create -f 54-G7E-GLM53-TraceLab-Users-Offload.yaml`.

Before `start`: the AIPerf clients run 0.13.0 (`../k8s/02-aiperf-clients.yaml` applied), the
integration run passed (plan Task 6), GPU and load nodes up and tagged AlwaysOn. On Akamas 4.1 the
constraints, `parametersSelection` and windowing cannot change once the study runs (finish + update +
start restarts from zero): the SLO thresholds and the ramp are fixed at the gate. A change to the
workflow means delete and recreate it.

No placeholders are left in these files.
