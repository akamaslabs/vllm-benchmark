# akamas/ — study 53's Akamas resources

Created on 2026-11-18 with the `akamas-study-manager` plugin (0.3.0), for **Akamas 4.1.3**
(`akamas-41`), from study 52's resources (components and telemetry copied and renamed). Goal and
method: `../README.md`. Design: `docs/superpowers/specs/2026-11-18-study-53-glm53-kv-offload-idle-sessions-design.md`.

## Versions

- **Packs:** vLLM 1.15.0, GPU 1.4.0, Kubernetes 1.9.0 (as study 52, checked 2026-11-17; the vLLM
  pack's `max_num_batched_tokens` domain reaches 131072, checked 2026-11-18).
- **Target:** `vllm/vllm-openai:v0.34.0`, `canada-quant/GLM-5.3-Flash-W4A16-MTP` (revision
  5681f7c2) served as `zai-org/GLM-5.3-Flash`, TP4; AIPerf 0.11.0.
- **Telemetry:** Prometheus (kube-prometheus-stack on `vllm-bench`).

## Resources

- **System** `vLLM_Benchmark_53_G7E_GLM53_Offload`, 12 components (`components/`), the same as
  study 52: `vllm` (vLLM), `vllm_workload` (Kubernetes Workload), `gpu0`-`gpu3` (GPU, one per
  physical GPU), `node_gpu` and `node_load` (Kubernetes Node), `container` and
  `container_loadtest` (Kubernetes Container), `cluster` (Kubernetes Cluster).
- **Telemetry** `Prometheus_53_G7E_GLM53_Offload` (`telemetry/prometheus.yaml`): study 52's full
  catalog (CLAUDE.md rule 9) with **`duration: 120`** instead of 30: the load sends one turn every
  ~15 s, so 30 s rate windows are often empty and a TTFT average of 0 would drag the mean down.
- **Workflow** `53-G7E-GLM53-Offload-Workflow`: `Write config` (FileConfigurator,
  `k8s/params.env.template`), `Apply config` (`k8s/apply_config.sh`, 140 min timeout for the
  node-replacement path; ~4 min normally), `Load` (`k8s/run_load.sh`, 40 min).
- **Study** `53-G7E-GLM53-Offload-Idle-Sessions`:
  - goal: minimize `vllm.time_to_first_token_avg` (server; the pack has no TTFT p50);
  - constraints: `vllm.request_success_rate >= 0.06` (90 % of the offered 0.0667 turns/s: a
    configuration that falls behind is infeasible) and `vllm.time_per_output_token_p95 <= 80`;
  - window: trim `[830s, 60s]` on `Load` (start-up + session creation + one 120 s rate window);
  - parameters: study 52's selection (same template); only the presets' values are used;
  - steps (10, presets only): `baseline` (gmu 0.95, MTP 5, no offload: the target's arguments),
    `Offload 32`, `Offload 64`, `Offload 128`, `Offload 256`, `Offload 512`, `Prefill 16k`
    (offload 512, `max_num_batched_tokens` 16384), `Prefill 32k` (offload 512, 32768),
    `Prefill 32k no offload`, `Baseline repeat`;
  - 8 KPIs: TTFT average, p90, p95, prompt from cache incl. offload, prompt from CPU offload,
    offload read-back GB/s, completed requests per s, TPOT p95.

## Setup & run

From the toolbox (`kubectl -n akamas-41 exec -it deploy/toolbox -c toolbox -- bash`), with the
study copied to `/work/study53/53-g7e-glm53-kv-offload-idle-sessions` (tar, as studies 45-52):

```bash
cd /work/study53/53-g7e-glm53-kv-offload-idle-sessions/akamas
akamas create system             system.yaml
for f in components/*.yaml; do akamas create component "$f" vLLM_Benchmark_53_G7E_GLM53_Offload; done
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_53_G7E_GLM53_Offload
akamas create workflow           53-G7E-GLM53-Offload-Workflow.yaml
akamas create study              53-G7E-GLM53-Offload-Idle-Sessions.yaml
akamas describe study 53-G7E-GLM53-Offload-Idle-Sessions

akamas start study 53-G7E-GLM53-Offload-Idle-Sessions
```

Bulk alternative (every file has `kind:` and `system:`), same dependency order:
`akamas create -f system.yaml`, then `akamas create -f components/`, `akamas create -f telemetry/`,
`akamas create -f 53-G7E-GLM53-Offload-Workflow.yaml`, `akamas create -f 53-G7E-GLM53-Offload-Idle-Sessions.yaml`.

Before `start`: study 52 finished (paused) and nothing of it still running on the toolbox; GPU node
and load node up; the smoke passed (`../README.md`, "Smoke"). On Akamas 4.1 constraints cannot be
edited while the study runs (finish + update + start restarts from zero).

After the study: resume study 52 (`akamas resume study -w default "52-G7E-GLM53-Coding-Sessions" -m DEL`);
its next Apply re-renders the shared Deployment.

No placeholders are left in these files.
