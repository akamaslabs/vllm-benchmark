# k8s/ — 30-l40s-gemma4-26b-tps

Study 28's single-vLLM machinery without the MIG part: one StatefulSet replica (`vllm-0`)
on the whole L40S, one AIPerf Job, everything in namespace `llm-l40s`.

| File | What |
|---|---|
| `params.env.template` | Rendered by the workflow's FileConfigurator (`${vllm.*}` tokens) into `params.env`. |
| `render_statefulset.sh` | Validates `params.env` and writes the vLLM flags into `01-statefulset_template.yaml` (booleans as `--x` / `--no-x`; no backend flag when a backend is absent or `auto`). |
| `apply_config.sh` | Apply config task: validate, free the GPU, start `vllm-0`, wait (fail fast on a crash loop), 8 warm-up requests, health check, full logs. |
| `run_test.sh`, `render_job.sh`, `05-job_template.yaml`, `lib_watchdog.sh` | RunTest task: ShareGPT replay with study 27's open-loop linear rate ramp (`RT_RATE`, `RT_RAMP_S`), watchdog past 2x the SLA, fail-fast guards, full logs. |
| `lib_health.sh` | The post-warm-up health decision. |
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml`, `02-service.yaml`, `03-hf-secret.yaml` | AIPerf results / ShareGPT cache, AIPerf HF cache, Services (`vllm`, `vllm-headless`), optional HF token template (never commit a real token). |
| `monitoring/` | ServiceMonitor `vllm-l40s`, the dcgm-exporter values with this node role added, counters and kube-prometheus values as deployed. |
| `tests/` | `bash tests/test_*.sh` (yq v4 needed): renderers, guards, watchdog, run_test with a stub kubectl. |
