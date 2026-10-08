# k8s/ — 31-l40s-gemma4-26b-tps-thinking

Study 30's `k8s/` with Gemma 4's thinking mode on (itself study 28's single-vLLM machinery
without the MIG part): one StatefulSet replica (`vllm-0`) on the whole L40S, one AIPerf Job,
everything in namespace `llm-l40s`. The same namespace, node, pod, Job, PVCs, Services and
ServiceMonitor as study 30, which runs before this study on the same node; the served model
name (`gemma4-26b-l40s-think`) keeps the two studies' vLLM series and ShareGPT caches apart.

What differs from study 30: `01-statefulset_template.yaml` (`enable_thinking` true,
`--reasoning-parser=gemma4`, `--max-model-len=16384`), `05-job_template.yaml` (requests
without `max_tokens`, a count-based warm-up, a `LENGTHS` summary at the end of a completed
run), `render_job.sh`
(`--closed` mode for the length run), `run_test.sh` (provisional ramp 0 -> 6 req/s over
6000 s).

| File | What |
|---|---|
| `params.env.template` | Rendered by the workflow's FileConfigurator (`${vllm.*}` tokens) into `params.env`. |
| `render_statefulset.sh` | Validates `params.env` and writes the vLLM flags into `01-statefulset_template.yaml` (booleans as `--x` / `--no-x`; no backend flag when a backend is absent or `auto`). |
| `apply_config.sh` | Apply config task: validate, free the GPU, start `vllm-0`, wait (fail fast on a crash loop), 8 warm-up requests, health check, full logs. |
| `run_test.sh`, `render_job.sh`, `05-job_template.yaml`, `lib_watchdog.sh` | RunTest task: ShareGPT replay without `max_tokens` with study 27's open-loop linear rate ramp (`RT_RATE`, `RT_RAMP_S`), watchdog past 2x the SLA, fail-fast guards, full logs. `render_job.sh --closed C N` renders the smoke run's closed-loop length run. |
| `lib_health.sh` | The post-warm-up health decision. |
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml`, `02-service.yaml`, `03-hf-secret.yaml` | AIPerf results / ShareGPT cache, AIPerf HF cache, Services (`vllm`, `vllm-headless`), optional HF token template (never commit a real token). |
| `monitoring/` | ServiceMonitor `vllm-l40s`, the dcgm-exporter values with this node role added, counters and kube-prometheus values as deployed. |
| `tests/` | `bash tests/test_*.sh` (yq v4 needed): renderers, guards, watchdog, run_test with a stub kubectl. `bash tests/dry_run_thinking.sh` (network, ~6 min): the rendered Job script with real AIPerf 0.11.0 against `mock_openai.py` streaming reasoning deltas: requests leave without `max_tokens`, a warm-up of long requests passes, `LENGTHS` counts reasoning tokens. |
