# k8s/ — 33-l40s-gemma4-26b-awq-no-thinking

Study 32's `k8s/` with thinking turned off (copied 2026-10-12, with study 32's `Stop AIPerf`
task). Study 32's own was study 31's (itself study 30's, study 28's single-vLLM machinery
without the MIG part) with a customer's Docker Compose setup and a synthetic multi-turn load:
one StatefulSet replica (`vllm-0`) on the whole L40S, one AIPerf Job, everything in namespace
`llm-l40s`. The same namespace, node, pod, Job, PVCs, Services and ServiceMonitor as studies
30-32, which run before this study on the same node; the served model name
(`gemma4-26b-awq-nothink`) keeps the studies' vLLM series apart.

What differs from study 32:
- `01-statefulset_template.yaml`: `--default-chat-template-kwargs '{"enable_thinking": false}'`
  (study 32: true), served model name `gemma4-26b-awq-nothink`.
- `run_test.sh`: ramp 0 -> **6** req/s over 6000 s (study 32: 2), provisional (study README
  "Load").
- `05-job_template.yaml`: the same load profile; comments only (the history holds answers, no
  reasoning). Results folder `/benchmarks/study33`; `stop_aiperf.sh` saves to
  `/work/aiperf-exports/33-l40s-gemma4-26b-awq-no-thinking/`.
- `tests/`: `enable_thinking false` asserted; the dry run's mock streams no reasoning and checks
  that no reply in the history holds any.

What study 32 changed from study 31 (kept here):
- `01-statefulset_template.yaml`: the compose's checkpoint and revision
  (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, `0ef577a`), `--max-model-len=96000`, tool calling
  (`--enable-auto-tool-choice --tool-call-parser=gemma4`), no `--language-model-only`, prefix
  caching at vLLM's default, the compose's two environment variables.
- `render_statefulset.sh`: an empty value means no flag (vLLM's default), for the parameters
  the baseline steps leave unrendered (`doNotRenderParameters`); `gpu_memory_utilization`,
  `max_num_seqs`, `max_num_batched_tokens` are always rendered.
- `05-job_template.yaml`, `render_job.sh`: AIPerf's synthetic multi-turn chat instead of the
  ShareGPT replay (shared system prompt, conversation history, input per request median ~3000 /
  p90 ~6000 tokens), a conversation pool sized from R x D, the input length in the `LENGTHS`
  summary.
- `run_test.sh`: ramp 0 -> 2 req/s over 6000 s (study 33: 6).
- `apply_config.sh`: the startup summary also prints the linear kernel and prefix caching.
- `stop_aiperf.sh` (2026-10-12): the workflow's `Stop AIPerf` task; `run_test.sh` leaves the Job
  to it with `RT_STOP_AIPERF=0`.

| File | What |
|---|---|
| `params.env.template` | Rendered by the workflow's FileConfigurator (`${vllm.*}` tokens) into `params.env`. |
| `render_statefulset.sh` | Validates `params.env` and writes the vLLM flags into `01-statefulset_template.yaml` (booleans as `--x` / `--no-x`; no flag for an empty value, nor for a backend absent or `auto`). |
| `apply_config.sh` | Apply config task: validate, free the GPU, start `vllm-0`, wait (fail fast on a crash loop), 8 warm-up requests, health check, full logs. |
| `run_test.sh`, `render_job.sh`, `05-job_template.yaml`, `lib_watchdog.sh` | RunTest task: synthetic multi-turn chat without `max_tokens` with study 27's open-loop linear rate ramp (`RT_RATE`, `RT_RAMP_S`), watchdog past 2x the SLA, fail-fast guards, full logs. `render_job.sh --closed C N` renders the smoke run's closed-loop length run. |
| `stop_aiperf.sh` | Stop AIPerf task (after RunTest, `alwaysRun`): SIGINT to the aiperf process (graceful: it writes its files), Job deleted, the measured run's `profile_export.jsonl` read from the results PVC by a short-lived pod and saved on the toolbox as `/work/aiperf-exports/33-l40s-gemma4-26b-awq-no-thinking/<epoch>.jsonl` (`SA_DEST`; skipped below `SA_MIN_FREE_MB` = 150 MB free). |
| `lib_health.sh` | The post-warm-up health decision. |
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml`, `02-service.yaml`, `03-hf-secret.yaml` | AIPerf results, AIPerf HF cache (tokenizer), Services (`vllm`, `vllm-headless`), optional HF token template (never commit a real token). |
| `monitoring/` | ServiceMonitor `vllm-l40s`, the dcgm-exporter values with this node role added, counters and kube-prometheus values as deployed. |
| `tests/` | `bash tests/test_*.sh` (yq v4 needed): renderers, guards, watchdog, run_test and stop_aiperf with a stub kubectl (`tests/stub/`, `tests/stub-stop/`). `bash tests/dry_run_multiturn.sh` (network, ~8 min): the rendered Job script, under dash, with real AIPerf 0.11.0 against `mock_openai.py` streaming content only (thinking off): no request carries `max_tokens`, every measured request shares one system prompt, later turns carry the history (no reasoning in it), `LENGTHS` reports the input length and no reasoning tokens; then the ramp exactly as `render_job.sh` writes it (shorter), with the requests following it. |
