# k8s/ — 34-l40s-gemma4-26b-awq-customer-replay

Study 33's `k8s/` (itself study 32's with thinking off) with thinking back on and the
customer's own AIPerf replay as the load (copied 2026-10-12). One StatefulSet replica (`vllm-0`)
on the whole L40S, one AIPerf Job, everything in namespace `llm-l40s`: the same namespace, node,
pod, Job, PVCs, Services and ServiceMonitor as studies 30-33, which run before this study on the
same node; the served model name (`aiservices_gemma`) keeps the studies' vLLM series apart.

What differs from study 33:
- `01-statefulset_template.yaml`: `--default-chat-template-kwargs '{"enable_thinking": true}'`
  (study 32's), served model name `aiservices_gemma` (the name the customer's payloads carry).
- `run_benchmark.sh` (new): **the customer's script, byte-identical** to what they sent
  (2026-10-12; `tests/test_render_job.sh` checks its sha256). Never edit it: the Job wraps it.
- `05-job_template.yaml`, `render_job.sh`: the Job installs AIPerf 0.11.0 where their script
  looks for it (`../llm-coding-benchmark/.venv`), renders the dataset, writes their
  `benchmark.env` (ENDPOINT vllm-0, MODEL `aiservices_gemma`, TOKENIZER, DATASET, USERS, RAMP =
  DURATION, CAP 60, GRACE 60), prints the watchdog's marker and runs `bash run_benchmark.sh
  NAME=<epoch>`; then the `LENGTHS` lines from their run folder. Their flags: `/v1/responses`,
  `mooncake_trace` payloads sent verbatim, `--concurrency` users with
  `--concurrency-ramp-duration`, sticky user sessions, think-time cap, seed 42, server token
  counts, records export. No synthetic load, no rate ramp, no warm-up in the Job.
  `render_job.sh <users> <ramp_s> <out>`; the Job renders USERS + 20 % users (at least + 10).
- `render_dataset.py` (new): fills the template's placeholders with text of the same length
  from AIPerf's `shakespeare.txt`, one copy of a template session per user (texts shared:
  see its docstring).
- `run_test.sh`: users 1 -> **100** over 6000 s (`RT_USERS`, `RT_RAMP_S`), provisional (study
  README "Load"); the template read from the toolbox (`RT_TEMPLATE`, `../load/README.md`) and put,
  gzipped, with `render_dataset.py` and `run_benchmark.sh` into the ConfigMap `aiperf-replay`
  the Job mounts.
- `stop_aiperf.sh`: SIGINT to the top aiperf process (now the child of their script, not of the
  Job's sh) and the records read from their run folder
  (`/benchmarks/study34/replay-benchmark/data/results/run_<date>_<epoch>`).
- `tests/`: `test_render_dataset.sh` (new, on the made-up `fixture_template.jsonl`),
  `test_render_job.sh` and `test_run_test.sh` rewritten, `enable_thinking true` asserted;
  `dry_run_replay.sh` replaces `dry_run_multiturn.sh`; `mock_openai.py` speaks `/v1/responses`.
- `smoke/` dropped (study 32's synthetic smoke run).

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
- `run_test.sh`: ramp 0 -> 2 req/s over 6000 s (study 33: 6; study 34 ramps users instead).
- `apply_config.sh`: the startup summary also prints the linear kernel and prefix caching.
- `stop_aiperf.sh` (2026-10-12): the workflow's `Stop AIPerf` task; `run_test.sh` leaves the Job
  to it with `RT_STOP_AIPERF=0`.

| File | What |
|---|---|
| `params.env.template` | Rendered by the workflow's FileConfigurator (`${vllm.*}` tokens) into `params.env`. |
| `render_statefulset.sh` | Validates `params.env` and writes the vLLM flags into `01-statefulset_template.yaml` (booleans as `--x` / `--no-x`; no flag for an empty value, nor for a backend absent or `auto`). |
| `apply_config.sh` | Apply config task: validate, free the GPU, start `vllm-0`, wait (fail fast on a crash loop), 8 warm-up requests, health check, full logs. |
| `run_test.sh`, `render_job.sh`, `render_dataset.py`, `run_benchmark.sh`, `05-job_template.yaml`, `lib_watchdog.sh` | RunTest task: the customer's `run_benchmark.sh` unchanged (their template rendered in the Job, users ramping 1 -> `RT_USERS` over `RT_RAMP_S`, closed loop), watchdog past 2x the SLA, fail-fast guards, full logs. |
| `stop_aiperf.sh` | Stop AIPerf task (after RunTest, `alwaysRun`): SIGINT to the aiperf process (graceful: it writes its files), Job deleted, the measured run's `profile_export.jsonl` read from the results PVC by a short-lived pod and saved on the toolbox as `/work/aiperf-exports/34-l40s-gemma4-26b-awq-customer-replay/<epoch>.jsonl` (`SA_DEST`; skipped below `SA_MIN_FREE_MB` = 150 MB free); the copy travels gzipped and must match the pod's sha256, up to `SA_COPY_TRIES` = 3 tries (a plain `kubectl exec cat` truncated a 13 MB export, 2026-10-13). |
| `lib_health.sh` | The post-warm-up health decision. |
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml`, `02-service.yaml`, `03-hf-secret.yaml` | AIPerf results, AIPerf HF cache (tokenizer), Services (`vllm`, `vllm-headless`), optional HF token template (never commit a real token). |
| `monitoring/` | ServiceMonitor `vllm-l40s`, the dcgm-exporter values with this node role added, counters and kube-prometheus values as deployed. |
| `aiperf_to_csv.py` | **Not wired yet.** Copy, byte for byte, of the AIPerf optimization pack's `tools/aiperf_to_csv.py` (pack 1.0.0, `gitlab.com/akamas/optimization-packs/aiperf`; sha256 `98241fdf05f9cb458197aa44bffcc1c954d462b1482414e785b47bcefa9f487c`): turns a saved `<epoch>.jsonl` into the CSV (one row per 30 s, columns = the AIPerf component type's metrics, all `aiperf_*`) for a CSV File telemetry instance. Edit it in the pack and copy it again; its tests live there. |
| `tests/` | `bash tests/test_*.sh` (yq v4 needed; `test_render_dataset.sh` needs AIPerf's corpus: `CORPUS=<path>` or aiperf installed): renderers, guards, watchdog, run_test and stop_aiperf with a stub kubectl (`tests/stub/`, `tests/stub-stop/`). `bash tests/dry_run_replay.sh [users] [ramp_s]` (network, ~3 min): the rendered Job script, under dash, with real AIPerf 0.11.0 against `mock_openai.py`'s `/v1/responses` (reasoning, then answer deltas) on `fixture_template.jsonl`: payloads as rendered (model `aiservices_gemma`, no placeholder, no `text`, no output cap), instructions shared by length, histories per user, reasoning tokens and the time to the first answer token in `LENGTHS`. `DRY_TEMPLATE=../load/dataset_u000.template.jsonl` runs it on the customer's template. |
