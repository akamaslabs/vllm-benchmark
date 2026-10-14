# k8s/: 34-l40s-gemma4-26b-awq-customer-replay

Study 35's `k8s/` (2026-10-13) back on the vllm-bench cluster, with output caps: every script calls
`kubectl --context $KUBE_CONTEXT` (default `default`, the toolbox's in-cluster context), namespace
`llm-l40s`, GPU node `node-role=llm-serving-l40s-1xl` (studies 30-33's, checkpoint cached in
`/var/lib/hf-cache-study30`), AIPerf on `node-role=system-m8a`, results under `/benchmarks/study34`.
**The difference from study 35:** the Job runs `caps.py apply` after the render, so every request
asks for its calibrated output length with `ignore_eos` (`../README.md` "Load").

| File | Role |
|---|---|
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml`, `02-service.yaml`, `03-hf-secret.yaml`, `monitoring/` | The namespace's layer, already in place from studies 30-33 (`../infra/`): results and tokenizer volumes, Services `vllm` / `vllm-headless`, ServiceMonitor `vllm-l40s`, optional HF token, the monitoring stack's values |
| `01-statefulset_template.yaml`, `render_statefulset.sh`, `params.env.template`, `apply_config.sh`, `lib_health.sh` | Apply config task: the customer's compose (thinking off by default) plus the 17 tuned flags; empty value = no flag = vLLM's default; refuses to start while another vLLM holds the GPU |
| `run_test.sh`, `render_job.sh`, `05-job_template.yaml`, `run_benchmark.sh`, `lib_watchdog.sh` | RunTest task: their `run_benchmark.sh` unchanged (sha256 checked), users 1 -> `RT_USERS` over `RT_RAMP_S`, watchdog on vLLM's e2e p50 (150 s) > 6 s for 120 s, fail-fast guards |
| `render_dataset_it.py`, `build_corpus.py`, `mix.json` | The template rendered with Italian texts in the Job (`../README.md` "Load"); the corpus is built once on the results volume; `mix.json` overrides the task weights (empty: the calibrated defaults) |
| `caps.py`, `calibrate_caps.sh`, `calibration.params.env`, `07-calibration_pod_template.yaml` | **Study 34's output caps.** `calibrate_caps.sh` (once, by hand, before the study): vLLM with the compose's numerics (`calibration.params.env`), the dataset rendered as the Job renders it, `caps.py measure` on every request the ramp can reach (uncapped, not streamed) -> `caps.json.gz` (output tokens per `<session>/<turn>`, task medians, the dataset's sha256) next to the template on the toolbox. `run_test.sh` ships it in the ConfigMap (exit 2 without it); the Job's `caps.py apply` refuses another dataset and writes `max_output_tokens` + `ignore_eos` into every payload (one `CAPS:` line) |
| `stop_aiperf.sh` | Stop AIPerf task: SIGINT to AIPerf, Job deleted, records saved on the toolbox as `/work/aiperf-exports/<study>/<epoch>.jsonl.gz` (gzipped, sha256-checked copy) |
| `aiperf_csv.sh`, `aiperf_to_csv.py` | AIPerf to CSV task: the newest records (written in the last 30 min) -> `/work/aiperf-csv/<study>/<epoch>.csv` for the CSV File telemetry; `aiperf_to_csv.py` is the AIPerf pack 1.0.0's `tools/` script, byte for byte |
| `tests/` | `test_*.sh` (stub kubectl; yq v4 for the render tests; `test_caps.sh`: measure against a local non-streaming mock, apply, refusal of another dataset), `dry_run_replay.sh` (the Job script with real AIPerf 0.11.0 against `mock_openai.py`; `DRY_CORPUS=<corpus_it.json.gz>` needed: it computes the caps from its own render first) |

`render_dataset_it.py` and `build_corpus.py` are copies of `../../../probes/gemma4-customer-dataset-it/`
(where the calibration lives): change both together.
