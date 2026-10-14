# k8s/: 36-l40s-gemma4-26b-awq-replay-users-v2

Study 35's `k8s/` moved back to the vllm-bench cluster (study 34's settings): every script calls
`kubectl --context $KUBE_CONTEXT` (default `default`, the toolbox's in-cluster context), namespace
`llm-l40s`, GPU node `node-role=llm-serving-l40s-1xl`, AIPerf on `node-role=system-m8a`.
`wna16_activation_dtype` is rendered as environment (`VLLM_MARLIN_INPUT_DTYPE`,
`VLLM_HUMMING_INPUT_QUANT_CONFIG`) at the template's TUNED_ENV line.

| File | Role |
|---|---|
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml`, `02-service.yaml`, `03-hf-secret.yaml`, `monitoring/servicemonitor.yaml` | The study's layer (`../infra/provision.sh`): results and tokenizer volumes, Services `vllm` / `vllm-headless`, ServiceMonitor `vllm-l40s`, optional HF token |
| `01-statefulset_template.yaml`, `render_statefulset.sh`, `params.env.template`, `apply_config.sh`, `lib_health.sh` | Apply config task: the customer's compose (thinking off by default) plus the tuned flags and environment; empty value = no flag = vLLM's default; frees the GPU from any earlier StatefulSet of the namespace first |
| `run_test.sh`, `render_job.sh`, `05-job_template.yaml`, `run_benchmark.sh`, `lib_watchdog.sh` | RunTest task: their `run_benchmark.sh` unchanged (sha256 checked), users 1 -> `RT_USERS` over `RT_RAMP_S`, watchdog on vLLM's e2e p50 (150 s) > 6 s for 120 s, fail-fast guards |
| `render_dataset_it.py`, `build_corpus.py`, `mix.json` | The template rendered with Italian texts in the Job (`../README.md` "Load"); the corpus is built once on the results volume; `mix.json` overrides the task weights (empty: the calibrated defaults) |
| `stop_aiperf.sh` | Stop AIPerf task: SIGINT to AIPerf, Job deleted, records saved on the toolbox as `/work/aiperf-exports/<study>/<epoch>.jsonl.gz` (sha256-checked copy) |
| `aiperf_csv.sh`, `aiperf_to_csv.py` | AIPerf to CSV task: the newest records (written in the last 30 min) -> `/work/aiperf-csv/<study>/<epoch>.csv` for the CSV File telemetry; `aiperf_to_csv.py` is the AIPerf pack 1.0.0's `tools/` script, byte for byte |
| `tests/` | `test_*.sh` (stub kubectl; yq v4 for the render tests), `dry_run_replay.sh` (the Job script with real AIPerf 0.11.0 against `mock_openai.py`) |

`render_dataset_it.py` and `build_corpus.py` are copies of `../../../probes/gemma4-customer-dataset-it/`
(where the calibration lives): change both together.
