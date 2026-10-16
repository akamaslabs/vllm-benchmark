# smoke/ — study 39 manual smoke run

The Akamas Smoke study's work (`39-C8i-Qwen3-06B-CPU-TPS-Smoke`), done outside Akamas when it is
not at hand: the workflow's own `../k8s/apply_config.sh` (baseline values: vLLM 0.31.0 CPU
defaults except gpu_memory_utilization 0.90) and `../k8s/run_test.sh` with the smoke ramp (default 0 -> 40 req/s over 900 s, as
the Smoke workflow; `SMOKE_RATE` / `SMOKE_RAMP_S` to try the probe's proposal), run from the
workstation.

- `smoke_manual.sh`: the run; the watchdog reads Prometheus through a self-restarting
  port-forward (`RT_PROM`). `SMOKE_OUT=<dir>` for the output. Ends with `smoke_analyze.py`.
- `smoke_analyze.py <prometheus> <start> <end> [csv]`: the study's telemetry queries at 30 s
  and its scoring (best 6-sample window with TTFT p95 :max <= 3000 and TPOT p95 :max <= 100),
  plus prefix-cache hit rate, KV usage, preemptions, CPU of vLLM / AIPerf and vLLM's memory
  working set.
- `results/` (written by the run): `summary.txt`, `timeseries.csv`, `run_test.log`,
  `apply_config.log`, `params.env`, `times.txt`.

TODO(calibrate): the main workflow's `RT_RATE` / `RT_RAMP_S` are placeholders until the probe
and this smoke run locate the baseline's knee.
