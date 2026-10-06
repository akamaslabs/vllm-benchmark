# smoke/ — study 30 manual smoke run

The Akamas smoke study's work, done outside Akamas on 2026-10-06 while the Akamas 4.1 instance
was being installed: the workflow's own `../k8s/apply_config.sh` (baseline values) and
`../k8s/run_test.sh` with the smoke ramp (0 -> 40 req/s over 900 s), run from the workstation.

- `smoke_manual.sh`: the run; the watchdog reads Prometheus through a self-restarting
  port-forward (`RT_PROM`). `SMOKE_OUT=<dir>` for the output.
- `smoke_analyze.py <prometheus> <start> <end> [csv]`: the study's telemetry queries at 30 s
  and its scoring (best 6-sample window with TTFT p95 :max <= 1500 and ITL p95 :max <= 300).
- `results/`: `summary.txt` (table and score), `timeseries.csv`, `run_test.log`,
  `apply_config.log`, `params.env`, `times.txt`.

Results: study README, "Smoke run".
