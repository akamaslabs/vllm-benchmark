# smoke/ — study 31 manual smoke run

Run outside Akamas before the study is created, from the workstation, as study 30's: one vLLM
start with the baseline values through the workflow's own `../k8s/apply_config.sh`, then two
AIPerf Jobs.

1. **Length run** (closed loop, 32 concurrent, 320 requests): how many tokens Gemma 4 reasons
   and answers on ShareGPT prompts once `max_tokens` is gone. The Job log ends with the
   `LENGTHS` lines (reasoning / answer tokens, TTFT, time to the first answer token, outputs
   that may have hit `max_model_len`). They decide `--max-model-len` (16384 is
   provisional) and give the expected knee.
2. **Smoke ramp**: `../k8s/run_test.sh` with a steep ramp over 1800 s and wide first-trial
   guards; the watchdog ends it past the knee. The top rate comes from the length run, 2.5 x
   2900 / mean output tokens per request (2900 = study 30's baseline generated tokens/s at
   its knee), so the knee falls mid-ramp even if thinking halves the tokens/s at the knee.

- `smoke_manual.sh`: the run (`SMOKE_OUT=<dir>`; `SMOKE_LEN_CONC`, `SMOKE_LEN_COUNT`,
  `SMOKE_RATE`, `SMOKE_RAMP_S` override the defaults; `SMOKE_PHASES=length` or `ramp` runs
  one phase, `SMOKE_SKIP_APPLY=1` reuses the running vLLM). The watchdog reads Prometheus through
  a self-restarting port-forward (`RT_PROM`).
- `smoke_analyze.py <prometheus> <start> <end> [csv]`: the study's telemetry queries at 30 s
  and its scoring (best 6-sample window with TTFT p95 :max <= 1500 and ITL p95 :max <= 300).
  Run it on the ramp's window (`RUNTEST START` / `RUNTEST END` in `times.txt`).
- `results/` (after the run): `length_run.log`, `run_test.log`, `apply_config.log`,
  `params.env`, `times.txt`, and `summary.txt` / `timeseries.csv` from `smoke_analyze.py`.

Results and the R / D decision: study README, "Smoke run".
