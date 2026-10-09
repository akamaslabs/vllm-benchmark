# smoke/ — study 32 manual smoke run

Run outside Akamas before the study is created, from the workstation, after the probe
(`../probe/`), as study 31's: one vLLM start through the workflow's own `../k8s/apply_config.sh`,
then up to two AIPerf Jobs, with the synthetic multi-turn load of `../k8s/05-job_template.yaml`.
`SMOKE_CONFIG` picks the vLLM configuration: `compose` (default, the study's baseline: the
compose's flags only, the other lines empty = vLLM's defaults) or `large` (the `kv fp8 large
batch` preset).

1. **Length run** (`compose`; closed loop, 32 concurrent conversations, 320 requests): the
   `LENGTHS` lines of the Job log give the input per request (`input_sequence_length`: does
   the profile hit the customer's median ~3000 / p90 ~6000 tokens with this checkpoint's
   replies? If not, retune `PROMPT` / `TURNS` in `05-job_template.yaml`), the reasoning and
   answer tokens, TTFT, the time to the first answer token, and the requests near
   `max_model_len` 96000.
2. **Smoke ramp**, twice (`compose`, then `large`): `../k8s/run_test.sh` with a steep ramp over
   1800 s and wide first-trial guards; the watchdog ends it past the knee. The top rate, unless
   `SMOKE_RATE` is given, comes from the length run and the KV cache size vLLM reports (Little's
   law: requests that fit = min(max_num_seqs, KV tokens / (mean input + mean output)), over a
   request's life of mean output x 80 ms, times 2). The study's R must let both
   configurations reach their knee (the compose caps at 64 sequences, the large preset holds
   several times more).

- `smoke_manual.sh`: the run (`SMOKE_OUT=<dir>`; `SMOKE_CONFIG`, `SMOKE_LEN_CONC`,
  `SMOKE_LEN_COUNT`, `SMOKE_RATE`, `SMOKE_RAMP_S` override the defaults; `SMOKE_PHASES=length` or
  `ramp` runs one phase, `SMOKE_SKIP_APPLY=1` reuses the running vLLM, `SMOKE_LEN_LOG` points the
  `large` ramp at the compose's length run). The watchdog reads Prometheus through a
  self-restarting port-forward (`RT_PROM`).
- `smoke_analyze.py <prometheus> <start> <end> [csv]`: the study's telemetry queries at 30 s
  and its scoring (best 6-sample window by completed requests/s with the e2e p95 :max <= 30 s),
  plus TTFT / ITL p95s and the prefix cache hit rate. Run it on each ramp's window (`RUNTEST START` / `RUNTEST END`
  in `times.txt`).
- `results/` (after the run): per configuration `length_run.log`, `run_test.log`,
  `apply_config.log`, `params.env`, `times.txt`, and `summary.txt` / `timeseries.csv` from
  `smoke_analyze.py`.

Results and the R / D decision: study README, "Smoke run".
