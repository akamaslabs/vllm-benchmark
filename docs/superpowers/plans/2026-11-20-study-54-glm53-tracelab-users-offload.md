# Study 54 (GLM-5.3-Flash, TraceLab users, CPU KV offload) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build study 54 as a self-contained study folder that, from one Akamas study on akamas-41,
measures how many TraceLab coding users one GLM-5.3-Flash replica holds within the SLO, for the
baseline, four offload presets and 30 optimized configurations, ending each run at the knee.

**Architecture:** Study 53's apply path (params.env -> `VLLM_ADDITIONAL_ARGS`, Recreate Deployment,
proven arguments) plus an LMCache mode; study 38's scoring path (ramp, TTFT watchdog, graceful stop,
per-request records -> 30 s CSV -> CSV File telemetry, `stability` windowing on users) with AIPerf
0.13 replaying TraceLab pools from the two long-lived client pods. The CSV is cut at the first
sustained SLO breach, so Akamas can only score windows before the knee.

**Tech Stack:** bash + python3 (stdlib, PyYAML for checks), kubectl, AIPerf 0.13.0, vLLM v0.34.0,
LMCache 0.5.8rc1, Akamas 4.1.3 (vLLM, AIPerf, GPU 1.4.0, Kubernetes 1.9.0 packs), Prometheus.

**Spec:** `docs/superpowers/specs/2026-11-20-study-54-glm53-tracelab-users-offload-design.md`
(approved; "Calls for the build" holds the open calls, confirmed at the gate).

## Global Constraints

- Study folder `studies/54-g7e-glm53-tracelab-users-offload/`; Akamas study
  `54-G7E-GLM53-TraceLab-Users-Offload`, system `vLLM_Benchmark_54_G7E_GLM53_TraceLab`, workflow
  `54-G7E-GLM53-TraceLab-Workflow`; toolbox copy `/work/study54/54-g7e-glm53-tracelab-users-offload`.
- Repo files and commits in English; no secrets, no target hostnames/names; commit on master, no push.
- Akamas YAML through the akamas-study-manager plugin; full component and telemetry catalog (CLAUDE.md
  rule 9, base: study 53's catalog) plus the `aiperf` component and its CSV File telemetry.
- No toolbox sync, `akamas create` or start before the user's go at the gate (Task 7).
- SLO: client TTFT p95 over 150 s <= 5000 ms and ITL p95 over 150 s <= 50 ms; each threshold is one
  value in `workload.env` (`SLO_TTFT_MS`, `SLO_ITL_MS`) mirrored in the study's constraints.
- Load: TraceLab v0.0.2, two disjoint pools of 600 sessions (seed 54, horizon 80 min, think time cap
  1200 s, max input 250,000, >= 2 rounds), K = 8,000 shared system prompt tokens subtracted at write,
  `sys8k.txt` (8,000 GLM tokens, md5 7342a71a76a5f5314f8e238a2767f050); AIPerf flags exactly as
  `pod_run.sh` of the calibration (`--custom-dataset-type tracelab --no-fixed-schedule
  --dataset-sampling-strategy shuffle --random-seed 1 --inter-turn-delay-cap-seconds 1200
  --extra-inputs ignore_eos:true --benchmark-grace-period 30`), `AIPERF_DATASET_MMAP_PREFAULT=false`,
  both configure timeouts 1800.
- Ramp: `USERS_PER_CLIENT` and `RAMP_S` in `workload.env` (candidate 50 and 4200: 2 -> 100 users over
  70 min), fixed after the slow-slope ramps; benchmark duration `RAMP_S + 300`.
- Watchdog: server TTFT p95 over 150 s (Prometheus) > 10,000 ms for 120 s, armed 300 s after the first
  completed request; stall 900 s; deadline `RAMP_S + 900`. Experiment <= ~1 h 40.
- Server: `vllm/vllm-openai:v0.34.0`, checkpoint `canada-quant/GLM-5.3-Flash-W4A16-MTP` rev 5681f7c2,
  study 53's fixed arguments; LMCache 0.5.8rc1 via `pip install --no-deps` at start, server
  `--port 5555 --l1-size-gb <size> --eviction-policy LRU`.
- Steps: baseline (gmu 0.95, MTP 5, nothing else rendered); presets Native 128 / 256 / 512, LMCache 512
  (`runOnFailure: true`); optimize 30 experiments, AKAMAS, domains `kv_offloading_size` [0, 512],
  `gpu_memory_utilization` [0.85, 0.97], `max_num_batched_tokens` [2048, 32768], `max_num_seqs`
  [16, 1024], `spec_tokens` [0, 5]; `kv_offloading_backend` rendered at its default (native) outside
  the presets; constraint `max_num_batched_tokens >= max_num_seqs`.
- Akamas rules from memory: step names `^[a-zA-Z\s][a-zA-Z0-9_\s]*$`; `doNotRenderParameters` renders
  "" (empty = no flag); placeholders without underscores; <= 8 KPIs.

## Review Focus

1. A late window that dips back under the SLO after the knee: the CSV for Akamas ends at the first
   sustained breach (4 bad buckets, >= 20 requests in the window); a test pins it (Task 3).
2. A pool that runs out (a session started twice by one client): the run is not scored; the converter
   exits 3 and names the session (Task 3).
3. AIPerf left running in a client pod after Akamas kills RunTest or the study is finished (study 47):
   RunTest reaps both clients before starting and Stop AIPerf after (Task 4, stub test).
4. vLLM restarted or replaced mid-run (spot reclaim, OOM): RunTest fails the trial (Task 4, stub test).
5. LMCache preset whose server never listens, or a worker that dies in warmup: Apply fails with the
   reason line instead of hanging (Task 2, fixture log test).

---

### Task 1: Study folder, load assets and client setup

**Files:**
- Create: `studies/54-g7e-glm53-tracelab-users-offload/` from `studies/_TEMPLATE/` (README, akamas/,
  k8s/, load/, results/, infra/ pointer to study 53's infra)
- Create: `load/prepare_tracelab.py` (copy of the spike's), `load/prepare_pool.py` (the calibration
  copy), `load/sys8k.txt`, `load/prepare_load.sh`, `load/README.md` (source, licence credit, md5s)
- Create: `k8s/workload.env`, `k8s/client_run.sh` (from `pod_run.sh`), `k8s/00-base.yaml`,
  `k8s/02-aiperf-clients.yaml` (study 53's, AIPerf pinned to 0.13.0), `k8s/03-download-checkpoint.yaml`,
  `k8s/download_checkpoint.sh`, `k8s/monitoring/` (copies of study 53's)
- Test: `load/test_prepare_pool.sh`

**Interfaces:**
- Produces: `workload.env` variables `NS URL MODEL USERS_PER_CLIENT RAMP_S SLO_TTFT_MS SLO_ITL_MS
  SYS_TOKENS POOL_SESSIONS POOL0_SHA POOL1_SHA SYS_MD5 LOAD_DIR` (toolbox path of the staged pools,
  default `/work/study54-load`); `client_run.sh POOL RUN CONC RAMP_S DUR_S` in the pod at
  `/tmp/client_run.sh` writing `/tmp/a13/runs/RUN/{aiperf.log,cgroup.log,exit}` and the export under
  `/tmp/a13/runs/RUN/aiperf/`; `prepare_load.sh OUT_DIR` writing `pool_0.jsonl.gz`, `pool_1.jsonl.gz`,
  `sys8k.txt` and checking them.

- [ ] **Step 1: Write `test_prepare_pool.sh`** on a 6-session synthetic TraceLab fixture
  (`load/fixtures/trace_small.jsonl.gz`, written by the test): with `--sessions 2 --pools 2
  --sys-tokens 1000` it asserts two disjoint pools of 2 sessions, every written `input_tokens_total` =
  recorded - 1000, `prefix_tokens` floored at 0, a session with a round under 1,200 dropped, a session
  whose round 3 exceeds 262,144 with its output cut before round 3; same output on a second run
  (sha256 of the decompressed files).
- [ ] **Step 2: Run it** — `bash studies/54-*/load/test_prepare_pool.sh` → FAIL (no script in place).
- [ ] **Step 3: Copy `prepare_pool.py`/`prepare_tracelab.py`**, the spike path argument replaced by
  the script's own directory; `prepare_load.sh` downloads TraceLab (URL in the spike README) into
  OUT_DIR if missing, runs `prepare_pool.py ... --sessions 600 --pools 2 --seed 54 --horizon-min 80
  --cap-s 1200 --sys-tokens 8000`, copies `sys8k.txt`, and fails unless the decompressed sha256s and
  the prompt md5 equal `workload.env`'s (recorded from the calibration pools).
- [ ] **Step 4: Run the test** → PASS; run `prepare_load.sh` once locally against the calibration
  download and check it reproduces the calibration pools' decompressed sha256.
- [ ] **Step 5: `client_run.sh`** = `pod_run.sh` with `aiperf` from PATH, `--url $URL` and the
  prompt at `/tmp/sys8k.txt`; `02-aiperf-clients.yaml` = study 53's with `aiperf==0.13.0`.
  `bash -n` on every script.
- [ ] **Step 6: Commit** — `Study 54: folder, TraceLab pool preparation (K = 8,000), client run
  script, workload.env`.

### Task 2: Apply config with the LMCache mode

**Files:**
- Create: `k8s/apply_config.sh`, `k8s/lib_apply.sh`, `k8s/lib_checks.sh`, `k8s/01-vllm_template.yaml`,
  `k8s/params.env.template` (from study 53)
- Test: `k8s/test_apply_config.sh`, `k8s/test_checks.sh`, `k8s/fixtures/` (from study 53, extended)

**Interfaces:**
- Consumes: `workload.env` (`NS`).
- Produces: `params.env` variables = study 53's plus `KV_OFFLOADING_BACKEND`; `RENDER_ONLY=1` prints the
  args line then `shm=<Gi> mem=<Gi> lmcache=<0|1>`; exit codes as study 53 (2, 3, 4, 6, 8, 9).

- [ ] **Step 1: Extend the tests** (study 53's cases kept): `KV_OFFLOADING_BACKEND=` with size 512 ->
  args end `--kv-offloading-size 512 --kv-offloading-backend native`, `lmcache=0`;
  `KV_OFFLOADING_BACKEND=lmcache` size 512 -> `--kv-offloading-backend lmcache`, `lmcache=1`,
  `mem=608` (96 + 512); `lmcache` with size 0 -> exit 2 ("lmcache needs a size"); backend `foo` ->
  exit 2; size 0 + native -> no offload flags. `startup_failure_reason` on a fixture log with
  `[lmcache] server not listening after 300 s` prints that line.
- [ ] **Step 2: Run** `bash k8s/test_apply_config.sh` → the new cases FAIL.
- [ ] **Step 3: Implement.** The template gets an `@LMCACHE_START@` block before `vllm serve` (the smoke's
  five lines, `--l1-size-gb @OFFLOAD_GI@`), rendered empty unless `lmcache=1`; the annotation becomes
  `study54/applied-at`; the proof step also checks `--kv-offloading-backend` in the pod's arguments
  and, for lmcache, `[lmcache] server listening on 5555` in the log. Rig-specific values (namespace,
  image, model path, served name, `OFFLOAD_MAX`, `MEM_BASE`, node group) come from `workload.env`, not
  from the script body (design "Portability").
- [ ] **Step 4: Run** both test scripts → PASS.
- [ ] **Step 5: Commit** — `Study 54: apply_config with the LMCache 0.5.8rc1 server mode`.

### Task 3: AIPerf 0.13 records -> CSV, cut at the knee, offline score

**Files:**
- Create: `k8s/aiperf_to_csv.py` (from study 38's), `k8s/score_offline.py`
- Test: `k8s/test_aiperf_to_csv.sh`, `k8s/fixtures/export_c0.jsonl`, `export_c1.jsonl`, `pool_c0.jsonl.gz`,
  `pool_c1.jsonl.gz` (small, synthetic)

**Interfaces:**
- Produces: `aiperf_to_csv.py --pair EXPORT POOL [--pair EXPORT POOL] --out CSV --full-out CSV
  [--slo-ttft-ms 5000] [--slo-itl-ms 50] [--min-window-reqs 20] [--breach-buckets 4]` -> exit 0, 2 (no
  usable record) or 3 (a session started twice by one client); stdout line `knee: bucket <TS> users <n>
  (<TTFT|ITL|TTFT and ITL>)` or `knee: none`. Columns = study 38's (the AIPerf pack's metrics), except
  `aiperf_active_sessions` = sessions started - sessions finished at the bucket's end over both clients
  (finished = the pool's last round of that session ended; flattened-agent conversations count as
  sessions). `score_offline.py CSV [--width 6]` prints `score <users> window <TS..TS>` (max average users
  over windows of 6 rows whose average TTFT p95_150s and ITL p95_150s are within the SLO; empty cells
  ignored) or `score none`.

- [ ] **Step 1: Write the tests**: two clients, a ramp of synthetic sessions whose ITL breaks for 4
  buckets at minute 10 and recovers at minute 14 -> `--out` stops before minute 10's bucket,
  `--full-out` keeps every bucket, `score_offline.py --out` < score on `--full-out`; a breach of 3
  buckets only -> no cut; a bucket with 19 requests in the window never counts as bad; a conversation
  id with `turn_index` 0 twice in one client -> exit 3; users rise by one when a session starts and fall
  when its last pool round ends, not between its rounds.
- [ ] **Step 2: Run** `bash k8s/test_aiperf_to_csv.sh` → FAIL.
- [ ] **Step 3: Implement** (study 38's aggregation; the 0.13 record fields `metadata.conversation_id`,
  `turn_index`, `request_start_ns`, `request_end_ns`, top-level `error`; the knee rule of the
  calibration's `ramp_buckets.py`).
- [ ] **Step 4: Run** → PASS. **Then replay the calibration exports** (`ramps/r1..r3`, `s-*`): the
  knee line must match `ramp_buckets.py`'s for each; record the scores in the design (scoring check).
- [ ] **Step 5: Commit** — `Study 54: AIPerf 0.13 CSV converter cut at the first sustained SLO breach,
  offline scorer`.

### Task 4: RunTest, Stop AIPerf, AIPerf to CSV

**Files:**
- Create: `k8s/run_test.sh`, `k8s/lib_watchdog.sh` (study 38's, generalized to a threshold variable),
  `k8s/stop_aiperf.sh`, `k8s/aiperf_csv.sh`, `k8s/prom_query.sh`
- Test: `k8s/test_watchdog.sh`, `k8s/test_run_test.sh` (stub `kubectl` on PATH)

**Interfaces:**
- Consumes: `workload.env`, `client_run.sh`, `aiperf_to_csv.py`, `score_offline.py`.
- Produces: `run_test.sh` exit 0 (watchdog fired or ramp completed), 1 (vLLM restarted/replaced, stall,
  deadline, a client exited non-zero before the watchdog), 6 (a client not ready: AIPerf not 0.13.0,
  pool or prompt checksum wrong); the run name `RUN=s54-<epoch>` written to
  `/work/study54-runs/current`; `stop_aiperf.sh` sends SIGINT to AIPerf in both pods (300 s), reaps
  anything left, copies each client's `profile_export.jsonl` to `/work/study54-runs/<RUN>/c<i>.jsonl.gz`;
  `aiperf_csv.sh` writes `/work/aiperf-csv/study54/<RUN>.csv` (cut) and
  `/work/aiperf-csv-full/study54/<RUN>.csv`, prints the knee and the offline score.
  `prom_query.sh '<promql>'` = instant query through `kubectl get --raw` to
  `monitoring/kube-prometheus-stack-prometheus:http-web`.

- [ ] **Step 1: Write the tests**: `wd_over 10001` true / `10000` false / empty false; with the stub,
  RunTest reaps a leftover `aiperf profile` in both pods before launching, fails with exit 1 when the
  stub's pod uid changes, exits 6 when the stub's pool sha is wrong, launches both clients with
  `client_run.sh /tmp/pool.jsonl.gz <RUN> 50 4200 4500`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** (study 38's loop; arming on vLLM's `request_success_total` rising, then
  `WD_ARM_DELAY_S`; the client checks copy pool/prompt/`client_run.sh` into a pod whose checksum
  differs; logs: both clients' `aiperf.log` tails and `cgroup.log` tail dumped unconditionally).
- [ ] **Step 4: Run** → PASS; `bash -n` and `shellcheck` if available.
- [ ] **Step 5: Commit** — `Study 54: RunTest with the TTFT watchdog, graceful stop, CSV task`.

### Task 5: Akamas resources (akamas-study-manager plugin)

**Files:**
- Create: `akamas/system.yaml`, `akamas/components/*.yaml` (study 53's catalog renamed + `aiperf.yaml`),
  `akamas/telemetry/prometheus.yaml` (study 53's, renamed), `akamas/telemetry/aiperf-csv.yaml`
  (study 38's, pattern `/work/aiperf-csv/study54/*.csv`), `akamas/54-G7E-GLM53-TraceLab-Workflow.yaml`,
  `akamas/54-G7E-GLM53-TraceLab-Users-Offload.yaml`, `akamas/README.md`
- Test: `k8s/test_akamas.sh` (study 53's, adapted)

**Interfaces:**
- Consumes: script paths and exit codes of Tasks 2-4; `params.env.template` tokens.
- Produces: workflow tasks `Write config`, `Apply config` (timeout 140m), `RunTest` (timeout =
  deadline + 10 min), `Stop AIPerf` (alwaysRun, critical false, 20m), `AIPerf to CSV` (10m); goal
  maximize `aiperf.aiperf_active_sessions`; constraints on `aiperf.aiperf_time_to_first_token_p95_150s`
  and `aiperf.aiperf_inter_token_latency_p95_150s`; windowing `stability` width 6, maxStdDev 300000000,
  `when` max on users; 8 KPIs (users, TTFT p95 150 s, ITL p95 150 s, completed requests/s, prefix hit
  incl. offload, CPU offload hit, offload read-back GB/s, GPU KV usage).

- [ ] **Step 1: Write `test_akamas.sh`**: goal and constraint thresholds equal `workload.env`'s SLO values;
  every step name matches the regex; presets carry `runOnFailure: true`; offload values [none, 128, 256,
  512, 512] and backend lmcache only on `LMCache 512`; optimize step `numberOfExperiments: 30` and
  `renderParameters` includes `vllm.kv_offloading_backend`; every `${vllm.x}` token in
  `params.env.template` is in `parametersSelection` or a step's `renderParameters`; every workflow
  command path exists under the study; RunTest timeout > `RAMP_S + 900` s; <= 8 KPIs; every
  telemetry metric a KPI uses is declared.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Generate the YAML with `/akamas-study-manager:build`** (pack vocabulary read on akamas-41
  with `akamas describe component-type`); README "Setup & run" with the create/start commands.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `Study 54: Akamas system, catalog, telemetry, workflow and study`.

### Task 6: Integration on the node (after the slow-slope ramps)

- [ ] **Step 1:** Announce, then apply `02-aiperf-clients.yaml` (clients restart with AIPerf 0.13.0;
  the dataset cache rebuilds on the first run, ~200 s each).
- [ ] **Step 2:** From the laptop, with `STUDY_DIR` = the repo folder and a hand-written `params.env`
  for `LMCache 512`: `apply_config.sh` -> exit 0, pod proven; then `run_test.sh` with
  `USERS_PER_CLIENT=5 RAMP_S=300` and `WD_TTFT_MS=1` (forces the watchdog) -> exit 0 within ~10 min;
  `stop_aiperf.sh` -> both exports saved; `aiperf_csv.sh` -> CSV with rows, knee and score lines.
  Expected: every step exits 0 and the CSV's TS spans the run in UTC.
- [ ] **Step 3:** Same with a Native params.env for one apply only (exit 0, `--kv-offloading-backend
  native` proven).
- [ ] **Step 4: Commit** fixes, the README's stack and the integration notes.

### Task 7: Scoring check and the gate

- [ ] **Step 1:** Slow-slope ramps through `ramp_buckets.py`, `aiperf_to_csv.py` and `score_offline.py`:
  knees per configuration, offline scores, baseline vs Native 512 before the KV fills.
- [ ] **Step 2:** Fix `USERS_PER_CLIENT`/`RAMP_S` in `workload.env` (best knee in the first two thirds
  of the ramp), update the design and the README; `bash k8s/test_*.sh` all PASS; commit.
- [ ] **Step 3: The gate:** one message to the user (Italian) with the slow-slope numbers, the values that
  freeze (SLO thresholds, N and R, backend, domains) and my recommendation; ask one go for toolbox
  sync, `akamas create` and start.
- [ ] **Step 4 (after the go):** sync to `/work/study54`, stage the pools in `/work/study54-load`,
  `akamas create` in dependency order, `akamas describe`, start, check the first experiment's Apply
  and RunTest logs and the first CSV rows against the trial window.
