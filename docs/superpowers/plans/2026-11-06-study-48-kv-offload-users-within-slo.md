# Study 48 — Users within TTFT SLO with CPU KV offload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and start the Akamas study `48-G7-KV-Offload-Users`. For each configuration, a
user ramp finds the most users served with client TTFT p95 ≤ 500 ms. A Hold task re-runs that
level, and Akamas scores it from Prometheus.

**Architecture:**
- The study folder is a copy of study 47.
- `run_load.sh`'s single AIPerf run becomes a reusable `run_level` in `k8s/lib_load.sh`. It
  keeps the reaper, the counter snapshots and the RESULT parser.
- `run_ramp.sh` walks the fixed levels with a 500 ms SLO check in `lib_checks.sh`, and writes
  `level.current`.
- `run_hold.sh` re-runs that level; the study windows on it.
- Akamas workflow: Write config → Apply config → Ramp → Hold.

**Tech Stack:** bash, kubectl, python3 (inside the AIPerf pod), AIPerf 0.11.0, vLLM 0.29.0,
Akamas 4.1 (`akamas-41`, toolbox), vLLM pack 1.15.0, Kubernetes pack 1.9.0, EKS `vllm-bench`.

**Spec:** `docs/superpowers/specs/2026-11-06-study-48-kv-offload-users-within-slo-design.md`

## Global Constraints

- Per-user workload identical to study 47:
  - TURNS=10, ISL=600, OSL=200 (`ignore_eos`), history re-sent;
  - one turn per user every 64 s, so `RATE = USERS / 64`.
- Levels `16 24 32 48 64 80 96 112 128`. Per level, a fresh AIPerf run of 960 s plus 60 s of grace.
- Each level is judged on requests started ≥ 600 s after its first request. It passes when:
  - client TTFT p95 ≤ 500 ms;
  - completed ≥ 95 % of sent;
  - errors < 1 %;
  - sent ≥ 95 % of the expected count.
- The Ramp stops at the first failing level.
  - If no level passes, Hold runs at 16.
  - If 128 passes, it prints a WARNING that the score is capped.
- Hold: the same run at `level.current`. Its client check fails the trial only on errors,
  completion or falling behind, never on TTFT.
- Akamas windowing: trim `[620s, 90s]` on task `Hold`, one unit per value.
- Goal: maximize `64 * vllm.request_success_rate - vllm.time_to_first_token_p95 / 1000`.
- Constraints (absolute):
  - `vllm.time_to_first_token_p95 <= 600`;
  - `vllm.inter_token_latency_p95 <= 120`.
- Parameters and domains exactly as in study 47:

  | Parameter | Domain |
  |---|---|
  | `kv_offloading_size` | [0, 28] |
  | `kv_cache_dtype` | {auto, fp8} |
  | `gpu_memory_utilization` | [0.80, 0.94] |
  | `max_num_seqs` | [16, 256] |
  | `max_num_batched_tokens` | [1024, 16384] |
  | `block_size` | {"16","32","64","128"} |

  Plus mnbt ≥ mns.
- Steps, with values and `doNotRenderParameters: [vllm.block_size]` on the 5 fixed steps as in
  study 47:
  - `baseline`;
  - `Offload only` (24);
  - `FP8 KV only`;
  - `Offload and FP8`;
  - `Baseline repeat`;
  - `optimize`: AKAMAS, 3 init, **10 experiments**, 5 max failed.
- AIPerf pod memory 8 Gi. vLLM pod memory = offload + 20 Gi, shm = offload + 1 Gi (2 Gi when off).
- Same node group `llm-serving-g7-kv`, namespace `kv-offload` (study 47 is finished), toolbox
  path `/work/study48/48-g7-kv-offload-users-within-slo`.
- Akamas YAML only through the `akamas-study-manager` plugin. Validate with `akamas create`.
- Repo and commits in English. No secrets. Commit on branch `study-48-kv-offload-users`, no push.
- Tag the node AlwaysOn while the study runs. Scale it to 0 at the end.

## Review Focus

1. **vLLM OOM or eviction at a high level during the Ramp.** Expected: the run exits 5 with the
   reason, as in study 47, and the experiment fails visibly. It must never silently score a
   lower level. Test in Task 3 (`ramp: restart -> exit 5`).
2. **AIPerf exits non-zero, or produces no RESULT, at one level.** Expected: that level counts
   as failing, the ramp stops there, and `level.current` holds the last passing level. Test in
   Task 3.
3. **Node memory:** 28 + 20 + 8 GiB plus the daemonset requests must fit the 57 GiB allocatable.
   Expected: the vLLM pod schedules at offload 28. Checked in the Task 5 smoke run, with
   `kubectl describe node` Allocated resources.
4. **Stale AIPerf processes between levels.** Expected: every `run_level` reaps first, so
   level N+1 never runs alongside level N. This is already covered by study 47's reaper, now in
   `run_level`, and checked by the smoke run's "stale processes killed: 0".
5. **Hold at a level whose TTFT lands between 500 and 600 ms.** Expected: VALID in Akamas,
   flagged in the README. No test; it is a reporting rule in Task 6.

---

### Task 1: Scaffold study 48 from study 47

**Files:**
- Create `studies/48-g7-kv-offload-users-within-slo/` by copying from study 47:
  - `infra/`;
  - `k8s/` without `params.env`, `01-vllm.yaml` and `run_load.sh`;
  - `akamas/` as a starting point, rewritten in Task 4.
- Modify in the copy:
  - `k8s/apply_config.sh`: default `STUDY_DIR`;
  - `k8s/01-vllm_template.yaml`: annotation `study48/applied-at`, hostPath
    `/var/lib/hf-cache-study48`;
  - `k8s/02-aiperf.yaml`: memory 8Gi;
  - `infra/eks/gpu-nodegroup.sh`: tag `Study=48-kv-offload-users`;
  - comments that say "study 47".
  - The node group and namespace stay the same.

**Interfaces:**
- Produces the folder layout used by every later task, at `STUDY=studies/48-g7-kv-offload-users-within-slo`.

- [ ] Step 1: `git switch -c study-48-kv-offload-users`, then copy the folder.
- [ ] Step 2: Apply the renames above.
  - Run: `grep -rn "study47\|47-g7\|/study47" $STUDY`
  - Expected: no matches, except README references to study 47 as a prior study.
- [ ] Step 3: Run `bash $STUDY/k8s/test_apply_config.sh` and `bash $STUDY/k8s/test_checks.sh`.
  - Expected: 26 passed and 7 passed.
- [ ] Step 4: Commit "Study 48: scaffold from study 47 (same node group and namespace)".

### Task 2: SLO-aware level check

**Files:**
- Modify: `$STUDY/k8s/lib_checks.sh`
- Test: `$STUDY/k8s/test_checks.sh`

**Interfaces:**
- Produces `check_level SUMMARY EXPECTED_SENT SLO_MS`.
  - Returns 0 on pass, 1 on fail, with the reason on stdout.
  - It applies `check_result` first, then `ttft_p95_ms <= SLO_MS`.
  - An `NA` or `nan` TTFT fails.

- [ ] Step 1: Add the failing tests to `test_checks.sh`, using its `expect`-style helper for
  `check_level`:
  - `level ok`: `RESULT sent=360 completed=360 errors=0 ttft_p50_ms=190 ttft_p95_ms=480 itl_p95_ms=15`, expected 360, SLO 500 → 0;
  - `level at SLO`: ttft_p95_ms=500 → 0;
  - `level over SLO`: ttft_p95_ms=501 → 1;
  - `level ttft nan`: ttft_p95_ms=nan with sent=360 → 1;
  - `level fell behind`: sent=300, expected 360 → 1.
- [ ] Step 2: Run `bash test_checks.sh`. Expected: the 5 new tests FAIL (`check_level: command not found`).
- [ ] Step 3: Implement `check_level` in `lib_checks.sh`.
- [ ] Step 4: Run `bash test_checks.sh`. Expected: 12 passed, 0 failed.
- [ ] Step 5: Commit "Study 48: check_level (client TTFT p95 SLO on top of the trial checks)".

### Task 3: lib_load.sh (run_level) and the Ramp/Hold scripts

**Files:**
- Create:
  - `$STUDY/k8s/lib_load.sh`;
  - `$STUDY/k8s/run_ramp.sh`;
  - `$STUDY/k8s/run_hold.sh`;
  - `$STUDY/k8s/test_ramp.sh`.
- Modify: `$STUDY/k8s/workload.env`.
  - Keep TURNS, ISL and OSL.
  - Add `LEVELS="16 24 32 48 64 80 96 112 128"`, `LEVEL_SECONDS=960`, `STEADY_FROM=600`,
    `SLO_TTFT_MS=500`, `SECONDS_PER_TURN=64`.
  - Drop USERS, RATE and DURATION.
- Modify: `$STUDY/k8s/.gitignore` to add `level.current`.

**Interfaces:**
- `lib_load.sh` provides:
  - `run_level USERS SECONDS LABEL`: prints the summary of study 47's `run_load.sh`, unchanged:
    the "first request" line, the RESULT line, the per-minute table, the counter deltas and
    the AIPerf log tail;
    - it reaps first, with exit 6 if the reaper fails;
    - `RATE = USERS / SECONDS_PER_TURN`, with 4 decimals;
    - it returns AIPerf's exit code and never exits on a failed check;
  - `restarts`: same as study 47;
  - `expected_sent USERS`: `(LEVEL_SECONDS - STEADY_FROM) * USERS / SECONDS_PER_TURN`, an
    integer;
  - `t` and `die` helpers;
  - `$NS`, `$POD`.
- `run_ramp.sh`:
  - takes overrides from env (`LEVELS`, `LEVEL_SECONDS`, `STEADY_FROM`) for smoke runs;
  - sources `${LIB_LOAD:-$HERE/lib_load.sh}`, so tests can inject a stub;
  - writes `$HERE/level.current`;
  - prints `RAMP level=<L> pass|fail <reason>` per level, then
    `RESULT ramp best=<L|none> hold=<L>`;
  - exit codes: 0 normally; 5 if `restarts` changed; 3 if there is no Running pod.
- `run_hold.sh`:
  - reads `level.current` and runs `run_level` once;
  - exits 6 if `check_result` fails, never on TTFT;
  - exits 5 on a restart.

- [ ] Step 1: Write `test_ramp.sh`, with a stub `LIB_LOAD` whose `run_level` prints a RESULT
  line whose `ttft_p95_ms` comes from a per-level table in the env. Tests:
  - `ramp stops at first fail`: TTFT 100,100,100,700 for levels 16,24,32,48 → `level.current`
    = 32, output has `RAMP level=48 fail`, 48 is the last level run;
  - `ramp none passes`: first level 700 → `level.current` = 16, `RESULT ramp best=none hold=16`;
  - `ramp top passes`: all 100 → `level.current` = 128 and a `WARNING` line;
  - `ramp aiperf failure`: stub returns rc 1 with no RESULT at level 24 → `level.current` = 16;
  - `ramp restart -> exit 5`: the stub's `restarts` changes after level 24 → exit 5.
- [ ] Step 2: Run `bash test_ramp.sh`. Expected: FAIL (`run_ramp.sh: No such file`).
- [ ] Step 3: Write `lib_load.sh` by moving `run_load.sh`'s body into `run_level` and its
  helpers, then write `run_ramp.sh` and `run_hold.sh`.
- [ ] Step 4: Run `bash test_ramp.sh && bash test_checks.sh && bash test_apply_config.sh`.
  Expected: all pass (5, 12, 26).
- [ ] Step 5: Run `bash -n` and `shellcheck` if installed on every `.sh`. Expected: clean.
- [ ] Step 6: Commit "Study 48: run_level library, Ramp (500 ms SLO, fixed levels) and Hold scripts".

### Task 4: Akamas resources (akamas-study-manager plugin)

**Files:** `$STUDY/akamas/`:
- `system.yaml`;
- `components/vllm.yaml` and `components/container.yaml`;
- `telemetry/prometheus.yaml`;
- `48-G7-KV-Offload-Users-Workflow.yaml`;
- `48-G7-KV-Offload-Users.yaml`;
- `README.md`.

**Interfaces:**
- Names:

  | Resource | Name |
  |---|---|
  | System | `vLLM_Benchmark_48_G7_KV_Offload_Users` |
  | Telemetry | `Prometheus_48_G7_KV_Offload_Users` (study 47's queries; namespace `kv-offload`) |
  | Workflow | `48-G7-KV-Offload-Users-Workflow` |
  | Study | `48-G7-KV-Offload-Users` |

- Workflow tasks:

  | Task | Operator | Timeout | Command or path |
  |---|---|---|---|
  | Write config | FileConfigurator | – | `params.env.template` → `params.env` |
  | Apply config | Executor | 60m | `apply_config.sh` |
  | Ramp | Executor | 180m | `run_ramp.sh` |
  | Hold | Executor | 25m | `run_hold.sh` |

  All paths are under `/work/study48/48-g7-kv-offload-users-within-slo/k8s/`.
- Study: goal, constraints, windowing, parameters and steps from Global Constraints. 8 KPIs:
  - Users: `64 * vllm.request_success_rate`;
  - `vllm.total_token_throughput`;
  - `vllm.time_to_first_token_p95`;
  - `vllm.inter_token_latency_p95`;
  - `vllm.prefix_cache_hit_rate`;
  - `vllm.cpu_offload_prefix_hit_rate`;
  - `vllm.kv_cache_usage_max`;
  - `container.container_memory_working_set`.

- [ ] Step 1: Run `/akamas-study-manager:build` in Modify mode on the copied `akamas/`, with the
  values above. Re-check the domains with `akamas describe optimization-pack vLLM`.
- [ ] Step 2: Validate with the CLI after Task 5's toolbox sync: `akamas create -f` in
  dependency order, then `akamas describe study "48-G7-KV-Offload-Users"`.
  - Expected: created. Steps listed as 5 + optimize; windowing on Hold.
- [ ] Step 3: Commit "Study 48: Akamas system, telemetry, workflow and study".

### Task 5: Node, toolbox sync and smoke run

- [ ] Step 1: Run `AWS_PROFILE=lab $STUDY/infra/eks/gpu-nodegroup.sh --up --always-on`.
  - Expected: one Ready node, `node-role=llm-serving-g7-kv`, tagged AlwaysOn.
- [ ] Step 2: Sync the folder to `/work/study48/` on the toolbox with tar, as in study 47.
- [ ] Step 3: Smoke the Apply config at offload 28 + fp8 (worst-case memory):
  - run `apply_config.sh` from the toolbox with a hand-written `params.env`;
  - Expected: vLLM Ready;
  - the vLLM pod, the AIPerf pod and the daemonsets fit, per `kubectl describe node`.
- [ ] Step 4: Smoke the Ramp and Hold with
  `LEVELS="16 24" LEVEL_SECONDS=180 STEADY_FROM=60 bash run_ramp.sh && bash run_hold.sh`, short
  durations, on the toolbox.
  - Expected: two `RAMP level=` lines, `RESULT ramp`, `level.current` written;
  - Hold prints RESULT;
  - "stale processes killed: 0".
- [ ] Step 5: Task 4 Step 2 (create). Then `yes y | akamas start study "48-G7-KV-Offload-Users"`.
  - Expected: experiment 1 RUNNING.
  - If the start fails with "Could not start the study", run `finish` and start again.

### Task 6: Docs and tracking

**Files:**
- `$STUDY/README.md`: Objective, Stack & versions, Parameters tuned, the ramp/SLO rules, the
  reporting rule for Holds between 500 and 600 ms, Status RUNNING.
- `$STUDY/akamas/README.md`.
- `studies/README.md`: row 48, RUNNING.
- `ROADMAP.md`: backlog row 48.

- [ ] Step 1: Write the files.
- [ ] Step 2: Commit "Study 48: README, recap and roadmap rows (RUNNING)".
- [ ] Step 3: Run the final whole-branch review (executing-plans) and the fix pass.
