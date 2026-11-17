# Study 52 — GLM-5.3-Flash coding sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ready-to-run Akamas study (folder, manifests, scripts, telemetry, Akamas YAML) that
finds how many coding sessions GLM-5.3-Flash (FP8, TP4, vLLM v0.34.0) serves within TTFT p95 <= 5 s
and ITL p95 <= 50 ms on one g7e.24xlarge spot, built while the node is not available yet.

**Architecture:** Study 48's single-node Ramp + Hold harness (one fresh vLLM pod per experiment,
client-side level checks, score from Prometheus on the Hold window), with study 51's two AIPerf
clients on a CPU node and its full telemetry catalog. The vLLM pod keeps the target deployment's
container shape: `vllm serve /mnt/models ... ${VLLM_ADDITIONAL_ARGS}`.

**Tech Stack:** bash + kubectl (toolbox), EKS `vllm-bench`, vLLM `vllm/vllm-openai:v0.34.0`,
AIPerf 0.11.0, Prometheus (kube-prometheus-stack, dcgm-exporter, node-exporter,
kube-state-metrics), Akamas 4.1 (akamas-41) with the vLLM, GPU and Kubernetes packs.

**Spec:** `docs/superpowers/specs/2026-11-17-study-52-glm53-coding-sessions-design.md`

## Global Constraints

- Study folder `studies/52-g7e-glm53-coding-sessions-within-slo`; namespace `glm53-coding`;
  node role `llm-serving-g7e-24xl-spot` (vLLM), the AIPerf clients on study 51's CPU node role.
- Image `vllm/vllm-openai:v0.34.0`; checkpoint `zai-org/GLM-5.3-Flash` (FP8, 328 GB) on hostPath
  `/var/lib/hf-cache-study52`, served as `zai-org/GLM-5.3-Flash`.
- Fixed arguments (the target's, plus text-only): `--tensor-parallel-size 4 --max-model-len 262144
  --kv-cache-dtype fp8 --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice
  --language-model-only`. Container command: `vllm serve /mnt/models --served-model-name
  "zai-org/GLM-5.3-Flash" --port 8000 --disable-access-log-for-endpoints /health,/metrics,/ping
  --shutdown-timeout 40 ${VLLM_ADDITIONAL_ARGS}`.
- Workload: shared system prompt 20000 tokens, per-session context 10000, 16 turns, ISL 4000,
  OSL 400 (`ignore_eos`), one turn per session every 30 s.
- Ramp levels `4 8 12 16 24 32 48 64`; level 960 s + 60 s grace; steady from 500 s.
- SLO: client TTFT p95 <= 5000 ms and client ITL p95 <= 50 ms (each the worse of the two
  clients); errors < 1 %; >= 95 % completed; >= 95 % of expected sent. Each level also prints the
  worst session's TTFT p95 and the number of sessions with TTFT p95 > 5000 ms (report only).
- Hold 1080 s, Akamas trim `[520s, 90s]` (single unit each: memory "akamas-trim-single-unit").
- Score `30 * vllm.request_success_rate` (no TTFT tie-break, the user's call); constraints
  server TTFT p95 <= 6 s, ITL p95 <= 60 ms (the ITL one written only after the smoke compares
  server and client ITL under MTP).
- Smoke go / no-go: FP8 TP4 must start with `--max-model-len` >= 131072; offload must start and
  its counters move; otherwise stop and report to the user before any Akamas resource.
- `kv_offloading_size` <= 512 GiB; `/dev/shm` = offload + 1 GiB.
- Files and commits in English; no customer names, hostnames or internal addresses in the repo;
  no secrets.
- Akamas YAML only through the `akamas-study-manager` plugin; no `akamas create`, no toolbox sync
  until the user confirms; step names match `^[a-zA-Z\s][a-zA-Z0-9_\s]*$`; telemetry placeholders
  alphanumeric only.
- Full component and telemetry catalog (CLAUDE.md rule 9).

## Review Focus

- **Spot node reclaimed** between or during experiments: Apply must wait for a node and re-download
  the checkpoint instead of failing in seconds (study 42), Ramp must exit 5 (not an SLO verdict).
  Test in Task 3 (`wait_for_node` with a stubbed `kubectl`).
- **A parameter left out by Akamas** (`doNotRenderParameters` renders an empty string): no flag,
  never `--flag=`. `spec_tokens` 0 or empty means no speculative config at all. Test in Task 3.
- **ITL not available** (every request failed, or no streamed token in the window): the level
  fails with a reason, never passes. Test in Task 2.
- **First start on an empty node**: 328 GB download + ~4 min JIT + weight load must fit the
  startup budget (startupProbe and the apply wait at least 3600 s). Asserted in Task 3's render
  test on the manifest.
- **`max_num_seqs` above the Mamba block count** makes vLLM refuse to start: Apply exits 4 with the
  log tail that names it, so the experiment fails visibly. Test in Task 3 (log-tail grep on a
  fixture).

---

### Task 1: Study folder, infra and README skeleton

**Files:**
- Create: `studies/52-g7e-glm53-coding-sessions-within-slo/` from `studies/_TEMPLATE` (use the
  `new-study` skill)
- Create: `.../infra/eks/nodegroup-g7e-24xl-spot.yaml` (copy of
  `docs/spikes/2026-11-17-glm53-flash-phase0/infra/nodegroup-g7e-24xl-spot.yaml`)
- Create: `.../infra/eks/nodegroups.sh` (from study 51's: subcommands `probe`, `create`, `up`,
  `down`, `status`; one spot node, AZ us-east-2c, no placement group, no role labels)
- Create: `.../k8s/monitoring/dcgm-exporter-values.yaml` (copy of the spike's, Helm revision 30)
- Create: `.../README.md` (objective, stack, method from the spec; status PREPARED)

**Interfaces:**
- Produces: `nodegroups.sh up` leaves one Ready node labelled `node-role=llm-serving-g7e-24xl-spot`
  with the ASG and instance tagged `AlwaysOn=true`; `down` scales to 0 and removes the tag.

- [ ] **Step 1:** Scaffold with the `new-study` skill; copy the two YAML files.
- [ ] **Step 2:** Write `nodegroups.sh`; `probe` asks the spot placement score and price for
  g7e.24xlarge in us-east-2 (no capacity reservation: spot).
- [ ] **Step 3:** Run `bash -n infra/eks/nodegroups.sh` and `AWS_PROFILE=lab ./nodegroups.sh status`.
  Expected: syntax OK; status prints the node group at 0 or "not found".
- [ ] **Step 4:** README skeleton; commit `Study 52: folder, spot node group, README skeleton`.

### Task 2: Workload and level checks (TTFT and ITL)

**Files:**
- Create: `.../k8s/workload.env`, `.../k8s/lib_checks.sh`, `.../k8s/test_checks.sh` (from study 48)

**Interfaces:**
- Produces: `check_result SUMMARY EXPECTED_SENT` (unchanged from 48) and
  `check_level SUMMARY EXPECTED_SENT SLO_TTFT_MS SLO_ITL_MS` → 0 pass / 1 fail, reason on stdout.
  `workload.env` exports `TURNS=16 ISL=4000 OSL=400 SECONDS_PER_TURN=30 SHARED_PROMPT=20000
  USER_CONTEXT=10000 LEVELS="4 8 12 16 24 32 48 64" LEVEL_SECONDS=960 STEADY_FROM=500
  SLO_TTFT_MS=5000 SLO_ITL_MS=50 EXTRA_INPUTS="ignore_eos:true"`.

- [ ] **Step 1: Write the failing tests** in `test_checks.sh`, study 48's cases plus:

```bash
expect_level "both ok"        0 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=900 ttft_p95_ms=4800 itl_p95_ms=42" 360 5000 50
expect_level "itl at SLO"     0 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=900 ttft_p95_ms=4800 itl_p95_ms=50" 360 5000 50
expect_level "itl over SLO"   1 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=900 ttft_p95_ms=4800 itl_p95_ms=50.1" 360 5000 50
expect_level "itl nan"        1 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=900 ttft_p95_ms=4800 itl_p95_ms=nan" 360 5000 50
expect_level "ttft over SLO"  1 "RESULT sent=360 completed=360 errors=0 ttft_p50_ms=900 ttft_p95_ms=5001 itl_p95_ms=42" 360 5000 50
```

- [ ] **Step 2:** `bash k8s/test_checks.sh` → FAIL on the ITL cases.
- [ ] **Step 3:** Add the ITL criterion to `check_level` (same awk shape as TTFT; a non-numeric ITL
  fails with "client ITL p95 not available").
- [ ] **Step 4:** `bash k8s/test_checks.sh` → `N passed, 0 failed`.
- [ ] **Step 5:** Commit `Study 52: workload and TTFT + ITL level checks`.

### Task 3: vLLM manifest and apply_config.sh

**Files:**
- Create: `.../k8s/00-base.yaml` (namespace `glm53-coding`, Service `vllm`, ServiceMonitor every 5 s)
- Create: `.../k8s/01-vllm_template.yaml`, `.../k8s/params.env.template`, `.../k8s/apply_config.sh`,
  `.../k8s/test_apply_config.sh`, `.../k8s/fixtures/mamba-blocks.log`

**Interfaces:**
- Consumes: Task 1's node role.
- Produces: `params.env` variables `KV_OFFLOADING_SIZE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS
  MAX_NUM_BATCHED_TOKENS SPEC_TOKENS DECODE_CONTEXT_PARALLEL_SIZE MOE_BACKEND
  MAX_CUDAGRAPH_CAPTURE_SIZE ASYNC_SCHEDULING PERFORMANCE_MODE OPTIMIZATION_LEVEL`
  (`${vllm.<name>}` tokens); `RENDER_ONLY=1` prints the `VLLM_ADDITIONAL_ARGS` value on one line,
  then `shm=<Gi> mem=<Gi>`. Template tokens: `@ADDITIONAL_ARGS@ @SHM_GI@ @MEM_GI@ @APPLIED_AT@`.
  Functions `wait_for_node TIMEOUT_S` (0 when a Ready node with the role exists, 3 after
  TIMEOUT_S) and `startup_failure_reason LOGFILE` (prints the vLLM error line naming Mamba blocks,
  OOM or "No available memory for the cache blocks").

- [ ] **Step 1: Write the failing tests** (`test_apply_config.sh`, study 48's `render` helper):
  - all empty → the args line is exactly the fixed arguments (no tuned flag); `shm=2 mem=96`;
  - the baseline values (`GPU_MEMORY_UTILIZATION=0.95 SPEC_TOKENS=5`, rest empty) → the target's
    argument line plus `--language-model-only`;
  - `SPEC_TOKENS=5` → contains `--speculative-config.method mtp
    --speculative-config.num_speculative_tokens 5`; `SPEC_TOKENS=0` and empty → no
    `--speculative-config`;
  - `KV_OFFLOADING_SIZE=256` → `--kv-offloading-size 256 --kv-offloading-backend native`,
    `shm=257 mem=352`; `KV_OFFLOADING_SIZE=513` → exit 2;
  - `DECODE_CONTEXT_PARALLEL_SIZE=3` → exit 2 (allowed 1, 2, 4);
  - `MAX_NUM_BATCHED_TOKENS=64 MAX_NUM_SEQS=128` → exit 2;
  - `GPU_MEMORY_UTILIZATION=9.5e-1` → `--gpu-memory-utilization 0.950000`;
  - `MOE_BACKEND=nonsense` → exit 2 (allowed list from Task 6's smoke, initially
    `auto triton cutlass flashinfer_cutlass deep_gemm`);
  - the rendered manifest has `failureThreshold` × `periodSeconds` >= 3600 on the startupProbe;
  - `startup_failure_reason fixtures/mamba-blocks.log` prints the line with
    "exceeds available Mamba cache blocks";
  - `wait_for_node 1` with a stub `kubectl` on PATH returning no node → exit 3.
- [ ] **Step 2:** `bash k8s/test_apply_config.sh` → failures.
- [ ] **Step 3:** Implement. `01-vllm_template.yaml`: a Pod (study 48's Deployment shape is fine),
  4 GPUs, `requests == limits`, memory `@MEM_GI@Gi` = offload + 96, `/dev/shm` `@SHM_GI@Gi`;
  init container downloads `zai-org/GLM-5.3-Flash` with `hf download` if the snapshot is
  missing; hostPaths for the checkpoint and the compile caches (`/root/.cache`,
  `/root/.tilelang`, `/root/.triton`, as in the spike's `04-vllm-customer-like.yaml`); the
  container command above with `VLLM_ADDITIONAL_ARGS` = fixed args + `@ADDITIONAL_ARGS@`.
  `apply_config.sh`: study 48's structure, plus `wait_for_node 1800` before apply, and after
  Ready the study-49 check: the pod's real `VLLM_ADDITIONAL_ARGS` equals the rendered one and
  `vllm:cache_config_info` reports the rendered gmu and `num_gpu_blocks` > 0; on a startup
  failure print `startup_failure_reason` and exit 4.
- [ ] **Step 4:** `bash k8s/test_apply_config.sh` → `N passed, 0 failed`.
- [ ] **Step 5:** Commit `Study 52: vLLM manifest in the target's container shape, apply_config`.

### Task 4: Load, Ramp and Hold

**Files:**
- Create: `.../k8s/02-aiperf-clients.yaml` (study 51's two clients, CPU node, results volume
  sized for 100k-token prompts and cleaned per level), `.../k8s/lib_load.sh`, `.../k8s/run_ramp.sh`,
  `.../k8s/run_hold.sh`, `.../k8s/test_ramp.sh`, `.../k8s/test_load.sh`

**Interfaces:**
- Consumes: Task 2's `check_level` (4 arguments) and `workload.env`; Task 3's Service
  `vllm.glm53-coding.svc:8000`.
- Produces: `run_level SESSIONS SECONDS LABEL` printing the study-48 `RESULT` line (merged over the
  two clients, as in study 51); `run_ramp.sh` writes `k8s/level.current`; `run_hold.sh` runs the
  level for 1080 s; exit codes as in study 48 (3 no pod, 5 vLLM restarted, 6 AIPerf failure,
  7 no level passed).

- [ ] **Step 1: Write the failing tests:** `test_ramp.sh` (study 48's stub `LIB_LOAD`) with a level
  that passes TTFT but fails ITL → `RAMP level=<L> fail client ITL p95 ...` and the previous level
  in `level.current`; `test_load.sh` checks the rendered AIPerf command contains
  `--shared-system-prompt-length 20000`, `--user-context-prompt-length 10000`,
  `--conversation-turn-mean 16`, `--isl 4000`, `--osl 400`, `--tokenizer zai-org/GLM-5.3-Flash`,
  and for 12 sessions each of the two clients gets `--num-users 6 --user-centric-rate 0.2000`;
  the parser prints `worst_session_ttft_p95_ms` and `sessions_over_slo` on the RESULT line (from
  `x_correlation_id` in the export).
- [ ] **Step 2:** Run both → failures.
- [ ] **Step 3:** Implement from study 48 (`run_level` gains `DRY_RUN=1` printing the AIPerf command
  for `test_load.sh`) and study 51 (two clients, half the sessions each, results merged); counter
  snapshot adds `vllm:spec_decode_*` and the offload counters.
- [ ] **Step 4:** Run both → `N passed, 0 failed`.
- [ ] **Step 5:** Commit `Study 52: AIPerf coding sessions, Ramp and Hold`.

### Task 5: Smoke script (runs once the node is up)

**Files:**
- Create: `.../k8s/smoke.sh`

**Interfaces:**
- Consumes: Tasks 3 and 4.
- Produces: one report with: start-up time (cold and warm caches), KV tokens and Mamba blocks from
  the log, MoE backend candidates (`potential backends` line), MTP acceptance after 20 requests,
  DCP 4 start or failure, offload 256 GiB counters after one 4-session level of 300 s, server vs
  client ITL p95 with MTP on, and the go / no-go verdict from the Global Constraints.

- [ ] **Step 1:** Write `smoke.sh` as a sequence of `apply_config.sh` + short `run_level` calls with
  override env (`LEVELS`, `LEVEL_SECONDS`, `STEADY_FROM`), each step printing PASS/FAIL.
- [ ] **Step 2:** `bash -n k8s/smoke.sh` → OK. (Running it waits for the node.)
- [ ] **Step 3:** Commit `Study 52: smoke script`.

### Task 6: Akamas resources and telemetry

**Files:**
- Create (with `/akamas-study-manager:build`): `.../akamas/system.yaml`,
  `.../akamas/components/{vllm,gpu0,gpu1,gpu2,gpu3,node_gpu,node_load,container,container_loadtest,vllm_workload,cluster}.yaml`,
  `.../akamas/telemetry/prometheus.yaml`, `.../akamas/52-G7E-GLM53-Coding-Workflow.yaml`,
  `.../akamas/52-G7E-GLM53-Coding-Sessions.yaml`, `.../akamas/README.md`

**Interfaces:**
- Consumes: Task 3's params tokens; Task 4's scripts (Ramp, Hold task commands); the toolbox path
  `/work/study52/52-g7e-glm53-coding-sessions-within-slo`.
- Produces: system `vLLM_Benchmark_52_G7E_GLM53_Coding`, telemetry
  `Prometheus_52_G7E_GLM53_Coding`, workflow `52-G7E-GLM53-Coding-Workflow`, study
  `52-G7E-GLM53-Coding-Sessions`.

- [ ] **Step 1:** Telemetry: start from study 51's 113 metrics; vLLM filtered on namespace
  `glm53-coding`; add the offload and spec-decode metrics; one GPU component per GPU index
  (`gpu` label 0-3) on the node role; Node components with the pack's CPU, PSI, load, memory,
  swap, disk, filesystem, per-interface network and TCP retransmit metrics; Container components
  for vLLM and AIPerf; Workload; Cluster.
- [ ] **Step 2:** Study: goal, constraints, windowing trim and steps from the Global Constraints;
  parameters and domains from the spec, each checked against `akamas describe
  optimization-pack` (installed pack version written in `akamas/README.md`); KPIs: sessions, output
  tokens/s, TTFT p95, ITL p95, combined prefix hit rate, MTP acceptance, KV usage, GPU 0
  utilization. Steps: `baseline` (gmu 0.95, `spec_tokens` 5, or 0 if the smoke shows no MTP
  acceptance), presets `Offload 256` (`kv_offloading_size` 256), `DCP four`
  (`decode_context_parallel_size` 4), `MTP three` and `MTP one` (`spec_tokens` 3 and 1),
  `Baseline repeat`, then `optimize` with 10 experiments. Presets only for the levers the smoke
  confirms (Task 5); the others stay out with a note in the README.
- [ ] **Step 3:** Validate offline with the plugin's schema check; `akamas create` waits for the
  user's go (then `akamas create -f` each file and `akamas describe` the study).
- [ ] **Step 4:** Commit `Study 52: Akamas system, telemetry, workflow, study`.

### Task 7: Records

**Files:**
- Modify: `studies/README.md` (row 52, status PREPARED), `ROADMAP.md` (study 52 in the plan, pack
  gaps `mamba_ssm_cache_dtype`, `kv_cache_dtype fp8_ds_mla`, `language_model_only`)
- Modify: `.../README.md` (TODO list: pack gaps, smoke pending, quota case)

- [ ] **Step 1:** Write the rows; run every `test_*.sh` once more → all pass.
- [ ] **Step 2:** Commit `Study 52 prepared: waiting for the spot quota and the smoke`.
