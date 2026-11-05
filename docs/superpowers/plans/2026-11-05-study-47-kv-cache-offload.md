# Study 47 — KV cache offload TTFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runnable Akamas study that minimizes TTFT p95 of a fixed multi-turn chat load on one g7.4xlarge, tuning CPU KV offload together with the other KV-memory levers.

**Architecture:** Self-contained study folder `studies/47-g7-kv-cache-offload-ttft/` shaped like study 46: `infra/` (node group script), `k8s/` (vLLM Deployment template, AIPerf client pod, ServiceMonitor, `apply_config.sh`, `run_load.sh`), `akamas/` (system, components, telemetry, workflow, study). The workflow runs on the akamas-41 toolbox (study copied to `/work/study47` with tar): Write config → Apply config → Load; the study windows on Load.

**Tech Stack:** EKS 1.36 (`vllm-bench`), vLLM `vllm/vllm-openai:v0.29.0`, Qwen/Qwen3-8B-FP8, AIPerf 0.11.0, Akamas 4.1 (akamas-41), vLLM pack 1.15.0, Kubernetes pack 1.9.0, Prometheus (kube-prometheus-stack).

**Spec:** `docs/superpowers/specs/2026-11-05-study-47-kv-cache-offload-design.md` (approved 2026-11-05). Spike: `docs/spikes/2026-11-05-kv-offload/README.md`.

## Global Constraints

- Node group `llm-serving-g7-kv` (label `node-role=llm-serving-g7-kv`, taint `nvidia.com/gpu=present:NoSchedule`), one g7.4xlarge; desired 0 outside runs; AlwaysOn tag while a study runs (17:00 UTC stop), removed after.
- vLLM fixed flags: `--model=Qwen/Qwen3-8B-FP8 --served-model-name=qwen3-8b --port=8000 --host=0.0.0.0 --max-model-len=16384 --enable-prefix-caching`. Everything else only when tuned.
- Workload (frozen after calibration, Task 4): `--user-centric-rate 1 --num-users 64`, `--conversation-turn-mean 10 --conversation-turn-stddev 0`, `--isl 600 --isl-stddev 0 --osl 200 --osl-stddev 0 --extra-inputs ignore_eos:true`, synthetic, 1200 s + 60 s grace.
- Calibration rule (spec): baseline keeps up if completed ≥ 95 % of sent and TTFT p95 over minutes 6–20 < 10 s; else users 64 → 56 → 48 at 64 s per user (rate = users / 64).
- Goal: minimize `vllm.time_to_first_token_p95`. Constraints absolute: `vllm.inter_token_latency_p95 <= 50`; client-side, `run_load.sh` fails the trial if errors ≥ 1 % or completed < 95 % of sent.
- Window: trim on task `Load`, `[360s, 90s]` (single unit per value).
- Offload size 0 = no offload flags. `/dev/shm` = offload + 1 GiB; vLLM memory request = limit = offload + 20 GiB; vLLM CPU 10; AIPerf pod 4 CPU / 6 GiB on the same node.
- Akamas YAML through the `akamas-study-manager` plugin; validate with `akamas create` in the toolbox. Watermark/tiny-real guard and scientific-notation conversion for every real (study 46 lesson).
- Repo files and commits in English; no secrets; commit on a study branch, no push.

## Review Focus

- Offload 0 rendered as an empty string or `0` → no `--kv-offloading-*` flag at all, shm stays 2 GiB, memory 20 GiB.
- A real rendered in scientific notation or without its exponent (`gpu_memory_utilization`) → decimal, or out-of-domain value refused with a clear error, never a silent default.
- vLLM crash-looping at startup (OOM, shm) → apply_config fails fast with the vLLM log on stdout, not after the 40-min timeout.
- AIPerf run that completes with many errors or falls behind (open loop) → trial fails, so the optimizer sees it as infeasible.
- Node missing (scaled to 0, spot-free but stopped) → apply_config fails in seconds with "no node llm-serving-g7-kv".

---

### Task 1: Infra and k8s manifests

**Files:**
- Create: `studies/47-g7-kv-cache-offload-ttft/infra/eks/gpu-nodegroup.sh` (from study 46's: NG `llm-serving-g7-kv`, `g7.4xlarge`, `--up`, `--down`, `--always-on`; `--down` sets desired 0 and removes the ASG AlwaysOn tag), `infra/README.md`.
- Create: `k8s/01-vllm_template.yaml` (Namespace `kv-offload`, Service `vllm` label `app: vllm`, Deployment `vllm`, Recreate, tokens `@TUNED_ARGS@ @SHM_GI@ @MEM_GI@`), `k8s/02-aiperf.yaml` (Pod `aiperf`, as the spike's), `k8s/monitoring/servicemonitor.yaml` (`vllm-kv-offload`, namespace `kv-offload`), `k8s/.gitignore` (rendered files, params.env).

- [ ] **Step 1:** Write the files. Template from the spike's `vllm-B.yaml` with the fixed flags above, HF cache hostPath `/var/lib/hf-cache-study47`, startup probe 120 × 10 s.
- [ ] **Step 2:** `kubectl apply --dry-run=server -f` the AIPerf pod and ServiceMonitor; render the template by hand with offload 24 and dry-run it. Expected: no error.
- [ ] **Step 3:** Commit.

### Task 2: `apply_config.sh` + `params.env.template`

**Files:** Create `k8s/params.env.template`, `k8s/apply_config.sh`.

**Interfaces:**
- Produces: `params.env` variables `KV_OFFLOADING_SIZE KV_CACHE_DTYPE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE` (all may be empty = vLLM default); `RENDER_ONLY=1` prints one flag per line plus `shm=<Gi> mem=<Gi>` and exits 0; `STUDY_DIR` overridable.
- Exit codes: 2 invalid parameters, 3 node missing, 4 vLLM not ready / crash loop.

- [ ] **Step 1: Failing render tests** in `k8s/test_apply_config.sh` (plain bash, temp dir per case, asserts with grep):
  - all empty → output has no `--kv-offloading`, no `--gpu-memory-utilization`, `shm=2 mem=20`.
  - `KV_OFFLOADING_SIZE=0` → no `--kv-offloading`, `shm=2 mem=20`.
  - `KV_OFFLOADING_SIZE=24` → `--kv-offloading-size=24`, `--kv-offloading-backend=native`, `shm=25 mem=44`.
  - `GPU_MEMORY_UTILIZATION=9.2E-1` → `--gpu-memory-utilization=0.920000`; `=2.19` → exit 2.
  - `MAX_NUM_SEQS=128 MAX_NUM_BATCHED_TOKENS=64` → exit 2 (mnbt < mns).
  - `KV_OFFLOADING_SIZE=40` → exit 2 (above 28, node RAM).
  - `BLOCK_SIZE=24` → exit 2.
- [ ] **Step 2:** Run `bash k8s/test_apply_config.sh`. Expected: FAIL (script missing).
- [ ] **Step 3: Implement `apply_config.sh`**: section 0 parse/validate/render (pattern of study 46: log raw params.env, sci-notation conversion, domain checks, `flag()` builder); then: node present check; apply ServiceMonitor, AIPerf pod (idempotent, wait Ready 600 s); render template, `kubectl apply`, `rollout status` 2400 s while watching restartCount (≥ 2 → dump `--previous` log, exit 4); on success print the startup summary lines (KV cache size, offload mmap) and the full vLLM log.
- [ ] **Step 4:** Run the tests. Expected: all PASS.
- [ ] **Step 5:** Commit.

### Task 3: `run_load.sh`

**Files:** Create `k8s/run_load.sh`, `k8s/workload.env` (USERS, RATE, DURATION, frozen by Task 4).

**Interfaces:** Consumes `workload.env`; prints `RESULT sent=<n> completed=<n> errors=<n> ttft_p50_ms=<x> ttft_p95_ms=<x> itl_p95_ms=<x>` computed client-side from `profile_export.jsonl` over requests starting ≥ 360 s after the first request, the per-minute TTFT table, the vLLM offload counters delta, the AIPerf log tail; exits 6 if errors ≥ 1 % or completed < 95 % of sent, 5 if vLLM restarted.

- [ ] **Step 1:** Write it (AIPerf command from Global Constraints; parser in Python run inside the AIPerf pod, as the spike's `summ.sh` and study 46's `lib_load.sh`).
- [ ] **Step 2:** Smoke: `DURATION=120` against a running vLLM (Task 4's node). Expected: RESULT line, exit 0.
- [ ] **Step 3:** Commit.

### Task 4: Calibration (freezes the workload)

- [ ] **Step 1:** `infra/eks/gpu-nodegroup.sh --up`; copy the study to the toolbox (`/work/study47`, tar without macOS xattrs); empty `params.env` (baseline) → `apply_config.sh`.
- [ ] **Step 2:** Read vLLM's actual defaults from the startup log (`max_num_seqs`, `max_num_batched_tokens`, `block_size`, `gpu_memory_utilization`, KV dtype) — they become the baseline step's recorded values.
- [ ] **Step 3:** `run_load.sh` with 64 users. Apply the calibration rule; if it fails, step users down and repeat. Write the frozen values in `workload.env` with a comment (date, numbers).
- [ ] **Step 4:** Scale the node group to 0 if the study is not created the same day. Commit `workload.env` + calibration notes in the study README.

### Task 5: Akamas resources (akamas-study-manager plugin)

**Files:** `akamas/system.yaml`, `akamas/components/{vllm,container}.yaml`, `akamas/telemetry/prometheus.yaml`, `akamas/47-G7-KV-Offload-Workflow.yaml`, `akamas/47-G7-KV-Offload-TTFT.yaml`, `akamas/README.md`.

- [ ] **Step 1:** System `vLLM_Benchmark_47_G7_KV_Offload`; components `vllm` (vLLM, pod `^vllm-.*`, model `qwen3-8b`) and `container` (Kubernetes Container, namespace-scoped queries).
- [ ] **Step 2:** Telemetry: study 46's vLLM and container queries scoped to `namespace="kv-offload"`, plus the four pack 1.15.0 offload metrics with the README's PromQL (`prefix_cache_hit_rate_with_offload` must work with offload off: `or` fallback to 0 on the external term).
- [ ] **Step 3:** Workflow: Write config (FileConfigurator), Apply config (60 m), Load (DURATION + 15 m).
- [ ] **Step 4:** Study: goal/constraint from Global Constraints; windowing trim `[360s, 90s]` task `Load`; parameters `vllm.kv_offloading_size [0, 28]` (spec said 0–32: 28 + 20 GiB vLLM + 6 GiB AIPerf fits the 57 GiB allocatable), `vllm.kv_cache_dtype {auto, fp8}`, `vllm.gpu_memory_utilization [0.80, 0.94]`, `vllm.max_num_seqs [16, 256]`, `vllm.max_num_batched_tokens [1024, 16384]`, `vllm.block_size {16, 32, 64, 128}`; constraint mnbt ≥ mns; steps: `baseline` (values = vLLM's actual defaults from Task 4, every vLLM parameter in `doNotRenderParameters` so no tuned flag is rendered), `Offload only` (24), `FP8 KV only`, `Offload and FP8`, `Baseline repeat`, `optimize` (4 init, 25 experiments, 10 max failed); ≤ 8 KPIs (TTFT p95/avg, ITL p95, prefix_cache_hit_rate, prefix_cache_hit_rate_with_offload, cpu_offload_prefix_hit_rate, kv_offload_cpu_to_gpu_throughput, preemption_rate).
- [ ] **Step 5:** `akamas create -f` each resource in the toolbox (system, components, telemetry, workflow, study). Expected: all created; fix and re-create on error.
- [ ] **Step 6:** Commit.

### Task 6: Smoke, start, records

- [ ] **Step 1:** One manual Write-config-equivalent with offload 24 + `apply_config.sh` + `run_load.sh DURATION=300` to prove the offload path end to end (shm, memory, metrics non-empty in Prometheus for all four offload queries).
- [ ] **Step 2:** `gpu-nodegroup.sh --always-on`; `akamas start study` (finish + `yes y | start` if it hangs).
- [ ] **Step 3:** Study README (objective, stack & versions, parameters, calibration), `studies/README.md` row (RUNNING), ROADMAP backlog row. Commit.
- [ ] **Step 4:** Watch experiment 1 to FINISHED and check its score is non-empty.
