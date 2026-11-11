# Study 50 — MoE parallelism layouts across two g7e nodes over EFA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run study 50 in `studies/50-g7e-efa-moe-parallelism-users-ramp/`: seven
parallelism layouts of `openai/gpt-oss-120b` across two g7e.8xlarge over EFA, each scored on
the chat users it holds within client TTFT p95 <= 500 ms.

**Architecture:**
- Two StatefulSets, `vllm-a` and `vllm-b`, one per GPU node. `k8s/apply_config.sh` deletes and
  re-creates them in the layout the ten Akamas parameters describe (`k8s/lib_layout.sh`), then
  proves the layout on the running pods (arguments, `vllm:cache_config_info`, ranks, expert
  parallelism, NCCL transport).
- Two AIPerf client pods on the load node. `k8s/run_ramp.sh` finds the best level, then
  `k8s/run_hold.sh` repeats it and Akamas scores that window.
- Telemetry: vLLM, DCGM per physical GPU, the study's own EFA byte exporter, node-exporter.
- One Akamas study with seven preset steps and no optimize step.

**Tech Stack:** bash, kubectl, eksctl (`efaEnabled`), Python 3, AIPerf 0.11.0, vLLM 0.29.0 (AWS
Deep Learning Container), NCCL 2.29.7 + aws-ofi-nccl 1.18.0, dcgm-exporter, Akamas 4.1.0, packs
vLLM 1.15.0, GPU 1.4.0, Kubernetes 1.9.0.

**Spec:** `docs/superpowers/specs/2026-11-09-study-50-g7e-efa-moe-parallelism-design.md`
(go on 2026-11-10). **Spike:** `docs/spikes/2026-11-10-efa-eks/README.md`.

## Global Constraints

- Layouts (replicas / TP / DP / PP / EP): single `1/1/1/1/false`, replicas `2/1/1/1/false`,
  dp2 `1/1/2/1/false`, dp2ep `1/1/2/1/true`, tp2 `1/2/1/1/false`, tp2ep `1/2/1/1/true`,
  pp2 `1/1/1/2/false`.
- Same vLLM flags in every step: `gpu_memory_utilization` 0.90, `max_num_seqs` 512,
  `max_num_batched_tokens` 8192, `kv_cache_dtype` auto, `moe_backend` auto.
- Workload (`k8s/workload.env`): 10 turns, 600 input and 200 output tokens per turn, one turn
  every 64 s, levels 32 64 128 192 256 384 512 768, 960 s per level judged from second 600,
  SLO client TTFT p95 <= 500 ms.
- Every cross-node trial must show NCCL on EFA, or the trial fails.
- NVIDIA DCGM metrics in telemetry, one Akamas component per physical GPU.
- The node group is always created with `--install-nvidia-plugin=false`.
- GPU nodes and their ASG tagged `AlwaysOn=true` while the study runs, removed at teardown.
- No secrets in the repo. Paid AWS actions only within the user's go (2x g7e.8xlarge).
- Smoke checks read the effective pod configuration before `akamas start study`.

## Review Focus

- A layout that starts but silently falls back to TCP: covered by the transport check in
  `test_checks.sh` and `test_apply_config.sh`.
- A template token rendered inside a comment (study 49's failure): covered by `test_render.sh`.
- A parameter combination outside the seven layouts: `lib_layout.sh` rejects it
  (`test_apply_config.sh`), and the study's parameter constraints exclude it.
- A ramp where the first level already fails: `test_ramp.sh` (Hold runs the lowest level).
- Not covered offline, left to the smoke checks: every telemetry query returning data, dp2
  without expert parallelism under external load balancing, AIPerf client CPU at 768 users.

---

### Task 1: EFA spike (done 2026-11-10)

- [x] 2x g6.8xlarge with EFA on `vllm-bench`, AWS vLLM image as it is.
- [x] NCCL over EFA against TCP, every multi-node layout started on gpt-oss-20b.
- [x] Torn down, recorded in `docs/spikes/2026-11-10-efa-eks/README.md`, committed (3fcc4e7).

### Task 2: Kubernetes layer (done 2026-11-10)

**Files:** `k8s/*.sh`, `k8s/*.yaml`, `k8s/monitoring/`, `infra/eks/`, `infra/efa/`.

- [x] Layout, render, checks, load and ramp libraries with their tests.
- [x] `bash k8s/test_checks.sh && bash k8s/test_ramp.sh && bash k8s/test_render.sh && bash
  k8s/test_load.sh && bash k8s/test_apply_config.sh`. Expected: 214 assertions, 0 failed.
- [x] dcgm-exporter Helm release upgraded with the node role (revision 27).

### Task 3: Akamas resources (done locally 2026-11-10)

**Files:** `akamas/system.yaml`, `akamas/components/*.yaml`, `akamas/telemetry/prometheus.yaml`,
`akamas/50-G7E-EFA-MoE-Parallelism-Workflow.yaml`, `akamas/50-G7E-EFA-MoE-Parallelism.yaml`.

- [x] System, eight components, telemetry (115 metrics), workflow, study with seven presets.
- [x] Structural check: template parameters = study parameters, steps satisfy the constraints.
- [x] Sync to the toolbox (`/work/study50`) and `akamas create` each resource. Expected: every
  resource accepted by Akamas 4.1.0. Done 2026-11-10; the five script tests also pass in the
  toolbox (bash 5.3), and Prometheus parses all 115 telemetry queries.

### Task 4: Nodes

- [ ] `nodegroups.sh wait-up 10`. Expected: two nodes Ready, labelled `study50/role=a|b`.
  Blocked: no capacity in any zone of us-east-2 from 20:34 UTC on 2026-11-10 to 06:29 UTC on
  2026-11-11 (111 probes). A second wait (`wait-up 24`) started at 08:07 UTC on 2026-11-11, no
  capacity by 08:44 UTC. Fallback chosen by the user on 2026-11-11: the same design without EFA
  on two g7e.4xlarge, as study 51 (`studies/51-g7e-tcp-moe-parallelism-users-ramp`). The second
  wait is stopped when study 51's nodes are created (64 vCPU of G quota); start it again for
  this study after study 51's teardown.
- [ ] `load-node.sh up`, `nodegroups.sh status`. Expected per node: `nvidia.com/gpu: 1`,
  `vpc.amazonaws.com/efa: 1`, `hugepages-2Mi` >= 5120Mi, a dcgm-exporter pod.
- [ ] `kubectl apply -f k8s/00-base.yaml -f k8s/02-model-download.yaml`. Expected: both download
  Jobs complete.

### Task 5: Smoke checks (README "Before starting: smoke checks")

- [ ] Checks 1 to 6, each with its evidence written into the study README.
- [ ] Check 7 (Nsight Perf SDK for AI): one hour, non-blocking.
- [ ] Every query of `akamas/telemetry/prometheus.yaml` returns data through the Prometheus
  proxy. Fix the telemetry and re-create the Akamas resources if one does not.

### Task 6: Run

- [ ] `akamas start study "50-G7E-EFA-MoE-Parallelism"`. Expected: seven experiments, about 15
  to 17 hours.
- [ ] TCP controls by hand: `NCCL_ENV="NCCL_NET=Socket" bash k8s/smoke.sh tp2 dp2ep` at the
  full levels, results in `results/tcp-controls/`.

### Task 7: Close

- [ ] Export and `study-recap` skill.
- [ ] Teardown per `infra/README.md`. Expected: no G instance left in EC2.
- [ ] Report and HTML artifact with the results.
