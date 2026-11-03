# Study 46 — Users per dollar on a split RTX PRO 6000 via DRA (design)

**Date:** 2026-11-03 · **Status:** approved 2026-11-03 (user: "vai") · **Study folder (to create):**
`studies/46-g7e-dra-mig-users-per-dollar/`

## Intent and the why

A generic, demonstrative customer case: a customer serves a **small model on a GPU that is
far too big for it** and pays for 96 GB while using ~20. The study shows how many users the
same GPU serves within a reasonable SLA when it is **split with MIG, requested through
Kubernetes Dynamic Resource Allocation (DRA)**, versus kept whole. The sentence the study
should let us write in the README is of the form: *"one RTX PRO 6000 split in N slices
serves X users within the SLA, Y % more than the whole GPU, i.e. Z users per $/h."*

Why now: the `vllm-bench` cluster was upgraded to Kubernetes **1.36** on 2026-11-02/03. On
1.36 `DRAPartitionableDevices` is on by default, which is what the NVIDIA DRA driver's
`DynamicMIG` feature needs; on 1.35 it was out of reach (EKS does not let us set
control-plane feature gates — ROADMAP "DRA prerequisites and risk").

The value of DRA here is the **mechanism**, not a new sharing mode (ROADMAP note, study 25
design): each pod declares the slice it needs in a `ResourceClaim`, and the driver creates
the MIG partition on demand — instead of study 25/26's relabel-node + reconfigure
device-plugin + manual `nvidia-smi mig` sequence.

## Prior results this builds on (read before planning)

- **Study 25** (RTX PRO 4500, 165 W): splitting a power-capped GPU adds no throughput;
  MPS ≈ whole GPU, time-slicing −14 %, MIG −5 %.
- **Study 26** (same GPU): MIG costs ~3–6 %; throughput scales with slice size, **users
  within SLO do not** (KV shrinks faster than the slice); `max_num_seqs` moves the user
  ceiling, not tokens/s. Its "users within SLO" reading is the closest precedent for this
  study's goal.
- **Study 28**: Qwen3-8B-FP8 measured memory footprint (~8.9 GiB weights) and the AIPerf
  open-loop setup; **study 45**: tokens/s per $/h across g5/g6/g6e — the comparison point
  for "would a smaller GPU have been cheaper".

The RTX PRO 6000 is 600 W class; study 25's power-cap conclusion may not transfer. That is
one of the things this study finds out — it is not assumed either way.

## Goals

1. Measure **total users served within the SLA** on one RTX PRO 6000 for each split: whole
   GPU, 2 × `2g.48gb`, 4 × `1g.24gb`, with vLLM settings tuned per split.
2. Do it with slices **requested via DRA `ResourceClaim`s** (DynamicMIG), proving the
   mechanism works on this GPU/AMI/cluster.
3. Report users per $/h (g7e price) in the README, comparable with study 45's GPUs.

## Non-goals

- MPS and time-slicing (study 25 already measured them; `MPSSupport` is mutually
  exclusive with `DynamicMIG` in the driver).
- Mixed layouts (e.g. one `2g.48gb` + two `1g.24gb`), multiple models, or heterogeneous
  vLLM settings across replicas of the same experiment.
- Comparing g7e instance sizes against each other.
- Counting the load-generator node in the cost.

## Anti-goals (a failure even if it runs)

- A result driven by **CPU starvation** of vLLM on the 8-vCPU node or by the **load
  generator** being the bottleneck, presented as a GPU-split effect.
- Claiming that splitting increases **throughput** (study 25 says it shouldn't); the claim
  is about users within SLA per dollar.
- Silently falling back to the classic device plugin: then the study is a re-run of
  study 26 on a bigger GPU, not a DRA study, and must be named and reported as such.

## Decided (by the user, in conversation)

- Generic demo case, not a specific customer.
- GPU: **RTX PRO 6000 Blackwell 96 GB** (AWS g7e).
- Metric: **users served within a reasonable SLA**, expressed per cost.
- Load generator on a **separate, non-GPU node**.

## Decided by Claude (the user can override; marked as my call)

- **Split mechanism: MIG only**, via DRA `DynamicMIG`; whole GPU as baseline. Reasons: MPS
  and time-slicing already measured (study 25); MIG is what enterprise clusters allocate
  (study 26 note); `DynamicMIG` + `MPSSupport` cannot coexist.
- **Instance: `g7e.2xlarge`** (8 vCPU / 64 GiB, 3.36 USD/h on-demand, us-east-2). Only size
  with capacity in both probes on 2026-11-03 (4xlarge: none anywhere; 8xlarge: 2a only in
  the morning, gone at 13:25 UTC). Cheapest instance with this GPU, which fits the cost
  story. Risk: 4 vLLM replicas on 8 vCPU (see "CPU fairness").
- **Model: `Qwen/Qwen3-8B-FP8`**, vLLM `v0.29.0` (as studies 27/28/45), `--max-model-len
  4096`, prefix caching off (as study 28). Fits a `1g.24gb` slice with ~10 GiB of KV.
- **SLA: TTFT p95 ≤ 1500 ms, ITL p95 ≤ 300 ms, ≥ 95 % of requests completed** (studies
  26/28), checked **per replica**.
- **Goal formula: users within SLA = N_replicas × min over replicas of (highest concurrency
  level at which that replica meets the SLA).** Using the minimum, not the sum, means the
  number is what every tenant on the GPU is guaranteed — a split where one slice breaks
  early is scored by that slice. Akamas maximizes this; cost is constant (one g7e), so
  per-$ is computed in the README, not in Akamas.
- **Load: closed-loop concurrency ramp per replica** (AIPerf, ShareGPT, studies 25/26
  style), **one AIPerf Job per replica** with a fixed per-replica seed, all replicas
  loaded at the same time (a slice is always measured with busy neighbours — study 26
  rule; MIG does not isolate power).
- **Load-generator node:** existing node group **`m8a-2xlarge`** (m8a.2xlarge, 8 vCPU /
  32 GiB, taint `dedicated=m8a-2xlarge`, subnets 2a/2b/2c), scaled 0 → 1 for the study,
  back to 0 after. AIPerf pods tolerate the taint and select `node-role: m8a-2xlarge`.
  Cross-AZ placement vs the GPU node is accepted (~1 ms) and noted in the README.
- **GPU node group: new `llm-serving-g7e-2xl`**, `g7e.2xlarge`, `AL2023_x86_64_NVIDIA`
  release for **1.36** (no pinned AMI), subnets **2a/2b/2c**, label
  `node-role=llm-serving-g7e-2xl`, taint `nvidia.com/gpu=present:NoSchedule`, min 0 / max 1.
  Created with `aws eks create-nodegroup` like `llm-serving-g7-4500` (memory/precedent).
  Tag node + ASG `AlwaysOn=true` (17:00 UTC nightly stop). The old `llm-serving-g7e`
  (4xlarge, AMI pinned to 1.35, 2a/2b only) is left untouched.
- **DRA driver: standalone `dra-driver-nvidia-gpu` Helm chart** (driver docs' DynamicMIG
  install: `--set featureGates.DynamicMIG=true --set gpuResourcesEnabledOverride=true`),
  scheduled **only on the g7e node**, using the AMI's preinstalled NVIDIA driver and
  container toolkit. Not the GPU Operator `GPUCluster` path: that path is new-cluster-only,
  takes over the device plugin cluster-wide and would touch every other GPU node group.
- **Classic device plugin off on that node**: both the `kube-system`
  `nvidia-device-plugin-daemonset` (currently CrashLooping on non-GPU nodes — restrict it
  to GPU node groups other than this one) and the Helm release `nvdp-g7` must not schedule
  on `llm-serving-g7e-2xl`. `dcgm-exporter`'s node affinity must be extended to it.
- **Tuned parameters (Akamas):**
  - `gpu.mig_profile` ∈ {`none`, `2g.48gb`, `1g.24gb`} — GPU pack **1.4.0 is installed on
    Akamas 4.1 and its domain already contains `1g.24gb` and `2g.48gb`**; `none` = whole
    GPU (1 replica), `2g.48gb` → 2 replicas, `1g.24gb` → 4 replicas. Replica count is
    derived, not a separate parameter.
  - vLLM, applied identically to every replica: `max_num_seqs`,
    `max_num_batched_tokens`, `gpu_memory_utilization`, `kv_cache_dtype` {auto, fp8}.
    Domains to be checked against the installed vLLM pack (`akamas describe
    optimization-pack vLLM`) and sized so a `1g.24gb` replica can start.
- **Steps:** `baseline` (whole GPU, vLLM defaults except `gpu_memory_utilization` 0.90),
  two presets (`2g.48gb` and `1g.24gb` at the same vLLM settings), a final repeat of the
  baseline as a drift check (study 26 found ~5 % drift with GPU temperature), then an
  `optimize` step of **~25 experiments**. Budget estimate: ~30 experiments × ~75 min ≈
  38 h, ≈ 130–150 USD (GPU + load node), excluding the spike.
- **Akamas 4.1** (`akamas-41`), all Akamas YAML generated with the `akamas-study-manager`
  plugin, validated with `akamas create` before "done" (repo rules).

- **Comparison with smaller GPUs (goal 3):** no reference arm in this study. Study 45
  measured tokens/s per $/h, not users within SLA, so the README compares on tokens/s per
  $/h (which this study also records) and states that a users-within-SLA reference on
  L4/L40S is a follow-up, added to the ROADMAP backlog.
- **Shared cluster components are safe to touch now:** on 2026-11-03 Akamas 4.1 has no
  RUNNING study (63 FINISHED, 2 CREATED, 2 ERROR) and no GPU node is up;
  `studies/README.md` rows still saying RUNNING (28, 30–33) are stale. Edits to the device
  plugin DaemonSet, `nvdp-g7` and `dcgm-exporter` only **add** an exclusion/inclusion for
  `llm-serving-g7e-2xl`; no other node group's scheduling changes.
- **Pause points for the user's OK:** (1) after Phase 0 (pass/fail + CPU reading),
  (2) before toolbox sync / `akamas create` / `akamas start study` (repo practice), (3)
  before any fallback (static MIG, 2 replicas max, other instance size).

- **"User" = one request always in flight** (closed loop, no think time), as in studies
  25/26 — the scored number. The README also translates it into people served under a
  stated think time (e.g. 20 s between messages), clearly labelled as a derived figure.
  Caveat known up front: under this definition study 26 found slices *lose* users vs the
  whole GPU; if that repeats, it is reported as the finding. (User given the choice
  2026-11-03, said "vai" without picking: Claude's recommended default applies.)
- **No deadline set** for g7e capacity; the builder probes capacity before each attempt
  and reports to the user if the node cannot be obtained within a working day.

## CPU fairness (anti-goal guard)

- Each vLLM pod gets a CPU request/limit of ⌊node allocatable CPU / N_replicas⌋ (whole GPU
  gets all of it), so the comparison is per-slice fair.
- Telemetry must include **CPU throttling of every vLLM container** and **CPU of every
  AIPerf pod**. An experiment where a vLLM container is throttled > ~10 % of the time or an
  AIPerf pod is near its limit at the level that sets the score is flagged in the README
  (and, if systematic, the builder stops and reports — the fallback is 2 replicas max or
  waiting for g7e.8xlarge capacity, the user's call).

## Phase 0 — DRA spike (gate before any Akamas resource is created)

Throwaway, on the new node, a few hours. Pass criteria, all required:

1. `nvidia-smi` on the node: driver **≥ 580**, MIG capable; record version, power limit.
2. Driver installed with `DynamicMIG`; `ResourceSlice`s published for the node.
3. A pod with a claim for a `1g.24gb` device (`mig.nvidia.com` DeviceClass + CEL on
   `profile`) gets it; **four** such pods at once; then delete and request **two**
   `2g.48gb`; then **one whole GPU** (`gpu.nvidia.com`). Record the time to re-partition.
4. vLLM with Qwen3-8B-FP8 starts on a `1g.24gb` slice and reports its KV cache size.
5. dcgm-exporter shows per-MIG-instance series.
6. **CPU check:** four `1g.24gb` vLLM replicas under a short AIPerf load at a mid
   concurrency level; record CPU throttling per container. If throttling is material, the
   options (accept and flag / cap at 2 replicas / move to g7e.8xlarge when capacity
   appears) go to the user at pause point 1 — the `1g.24gb` arm is not dropped silently.

Open question the spike answers: with `DynamicMIG` on, can a **full-GPU claim** still be
satisfied on Blackwell (docs say Hopper+ toggles MIG on demand; Ampere cannot)? If not,
the "whole GPU" arm becomes `4g.96gb` (MIG, ~3–6 % overhead per study 26) and that is
stated in the README.

**If DynamicMIG fails:** fall back to **static MIG through DRA** (GA in the driver;
partitions created by the apply script, kubelet plugin restarted). **If DRA fails
entirely:** stop and report to the user; do not silently use the device plugin (anti-goal).

## How an experiment is applied (sketch — details left to the builder)

Same shape as study 26's `apply_config.sh`, minus the device-plugin steps: render params
→ delete vLLM pods and their claims (driver tears down partitions) → render one
`ResourceClaimTemplate` for the chosen profile and a StatefulSet with N replicas
consuming it → wait Ready with crash-loop fail-fast → start N AIPerf Jobs on the load
node → compute the score. Study 25/26 guards carry over (fail on Job failure, replica
restart, or no completions for 15 min).

## Left to the builder

- Concurrency ramp levels and dwell per level (resolution ≤ ~15 % around the expected
  break point; whole GPU may need levels well above 512 per replica).
- How "highest passing level" reaches Akamas (CSV/Prometheus/pushgateway telemetry);
  Akamas windowing choice.
- Exact vLLM parameter domains, `parameterConstraints`, and KPI selection (≤ 8 KPIs).
- Namespace (new one, e.g. `dra-mig`; the stale `gpu-sharing/vllm-0` pod is left alone
  or cleaned up with the user's OK), served model name unique to this study.
- Scripts for node-group creation/scaling, probing capacity first
  (`studies/17-…/infra/eks/gpu-capacity-fallback.sh probe`).
- `infra/` folder per repo convention; update the reference `cluster.yaml` version only
  if the user asks (separate change).

## Risks

- **g7e capacity** moves between AZs within hours (2026-11-03 probes); the study may wait.
- **`DynamicMIG` is alpha**; Phase 0 exists for this.
- **CPU on 8 vCPU** with 4 replicas — guarded above.
- Nightly EC2 stop (tag `AlwaysOn`), Akamas 4.1 "Could not start the study" DAG timeout
  (known workaround), AIPerf results PVC size (study 41: size/clean it).
