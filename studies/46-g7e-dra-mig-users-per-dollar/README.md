# 46-g7e-dra-mig-users-per-dollar

**Status:** TODO — scaffolded 2026-11-03. Next: Phase 0 (DRA spike) on a new
`g7e.2xlarge` node; Akamas resources are generated only after Phase 0 passes.
**Dates:** 2026-11-03 –

Design: [`docs/superpowers/specs/2026-11-03-study-46-g7e-dra-mig-users-per-dollar-design.md`](../../docs/superpowers/specs/2026-11-03-study-46-g7e-dra-mig-users-per-dollar-design.md)
(approved 2026-11-03). This README restates what is needed to run the study.

## Objective

**A generic customer case: a small model served on a GPU that is far too big for it.**
How many users does one RTX PRO 6000 Blackwell 96 GB serve within a reasonable SLA when
it is split with MIG, with the slices **requested through Kubernetes Dynamic Resource
Allocation (DRA)**, compared with the same GPU kept whole?

- **Goal (maximize):** users within the SLA on the whole GPU =
  **N_replicas × min over replicas of the highest concurrency level at which that
  replica meets the SLA**. The minimum is used instead of the sum so that the score is
  what every tenant on the GPU is guaranteed.
- **"User"** = one request always in flight (closed loop, no think time), as in studies
  25/26. The README will also translate the result into people served with a stated
  think time (for example 20 s between messages), labelled as a derived number.
- **SLA, checked per replica:** TTFT p95 <= 1500 ms, ITL p95 <= 300 ms, and at least
  95 % of requests completed (same values as studies 26/28).
- **Cost:** one g7e.2xlarge at 3.36 USD/h on-demand (us-east-2, 2026-11-03). The cost is
  the same in every experiment, so users per $/h is computed here, not in Akamas. The
  load-generator node is not counted.

Why now: the `vllm-bench` cluster moved to Kubernetes 1.36 on 2026-11-02/03. On 1.36
`DRAPartitionableDevices` is enabled by default, and the NVIDIA DRA driver's
`DynamicMIG` needs it. On 1.35 it was out of reach because EKS does not expose
control-plane feature gates (ROADMAP, "DRA prerequisites and risk"). DRA brings a
different **mechanism**, not a new way to share the GPU: each pod asks for its slice in
a `ResourceClaim`, and the driver creates the MIG partition. Studies 25/26 needed node
relabelling, a device-plugin reconfiguration and `nvidia-smi mig` instead.

Prior results this study builds on:

- **Study 25**, on a 165 W RTX PRO 4500: splitting a power-capped GPU adds no
  throughput.
- **Study 26**: MIG costs about 3–6 %. Users within the SLO do **not** scale with the
  slice, because the KV cache per slice shrinks faster. If slices lose users here too,
  that is the finding and gets reported as such.

The RTX PRO 6000 is a 600 W-class GPU, so study 25's power-cap result may not carry over.
Nothing is assumed about it in either direction.

**Out of scope:** MPS and time-slicing (already measured in study 25, and
`MPSSupport` cannot be used together with `DynamicMIG`), mixed slice layouts, several
models at once, different vLLM settings within one experiment, and comparing g7e sizes.

## Stack & versions

- **Akamas version:** 4.1.0 (`akamas-41`, <https://akamas41.lab.akamas.io>).
- **Optimization packs** (checked on Akamas 4.1 on 2026-11-03): GPU **1.4.0**, whose
  `mig_profile` domain already contains `none`, `2g.48gb` and `1g.24gb`; vLLM
  **1.14.0**; Kubernetes **1.9.0**.
- **Workload under test:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8` (about 8.9 GiB
  of weights, measured in study 28), `--max-model-len 4096`, prefix caching off. The
  served model name will be unique to this study. One pod per MIG slice, each with its
  own `ResourceClaim`.
- **Cluster / hardware:** EKS `vllm-bench`, us-east-2, **Kubernetes 1.36.5**.
  - **GPU node:** new node group `llm-serving-g7e-2xl`, 1x **g7e.2xlarge** (8 vCPU /
    64 GiB, 1x RTX PRO 6000 Blackwell Server Edition 96 GB).
  - **AMI:** `AL2023_x86_64_NVIDIA` for 1.36 (not pinned).
  - **Subnets:** 2a/2b/2c, because g7e capacity moved between zones within hours on
    2026-11-03.
  - **Labels, taints and tags:** label `node-role=llm-serving-g7e-2xl`, taint
    `nvidia.com/gpu=present:NoSchedule`, min 0 / max 1. Tag node and ASG
    `AlwaysOn=true` (EC2 instances are stopped every night at 17:00 UTC).
  - **MIG profiles on this GPU:** `1g.24gb` x4, `2g.48gb` x2, `4g.96gb` x1.
- **DRA:**
  - **Driver:** the standalone `dra-driver-nvidia-gpu` Helm chart with
    `featureGates.DynamicMIG=true`, scheduled on the g7e node only. It uses the AMI's
    NVIDIA driver, which must be 580 or newer.
  - **Not the GPU Operator `GPUCluster` path:** that path is for new installs only and
    takes over the device plugin on the whole cluster.
  - **Classic device plugin:** both the `kube-system` DaemonSet and the `nvdp-g7` release
    are kept off the g7e node.
- **Load generator:** AIPerf (same version as study 28, 0.11.0, unless Phase 0 shows a
  reason to change), ShareGPT, closed-loop concurrency ramp, **one Job per replica**
  with a fixed per-replica seed, all replicas under load at the same time. Pods run on
  the existing CPU node group **`m8a-2xlarge`** (m8a.2xlarge, taint
  `dedicated=m8a-2xlarge`), scaled from 0 to 1 for the study. If the load node and the
  GPU node end up in different zones (about 1 ms extra), this is accepted and recorded
  here.
- **Telemetry:** Prometheus (kube-prometheus-stack): per-replica vLLM metrics and
  dcgm-exporter per MIG instance (its affinity needs extending to the new node).
  **CPU-fairness guard:** CPU throttling of every vLLM container and CPU of every AIPerf
  pod. Each vLLM pod gets ⌊allocatable CPU / N⌋ CPU.

## Parameters tuned

| Component.parameter | Domain (to validate against the packs when generating YAML) | Baseline |
|---|---|---|
| `gpu.mig_profile` | `none` (whole GPU, 1 replica), `2g.48gb` (2 replicas), `1g.24gb` (4 replicas) | `none` |
| `vllm.max_num_seqs` | TBD from Phase 0 KV measurements (pack: 1–4096) | vLLM default |
| `vllm.max_num_batched_tokens` | TBD (pack: 1–65536) | 2048 |
| `vllm.gpu_memory_utilization` | about 0.85–0.95 (fraction of the slice under MIG) | 0.90 |
| `vllm.kv_cache_dtype` | `auto`, `fp8` | `auto` |

The replica count follows from `mig_profile` and is not a separate parameter. Every
replica in an experiment gets the same vLLM values.

**Steps (planned):**

1. `baseline`: whole GPU at defaults.
2. Preset `2g.48gb`.
3. Preset `1g.24gb`.
4. Baseline repeat, to check for drift. Study 26 saw about 5 % drift with GPU
   temperature.
5. `optimize`: about 25 experiments.

Estimate: about 30 experiments × 75 min ≈ 38 h, ≈ 130–150 USD, plus Phase 0.

## Phase 0: DRA spike (must pass before anything is created on Akamas)

1. `nvidia-smi` on the node reports driver 580 or newer and MIG support.
2. The driver is installed with `DynamicMIG` and publishes `ResourceSlice`s.
3. Claims are tested in this order: four `1g.24gb` at once, then two `2g.48gb`, then one
   whole GPU. Record the time each re-partition takes.
   - Open question: can the whole-GPU claim still be served with `DynamicMIG` enabled?
     If not, the whole-GPU arm becomes `4g.96gb`, and this README says so.
4. vLLM with Qwen3-8B-FP8 starts on a `1g.24gb` slice. Record its KV cache size.
5. dcgm-exporter publishes one series per MIG instance.
6. CPU check: four replicas under a short AIPerf load. Record CPU throttling for each.

**Fallbacks:**

- **`DynamicMIG` fails:** switch to static MIG through DRA.
- **DRA fails entirely:** stop and report. The study does not fall back silently to
  the device plugin.
- **CPU throttling is material:** ask the user to choose: accept it and flag it, cap
  the study at 2 replicas, or move to g7e.8xlarge.

**Pause points for the user's OK:**

1. After Phase 0.
2. Before syncing the toolbox, `akamas create` or `akamas start study`.
3. Before any fallback.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
