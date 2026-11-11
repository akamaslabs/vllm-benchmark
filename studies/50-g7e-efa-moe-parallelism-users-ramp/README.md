# 50-g7e-efa-moe-parallelism-users-ramp

**Status:** BUILT locally on 2026-11-10, not started. The EFA spike passed
(`docs/spikes/2026-11-10-efa-eks/README.md`). The scripts are tested locally (214 assertions).
**Blocked on capacity:** no availability zone of us-east-2 had two g7e.8xlarge for ten hours,
from 20:34 UTC on 2026-11-10 to 06:29 UTC on 2026-11-11 (`nodegroups.sh wait-up 10`, one probe
every 5 minutes, 111 probes, none with room). The wait ended without creating anything. A second
wait (`wait-up 24`) started at 08:07 UTC on 2026-11-11 and had found no capacity by 08:44 UTC.
**Fallback chosen by the user on 2026-11-11:** the same design without EFA on two g7e.4xlarge,
as study 51 (`../51-g7e-tcp-moe-parallelism-users-ramp/README.md`). The lab's G quota (64 vCPU)
does not hold both pairs: the second wait is stopped when study 51's nodes are created. This
study stays built for when a g7e.8xlarge pair is available. Nothing of this study has run on real
nodes yet: the smoke checks below are still to do. The Akamas resources exist on akamas-41 since
2026-11-10 (every file accepted by `akamas create`, study not started), and Prometheus parses
all 115 telemetry queries; whether each returns data is smoke check 6.

- **Design:** `docs/superpowers/specs/2026-11-09-study-50-g7e-efa-moe-parallelism-design.md`
  (go on 2026-11-10).
- **Plan:** `docs/superpowers/plans/2026-11-10-study-50-g7e-efa-moe-parallelism.md`.
- **Builds on:** study 44 (multi-node over TCP, the 5 Gbit/s single-flow cap), study 48 (ramp
  and hold of chat users), study 49 (two AIPerf clients, pod-level config checks) and the EFA
  spike.

## Objective

A customer with two plain GPU servers, one GPU each, asks how to serve a MoE model across them.
This study measures, **over EFA**, how many chat users each parallelism layout serves within the
SLO, and what each layout costs on the wire.

- **Goal:** the most users served with **client TTFT p95 ≤ 500 ms**, at the per-user load of
  studies 47 and 48 (one turn every 64 s, history re-sent).
- **Seven layouts** on the same two nodes: one GPU (node b idle, the "half the hardware"
  reference), two independent replicas, DP2, DP2 + expert parallel, TP2, TP2 + expert parallel,
  PP2.
- **Evidence for the differences:** EFA bytes per node, GPU utilization and memory per GPU,
  queue depths, ITL.
- **Every cross-node experiment proves its transport:** `apply_config.sh` fails the trial if
  NCCL did not pick EFA. Two TCP controls (TP2 and DP2 + expert parallel with
  `NCCL_NET=Socket`) are run by hand after the presets.
- **Not a goal:** cost per user (both nodes are up in every experiment), NCCL tuning.

The sentence this study should let us write: *"On two RTX PRO 6000 nodes over EFA 100 Gbit/s,
layout X serves N chat users within TTFT p95 ≤ 500 ms, against M on one GPU; the cross-node
layouts cost Y % because of Z."*

## Method

Each experiment runs four workflow tasks (`akamas/50-G7E-EFA-MoE-Parallelism-Workflow.yaml`):

1. **Write config:** the ten parameters into `k8s/params.env`.
2. **Apply config** (`k8s/apply_config.sh`): deletes and re-creates the StatefulSets `vllm-a`
   and `vllm-b` in the experiment's layout (`k8s/lib_layout.sh`), waits for vLLM, then checks on
   the running pods: the arguments, `vllm:cache_config_info`, the two ranks, the expert
   parallelism line and the NCCL transport. Any mismatch fails the trial.
3. **Ramp** (`k8s/run_ramp.sh`): levels 32, 64, 128, 192, 256, 384, 512, 768 users. Each level
   is a fresh 16-minute AIPerf run, judged on the requests started after minute 10 against
   client TTFT p95 ≤ 500 ms. It stops at the first failing level.
4. **Hold** (`k8s/run_hold.sh`): the best level again, fresh. Akamas scores this task.

**Load:** AIPerf multi-turn chat, 10 turns per conversation, 600 input tokens per turn, 200
output tokens (`ignore_eos`, `reasoning_effort: low`), history re-sent. Two AIPerf clients take
half the users each. Client 0 always targets node a. Client 1 targets node b when node b runs an
API server (two replicas, DP layouts), node a otherwise.

**Scoring:** users = 64 × completed requests per second over the Hold window, minus TTFT p95 in
seconds as a tie-break. Server-side safety nets: TTFT p95 ≤ 600 ms, ITL p95 ≤ 200 ms.

## Stack & versions

- **Hardware:** two g7e.8xlarge on demand, node group `llm-serving-g7e-8xl-efa`, one cluster
  placement group.
  - Per node: NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB, 32 vCPU, 256 GiB, 100 Gbit/s,
    EFA, 5.27 USD/h.
  - Load node: `m8a-2xlarge` (m8a.2xlarge). EKS `vllm-bench` 1.36, us-east-2. See
    `infra/README.md`.
- **Server:** AWS Deep Learning Container
  `763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.29.0-gpu-py312-cu130-ubuntu24.04-ec2`
  (vLLM 0.29.0, torch 2.13.0+cu130, NCCL 2.29.7, aws-ofi-nccl 1.18.0, libfabric 2.4). No custom
  image.
  - `openai/gpt-oss-120b` served as `gpt-oss-120b`, `--max-model-len 16384`, prefix caching on.
  - Per pod: 26 CPU, 160 GiB, one GPU, one EFA device, 5120 Mi of 2 MiB huge pages, `IPC_LOCK`,
    16 GiB memory-backed `/dev/shm`.
  - Multi-node layouts: `--nnodes 2 --node-rank <0|1> --master-addr <pod a IP>`. Node b is
    headless on TP and PP. On DP both nodes run an API server
    (`--data-parallel-external-lb`).
- **Load:** AIPerf 0.11.0, two client pods (3 CPU each) on the load node.
- **Akamas:** 4.1.0 with packs vLLM 1.15.0, GPU 1.4.0 and Kubernetes 1.9.0. See
  `akamas/README.md`.
- **Telemetry:** Prometheus.
  - vLLM scraped every 5 s on every pod with an API server.
  - **NVIDIA DCGM metrics** per physical GPU (`gpu_a`, `gpu_b`).
  - **EFA byte counters** from the study's own exporter (`k8s/monitoring/efa-exporter.yaml`):
    they exist only under the host's `/sys/class/infiniband`, node-exporter does not export
    them, and EFA traffic does not show in the ENA counters.
  - Node CPU, memory, ENA traffic and TCP retransmits per GPU node.

## Parameters tuned

Presets only: the first five parameters pick the layout, the other five are the same in every
step. The domains are the space a follow-up optimize study would search.

| Parameter | Domain | Presets |
|---|---|---|
| `vllm_workload.replicas` | 1–2 | 2 on "Two replicas", else 1 |
| `vllm.tensor_parallel_size` | 1–2 | 2 on the TP layouts |
| `vllm.data_parallel_size` | 1–2 | 2 on the DP layouts |
| `vllm.pipeline_parallel_size` | 1–2 | 2 on "PP two" |
| `vllm.enable_expert_parallel` | true, false | true on the two expert-parallel layouts |
| `vllm.gpu_memory_utilization` | 0.80–0.92 | 0.90 |
| `vllm.max_num_seqs` | 128–1024 | 512 |
| `vllm.max_num_batched_tokens` | 2048–32768 | 8192 |
| `vllm.kv_cache_dtype` | auto, fp8 | auto |
| `vllm.moe_backend` | auto, triton, marlin, flashinfer_cutlass | auto |

Constraints between parameters: `max_num_batched_tokens ≥ max_num_seqs`; replicas × TP × DP × PP
≤ 2; expert parallelism only with TP or DP at 2. The baseline is the one-GPU layout.

## Deviations from the design

- **DP layouts use external load balancing, not hybrid.** The spike showed that
  `--data-parallel-hybrid-lb` needs more than one GPU per node. With external load balancing
  each node serves its local engine and the two AIPerf clients take one node each. `DP_MODE=internal`
  (one API server, node b headless) stays available in `apply_config.sh` as a fallback.
- **Two AIPerf clients on every layout**, also where only node a serves, so the client side is
  the same everywhere.
- **Eight ramp levels (32 to 768)**, one more than the design's budget table assumed. Worst case
  per experiment: about 2.9 hours.
- **EFA bytes are stored under `gpu_nvlink_tx_bytes` / `gpu_nvlink_rx_bytes`** of the GPU
  components: no installed pack has a network-fabric metric. Pack TODO: a metric pair for it.
- **The TCP controls run after the presets**, not during the smoke, so the night goes to the
  Akamas study. Results will go to `results/tcp-controls/`.
- **Shared cluster objects changed for this study:** the dcgm-exporter Helm release in
  `monitoring` got one more node role (revision 27). On the spike, eksctl also replaced the
  shared `kube-system/nvidia-device-plugin-daemonset` (v0.18.0 with node-role exclusions became
  v0.19.0 without). It is **not rolled back**: that is the cluster owner's call (command in the
  spike README). Study 50's node group is created with `--install-nvidia-plugin=false`.
- **Nsight Perf SDK for AI:** still a one-hour, non-blocking attempt during the smoke.

## Before starting: smoke checks

None done yet. Each must pass on the real pair before `akamas start study`, and each prints its
evidence. `k8s/smoke.sh [layout ...]` applies a layout at the presets' flags and runs one short
level.

1. **Nodes:** both Ready and labelled `study50/role`, `nvidia.com/gpu: 1`,
   `vpc.amazonaws.com/efa: 1`, enough `hugepages-2Mi`, a dcgm-exporter pod on each, the model on
   both (`02-model-download.yaml`).
2. **NCCL over EFA and over TCP** on the pair (the spike's `nccl_bench.py`), numbers here.
3. **One GPU:** `smoke.sh single`. Effective pod arguments and `cache_config_info` read by
   `apply_config.sh`; AIPerf's output token count per request is 200 (else a `min_tokens` trial).
4. **Cross-node layouts:** `smoke.sh dp2 dp2ep tp2 tp2ep pp2`. Transport line `NET/OFI` on both
   pods, expert parallelism line where asked, the 65 GB load inside the 2400 s wait. If `dp2`
   without expert parallelism does not start under external load balancing: `DP_MODE=internal`
   for both DP layouts.
5. **Two replicas:** `smoke.sh replicas`, the two clients produce one RESULT.
6. **Telemetry:** every query of `akamas/telemetry/prometheus.yaml` returns data for vLLM, both
   GPUs, both nodes and the EFA counters; the toolbox may apply the EFA exporter and the
   ServiceMonitors; AIPerf client CPU below its limit at the top levels.
7. **Nsight:** one hour, non-blocking.

## Results

Not run yet.

## Conclusions

Not run yet.
