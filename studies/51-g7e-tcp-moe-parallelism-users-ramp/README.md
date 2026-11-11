# 51-g7e-tcp-moe-parallelism-users-ramp

**Status:** BUILT locally on 2026-11-11, not started. The scripts are tested locally (215
assertions). Nothing of this study has run on real nodes yet: the smoke checks below are still
to do.

This is **study 50's design run without EFA**. Study 50 needs two g7e.8xlarge (the smallest g7e
with EFA) and us-east-2 had no such pair from 20:34 UTC on 2026-11-10 onwards. On 2026-11-11 the
user chose to run the same seven layouts on two g7e.4xlarge over TCP instead. Study 50 stays
built and blocked, for when the pair is available.

- **Design:** `docs/superpowers/specs/2026-11-09-study-50-g7e-efa-moe-parallelism-design.md`
  (study 50's; the differences are in "Deviations from study 50" below).
- **Builds on:** study 44 (multi-node over TCP, the 5 Gbit/s single-flow cap), study 48 (ramp
  and hold of chat users), study 49 (two AIPerf clients, pod-level config checks), the EFA spike
  (`docs/spikes/2026-11-10-efa-eks/README.md`, which also measured TCP) and study 50 (every
  script and Akamas file here is a copy of study 50's, adapted).

## Objective

A customer with two plain GPU servers, one GPU each, asks how to serve a MoE model across them.
This study measures, **over plain TCP**, how many chat users each parallelism layout serves
within the SLO, and what each layout costs on the wire.

- **Goal:** the most users served with **client TTFT p95 ≤ 500 ms**, at the per-user load of
  studies 47 and 48 (one turn every 64 s, history re-sent).
- **Seven layouts** on the same two nodes: one GPU (node b idle, the "half the hardware"
  reference), two independent replicas, DP2, DP2 + expert parallel, TP2, TP2 + expert parallel,
  PP2.
- **Evidence for the differences:** network bytes per node, TCP retransmissions, GPU utilization
  and memory per GPU, queue depths, ITL.
- **Every cross-node experiment proves its transport:** `apply_config.sh` fails the trial if
  NCCL logged anything other than `Socket`.
- **Not a goal:** cost per user (both nodes are up in every experiment), NCCL tuning, the
  comparison with EFA (that needs study 50 on the same instance size).

The sentence this study should let us write: *"On two RTX PRO 6000 nodes over TCP, layout X
serves N chat users within TTFT p95 ≤ 500 ms, against M on one GPU; the cross-node layouts cost
Y % because of Z."*

## Method

Each experiment runs four workflow tasks (`akamas/51-G7E-TCP-MoE-Parallelism-Workflow.yaml`):

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

- **Hardware:** two g7e.4xlarge on demand, node group `llm-serving-g7e-4xl-tcp`, one
  availability zone, one cluster placement group (`study51-g7e-4xl-tcp`).
  - Per node: NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB, 16 vCPU, 128 GiB, 50 Gbit/s,
    no EFA, 4.00 USD/h.
  - Load node: `m8a-2xlarge` (m8a.2xlarge). EKS `vllm-bench` 1.36, us-east-2. See
    `infra/README.md`.
- **Server:** AWS Deep Learning Container
  `763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.29.0-gpu-py312-cu130-ubuntu24.04-ec2`
  (vLLM 0.29.0, torch 2.13.0+cu130, NCCL 2.29.7). The same image as study 50. No custom image.
  - `openai/gpt-oss-120b` served as `gpt-oss-120b`, `--max-model-len 16384`, prefix caching on.
  - Per pod: 13 CPU, 100 GiB, one GPU, 16 GiB memory-backed `/dev/shm` (inside the 100 GiB).
  - **`NCCL_NET=Socket`** in the pod environment: the image ships the OFI plugin, and the
    transport must not depend on what that plugin finds on a node without EFA.
  - Multi-node layouts: `--nnodes 2 --node-rank <0|1> --master-addr <pod a IP>`. Node b is
    headless on TP and PP. On DP both nodes run an API server
    (`--data-parallel-external-lb`).
- **Load:** AIPerf 0.11.0, two client pods (3 CPU each) on the load node.
- **Akamas:** 4.1.0 with packs vLLM 1.15.0, GPU 1.4.0 and Kubernetes 1.9.0. See
  `akamas/README.md`.
- **Telemetry:** Prometheus, 113 metrics.
  - vLLM scraped every 5 s on every pod with an API server.
  - **NVIDIA DCGM metrics** per physical GPU (`gpu_a`, `gpu_b`).
  - Node CPU, memory, network bytes and TCP retransmits per GPU node (node-exporter). NCCL runs
    on TCP, so the traffic between the two pods is in the nodes' network counters.
  - The study's role exporter (`k8s/monitoring/role-exporter.yaml`) only maps node names to
    roles a and b, for the joins.

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

## Deviations from study 50

| | Study 50 | Study 51 |
|---|---|---|
| Instances | 2 × g7e.8xlarge (32 vCPU, 256 GiB, 100 Gbit/s) | 2 × g7e.4xlarge (16 vCPU, 128 GiB, 50 Gbit/s) |
| Price per node | 5.27 USD/h | 4.00 USD/h |
| NCCL transport | EFA (OFI), proven per experiment | TCP (`NCCL_NET=Socket`), proven per experiment |
| vLLM pod | 26 CPU, 160 GiB, EFA device, huge pages, `IPC_LOCK` | 13 CPU, 100 GiB |
| Wire evidence | EFA byte counters (own exporter) | node network bytes and TCP retransmits (node-exporter) |
| Telemetry metrics | 115 | 113 (no EFA stand-ins) |
| KPI 6 | EFA bytes sent by node a | Network bytes sent by node a |
| TCP controls by hand | TP2 and DP2 + expert parallel | none: every layout is on TCP |

- **The pods have half the CPU of study 50's.** A later comparison of the two studies mixes
  transport and CPU: only a study 50 run on the same pod size isolates EFA.
- **Cluster placement group**, as in study 50: inside one a single TCP flow reaches 10 Gbit/s,
  outside 5 Gbit/s (study 44). `PLACEMENT=none` in `nodegroups.sh` is the fallback if the pair
  does not start inside one; the README records which one ran.
- **A node role of its own** (`llm-serving-g7e-4xl-tcp`), not the existing
  `llm-serving-g7e-4xl` group: that one is targeted by the DRA driver DaemonSet of study 46.
- **Shared cluster object changed for this study:** the dcgm-exporter Helm release in
  `monitoring` gets one more node role (`k8s/monitoring/dcgm-exporter-values.yaml`).

Kept from study 50: external load balancing on the DP layouts (`DP_MODE=internal` as a
fallback), two AIPerf clients on every layout, eight ramp levels (worst case about 2.9 hours per
experiment), Nsight Perf SDK for AI as a one-hour non-blocking attempt.

## Before starting: smoke checks

None done yet. Each must pass on the real pair before `akamas start study`, and each prints its
evidence. `k8s/smoke.sh [layout ...]` applies a layout at the presets' flags and runs one short
level.

1. **Nodes:** both Ready and labelled `study51/role`, `nvidia.com/gpu: 1`, in the placement
   group, a dcgm-exporter pod on each, the model on both (`02-model-download.yaml`).
2. **One GPU:** `smoke.sh single`. Effective pod arguments and `cache_config_info` read by
   `apply_config.sh`; AIPerf's output token count per request is 200 (else a `min_tokens` trial).
3. **Cross-node layouts:** `smoke.sh dp2 dp2ep tp2 tp2ep pp2`. NCCL transport `Socket` on both
   pods, expert parallelism line where asked, the 65 GB load inside the 2400 s wait. If `dp2`
   without expert parallelism does not start under external load balancing: `DP_MODE=internal`
   for both DP layouts.
4. **Two replicas:** `smoke.sh replicas`, the two clients produce one RESULT.
5. **Telemetry:** every query of `akamas/telemetry/prometheus.yaml` returns data for vLLM, both
   GPUs and both nodes; the toolbox may apply the role exporter and the ServiceMonitors; AIPerf
   client CPU below its limit at the top levels.
6. **Nsight:** one hour, non-blocking.

## Results

Not run yet.

## Conclusions

Not run yet.
