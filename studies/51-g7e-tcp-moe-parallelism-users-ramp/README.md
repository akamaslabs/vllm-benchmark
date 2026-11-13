# 51-g7e-tcp-moe-parallelism-users-ramp

**Status:** DONE. Study `51-G7E-TCP-MoE-Parallelism` ran on akamas-41 from 11:05 to 21:08 UTC on
2026-11-11: seven preset experiments, no optimize step, all valid. **PP two served 512 users, two
replicas 256, one GPU 64** (see Results). The nodes came up at 09:02 UTC in us-east-2a, inside
the placement group, and were deleted at 00:56 UTC on 2026-11-12, after the Nsight captures
(about 16 hours up). The scripts are tested locally (215 assertions).

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

Done on 2026-11-11 on the real pair, before `akamas start study`. `k8s/smoke.sh [layout ...]`
applies a layout at the presets' flags and runs one short level (32 users).

1. **Nodes: passed.** Both Ready and labelled `study51/role` (`ip-192-168-174-92` = a,
   `ip-192-168-182-253` = b), `nvidia.com/gpu: 1` each, both instances in the placement group
   `study51-g7e-4xl-tcp` (us-east-2a), a dcgm-exporter pod on each (Helm revision 28), the model
   on both (61 GB, 3 minutes per download Job).
2. **One GPU: passed.** The pod's arguments equal the preset's flags; `vllm:cache_config_info`
   shows `gpu_memory_utilization` 0.9 and prefix caching on; GPU KV cache 231,902 tokens.
   `vllm:generation_tokens_total` gives exactly 200 output tokens per request (AIPerf's own
   tokenizer counts about 191): no `min_tokens` needed.
3. **Cross-node layouts: passed**, all five. NCCL logged `Socket` on both pods (32 "Using network
   Socket" lines, no OFI or IB line), both ranks assigned, the expert parallelism line on both
   pods of the two EP layouts (64 local of 128 experts). `dp2` without expert parallelism started
   under external load balancing: `DP_MODE=internal` not needed. vLLM was ready in 142 to 190 s.
4. **Two replicas: passed.** A ten-minute level: the two clients produced one RESULT (269 of 269
   requests).
5. **Telemetry: passed.** Checked while both replicas were under load: 111 of the 113 queries of
   `akamas/telemetry/prometheus.yaml` return data. The two empty ones are `gpu_xid_errors` and
   `gpu_fan_speed`: the lab's dcgm-exporter exports neither series for any GPU, and neither is
   in the goal, the constraints or the KPIs. The toolbox applied the role exporter and the
   ServiceMonitors. AIPerf client CPU, read during the study: at most 0.11 of 3 cores per
   client, at every level.
6. **Nsight: not done before the start**, the smoke time went to the checks above. It was proven
   the same day on a third node (`docs/spikes/2026-11-11-nsight-perf/README.md`) and run on this
   pair after the last experiment (`nsight/README.md`).

One level of 32 users per layout (3 minutes, 10 for the two replicas), SLO client TTFT p95
≤ 500 ms. These are smoke numbers, not results:

| Layout | vLLM ready (s) | Requests completed | Client TTFT p95 (ms) | Client ITL p95 (ms) |
|---|---|---|---|---|
| One GPU | 140 | 60 of 60 | 144 | 8.0 |
| Two replicas | 143 | 269 of 269 | 90 | 6.1 |
| DP2 | 190 | 61 of 61 | 456 | 21.7 |
| DP2 + expert parallel | 190 | 61 of 61 | 433 | 14.1 |
| TP2 | 174 | 59 of 59 | 407 | 15.6 |
| TP2 + expert parallel | 174 | 60 of 60 | 441 | 14.9 |
| PP2 | 142 | 58 of 58 | 149 | 6.9 |

**Start:** the first `akamas start study` (10:59 UTC) left the study RUNNING with no experiment
(the late Airflow DAG of Akamas 4.1). `akamas finish study` and a second start at 11:05 UTC
created experiment 1.

## Results

All seven experiments ended valid. The study ran from 11:05 to 21:08 UTC on 2026-11-11. Raw data:
`results/` (`export.tar.gz`, the task logs in `logs/`, one row per load level in `levels.csv`,
node and GPU telemetry per level in `level_telemetry.csv`). The page with the charts is at
<https://claude.ai/artifact/BUisueGciJQbeTzk33poLN> (private to its owner).

**PP two serves 512 users, two replicas 256, every other layout 64 or fewer.**

| Exp | Layout | Akamas score | Users (best level) | GPU KV cache (tokens) | Hold: client TTFT p95 (ms) | Hold: client ITL p95 (ms) | First failing level |
|---|---|---|---|---|---|---|---|
| 1 | One GPU (baseline) | 66.5 | 64 | 231,902 | 152 | 8.2 | 128: TTFT p95 715 ms |
| 2 | Two replicas | 255.0 | 256 | 231,902 per replica | 394 | 22.3 | 384: 594 ms |
| 3 | DP two | 65.5 | 64 | 649,059 | 482 | 18.6 | 128: 537 ms |
| 4 | DP two + expert parallel | 64.1 | 64 | 652,049 | 476 | 21.6 | 128: 539 ms |
| 5 | TP two | 32.0 | 32 | 1,369,947 | 407 | 16.0 | 64: 524 ms |
| 6 | TP two + expert parallel | 61.0 | 64 | 1,402,755 | 526 | 36.8 | 128: 567 ms |
| 7 | **PP two** | **511.5** | **512** | 1,418,003 | 254 | 48.3 | 768: the server fell behind |

- **Against the baseline:** PP two serves 8 times the users of one GPU, two replicas 4 times.
  The levels are discrete: one GPU passes 64 and fails 128, so its real limit is somewhere
  between the two, and these ratios are upper bounds. The safe comparison is PP two against two
  replicas on the same two nodes: two to one.
- **Experiment 6 did not confirm its level.** TP two + expert parallel passed 64 users on the
  ramp (484 ms) and read 526 ms on the Hold, above the 500 ms goal. Akamas accepted it because
  its constraint is on the server-side TTFT (484 ms against 600). Read it as "32 to 64".
- **Latencies are the worse of the two AIPerf clients**, not a pooled percentile
  (`k8s/lib_load.sh`). One GPU at 128 users: 715 and 524 ms on the two clients.
- **No constraint was binding** on a valid experiment, no request was preempted, and the AIPerf
  clients used at most 0.11 of their 3 cores.

Client TTFT p95 per ramp level (ms, worse client; **bold** = above 500 ms, the ramp stops there):

| Users | One GPU | Two replicas | DP two | DP two + EP | TP two | TP two + EP | PP two |
|---|---|---|---|---|---|---|---|
| 32 | 156 | 95 | 445 | 436 | 389 | 430 | 111 |
| 64 | 161 | 96 | 476 | 466 | **524** | 484 | 170 |
| 128 | **715** | 107 | **537** | **539** | | **567** | 180 |
| 192 | | 351 | | | | | 183 |
| 256 | | 396 | | | | | 140 |
| 384 | | **594** | | | | | 214 |
| 512 | | | | | | | 252 |
| 768 | | | | | | | **82,594** |

### Why PP two beats two replicas: the KV cache

`gpt-oss-120b` takes about 61 GB of a 96 GB GPU. A replica is left with 15.9 GiB of KV cache
(231,902 tokens). PP two puts half of the layers on each GPU: each keeps 48.7 GiB of KV cache,
and because each GPU stores the KV of its own layers only, that is 1,418,003 tokens, six times a
replica's. The load re-sends the history at every turn, so the prefix cache decides the TTFT:

| Users | 32 | 64 | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|
| Prefix cache hit rate, one GPU | 77 % | 77 % | 16 % | | | | | |
| Prefix cache hit rate, two replicas | 77 % | 77 % | 77 % | 58 % | 16 % | 5 % | | |
| Prefix cache hit rate, PP two | 77 % | 77 % | 77 % | 77 % | 77 % | 77 % | 77 % | 23 % |

- 77 % is the most this load allows (the new turn is never cached). A replica loses it from 96
  users per replica onwards, and then recomputes every history: TTFT p50 goes from 99 to 266 ms.
- PP two keeps 77 % up to 512 users. At 768 the cache is exceeded too (23 % hits), the server
  completes 1,730 output tokens/s of the 2,400 offered, the queue grows and TTFT p95 reaches
  83 s. Its limit lies between 512 and 768 users; this run does not say whether memory or compute
  sets it.
- The DP and TP layouts also have large caches (649 thousand to 1.4 million tokens) and keep
  77 % at every level they ran. Their limit is elsewhere.

### Why DP and TP stall: the wire

Network bytes sent per GPU node over the last five minutes of each level (MB/s; DP and TP are
symmetric, PP two sends from node a to node b only):

| Users | DP two | DP two + EP | TP two | TP two + EP | PP two | One GPU, two replicas |
|---|---|---|---|---|---|---|
| 32 | 219 | 216 | 283 | 283 | 6.4 | below 0.5 |
| 64 | 407 | 407 | 488 | 494 | 12.5 | below 0.5 |
| 128 | 865 | 813 | | 969 | 24.5 | below 0.5 |
| 512 | | | | | 95.7 | |

- **DP and TP put 2 to 3 MB on the wire per generated token, in each direction** (6.4 to
  8.9 MB/s per user). PP two sends the hidden state of one layer boundary: about 0.19 MB/s per
  user, 35 to 45 times less.
- **It is latency, not bandwidth.** The highest rate, 969 MB/s, is 7.8 Gbit/s on a 50 Gbit/s
  node, with TCP retransmissions at zero. Every MoE layer of every decoding step waits for a
  round trip, so the TTFT starts at 390 to 445 ms with 32 users, where one GPU gives 156.
- **The GPUs wait.** DP and TP draw 140 to 210 W per GPU while DCGM reports 99 % utilization;
  PP two reaches 345 W and the replicas 489 W. Utilization alone does not show the stall, power
  does. The Nsight traces measure the wait: at 64 users the DP and TP layouts keep only an NCCL
  kernel on the GPU for 60 to 76 % of the time (see "Nsight traces" below).
- **Expert parallelism changed little over TCP:** no gain on DP two (64 users either way), and
  on TP two one ramp level that the Hold did not confirm.

### What PP two pays

| Users | 32 | 64 | 128 | 192 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|
| Client ITL mean, two replicas (ms) | 6.0 | 6.0 | 7.3 | 11.5 | 20.5 | 51.1 | |
| Client ITL mean, PP two (ms) | 7.2 | 6.9 | 9.8 | 13.0 | 17.4 | 29.1 | 46.0 |

Each token crosses both GPUs in sequence, so PP two streams slower than a replica at light load
(7.2 against 6.0 ms) and reaches 46 ms between tokens at 512 users, about 22 tokens/s per user.
From 256 users onwards it is faster than the replicas, which by then recompute their histories.

### Nsight traces

Nsight Perf SDK for AI did not run during the experiments. After the last one, each layout was
applied again with an Nsight variant of the pod, loaded at its best level for 13 minutes and
traced for 10 seconds on every node with a vLLM process. Recipe, files and caveats:
[`nsight/README.md`](nsight/README.md).

| Layout | Users | TTFT p95 with Nsight (ms) | TTFT p95 in the Hold (ms) | Zip size, node a / b (MB) |
|---|---|---|---|---|
| PP two | 512 | 260 | 254 | 9.3 / 10.0 |
| One GPU | 64 | 106 | 152 | 32.2 |
| Two replicas | 256 | 399 | 394 | 16.1 / 16.1 |
| TP two | 32 | 415 | 407 | 19.0 / 18.9 |
| DP two + expert parallel | 64 | 486 | 476 | 19.0 / 19.0 |
| TP two + expert parallel | 64 | 490 | 526 | 12.6 / 12.6 |
| DP two | 64 | 487 | 482 | 19.0 / 19.0 |

Every level passed with the Nsight service attached, so tracing cost nothing visible. These runs
judge a 3-minute window against the study's 6, so they are a check that the layout behaved as in
the study, not a second measurement. The 13 Perfetto zips (219 MB) are in `nsight/traces/`, kept
out of git.

A trace lists every kernel that ran on the GPU. That splits the GPU's time in three states: a
compute kernel is running, only NCCL kernels are on the GPU (they exchange data with the other
node, so the GPU is waiting for its peer), or no kernel at all. Mean of the two nodes, 10 seconds
per layout at its best level:

| Layout | Users | Compute kernel (%) | Only NCCL kernel (%) | No kernel (%) | GR Active (%) | SM Active (%) | Power (W) |
|---|---|---|---|---|---|---|---|
| Two replicas | 256 | 98.6 | 0.0 | 1.4 | 99.6 | 88.3 | 490 |
| One GPU | 64 | 82.9 | 0.0 | 17.1 | 85.2 | 63.0 | 337 |
| PP two | 512 | 73.6 | 26.1 | 0.3 | 100.0 | 63.9 | 397 |
| DP two + expert parallel | 64 | 37.8 | 59.9 | 2.3 | 99.2 | 28.0 | 247 |
| DP two | 64 | 36.6 | 61.1 | 2.3 | 99.1 | 27.8 | 245 |
| TP two | 32 | 24.1 | 43.5 | 32.4 | 68.7 | 16.0 | 201 |
| TP two + expert parallel | 64 | 22.6 | 76.1 | 1.3 | 99.4 | 18.1 | 236 |

- **The DP and TP layouts spend most of the time waiting for the other node:** 60 to 76 % at 64
  users, with compute kernels on the GPU for 22 to 38 %.
- **PP two waits for about a quarter of the time** (23 % on node a, 29 % on node b) and computes
  for the rest. One GPU and two replicas never wait: they exchange nothing.
- **GPU utilization does not see it.** GR Active, the counter behind `nvidia-smi` and DCGM
  utilization, reads 99 to 100 % on every layout at 64 users or more. On PP two it is 100 % both
  while a compute kernel runs and while only an NCCL kernel is on the GPU; SM Active is 82 to
  89 % in the first state and under 5 % in the second.
- **Limits:** one 10-second capture per layout, taken after the study. The trace shows that the
  GPU waits for its peer, not whether the time goes to the TCP wire or to the peer's own compute.

Per-node table, the query for ui.perfetto.dev and the scripts: [`nsight/README.md`](nsight/README.md),
section "What the traces show".

## Conclusions

- **Over TCP, pipeline parallelism is the one cross-node layout worth running for this model:**
  512 chat users within client TTFT p95 of 500 ms on two nodes, against 256 for two independent
  replicas and 64 for one GPU.
- **The gain comes from memory, not from compute.** Splitting the weights across the two GPUs
  triples the KV cache per GPU, and with a history re-sent at every turn the prefix cache is what
  holds the TTFT. Two replicas have twice the compute and lose their cache at 96 users each.
- **DP and TP across nodes over TCP are slower than one GPU.** They exchange 2 to 3 MB per
  generated token and pay a network round trip per MoE layer: 390 to 445 ms of TTFT before any
  load, GPUs at half power. This matches the EFA spike (small-message latency, not bandwidth).
- **Specific to this stack:** a model that fills about two thirds of the GPU, a multi-turn load
  with long re-sent histories, prefix caching on, one turn per user every 64 s. With a model that
  leaves the GPU most of its memory, or without KV pressure, replicas scale linearly (studies 44
  and 45) and PP two would only add latency between tokens.
- **Limits of this run:** one run per layout, eight discrete levels (the ratios move by one
  level), the worse-client rule, pods with 13 CPU. PP two's limit lies between 512 and 768.
- **Surprises:** DCGM utilization stays at 99 % while the GPUs wait for the other node (60 to
  76 % of the time on DP and TP at 64 users, 23 to 29 % on PP two, from the Nsight traces); DP two
  without expert parallelism still moves as many bytes as with it; the Akamas export carries no
  node network metric, so the wire numbers come from Prometheus (`results/level_telemetry.py`,
  10 days of retention).
- **Next:**
  - Study 50 (EFA, two g7e.8xlarge) to see whether DP and TP recover with a low-latency
    transport. It is built and waits for capacity.
  - The fair rival of PP two is two replicas with a larger KV cache: FP8 KV and CPU KV offload
    (studies 47 and 48) on each replica.
  - A finer ramp between 512 and 768 users on PP two, and the NCCL socket variables on the TCP
    layouts (`ROADMAP.md`, section E).
