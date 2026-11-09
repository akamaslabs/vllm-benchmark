# Study 50 — MoE parallelism layouts across two GPU nodes over EFA (design)

**Date:** 2026-11-09 · **Status:** draft, builder check pending · **Study folder (to create):**
`studies/50-g7e-efa-moe-parallelism-users-ramp/`

## Intent and the why

A customer with two "simple" GPU servers, one GPU each, asks how to serve a MoE model across
them: two independent replicas, data parallelism with expert parallelism, tensor parallelism,
pipeline parallelism. Study 44 showed that the inter-node link is the whole story on plain AWS
networking: a single TCP flow caps at ~5 Gbps. The user wants to see the same question **with
EFA**, which is what the customer would buy, and to see where the time goes with NVIDIA's new
Nsight Perf SDK for AI, read after the experiments.

The README sentence this study should let us write:

> *"On two RTX PRO 6000 nodes over EFA 100 Gbps, layout X serves N chat users within TTFT p95
> ≤ 500 ms, against M on one GPU; the cross-node layouts cost Y % because of Z."*

Three things the user asked for explicitly (2026-11-09): the EFA plumbing proven first on cheap
spot instances, the study on g7e.8xlarge, everything torn down at the end, and a report with an
artifact to read the results. The model is `openai/gpt-oss-120b`; the load is a ramp of users.

## Goals, non-goals, anti-goals

- **Goal:** for each parallelism layout on two nodes, the number of chat users served within the
  SLO, with the same per-user load as studies 47/48, and the network and GPU evidence that
  explains the differences (bytes over EFA, SM activity, queue depths).
- **Goal:** a working, documented EFA path on this EKS cluster (node group, device plugin, image,
  NCCL check), reusable by later studies.
- **Goal:** an attempt at Nsight Perf SDK traces during the hold of each experiment, strictly off
  the critical path.
- **Non-goal:** tuning NCCL itself (ROADMAP section E), or DeepEP/pplx kernels, which need
  NVLink/RDMA layouts we do not have.
- **Non-goal:** cost per user. Both nodes are up for every experiment; the single-GPU layout
  leaves one idle on purpose, as the "half the hardware" reference.
- **Anti-goal:** a result that attributes a difference to EFA without a TCP control, or a layout
  that silently fell back to sockets. Every cross-node experiment must prove which transport
  NCCL used.
- **Anti-goal:** repeating study 49's failure: a tuned flag that never reached the pod. The apply
  script verifies the effective arguments and `vllm:cache_config_info` on **every** pod on both
  nodes before the load starts.

## Facts verified on 2026-11-09 (the design rests on these)

- **vLLM 0.29.0** (`vllm/vllm-openai:v0.29.0`, the version the pack references) has
  `--nnodes`, `--node-rank`, `--master-addr` and `--headless` for multi-node TP/PP on the `mp`
  executor: **Ray is not needed**. Data parallel across nodes uses `--data-parallel-size`,
  `--data-parallel-size-local`, `--data-parallel-address`, `--data-parallel-rpc-port`,
  `--data-parallel-start-rank` and `--headless` on the worker. `--all2all-backend` defaults to
  `allgather_reducescatter`; `pplx`/`naive` fall back to it, `deepep_*` need DeepEP.
- In vLLM's data-parallel mode the **expert layers are always shared across the DP × TP group**
  (tensor-parallel by default, expert-parallel with `--enable-expert-parallel`), and "expert
  layers across all ranks are required to synchronize during every forward pass" (vLLM docs,
  data_parallel_deployment). So vLLM DP2 is *not* two independent replicas; study 14's
  observation stands. Two independent replicas are a Kubernetes layout, not a vLLM one.
- **vLLM pack 1.15.0** (installed on akamas-41) carries `tensor_parallel_size` [1,16],
  `pipeline_parallel_size` [1,16], `data_parallel_size` [1,8], `enable_expert_parallel`
  {"true","false"}, `moe_backend` (17 values), `max_num_seqs`, `max_num_batched_tokens`,
  `gpu_memory_utilization`, `kv_cache_dtype`. GPU pack 1.4.0, Kubernetes pack 1.9.0.
- **Instances (us-east-2):** g7e.8xlarge = 1× RTX PRO 6000 Blackwell Server Edition 96 GB,
  32 vCPU, EFA, 100 Gbps, on-demand 5.27 USD/h (spot ~2.46). g6.8xlarge = 1× L4 24 GB, 32 vCPU,
  EFA, 25 Gbps, on-demand 2.01, spot ~0.97. Quotas: "Running On-Demand G and VT" 64 vCPU, "All
  G and VT Spot Instance Requests" 64 vCPU (separate quotas). Two 8xlarge of either kind fill
  one quota exactly: **nothing else G-family can be up alongside**, not even a probe.
- **EFA needs a cluster placement group**, so both nodes land in one AZ. g7e capacity is known
  to move between AZs within hours (study 49 notes): probe for two instances before `--up`.
- **EKS AMI** for 1.36 (`1.36.5-20261031`, AL2023 NVIDIA) ships driver **580.178.04**
  (`nvidia-kmod` default; 595.91.07 is the "latest" variant). Nsight Perf SDK asks for
  Blackwell (ok: RTX PRO 6000) and driver 620+ (power metrics need R615+); with a 580 kernel
  driver it runs through the CUDA 13.5 compatibility user-mode driver, without GR-context
  tracing (no per-kernel GPU timings) and without power. The SDK is "Beta Preview", published
  2026-11-03, documented on Ubuntu 24.04; MIG is not mentioned.
- **AWS Deep Learning Containers** publish `763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.29.0-gpu-py312-cu130-ubuntu24.04-ec2`.
  Whether it carries libfabric + aws-ofi-nccl is **for the spike to verify**; if not, the spike
  builds `vllm/vllm-openai:v0.29.0` + EFA user-space libraries and pushes it to the lab ECR.
- **gpt-oss-120b** (`config.json`): 36 layers, 128 experts, top-4, 8 KV heads × 64, half the
  layers full attention and half sliding-window 128, MXFP4 experts (~61 GiB of weights, HF
  model card). KV ≈ 37 KB/token in bf16 on the full-attention layers plus a few MB per sequence
  on the sliding ones: with ~22 GiB left for KV on a 96 GB card, roughly 600k tokens, i.e.
  ~150 users of ~3.9k live tokens each per GPU before KV pressure. Expect the user ceiling to
  come from prefill/decode compute rather than KV, and to sit well above study 48's 128.

## Hardware and cluster (infra/)

- **Study node group `llm-serving-g7e-8xl-efa`:** 2× g7e.8xlarge, on-demand (study runs are
  long; spot interruptions killed study 42), one AZ chosen by a capacity probe, `efaEnabled`,
  taint `nvidia.com/gpu=present:NoSchedule`, created at desired 0. Defined as an **eksctl
  managed node group** (`infra/eks/nodegroups.yaml` + `nodegroups.sh create|up|down|delete|status`
  wrapping `eksctl` and the AlwaysOn tagging of study 49's `gpu-nodegroup.sh`): eksctl's
  `efaEnabled: true` creates the launch template with the EFA interface, the placement group,
  the security-group rules and installs the EFA device plugin, which the repo's plain AWS-CLI
  scripts cannot. GPUs are served by the standard `nvidia-device-plugin` (no DRA, no MIG: whole
  GPUs); the builder checks that study 46's DRA installer does not exclude this group.
- **Spike node group `llm-serving-g6-8xl-efa-spot`:** 2× g6.8xlarge **spot** (user's choice),
  same shape, same AZ rule. Exists only during the spike and is **deleted** at its end.
- **Load node:** `m8a-2xlarge` node group from study 49 (`load-node.sh`), AIPerf clients.
- **AlwaysOn** on both GPU ASGs and the load node for the whole study; removed at teardown.
- **Model weights:** ~61 GiB loaded on both nodes at every experiment. Each node keeps a
  `hostPath` HF cache warmed during the smoke (first pull ~10 min from HF, then local), so a
  vLLM start is bounded by load-from-disk (~2-3 min on the 8xlarge's EBS; the builder sizes the
  root volume, ≥ 200 GiB gp3). Both nodes pull once; the study clock counts loads at ~5 min.
- **Network policy (my call, builder check):** no Ray, so the ROADMAP's Ray caveat does not
  apply; the vLLM inter-node ports (RPC, torch distributed, NCCL) are plain and unauthenticated
  anyway, so the namespace gets a `NetworkPolicy` that allows ingress only from the namespace
  itself and from the load/monitoring namespaces. On EKS with the VPC CNI the policy is a no-op
  unless the network-policy agent is enabled: the builder checks `aws-node`'s
  `enable-network-policy`; if it is off, the study **does not** enable it (cluster-wide change
  on a shared lab cluster) and the README records the policy as present-but-unenforced, with
  the lab's security groups as the actual boundary. EFA/SRD traffic bypasses the kernel and no
  policy covers it.

## Spike (docs/spikes/2026-11-xx-efa-eks/): prove EFA before paying for g7e

On 2× g6.8xlarge spot. Timebox one working day. Outputs: a README with the measured numbers,
the node group config, the image choice, and the manifests the study reuses. Steps:

1. Node group up; `kubectl describe node` shows `vpc.amazonaws.com/efa: 1`; EFA device plugin
   running.
2. **nccl-tests** (all_reduce, alltoall, sendrecv, 2 nodes × 1 GPU) in a pod requesting
   `vpc.amazonaws.com/efa: 1` plus hugepages if the plugin needs them, with `NCCL_DEBUG=INFO`:
   the log must show `NET/OFI` (aws-ofi-nccl) and the bus bandwidth; then the same with
   `NCCL_NET=Socket` as the **TCP control**. Both numbers go in the README and later in the
   study README.
3. The vLLM image: try the AWS DLC `vllm:0.29.0-...-ec2` first (check `libnccl-net-ofi` /
   `aws-ofi-nccl` inside); otherwise build `vllm/vllm-openai:v0.29.0` + EFA installer
   (`--skip-kmod`) + aws-ofi-nccl and push to ECR (`vllm-efa:0.29.0`). Record which one.
4. A small model (Qwen3-8B-FP8, already cached in the lab) in each cross-node layout (DP2+EP
   is meaningless for a dense model, so: TP2 via `--nnodes 2`, PP2 via `--nnodes 2`, DP2 via
   `--data-parallel-*`) just to prove the multi-node start, the readiness logic and the NCCL
   transport line in vLLM's log. Not a benchmark.
5. Teardown: node group deleted, spot requests gone, verified in EC2.

What the spike does **not** need: gpt-oss-120b (does not fit an L4), Nsight (not Blackwell).

## Layouts under test (the presets)

All on the same two nodes, the same model, the same vLLM settings otherwise. `tp`, `dp`, `pp`,
`ep` are the pack parameters; the apply script derives the manifest from them.

| # | Layout | tp | dp | pp | ep | How it is run | Cross-node traffic |
|---|---|---|---|---|---|---|---|
| L1 | one GPU | 1 | 1 | 1 | false | one pod on node A, node B idle | none |
| L2 | two independent replicas | 1 | 2 | 1 | false, `replicas` | two pods, one per node, two AIPerf clients with half the users each (study 49 pattern) | none |
| L3 | vLLM DP2, experts tensor-parallel | 1 | 2 | 1 | false, `vllm_dp` | head + headless worker, internal LB | expert all-reduce every layer |
| L4 | vLLM DP2 + expert parallel | 1 | 2 | 1 | true | head + headless worker | expert all-to-all (allgather/reducescatter) |
| L5 | TP2 across nodes | 2 | 1 | 1 | false | `--nnodes 2`, mp executor | all-reduce per layer |
| L6 | TP2 + expert parallel | 2 | 1 | 1 | true | `--nnodes 2` | all-reduce + all-to-all |
| L7 | PP2 across nodes | 1 | 1 | 2 | false | `--nnodes 2` | activations between stages only |

L2 and L3 would share the same vLLM pack values. **Decision (my call, builder check):** L2 is
modeled with the **Kubernetes pack's workload `replicas`** parameter (vLLM dp=1, replicas=2),
L3 with vLLM dp=2 and replicas=1; the parameter constraint becomes
`replicas * tp * dp * pp <= 2`. In the optimize step dp=2 always means vLLM DP (hybrid LB), and
the optimizer may pick replicas=2 with dp=1, which is L2. If the Kubernetes pack 1.9.0 has no
usable `replicas` on the component type we use, the builder raises it as a pack TODO and L2
runs as a manual preset recorded in the README, as the TCP controls below. L3 and L4 use
`--data-parallel-hybrid-lb` with two pinned clients (see Load). An Akamas workflow task cannot
know which step it is in, so nothing is "written per step by the workflow".

**TCP controls (anti-goal 1, my call):** the two most network-bound cross-node layouts (L5 and
L4, by the nccl-tests numbers) are repeated once with `NCCL_NET=Socket`, **by hand during the
smoke** (same Ramp + Hold scripts, results in `results/tcp-controls/` and in the README's
table), not as Akamas experiments: no pack parameter exists for the NCCL transport, and adding
one is pack work outside this study (ROADMAP section E). Seven Akamas presets, two manual
controls.

**Transport proof:** for every cross-node experiment the apply script greps vLLM's NCCL init
lines for `NET/OFI` (or `NET/Socket` for the controls) and fails the trial if the transport is
not the intended one. The EFA byte counters in telemetry (below) are the second witness.

## Load: ramp of users, as study 48, re-scaled

- Per-user load unchanged from studies 47/48: ShareGPT-style multi-turn chat, 10 turns, 600-token
  user turns, 200-token answers (`ignore_eos`), one turn every 64 s, history re-sent. The
  `64 × request_success_rate` term therefore still reads as "users"; no re-derivation needed.
- **Reasoning (my call, builder check 2026-11-09):** gpt-oss-120b is a reasoning model. vLLM
  runs **without** `--reasoning-parser`, so every generated token is streamed as `content`:
  client TTFT is the first visible byte, AIPerf counts all 200 `ignore_eos` tokens, and the
  history re-sent at every turn grows by 200 tokens exactly as in study 48. The request carries
  `reasoning_effort: low` through `chat_template_kwargs` (AIPerf `--extra-inputs`, or vLLM's
  default chat-template kwargs if AIPerf cannot nest JSON; builder verifies) so the prompt looks
  like a real deployment. The README states that a production deployment with the parser on
  would show the user a later first `content` token; this study measures the engine, not the
  parser.
- **Prefix caching (my call):** **on**, as study 48 (the same load regime: the ceiling is where
  the live histories stop fitting the cache). To keep L3/L4 comparable with L2, the vLLM-DP
  layouts run with `--data-parallel-hybrid-lb` (each node its own API server queueing to its
  local engine, no `--headless`) and the **same two pinned clients as L2**, so every layout
  with two engines sees the same prefix-hit pattern; the only difference left between L2 and
  L3/L4 is the expert synchronization across nodes. Per-engine prefix hit rates are in telemetry
  and in the README.
- **Levels:** geometric, because the ceiling is unknown and probably in the hundreds, with a
  floor so that every layout yields a number: `32 64 128 192 256 384 512 768` users, each level
  960 s + 60 s grace, judged on requests started ≥ 600 s in (conversations at full size), stop
  at the first failing level; pass criteria as study 48 (client TTFT p95 ≤ 500 ms, errors < 1 %,
  ≥ 95 % completed, ≥ 95 % sent). The Hold reruns the last passing level and is the only scored
  task (trim `[620s, 90s]`). A layout failing 32 scores nothing and the README says so.
- **Ramp cost control:** on the optimize step (below) the ramp starts from the layout's best
  known level minus one step, read from a per-layout file the Ramp maintains; if that level
  fails, the ramp **steps down** until a level passes or the floor fails (my call: a bad trial
  costs at most the descent, and still scores). Presets always ramp from 32.
- **`max_model_len` 16384** (builder's deviation, accepted): 10 × 800 tokens plus the harmony
  template's system/developer/channel tokens and `max_tokens` 200 do not fit 8192, and a 400 on
  the tenth turn would fail levels on the error rule, not on performance. Study 48 used 16384.
- **Refinement:** the step between 256 and 768 is coarse (128-256 users). If two layouts pass
  the same level, the README compares their Hold TTFT, as study 48 does; a finer level set is a
  follow-up, not this study.
- With L2, L3 and L4 the users are split evenly over the two pinned clients. Aggregation
  (builder's deviation, accepted: study 49 only printed per-role RESULT lines): the Ramp judges
  a level on the **max** of the two clients' TTFT p95 and the **sum** of their sent/completed
  requests; the Hold pushes the same aggregate to Akamas.

## Goal, constraints, steps

```
maximize  64 * vllm.request_success_rate - vllm.time_to_first_token_p95 / 1000     # users held
subject to  vllm.time_to_first_token_p95 <= 600 ms   (server-side safety net, as study 48)
            vllm.inter_token_latency_p95  <= 200 ms   (see below)
            every pod on both nodes passed the effective-config and transport checks
```

- **Presets:** L1…L7 (the TCP controls are manual, see above), vLLM otherwise at the study
  baseline: `gpu_memory_utilization 0.90`, `max_num_seqs 512`, `max_num_batched_tokens 8192`,
  `kv_cache_dtype auto`, `moe_backend auto`, `max_model_len 16384`, prefix caching on. The
  baseline values are checked in the smoke on L1 before the study starts.
- **ITL safety net at 200 ms (my call):** study 48's 120 ms was "not degraded" for an 8B dense
  model; on cross-node layouts the ITL *is* the finding (36 layers of all-reduce per step), and
  a hard 120 ms would turn a layout's user count into CONSTRAINTS_VIOLATED. 200 ms is still a
  chat-acceptable floor (~5 tokens/s); ITL p95 per level goes in the README table regardless.
- **Optimize step (optional, user decides on cost):** 8 experiments over `max_num_seqs`
  [128, 1024], `max_num_batched_tokens` [2048, 32768], `gpu_memory_utilization` [0.80, 0.92],
  `kv_cache_dtype` {auto, fp8}, `moe_backend` {auto, triton, marlin, flashinfer_cutlass},
  **with tp/dp/pp/ep searched too**, bounded by `tp * dp * pp <= 2`, and seeded by the nine
  presets. Ray-free, so every combination is runnable. `tp == 2 || dp == 2 || pp == 2` is
  implied by the two-node layouts; `tp*dp*pp == 1` is L1.
- **Parameter constraints:** `replicas * tp * dp * pp <= 2`; `ep == "true"` only if
  `tp * dp >= 2`.
- **Windowing:** trim `[620s, 90s]` on the Hold task.
- **Failures that must not score:** AIPerf killed or no RESULT line → Ramp exit 6; no level
  passes → Hold exit 7; effective-config mismatch → exit 8; wrong NCCL transport → exit 9;
  vLLM crash loop → fail-fast within 7 minutes (study 13's guard).

## Telemetry (Prometheus, as studies 48/49)

- **vLLM** per pod (ServiceMonitor), aggregated into the `vllm` component (sum of rates, max
  of p95s over pods, as study 49's neighbours aggregation) plus `vllm:cache_config_info` per pod.
- **Client-side TTFT/ITL/success rate** from AIPerf's RESULT, pushed as in study 48.
- **DCGM** per GPU, both nodes, one `GPU` component each (`gpu_a`, `gpu_b`; house rule: one
  component per physical GPU): SM activity, tensor activity, DRAM bandwidth, PCIe tx/rx, power,
  memory used.
- **Network:** ENA bytes per node from node_exporter (`node_network_*_bytes_total`), and
  **EFA bytes per node** from the RDMA hardware counters
  (`/sys/class/infiniband/<efa>/ports/1/hw_counters/{tx_bytes,rx_bytes}`), which bypass the
  kernel and so never appear in node_exporter. A tiny exporter DaemonSet on the GPU nodes, in
  the style of study 49's slice-roles exporter, exposes them; `node_a`/`node_b` components of
  a Kubernetes-pack type carry them (or a custom metric on the GPU components if the pack has
  no fitting one: the builder checks the pack, and notes a TODO if neither fits).
- **Kubernetes Container** for the vLLM pods (CPU throttling guard, as study 49).

## Nsight Perf SDK for AI (timeboxed, off the critical path)

- **Where:** a DaemonSet on the GPU nodes, image built from the SDK repo on Ubuntu 24.04
  (`cmake` build; proprietary `nvidia-nsight-perf` binary downloaded at build), `hostPID: true`,
  `CAP_PERFMON`, GPU access, `hostPath /run/nvidia/nsight-perf` shared with the vLLM pods
  (global service scope). The vLLM pod is a PID-namespace descendant of the host, which is
  what the service needs to correlate processes. CUDA 13.5 forward-compat user-mode driver
  inside the Nsight image only; the vLLM image stays untouched.
- **When:** one hour during the smoke on node A, after the vLLM checks pass. Success = a
  `nvtrace_snapshot_consumer --duration-ms 60000` during a short load produces a Trace Stream
  Database that `nvtrace_perfetto_trace_exporter` turns into a `.perfetto-trace.zip` showing
  CUDA API calls and PM counters for the vLLM process. If it does not get there in the hour,
  the study runs without it and the README records what failed (driver, container, permission).
- **In the study, if it works:** the Hold task triggers a 60 s snapshot on both nodes in the
  middle of the scored window (`kubectl exec` into the DaemonSet pod), traces land on a PVC
  named by experiment, exported to Perfetto after the study and summarized in the report
  (kernel/API time split, SM activity during cross-node waits). Continuous tracing is not
  used: 20 kHz counters over hours would be gigabytes and the overhead claim covers only the
  API tracing.
- **Known limits to state in the README:** no per-kernel GPU timings and no power with the
  580 driver; Beta Preview; AL2023 host under an Ubuntu 24.04 container.

## Workflow and scripts (k8s/)

Study 49's shape, extended to two nodes: `apply_config.sh` reads `params.env`, picks the layout
from tp/dp/pp/ep/DP_MODE, renders the manifests from templates with `lib_render.sh` (the fixed
one: substitution only of the line holding the token alone, pre-apply check of the rendered
args), applies: head StatefulSet/pod on node A, worker on node B (nodeSelector by a
`study50/role` label set at `--up`), per-role Services, `NetworkPolicy`; waits for readiness of
the head's `/health`, then **verifies on every pod** the args (`kubectl get pod -o jsonpath
args`), `vllm:cache_config_info`, and the NCCL transport line; dumps logs. `run_ramp.sh` and
`run_hold.sh` from study 48, with the two-client aggregation from study 49 and the Nsight
snapshot hook. Tests as study 49 (`test_render.sh`, `test_apply_config.sh`, `test_ramp.sh`),
run locally before any toolbox sync; toolbox sync only after the user's confirmation.

Smoke checks before the study starts (each must pass, each prints evidence):

1. both nodes Ready, EFA resource visible, device plugins up, DCGM scraping both GPUs;
2. nccl-tests on the g7e pair over EFA and over Socket (numbers to the README);
3. L1 start + effective-config check + 5-minute load at 64 users;
4. L4 and L5 start, transport line `NET/OFI`, effective-config check on both pods, short load;
5. L2 two-client aggregation produces one RESULT;
6. Prometheus has vLLM, DCGM, ENA and EFA series with the expected labels, and the Akamas
   telemetry instance pulls them (a baseline scored with non-empty KPIs);
7. Nsight, one hour, non-blocking.

## Budget (user decides the scope)

Pair of g7e.8xlarge on-demand 10.54 USD/h + load node ~0.45 → ~11 USD/h. A full ramp is at most
7 levels × 17 min ≈ 2 h plus a 17-min Hold and ~5 min of loads: ~2.4 h worst case, ~1.7 h
typical (stop at first failure).

| Scope | Experiments | Hours | USD |
|---|---|---|---|
| Spike (2× g6.8xlarge spot, ~6 h) | – | 6 | ~15 |
| Smoke on g7e (checks 1-7) + 2 manual TCP controls (~2 h each, by hand) | – | ~7 | ~75 |
| Presets (L1…L7, Akamas) | 7 | ~13 | ~145 |
| + optimize step (ramp from best level − 1) | +8 | +8 | +90 |
| **Total, presets only** | 7 | ~26 | **~235** |
| **Total with optimize** | 15 | ~34 | **~325** |

Knobs: drop L6 and/or L3 (−2 experiments, −40 USD); drop the TCP controls (−45 USD, loses
anti-goal 1); shorten levels to 600 s with steady-from 300 s (histories at ~half size; about
−40 % everywhere, but a different load than study 48). The earlier "~90 USD" figure quoted in
chat assumed a 6-hour study and is superseded by this table.

**Sequencing:** the study pair fills the on-demand G quota, so it cannot overlap study 49's
pending rerun (one g7e.4xlarge). The spike uses the separate spot quota and **can** overlap it.

**Fallback if no AZ has two g7e.8xlarge (user decides):** g7e.8xlarge is the only g7e size
with EFA. Options: wait and re-probe (hours to days, as studies 46/49 saw); or 2× g6e.8xlarge
(L40S 48 GB, EFA 25 Gbps, 4.53 USD/h) with `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (29 GiB,
fits one card, known from studies 15/16; same layouts, no Nsight since Ada is not supported).

## Teardown and deliverables

- Teardown (user: "tira giù tutto"): both GPU node groups to 0 and the spot spike group
  **deleted**; AlwaysOn removed from every ASG; load node down; the ECR image kept (no
  meaningful cost) and named in the README as a cleanup item; EC2 verified empty of G-family
  instances.
- Deliverables: `studies/50-.../README.md` (study-recap skill), `results/` (export, experiments
  CSV, DCGM + network Prometheus series, nccl-tests logs, Perfetto traces if any),
  `studies/README.md` row, `ROADMAP.md` (H5/H6 updates, section C learnings, the EFA-on-EKS
  recipe as a reusable building block), the spike README, and an **HTML artifact** with the
  analysis (users per layout, TTFT/ITL curves per level, EFA vs ENA bytes, SM activity, what
  Nsight showed), built with the dataviz skill after the recap.

## Left to the builder

- eksctl config details (subnet, volume size, `maxPodsPerNode`, labels), the capacity probe,
  the AZ choice.
- Image choice (DLC vs custom), the Dockerfile, the ECR repository name.
- Exact manifests per layout (StatefulSet vs two Deployments, Services, readiness probes,
  ports), the role labels, the `NetworkPolicy` rules.
- The EFA counter exporter (language, port, labels) and its Akamas component type.
- How `DP_MODE` reaches the apply script, and the Akamas modeling of L2 vs L3.
- AIPerf flags for `reasoning_effort`, the two-client split, the RESULT aggregation.
- Nsight image build and DaemonSet details, within the one-hour timebox.
- Level file format for "ramp from best level − 1".

## Open questions for the user (the "theirs" pile)

1. Budget and scope: ~235 USD presets only, ~325 USD with the optimize step; which knobs, if
   any, to pull (see Budget). Seven presets as listed, or drop L3/L6.
2. Fallback if no AZ offers two g7e.8xlarge at start time: wait, or g6e.8xlarge + Qwen3-30B-A3B.
3. Sequencing with the study 49 rerun: 49 first (spike of 50 in parallel on spot), or 50 first.
4. Spot for the spike only (as agreed), on-demand for the study: confirm, given that the study
   pair costs 10.5 USD/h on-demand against ~5 on spot, and that a spot interruption would wipe
   a running experiment but not the study (Akamas marks it failed and continues).

## Calls made after the builder check (2026-11-09), for the user to veto

- Reasoning parser off; `reasoning_effort: low`; history grows 200 tokens/turn as in study 48.
- Prefix caching on; L3/L4 with `--data-parallel-hybrid-lb` and two pinned clients, so that
  only the expert synchronization separates them from L2.
- L2 modeled with the Kubernetes `replicas` parameter; TCP controls run by hand in the smoke.
- Level floor 32; ITL safety net 200 ms; optimize-step ramp steps down when "best − 1" fails.
- `max_model_len` 16384; two-client aggregation = max TTFT p95, sum of requests.
- NetworkPolicy applied but not enforced cluster-wide if the CNI agent is off.
