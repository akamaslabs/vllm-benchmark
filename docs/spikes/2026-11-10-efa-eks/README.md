# EFA on EKS spike (study 50 preparation), 2026-11-10

Throwaway check that multi-node vLLM can talk over EFA on this cluster before paying for two
g7e.8xlarge: does a pod get an EFA device, does NCCL pick it, do the multi-node vLLM layouts of
study 50 start and serve, and how far is TCP behind. No Akamas study, run by hand. Times are UTC.
Design: `docs/superpowers/specs/2026-11-09-study-50-g7e-efa-moe-parallelism-design.md`.

**Verdict: go.** The AWS vLLM image works over EFA as it is, every layout of study 50 starts across
two nodes, and EFA is measurably ahead of TCP on the layouts that talk per token.

## Setup

- **Nodes:** 2x g6.8xlarge (1x L4 23 GB, 32 vCPU, 128 GiB, 25 Gbit/s, EFA), **on-demand**, us-east-2a,
  one cluster placement group, node group `llm-serving-g6-8xl-efa` created with eksctl
  (`nodegroup-g6-8xl-efa.yaml`). Up 19:28 to 20:25, about 4 USD/h the pair.
  The spot variant (`nodegroup-g6-8xl-efa-spot.yaml`) was tried first: no AZ of us-east-2 had spot
  capacity for a g6, g5 or g4dn 8xlarge that evening.
- **Node allocatable:** cpu 31850m, memory ~116.8 GiB, `nvidia.com/gpu: 1`,
  `vpc.amazonaws.com/efa: 1`, `hugepages-2Mi: 7042Mi`. AL2023, kernel 6.18, containerd 2.3.7,
  NVIDIA driver 580.178.04.
- **Image:** `763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.29.0-gpu-py312-cu130-ubuntu24.04-ec2`
  (AWS Deep Learning Container; pull 2m46s, 8.9 GB). It ships `/opt/amazon/efa` (libfabric 2.4),
  aws-ofi-nccl 1.18.0, vLLM 0.29.0, torch 2.13.0+cu130, NCCL 2.29.7. **No custom image is needed.**
- **Pods:** `spike-pods.yaml`, one sleeping pod per node; every test is started with `kubectl exec`.
  What a pod needs for EFA: `vpc.amazonaws.com/efa: 1`, `hugepages-2Mi: 5120Mi`, capability
  `IPC_LOCK`, a memory-backed `/dev/shm`, plus `nvidia.com/gpu: 1`.
- **Model:** `openai/gpt-oss-20b` (MoE, 32 experts, fits one L4). Deviations from the design: the
  model (the design named Qwen3-8B, which is dense and cannot exercise expert parallelism) and the
  NCCL benchmark (`nccl_bench.py` over torch.distributed, the stack vLLM uses, instead of
  nccl-tests, which the image does not ship).

## What eksctl `efaEnabled: true` does to the cluster

Besides the launch template with the EFA interface, the placement group and the security-group
rules, eksctl touched two shared DaemonSets in `kube-system`:

- It created `aws-efa-k8s-device-plugin` (v0.5.4). Its node affinity lists 121 instance types and
  **no g7e**, and it does not tolerate `nvidia.com/gpu=present:NoSchedule`. On the spike the taint
  was removed from the two nodes by hand. Study 50 ships its own copy of the DaemonSet instead.
- It **replaced `nvidia-device-plugin-daemonset`**: v0.18.0 with the cluster's node-role exclusions
  became v0.19.0 with no affinity. Two pods now fail on the Akamas and system nodes, and the DRA
  node groups would get the classic plugin if started. The old template is DaemonSet revision 6:
  `kubectl -n kube-system rollout undo ds/nvidia-device-plugin-daemonset --to-revision=6`
  (not run: a change to a shared DaemonSet is the cluster owner's call).
  Next time: `eksctl create nodegroup --install-nvidia-plugin=false`.

## NCCL: EFA against TCP

`nccl_bench.py`, two ranks, one per node. Latency per call, and the bandwidth in Gbit/s for the
large messages.

| Test | EFA | TCP default | TCP 16 sockets |
|---|---|---|---|
| all_reduce 4 KB | 0.069 ms | 0.140 ms | 0.158 ms |
| all_reduce 64 KB | 0.088 ms | 0.283 ms | 0.281 ms |
| all_reduce 1 MB | 0.425 ms (19.8) | 0.923 ms (9.1) | 0.593 ms (14.2) |
| all_reduce 16 MB | 5.37 ms (25.0) | 10.09 ms (13.3) | 5.42 ms (24.8) |
| all_reduce 128 MB | 42.96 ms (25.0) | 77.39 ms (13.9) | 43.35 ms (24.8) |
| all_reduce 1 GB | 343.7 ms (25.0) | 613.9 ms (14.0) | 346.6 ms (24.8) |
| all_gather 64 KB | 0.101 ms | 0.186 ms | 0.181 ms |
| all_gather 16 MB | 5.37 ms (25.0) | 10.61 ms (12.7) | 5.42 ms (24.8) |
| all_gather 256 MB | 90.15 ms (23.8) | 159.86 ms (13.4) | 91.14 ms (23.6) |

- NCCL picks EFA with no setting: the log says `NET/OFI Selected provider is efa` and
  `Using network Libfabric`. `NCCL_NET=Socket` forces TCP (`Using network Socket`).
- "TCP 16 sockets" is `NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4`.
- **Bulk bandwidth is the same** once TCP uses many sockets: both fill the 25 Gbit/s NIC. The EFA
  gain is on **small messages, 2 to 3 times lower latency**, which is what a per-layer exchange pays.

## vLLM layouts across the two nodes

`run_layout.sh <layout> [TAG] [ENV=VALUE]` starts the layout, waits for `/health`, runs `load.py`
(completions, ~450-word prompt, 200 output tokens, `ignore_eos`) at concurrency 1 and 32, and
prints the NCCL transport of both nodes. Not a benchmark: it only makes layouts and transports
comparable on this pair.

| Layout | Ready | conc 1: tokens/s, ITL p50 | conc 32: tokens/s, TTFT p50 / p95, ITL p50 | KV cache tokens |
|---|---|---|---|---|
| single L4 | n/a | 57, 16.6 ms | 544, 873 / 2334 ms, 48.0 ms | 88,920 |
| TP2, EFA | ~130 s | 74, 12.7 ms | 647, 843 / 2037 ms, 36.7 ms | 612,059 |
| TP2, TCP | ~130 s | 57, 16.9 ms | 471, 1507 / 3770 ms, 50.8 ms | 612,059 |
| TP2 + EP, EFA | 132 s | 71, 13.1 ms | 645, 866 / 2046 ms, 37.7 ms | ~624,000 |
| PP2, EFA | 113 s | 55, 16.6 ms | 604, 768 / 1514 ms, 45.5 ms | ~462,000 |
| DP2, internal LB, EFA | 156 s | 56, 16.7 ms | 703, 1149 / 1834 ms, 35.9 ms | not captured |
| DP2 + EP, internal LB, EFA | 156 s | 54, 17.4 ms | 703, 1181 / 1796 ms, 35.7 ms | ~220,000 per engine |
| DP2 + EP, external LB, EFA | 135 s | 56 per node, 17.4 ms | 347 + 347, 1070 to 1244 / 1641 to 1999 ms, 36.8 ms | 263,504 per engine |
| DP2 + EP, external LB, TCP | 144 s | 48 per node, 20.0 ms | 290 + 290, 1479 / 2170 ms, 44.2 ms | 263,504 per engine |

The external-LB rows split the 32 workers 16 + 16 over the two API servers.

- **TP2 over TCP is slower than a single GPU** at concurrency 32 (471 against 544 tokens/s) and its
  TTFT p95 nearly doubles. Over EFA it is 19 % ahead of the single GPU and 37 % ahead of TCP.
- **DP2 + EP over TCP loses 16 %** of the throughput and 20 % on inter-token latency against EFA.
  So expert parallelism really exchanges tokens between the nodes on every step.
- TP and EP free GPU memory for the KV cache (weights are split), 3 to 7 times the single-GPU cache
  on this small GPU.
- Expert parallelism is on when asked: the log says `Expert parallelism is enabled. Expert placement
  strategy: linear. Local/global number of experts: 16/32`, with experts 0-15 on rank 0 and 16-31 on
  rank 1, MoE backend `MARLIN` (MXFP4 weights).

## How to launch vLLM 0.29 on two nodes

```
# node 0 (API server)
vllm serve MODEL ... <parallel flags> --nnodes 2 --node-rank 0 --master-addr <node 0 pod IP>
# node 1 (engine only)
vllm serve MODEL ... <parallel flags> --nnodes 2 --node-rank 1 --master-addr <node 0 pod IP> --headless
```

`<parallel flags>` is one of `--tensor-parallel-size 2`, `--pipeline-parallel-size 2`,
`--data-parallel-size 2`, each optionally with `--enable-expert-parallel`.

Data parallel has three load-balancing modes (read from `vllm/engine/arg_utils.py` and
`entrypoints/cli/serve.py`, then tried):

- **Internal** (default): the API server and the DP coordinator are on node 0, node 1 is headless.
  One endpoint, `/metrics` on node 0 carries `engine="0"` and `engine="1"`.
- **External** (`--data-parallel-external-lb`): both nodes run the same command **without**
  `--headless`, each has its own API server and serves its local engine; the DP rank comes from
  `--node-rank`. The client has to split the load. Needs an MoE model when dp > 1.
- **Hybrid** (`--data-parallel-hybrid-lb`): not usable with one GPU per node (it falls back to
  external) and rejected with `--headless`.

Study 50 uses the external mode for its DP layouts: one AIPerf client pinned to each node.

## Other findings

- **EFA byte counters exist only on the host**, under
  `/sys/class/infiniband/<device>/ports/1/hw_counters/` (`tx_bytes`, `rx_bytes`, `rdma_*`,
  `retrans_*`, `rx_drops`). node-exporter does not export them, and EFA traffic does not show in
  the ENA interface counters. Study 50 needs a small exporter on the host `/sys`.
- **AIPerf 0.11.0 and gpt-oss chat:** the model streams `delta.reasoning` chunks before any
  `delta.content`. AIPerf's `time_to_first_token` counts the first reasoning chunk (278 ms average
  in a short run), while `time_to_first_output_token` waits for the first content chunk (3.4 s).
  Study 50's client-side TTFT rule reads `time_to_first_token`.
- g7e.8xlarge (EC2 API): EFA supported, 100 Gbit/s, 32 vCPU, 256 GiB, 1x RTX PRO 6000 96 GB,
  cluster placement groups, offered in us-east-2a/2b/2c.

## Files

| File | What |
|---|---|
| `nodegroup-g6-8xl-efa.yaml` | the on-demand node group that ran the spike |
| `nodegroup-g6-8xl-efa-spot.yaml` | the spot variant (no capacity on 2026-11-10) |
| `spike-pods.yaml` | namespace, headless Service, two sleeping pods with GPU + EFA |
| `nccl_bench.py` | two-rank all_reduce / all_gather benchmark |
| `run_layout.sh`, `load.py` | start one vLLM layout on the two pods and load it |

Teardown: namespace `efa-spike` and the node group were deleted at the end of the spike. The
`kube-system/aws-efa-k8s-device-plugin` DaemonSet created by eksctl stays (0 pods without EFA nodes).
