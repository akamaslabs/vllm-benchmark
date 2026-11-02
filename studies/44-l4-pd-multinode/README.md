# 44-L4-PD-Multinode

**Status:** DONE — tappa A 2026-10-29 (manual runs A1-A6 + KV-transfer tuning), tappa B
2026-10-29/30 (B1, B2, B4, B5 from the toolbox; B3 skipped), tappa C 2026-10-30 (Akamas study
`44-C-L40S-A10G`, stopped with `akamas finish` after 7 completed experiments to make room for study
45). All study-44 node groups at 0. Tracker of the exploration:
<https://claude.ai/artifact/71LTADBPt2k3eiDwpUQ9GU>.
**Dates:** 2026-10-29 – 2026-10-30

## Objective

**What does prefill/decode disaggregation cost when prefill and decode run on different
nodes, and can different GPUs per role beat aggregated serving in tokens/s per $/h?**

Studies 18-29 ran P/D on ONE g6.12xlarge (4x L4, no GPU P2P, KV through host memory). The
best configuration stayed aggregated: study 27, P0D2 fp8 1980.22 against P1D1 fp8 1754.57
tokens/s per GPU. Real P/D deployments put the roles on different machines, often with
different GPUs (compute-heavy for prefill, bandwidth-heavy for decode). This study explores
that in cheap steps:

| Tappa | What | How |
|---|---|---|
| **A** | Same GPU on two nodes: does NIXL work over TCP between pods on different nodes, and what does the network add against study 27's single node? | manual runs: 2x g6.4xlarge, then 2x g6.xlarge |
| B | Different GPUs per role: prefill L40S (g6e.2xlarge), decode A10G (g5.2xlarge), plus controls | manual runs |
| C | Akamas study on the fixed B pair: topology (P/D or aggregated on both GPUs) and per-GPU flags, goal tokens/s per $/h (choosing the GPU type moved to study 45) | Akamas 4.1 |

**Tappa A questions, each with a number to measure:**

1. Real node-to-node bandwidth, burst and sustained (iperf3, 10 min): AWS lists g6.xlarge at
   2.5 Gbps baseline / 10 burst and g6.4xlarge at 10 / 25.
2. Does P1D1 work across nodes (KV pulled by the decode from the prefill pod's IP), and which
   UCX transport carries the KV (`UCX_PROTO_INFO=y`)?
3. KV transfer time per request and TTFT p95 / ITL p95 against study 27's P1D1 fp8 at equal
   load. Arithmetic: Qwen3-8B KV is 144 KiB/token in bf16, so a 4096-token prompt in fp8 is
   ~0.30 GB: ~1 s at 2.5 Gbps, ~0.25 s at 10 Gbps. A P1D1 tops out at ~0.8-0.9 req/s
   (study 27 learning 5), i.e. ~2 Gbps of KV, below even the xlarge baseline.
4. Score recomputed like study 27 (best valid 180 s window of router tokens/s per GPU), for
   P1D1 across nodes and for the P0D1 control on the same node class.

**Prediction, stated before the runs:** P1D1 across two g6.4xlarge scores at or below study
27's 1754.57 (the KV crosses a NIC instead of host memory, ~+0.25 s per prompt on TTFT),
still under the aggregated P0D1 control. On g6.xlarge the CPU/RAM (4 vCPU, 16 GiB, and a
host-side KV buffer with `kv_buffer_device=cpu`) may break before the network does.

## Stack & versions

- **Akamas version:** none in tappe A/B (manual runs). Tappa C: Akamas 4.1 on akamas-41, study
  `44-C-L40S-A10G`, system `vLLM_Benchmark_44_L40S_A10G_PD`, workflow `44-C-L40S-A10G-Workflow`
  (files in `akamas/`, scripts copied to `/work/study44` in the toolbox).
- **Optimization packs:** none in tappe A/B. Tappa C: vLLM pack 1.14.0 (`vLLM`,
  `vLLM_PD_Topology`); the hourly cost is a constant of the fixed node pair ($3.454/h). Choosing
  the GPU type with a cost from the AWS pack was done in study 45 (AWS pack 1.5.0).
- **Workload under test:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8`, NixlConnector,
  `kv_buffer_device=cpu`, fp8 KV, FLASHINFER. Flags of study 27 exp 8 (P1D1 kv fp8) and, for
  the control, of study 27's aggregated fp8 family (exp 20, approximate: see
  `k8s/20-pd-config.yaml`).
- **Cluster / hardware:** EKS `vllm-bench`, us-east-2, both GPU nodes pinned to us-east-2b
  (`infra/nodegroups.sh`): `llm-serving-l4-pd` 2x g6.4xlarge ($1.3232/h each),
  `llm-serving-l4-pd-xl` 2x g6.xlarge ($0.8048/h each); tappe B/C `llm-serving-l40s-pd`
  g6e.2xlarge (L40S) and `llm-serving-a10g-pd` g5.2xlarge (A10G). On-demand, tagged
  `AlwaysOn=true`.
- **Load generator:** AIPerf 0.11.0, study 27's job: synthetic 4096 in / 256 out
  (`ignore_eos`), open-loop gamma ramp (smoothness 4), seed 18, same slope as study 27
  (0.02 req/s per minute; default 0 -> 1.2 req/s in 3600 s), watchdog at TTFT p95 > 20 s or
  ITL p95 > 225 ms for 120 s.
- **Telemetry:** kube-prometheus-stack, ServiceMonitor `vllm-pd-mn` (5 s scrape); router
  metrics under `model_name=qwen3-8b-router`, as in study 27. `k8s/score.py` recomputes the
  score.

## Parameters tuned

None in tappe A/B: fixed configurations, compared against each other. Tappa C tuned the
topology and the vLLM flags of each GPU on the fixed L40S + A10G pair (see `akamas/44-C-L40S-A10G.yaml`).
Runs (results in the tables under Results):

| Run | Topology | Node group | Reference |
|---|---|---|---|
| A1 | iperf3, 600 s, 4 streams | 4xl | AWS 10 / 25 Gbps |
| A2 | P1D1 fp8 across 2 nodes | 4xl | study 27 exp 8: 1754.57 |
| A3 | P0D1 fp8 on 1 node | 4xl | study 27 exps 20/21/28: 1941-1975 |
| A4 | iperf3, 600 s | xl | AWS 2.5 / 10 Gbps |
| A5 | P1D1 fp8 across 2 nodes | xl | A2 |
| A6 | P0D1 fp8 on 1 node | xl | A3 |
| B2 | P1D1 L40S prefill → A10G decode | g6e.2xlarge + g5.2xlarge | B4 |
| B4 / B5 | aggregated on one L40S / one A10G | g6e.2xlarge / g5.2xlarge | — |
| C | Akamas, 7 experiments | g6e.2xlarge + g5.2xlarge | B2, B4 |

## How to run (tappa A)

```bash
cd studies/44-l4-pd-multinode
infra/nodegroups.sh create          # once: both groups at 0 nodes
infra/nodegroups.sh up 4xl          # 2x g6.4xlarge in us-east-2b
k8s/run.sh iperf 4xl                # A1
k8s/run.sh up 4xl pd && k8s/run.sh smoke && k8s/run.sh ramp     # A2
k8s/run.sh up 4xl agg && k8s/run.sh ramp                        # A3
infra/nodegroups.sh down 4xl; infra/nodegroups.sh up xl
k8s/run.sh iperf xl; k8s/run.sh up xl pd && k8s/run.sh smoke && k8s/run.sh ramp   # A4, A5
k8s/run.sh down; infra/nodegroups.sh down all
```

Logs land in `results/`. Every result also goes to the tracker.

## Results

Score = router tokens/s; in $ terms, tokens/s per on-demand $/h (us-east-2). Logs in `results/`
(iperf3, smoke, ramps, micro-benchmarks, tappa B sequence `runB.log`/`B*.out`).

**Tappa A — same GPU (L4) on two nodes, P1D1 vs aggregated:**

| Run | Configuration | tokens/s per GPU | req/s | tokens/s per $/h |
|---|---|---|---|---|
| A2 | P/D 2x g6.4xlarge | 1626 | 0.75 | 1229 |
| A3 | aggregated 1x g6.4xlarge | 1745 | 0.39 | 1318 |
| A5 | P/D 2x g6.xlarge | 1335 | 0.52 | 1659 |
| A6 | aggregated 1x g6.xlarge | 1967 | 0.43 | 2445 |

Network: iperf3 18.7 Gbps (4 flows) / 4.96 Gbps (1 flow) between g6.4xlarge; g6.xlarge 9.9 Gbps
for ~280 s, then 2.48 Gbps baseline. The KV of a 4096-token prompt (289 MB fp8) moves over **one TCP
flow** at ~540-600 MB/s (4xl) or ~310-390 MB/s (xl): +0.4-0.9 s TTFT per request. `--block-size 64`
(+6%) and NIXL `num_threads` 16 (+12%) do not lift the single-flow cap.

**Tappa B — different GPUs (g6e.2xlarge L40S, g5.2xlarge A10G):**

| Run | Configuration | tokens/s | req/s | tokens/s per $/h |
|---|---|---|---|---|
| B2 | P/D L40S → A10G | 5344 | 1.31 | 1547 |
| B4 | aggregated 1x L40S | 6477 | 1.44 | **2889** |
| B5 | aggregated 1x A10G | 2587 | 0.59 | 2134 |

**Tappa C — Akamas on fixed L40S + A10G ($3.454/h), topology and flags tuned:** baseline P/D 1449 /
repeat 1450 (+0.08%), aggregated on both GPUs (least-in-flight router) **2633**, P/D with decode
capped at 32 seqs 1497, aggregated with L40S Triton 2586, AKAMAS exps 6-7: 2447, 1928.

## Conclusions

- **P/D disaggregation over TCP never paid off on this load (4096 in / 256 out)**, neither with
  the same GPU on two nodes (−7% per GPU on 4xlarge, −32% per $ on xlarge vs aggregated) nor with
  a compute-heavy prefill GPU (L40S → A10G: −46% per $ vs one L40S; one L40S alone delivers more
  tokens/s than the P/D pair). The KV transfer adds ~1 s TTFT at low load and is capped by AWS's
  ~5 Gbps per-flow limit, not by the NIC.
- **NIXL P/D across Kubernetes nodes works with no special networking**: pod IP as
  `VLLM_NIXL_SIDE_CHANNEL_HOST`, UCX over TCP, `kv_buffer_device=cpu`; L40S ↔ A10G pass the NIXL
  compatibility hash (GPU type is not in it) and the A10G runs an fp8 KV cache with FlashInfer.
  Cost: a host copy of the KV region (12.7 GiB on a 16 GiB g6.xlarge, 33 GiB for an L40S prefill).
- **Cheaper nodes per GPU win per $:** one L4 on g6.xlarge beat g6.4xlarge by ~85% per $/h.
- Run-to-run noise between different hosts was ~13% (A3 vs A6 for the same aggregated config),
  but only 0.08% for a repeat on the same nodes (C exps 1-2).
- Follow-up done in study 45 (optimizer chooses the GPUs): same conclusions, L40S on top.
