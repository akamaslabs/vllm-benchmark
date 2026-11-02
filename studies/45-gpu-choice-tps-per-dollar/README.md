# 45-GPU-Choice-TPS-per-Dollar

**Status:** DONE — Akamas study `45-GPU-Choice-TPS-per-Dollar-v2` ran 2026-10-30 12:25 UTC →
2026-10-31 15:05 UTC (23 experiments, then `akamas finish`; experiment 24 aborted at the stop). All six
GPU node groups scaled to 0 and no GPU instance left running (checked 15:16 UTC). Analysis page:
<https://claude.ai/artifact/AsrF1yQyPXd1RmuRPTx3nM>. `45-GPU-Choice-TPS-per-Dollar` and
`45-GPU-Choice-Workflow` (v1) were created first and never started; delete them by hand.
**Dates:** 2026-10-30 – 2026-10-31

## Objective

**Given a load and its SLA, which GPU hardware serves it at the lowest cost per token?** The
optimizer chooses, per experiment:

- **the nodes**: one or two EC2 GPU instances among g5 (A10G), g6 (L4), g6e (L40S), each
  xlarge or 2xlarge (AWS pack 1.5.0's `aws_ec2_instance_type` / `aws_ec2_instance_size` on two
  `ec2` components, `node_a` and `node_b`);
- **the topology** (`vLLM_PD_Topology`): one node aggregated (P0 D1), two nodes aggregated
  behind a least-in-flight router (P0 D2), or prefill/decode disaggregated (P1 D1, node_a
  prefill → node_b decode, NixlConnector over TCP);
- **each pod's vLLM flags** (gmu, max_num_seqs, max_num_batched_tokens, linear backend; one KV
  dtype for both).

**Goal:** maximize `pd_topology.total_token_throughput / (node_a.aws_ec2_price +
node_b.aws_ec2_price)`: router tokens/s per on-demand $/h of the nodes in use (an unused node_b
costs 0), with TTFT p95 ≤ 10 s and ITL p95 ≤ 75 ms over 150 s at the router.

**Why:** study 44 (tappe A-C) measured these configurations by hand and with a fixed-hardware
Akamas study. Tokens/s per $/h: one L40S on g6e.2xlarge aggregated 2889, one L4 on g6.xlarge
2445, one A10G on g5.2xlarge 2134, L40S + A10G aggregated 2633, P/D L40S → A10G ~1450-1550.
This study asks whether the optimizer finds the best hardware on its own, how fast, and whether
a combination nobody measured (e.g. the L40S on a g6e.xlarge, ~3480 if B4 holds) does better.
**Prediction:** one L40S aggregated wins (on g6e.xlarge if there is capacity); P/D is never
chosen as the best.

## Stack & versions

- **Akamas:** 4.1.0 (akamas-41). **Packs:** AWS **1.5.0** (GPU families added 2026-10-30,
  MR akamas/optimization-packs/aws!4, installed before the merge), vLLM **1.14.0**.
- **Workload:** `vllm/vllm-openai:v0.29.0`, `Qwen/Qwen3-8B-FP8`, FLASHINFER, NixlConnector
  (pull, `kv_buffer_device=cpu`) when P/D, max_model_len 8192, no prefix caching.
- **Cluster / hardware:** EKS `vllm-bench`, us-east-2b, six node groups (min 0, max 2):

  | Instance | Node group | GPU | vCPU / RAM | $/h |
  |---|---|---|---|---|
  | g6.xlarge | llm-serving-l4-pd-xl (study 44) | L4 | 4 / 16 GiB | 0.8048 |
  | g6.2xlarge | llm-serving-g6-2xl | L4 | 8 / 32 GiB | 0.9776 |
  | g5.xlarge | llm-serving-g5-xl | A10G | 4 / 16 GiB | 1.006 |
  | g5.2xlarge | llm-serving-a10g-pd (study 44) | A10G | 8 / 32 GiB | 1.212 |
  | g6e.xlarge | llm-serving-g6e-xl | L40S | 4 / 32 GiB | 1.861 |
  | g6e.2xlarge | llm-serving-l40s-pd (study 44) | L40S | 8 / 64 GiB | 2.24208 |

  Prices: AWS Pricing API, on-demand, us-east-2, 2026-10-29. Every ASG is tagged
  `AlwaysOn=true` (propagated), so the lab's 17:00 UTC stop Lambda does not kill experiments.
- **Load generator:** AIPerf 0.11.0, study 27's open-loop gamma ramp (0 → 2.4 req/s over
  7200 s, smoothness 4, seed 18), 4096 in / 256 out, watchdog at TTFT p95 > 20 s or ITL p95 >
  225 ms for 120 s.
- **Telemetry:** Prometheus (`Prometheus_45_GPU_Choice`): study 44 tappa C's vLLM catalog plus
  `aws_ec2_price` ← `pd_node_price_dollars_per_hour{node}`, exported by the router from the
  prices `apply_config.sh` passes it.

## How it runs

Workflow `45-GPU-Choice-Workflow-v2`, on the toolbox, from `/work/study45/45-gpu-choice-tps-per-dollar`:

1. **Write config** — FileConfigurator renders `k8s/50-config_template.yaml`.
2. **Apply config** — `k8s/apply_config.sh`: topology and instances from the rendered config;
   scales the chosen node groups up (2 nodes when node_a and node_b are the same instance) and
   the others to 0; waits up to 15 min for the nodes (no capacity → the experiment fails);
   deploys `k8s/51-serving_template.yaml` (pods `vllm-pd-mn-a`/`-b` with anti-affinity, router
   with the node prices); fails fast on a crash loop; dumps every log.
3. **RunTest** — `k8s/run_test.sh` (study 44's ramp + watchdog + guards).
4. **Scale down** (`alwaysRun`, `critical: false`) — `k8s/scale_down.sh`: removes the pods and
   job, scales all six node groups to 0. Every experiment pays a fresh node start (~5-10 min) in
   exchange for nothing running after a stop or an error. Not critical, so a scaling hiccup
   cannot fail an experiment that already has a score.

**Idle guard:** `k8s/idle_guard.sh` runs on the toolbox (`setsid nohup`, log
`/work/study45/idle_guard.log`, 96 h): every 10 min, if no serving pod and no AIPerf job exist
for 30 min while a study-45 node group is still up, it runs `scale_down.sh`. It needs no Akamas
login and survives the laptop sleeping.

The toolbox's node role (`eksctl-vllm-bench-nodegroup-akamas-NodeInstanceRole-…`) has an inline
policy `study45-scale-gpu-nodegroups` (`eks:DescribeNodegroup`, `eks:UpdateNodegroupConfig` on
`nodegroup/vllm-bench/*/*`), added 2026-10-30 for this study and removed 2026-10-31 after it
finished: re-add it before running the workflow again.

## Parameters tuned

| Parameter | Domain | Baseline |
|---|---|---|
| `pd_topology.pd_prefill_instances` / `pd_decode_instances` | 0-1 / 1-2, sum ≤ 2 | 0 / 1 |
| `node_a`, `node_b` `.aws_ec2_instance_type` | g5, g6, g6e | g6 / (g6, unused) |
| `node_a`, `node_b` `.aws_ec2_instance_size` | xlarge, 2xlarge | xlarge / (xlarge) |
| `vllm_a`, `vllm_b` `.gpu_memory_utilization` | [0.7, 0.92] | 0.9 |
| `vllm_a`, `vllm_b` `.max_num_seqs` | [8, 256] | 256 |
| `vllm_a`, `vllm_b` `.max_num_batched_tokens` | [2048, 16384] | 4655 |
| `vllm_a`, `vllm_b` `.linear_backend` | marlin, humming, triton (g5: marlin only) | humming |
| `vllm_a.kv_cache_dtype` (both pods) | auto, fp8 | fp8 |

Steps: baseline one L4 g6.xlarge aggregated (= study 44 A6), 3 presets (A10G g5.2xlarge, L4
g6.2xlarge, two L4 g6.xlarge aggregated), then 33 AKAMAS experiments (`maxFailedExperiments:
15`). Baseline and presets use only instances probed available: on 2026-10-30 10:05 UTC neither
g6e.xlarge nor g6e.2xlarge had capacity in us-east-2b, and a failed baseline puts a 4.1 study in
ERROR. The L40S stays in the optimizer's space. ~1.2-2.3 h per experiment.

## Setup & run

```bash
# laptop, AWS_PROFILE=lab
studies/45-gpu-choice-tps-per-dollar/infra/nodegroups.sh create   # done 2026-10-30
cd studies && COPYFILE_DISABLE=1 tar --no-xattrs -czf - --exclude='results/*' 45-gpu-choice-tps-per-dollar \
  | kubectl -n akamas-41 exec -i deploy/toolbox -c toolbox -- sh -c 'mkdir -p /work/study45 && tar -xzf - -C /work/study45'

# toolbox (akamas-41), done 2026-10-30
cd /work/study45/45-gpu-choice-tps-per-dollar/akamas
akamas create system system.yaml
for f in components/*.yaml; do akamas create component $f vLLM_Benchmark_45_GPU_Choice; done
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_45_GPU_Choice
akamas create workflow 45-GPU-Choice-Workflow-v2.yaml
akamas create study 45-GPU-Choice-TPS-per-Dollar-v2.yaml

akamas start study 45-GPU-Choice-TPS-per-Dollar-v2   # "Could not start" -> finish + echo y | start
akamas finish study 45-GPU-Choice-TPS-per-Dollar-v2 --workspace default
```

After a stop, check that every node group is at 0 (`bash k8s/nodes.sh status` on the toolbox,
or `infra/nodegroups.sh status`); `infra/nodegroups.sh down` forces it.

## Results

Raw data in `results/`: `export.tar.gz` (Akamas export, complete: `study.json` + 106 metric
files), `exp45.csv` (one row per experiment: hardware, topology, score, throughput, latencies at
the scoring window, outcome), and the logs of the start/stop scripts. Score = router tokens/s per
on-demand $/h of the nodes in use.

**Best configuration (experiment 19, Akamas step) vs baseline (experiment 1):**

| Parameter | Baseline | Best |
|---|---|---|
| Topology (P, D) | 0, 1 (one node) | 0, 1 (one node) |
| node_a | g6.xlarge (L4) | **g6e.2xlarge (L40S)** |
| vllm_a.kv_cache_dtype | fp8 | fp8 |
| vllm_a.linear_backend | humming | triton |
| vllm_a.gpu_memory_utilization | 0.9 | 0.80 |
| vllm_a.max_num_seqs | 256 | 96 |
| vllm_a.max_num_batched_tokens | 4655 | 15766 |
| **Score** | 2217.5 | **2768.7 (+24.9%)** |
| Throughput / cost | 1785 tokens/s at $0.80/h | 6208 tokens/s at $2.24/h, 1.27 req/s |

The scoring window of the best has TTFT p95 0.49 s and ITL p95 53.5 ms; as in study 44 B4, the
L40S saturates in a step (ITL jumps past the 75 ms limit within minutes right after the window).

**Top experiments:**

| # | Configuration | Topology | Score | Note |
|---|---|---|---|---|
| 19 | L40S 2xlarge | 1 node | 2768.7 | best |
| 13 | 2x L40S xlarge | 2 nodes aggregated | 2738.4 | reached the ramp's 2.4 req/s: lower bound |
| 15 | 2x L40S xlarge | 2 nodes aggregated | 2736.9 | |
| 5 | L40S xlarge | 1 node | 2676.0 | bf16 KV, Marlin |
| 18 | 2x L40S xlarge | 2 nodes aggregated | 2638.5 | |
| 17 | A10G xlarge | 1 node | 2543.8 | best non-L40S |
| 12 | A10G xlarge | 1 node | 2529.5 | |
| 21 | L40S xlarge + L40S 2xlarge | 2 nodes aggregated | 2466.6 | reached the ramp's top |

Others: presets L4 xlarge 2217.5 (baseline), 2x L4 xlarge 2276.7, A10G 2xlarge 2108.1, L4 2xlarge
2015.9; mixed aggregated L40S + L4 2069.9-2281.0; the one P/D the optimizer tried (L40S xlarge
prefill → L4 xlarge decode) **654.9, the worst of the study**. 4 experiments failed (6, 7, 8, 11)
because no g6e instance was available in us-east-2b; none failed for any other reason.
Estimated node cost of the study: ~$72 (duration × on-demand price, failures at half).

**Binding constraints:** ITL p95 on most single-GPU runs (L4, A10G saturate on decode); TTFT p95 on
some aggregated pairs (one slow member queues); the load generator itself (2.4 req/s) on the two
L40S pairs that ended unsaturated (exps 13, 21).

## Conclusions

- **The optimizer found the right hardware by itself.** Its first own choice (experiment 5) was
  an L40S, and from then on the top of the ranking is L40S only: one L40S on g6e.2xlarge, one on
  g6e.xlarge, or two aggregated. It confirms study 44's manual ranking (B4 2889 > A6 2445 > B5
  2134 > P/D) without being told.
- **P/D disaggregation is never worth it on this load (4096 in / 256 out) and network (TCP, one
  ~5 Gbps flow per KV transfer):** tried once, worst result. Generalizes study 44 A-C to a search
  where the optimizer was free to pick it. Stack-specific: a longer prompt, a bigger model or
  RDMA could change it (Q6's thresholds still not met here).
- **Identical GPUs scale linearly when aggregated** (2x L4 ≈ 1x L4, 2x L40S ≈ 1x L40S per $), so
  the second node buys capacity, not efficiency. **Mixed GPUs aggregated are worse than the best
  one alone.**
- **Node size matters per GPU:** xlarge beats 2xlarge per $ for L4 and A10G, while for the L40S
  the 2xlarge (8 vCPU, 64 GiB) pays its ~20% premium. The vLLM flags move the score about as much
  as the hardware (same A10G xlarge: 2306-2544; same L4 xlarge: 1619-2218).
- **Operational:** per-experiment node scale-up/down from the workflow works (IAM policy on the
  toolbox role, `alwaysRun` scale-down, idle guard); it costs ~5-10 min per experiment. g6e
  capacity in us-east-2b came and went over the weekend: plan for it (presets on available
  instances, `maxFailedExperiments` headroom). Akamas 4.1 runs RunTest even after Apply config
  failed (fixed in `run_test.sh` with a router-ready check).
- **Next:** if the question is capacity at fixed cost rather than cost per token, rerun with a
  ramp above 2.4 req/s so that L40S pairs saturate; to test P/D fairly, a long-prompt load
  (e.g. 16-32k in) on the L40S pair.
