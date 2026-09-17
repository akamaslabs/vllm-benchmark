# 15-qwen3-30b-a3b-parallelism-goodput-per-gpu

**Status:** TODO (scaffolded locally on 2026-09-17; nothing created on the Akamas instance yet)
**Dates:** Created 2026-09-17 — not started

## Objective

Find the TP/DP/expert-parallel layout, and the vLLM settings around it, that give the most
**goodput per GPU actually holding weights** when serving a MoE model that *cannot* fit on
one GPU, on a 4x L4 node with PCIe-only interconnect (no NVLink).

```
maximize  (vLLM.prefill_token_throughput + vLLM.decode_token_throughput) / vLLM.active_gpus
subject to  vLLM.time_to_first_token_p95 <= 1500 ms
            vLLM.inter_token_latency_p95 <=  300 ms
```

### Why a different model, and why this one

Studies 10-14 all served `openai/gpt-oss-20b`, which **fits on a single L4**. That made
every parallelism result ambiguous: a topology could look bad simply because splitting a
model that doesn't need splitting is pure overhead. Study 14 also confirmed, from vLLM's
own behaviour, that **MoE expert weights are always sharded across the whole TP × DP
world** — vLLM offers no way to replicate them — so `data_parallel_size` on an MoE model
is not "independent replicas" and the huge inter-GPU traffic seen there was correct
behaviour, not a bug.

That left two options, discussed on 2026-09-17: drop TP/DP and scale with independent
Kubernetes replicas, or move to a model large enough that multi-GPU is forced. The team
chose the second: it is the realistic customer situation (fixed on-prem hardware, fixed
model — Leonardo/Zucchetti-style), and the interconnect itself may be tunable. The plan is
two phases, and **this study is phase 1 (topology only)**; phase 2 is the NCCL
environment-variable tuning (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_P2P_LEVEL`,
`NCCL_MIN_NCHANNELS`, `NCCL_BUFFSIZE`), which needs new parameters on the vLLM pack first
(ROADMAP.md section E).

`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` was picked because of where its weights land
relative to a 24 GB L4 (~22 GiB usable). Measured from the HF API and `config.json` on
2026-09-17:

| variant | weights | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|---|---|---|
| `-Instruct-2507` (bf16) | 56.87 GiB | no | no | 14.2 GiB/GPU |
| **`-Instruct-2507-FP8`** | **29.03 GiB** | **no** | **14.5 GiB/GPU** | **7.3 GiB/GPU** |

The FP8 variant is the one that makes the study answerable: single-GPU is impossible (so
multi-GPU is forced, as intended) *while* 2-GPU and 4-GPU layouts are both loadable, so
`active_gpus` varies and the per-GPU objective genuinely compares "half the hardware"
against "all of it". With the bf16 variant every legal layout would use all four GPUs,
`active_gpus` would be constant, and the objective would collapse into raw goodput.
The **Instruct** (non-thinking) variant is used rather than Thinking/base-hybrid: a
chain-of-thought channel would dominate output length under ShareGPT replay.

## Stack & versions

- **Akamas version:** 3.7.x
- **Optimization packs:** vLLM **1.9.1** and GPU **1.2.0**, both `INSTALLED` — verified on
  the instance with `akamas describe optimization-pack` on 2026-09-17; Kubernetes pack
  1.8.0-dev carried over from studies 13/14 (not re-verified).
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` — MoE, 128 experts / top-8, 48 layers, 32 attention
  heads / 4 KV heads, head_dim 128, hidden 2048, `moe_intermediate_size` 768, vocab 151936,
  FP8 e4m3 weights with `weight_block_size [128, 128]`, native context 262144 (pinned to
  8192, see below). Pinned flags: `--attention-backend TRITON_ATTN`, `--max-model-len 8192`,
  `--no-enable-prefix-caching`, `--enable-mfu-metrics`.
- **Cluster / hardware:** AWS `us-east-2`, EKS cluster `vllm-bench`, node group
  `llm-serving-l4` = 1x `g6.12xlarge` (4x NVIDIA L4 24GB, PCIe, **no NVLink**). All 4 GPUs
  are requested by the pod on every trial; unused ones idle. Provisioning in `infra/`.
  **The node is shared with study 14, which was still RUNNING when this was scaffolded.**
- **Load generator:** NVIDIA AIPerf 0.11.0, ShareGPT replay via cached `inputs.json`,
  closed-loop concurrency ramp of 12 levels **24 → 1024** (×√2 apart), 300 s each,
  `--goodput time_to_first_token:1500 inter_token_latency:300`. The floor moved down from
  studies 10-14's 150: this model's KV costs 96 KiB/token (2× gpt-oss-20b) and its weights
  take 14.5 GiB of each card in the 2-GPU layouts, so a 2-GPU configuration could already
  be queueing at the old first level — which would have made every 2-GPU cell infeasible
  and answered the study's question by construction.
- **Telemetry:** Prometheus (`kube-prometheus-stack`), `duration: 30`, `stability`
  windowing on `prefill_token_throughput` (width 6). Own telemetry instance, with the two
  queries study 14 was missing (`total_token_throughput`, `request_queue_time_p95`) added,
  so all 8 KPIs are populated.

## Parameters tuned

Study 14's 14 parameters and domains (`tensor_parallel_size` and `data_parallel_size` both
fully open), **plus `disable_custom_all_reduce`** — 15 in total. That one was added on
2026-09-17 after a review: it is the only interconnect knob vLLM pack 1.9.1 already models,
it was named explicitly in the thread that scoped this study, and `apply_config.sh` already
carried it in its boolean-rewrite list while nothing rendered it.

| Parameter | Domain | Baseline |
|---|---|---|
| `vLLM.tensor_parallel_size` | [1, 4] (`!= 3`) | 4 |
| `vLLM.data_parallel_size` | [1, 4] | 1 |
| `vLLM.enable_expert_parallel` | true / false | true |
| `vLLM.kv_cache_dtype` | auto / fp8 / fp8_e4m3 | vLLM default (`auto`) |
| `vLLM.gpu_memory_utilization` | [0.8, 0.9] | 0.85 |
| `vLLM.max_num_seqs` | [16, 1024] | vLLM default |
| `vLLM.max_num_batched_tokens` | [256, 8192] | vLLM default |
| `vLLM.max_cudagraph_capture_size` | [1, 1024] | vLLM default |
| `vLLM.optimization_level` | [0, 3] | vLLM default |
| `vLLM.block_size` | 16 … 128 (8 ordinals) | vLLM default |
| `vLLM.performance_mode` | balanced / interactivity / throughput | vLLM default |
| `vLLM.scheduling_policy` | fcfs / priority | vLLM default |
| `vLLM.enforce_eager` | true / false | vLLM default |
| `vLLM.async_scheduling` | true / false | vLLM default |
| `vLLM.disable_custom_all_reduce` | true / false | vLLM default (`false`) |

### Constraints (6)

1. `max_num_batched_tokens >= max_num_seqs` — vLLM's own scheduler rule.
2. **Sampler-warmup guard**, re-derived for this vocabulary:
   `gpu_memory_utilization × 22.03 + max_num_seqs × 0.00283 <= 21.63`
   (5 fp32 copies of 151936 logits = 0.00283 GiB/sequence, against gpt-oss-20b's 0.00375).
3. `tensor_parallel_size != 3` — 32 attention heads / 4 KV heads.
4. `tensor_parallel_size × data_parallel_size <= 4`.
5. **Memory fit, MoE-aware**: `gpu_memory_utilization × 22.5 − 29.03/(TP×DP) >= 4`, written
   multiplied out. Because the experts (~93% of the weights) are sharded over the whole
   TP × DP world, the per-GPU footprint follows the *product*. Side effect: TP×DP = 1 is
   never satisfiable, so **single-GPU layouts are excluded automatically** — the multi-GPU
   requirement is enforced by physics, not by a hand-written rule. TP×DP = 2 needs
   `gpu_memory_utilization >= ~0.845`.
6. **Expert parallelism required at 4 ranks**: `moe_intermediate_size` 768 over 4 ranks is
   192, not a multiple of the checkpoint's 128-wide quantization block, so a TP×DP = 4
   layout without expert parallelism is expected to fail at weight creation. Expert
   parallelism keeps each expert whole on one rank. **This is an inference, not a
   measurement** — `smoke_test.sh`'s `tp4-noep` configuration exists to confirm it; if it
   starts fine, drop this constraint.

### Initial design: 10 presets instead of Sobol

`numberOfInitExperiments: 0`, because any value > 0 makes the campaign service re-run its
own Sobol bootstrap on top of the presets. Study 13's Sobol head aliased `kv_cache_dtype`
with `enable_expert_parallel` perfectly (ROADMAP.md section C); the presets here fill all
four cells of that 2×2 and cover all five reachable topologies:

| | TP4/DP1 | TP2/DP2 | TP1/DP4 | TP2/DP1 | TP1/DP2 |
|---|---|---|---|---|---|
| **4 GPUs** | S1, S7, S10 | S2 | S3 | — | — |
| **2 GPUs** | — | — | — | S4, S6, S8 | S5, S9 |

`kv_cache_dtype` × `enable_expert_parallel`: (auto, on) S1-S5 · (auto, off) S6 ·
(fp8*, on) S7/S8/S10 · (fp8*, off) S9. Everything outside the three dimensions under test
is held fixed across S1-S9 (`gpu_memory_utilization` 0.88, `max_num_seqs` 768, …); S10 is
the one space-filling point. `disable_custom_all_reduce` is pinned to `false` (vLLM's own
default) in **all ten** presets, so the preset phase stays a clean topology comparison at
fixed interconnect behaviour — the optimizer is what explores that flag afterwards. Caveat
when reading its effect: vLLM disables the custom all-reduce kernel by itself when the
topology cannot support it (no GPU P2P, PCIe-only peers), which is plausible on this node
at TP4 and less so at TP2, so a null effect must be checked against the
"Custom allreduce is disabled" line in the Apply-config log before being read as
"measured no difference".

**Budget:** ≈ 85 min per experiment (30 min worst-case rollout + 60 min of ramp);
1 baseline + 10 presets + 100 optimize ≈ 6.5 days of node time.

## Open items before this study can start

1. **Study 14 owns the node.** It must finish or be stopped first — this also blocks the
   smoke test.
2. **Run `k8s/smoke_test.sh`.** It checks the three assumptions this study is built on:
   FP8 block-quant weights stay 8-bit on Ada (else nothing fits), expert parallelism is
   required at 4 ranks (constraint 6), and the sampler-warmup constant is right.
3. **Place the SSH key** at
   `/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/id_rsa`
   on the toolbox host — never committed (`.gitignore`).
4. **Nothing has been created on the Akamas instance yet**, and no YAML here has been
   validated with `akamas create`. It parses, every parameter/metric resolves against the
   installed vLLM 1.9.1 and GPU 1.2.0 packs, every domain is a subset of the pack's, and
   every preset satisfies all six constraints — all checked offline.
5. **Re-calibrate the ramp after the baseline**, from per-level `Waiting` and TTFT p95, as
   study 10 did for gpt-oss.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
