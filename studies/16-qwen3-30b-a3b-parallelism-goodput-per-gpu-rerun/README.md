# 16-qwen3-30b-a3b-parallelism-goodput-per-gpu-rerun

**Status:** FINISHED — run 2026-09-18 → 2026-09-20, stopped manually to cap cost after 27 of 100
optimizer experiments; analysed 2026-09-21 (see [Results](#results) and `results/report.html`)
**Dates:** Created 2026-09-18, successor to `15-qwen3-30b-a3b-parallelism-goodput-per-gpu`

> Same objective, model, hardware and search space as study 15 — read that study's README
> for the full rationale of the model choice and the topology design. This folder exists
> because study 15's memory constraint was wrong and the fix cannot be applied to a study
> that has already run (on Akamas 3.7 only the `goal` is editable in place). It **reuses
> study 15's system, telemetry instance and workflow**, so study 15's experiments can be
> imported and compared; `akamas/` keeps byte-identical copies of those for the record,
> and `k8s/`/`infra/` are snapshots — the running workflow reads study 15's folder.

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

Study 14's 14 parameters and domains, **plus `disable_custom_all_reduce`** (added
2026-09-17: the only interconnect knob vLLM pack 1.9.1 already models, and
`apply_config.sh` already carried it in its boolean-rewrite list while nothing rendered it)
**plus `pipeline_parallel_size`** (added 2026-09-18) — **16 in total**.

Pipeline parallelism is in scope because of this exact hardware: it splits the 48 layers
across stages and ships only the boundary activations, once per stage, against tensor
parallelism's two all-reduces *per layer*. On a node whose 4 L4s talk over PCIe with no
NVLink that is the cheapest collective pattern available — running a topology study here
without it would leave out the layout this hardware most plausibly favours.

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
| `vLLM.pipeline_parallel_size` | [1, 4] | 1 |

### Constraints (7)

1. `max_num_batched_tokens >= max_num_seqs` — vLLM's own scheduler rule.
2. **Sampler-warmup guard**, re-derived for this vocabulary:
   `gpu_memory_utilization × 22.03 + max_num_seqs × 0.00283 <= 21.63`
   (5 fp32 copies of 151936 logits = 0.00283 GiB/sequence, against gpt-oss-20b's 0.00375).
3. `tensor_parallel_size != 3` — 32 attention heads / 4 KV heads.
4. `tensor_parallel_size × data_parallel_size × pipeline_parallel_size <= 4` — the node
   has 4 L4s and the pod requests all of them.
5. **Memory fit, MoE-aware**: `gpu_memory_utilization × 22.5 − 29.03/(TP×DP×PP) >= 4`,
   written multiplied out. The experts (~93% of the weights) are sharded over the whole
   TP × DP world, and pipeline parallelism splits the layers, so the per-GPU footprint
   follows the *triple product*. Side effect: the product 1 is never satisfiable, so
   **single-GPU layouts are excluded automatically** — the multi-GPU requirement is
   enforced by physics, not by a hand-written rule. Product 2 needs
   `gpu_memory_utilization >= ~0.845`.
6. **Expert parallelism required at 4 ranks**: `moe_intermediate_size` 768 over 4 ranks is
   192, not a multiple of the checkpoint's 128-wide quantization block, so a TP×DP = 4
   layout without expert parallelism is expected to fail at weight creation. Expert
   parallelism keeps each expert whole on one rank. **This is an inference, not a
   measurement** — `smoke_test.sh`'s `tp4-noep` configuration exists to confirm it; if it
   starts fine, drop this constraint. Note it is written on `TP × DP`, **not** on the
   triple product: pipeline parallelism splits whole layers and never the 768-wide expert
   matrices, so `TP1/DP1/PP4` holds its experts intact and is the only 4-GPU layout that
   runs with expert parallelism off.
7. **No async scheduling with pipeline parallelism**:
   `pipeline_parallel_size == 1 || async_scheduling == "false"` — vLLM does not support the
   combination ([issue #32701](https://github.com/vllm-project/vllm/issues/32701), open as
   of 2026-09-18). Without it the optimizer would keep proposing configurations that fail
   or silently degrade.

### Initial design: 15 presets instead of Sobol

`numberOfInitExperiments: 0`, because any value > 0 makes the campaign service re-run its
own Sobol bootstrap on top of the presets. Study 13's Sobol head aliased `kv_cache_dtype`
with `enable_expert_parallel` perfectly (ROADMAP.md section C); the presets here fill all
four cells of that 2×2 and cover all five reachable topologies:

| GPU attive | layouts | presets |
|---|---|---|
| **2** | TP2/DP1 · TP1/DP2 · **PP2** | S4, S6, S8 · S5, S9 · **S11** |
| **3** | TP1/DP3 · **PP3** | **S14** · **S15** |
| **4** | TP4/DP1 · TP2/DP2 · TP1/DP4 · **TP2/PP2** · **PP4** | S1, S7, S10 · S2 · S3 · **S13** · **S12** |

All **11** reachable topologies on this 4-GPU node are covered except `TP1/DP2/PP2`, left
to the optimizer. Two of them are head-to-head pairs at equal `active_gpus` — the
comparison this study exists to make: **S4 (TP2) vs S11 (PP2)** on 2 GPUs, and
**S1 (TP4) vs S12 (PP4)** on 4. `TP1/DP3` (S14) was spotted on 2026-09-18 while enumerating:
it passes every constraint, no preset covered it, and study 9 had already run
`--data-parallel-size=3` on this same node.

Across S1-S10 the `kv_cache_dtype` × `enable_expert_parallel` table has all four cells
occupied: (auto, on) S1-S5 · (auto, off) S6 · (fp8*, on) S7/S8/S10 · (fp8*, off) S9.
Everything outside the dimensions under test is held fixed (`gpu_memory_utilization` 0.88,
`max_num_seqs` 768, …); S10 is the one space-filling point. **`async_scheduling` is `false`
in every preset**: PP cannot use it, so leaving it on in the PP=1 presets would confound
every TP-vs-PP comparison with a second changed variable — the optimizer explores it
afterwards, where constraint 7 allows. `disable_custom_all_reduce` is pinned to `false` (vLLM's own
default) in **all ten** presets, so the preset phase stays a clean topology comparison at
fixed interconnect behaviour — the optimizer is what explores that flag afterwards. Caveat
when reading its effect: vLLM disables the custom all-reduce kernel by itself when the
topology cannot support it (no GPU P2P, PCIe-only peers), which is plausible on this node
at TP4 and less so at TP2, so a null effect must be checked against the
"Custom allreduce is disabled" line in the Apply-config log before being read as
"measured no difference".

**Budget:** ≈ 85 min per experiment typical (rollout + 60 min of ramp), up to ~3.3 h worst
case (the workflow allows 90 m + 105 m); 1 baseline + 15 presets + 100 optimize ≈ **6.8
days** of node time as a floor.

## Open items before this study can start

1. **Study 14 owns the node.** It must finish or be stopped first — this also blocks the
   smoke test.
2. **Run `k8s/smoke_test.sh`** (10 configurations). It checks the four assumptions this
   study is built on: FP8 block-quant weights stay 8-bit on Ada (else nothing fits), expert
   parallelism is required at 4 expert-sharding ranks (constraint 6), pipeline parallelism
   starts at all on this stack (`pp2`/`pp4` — PP combined with DP was **not** verified from
   the source; if a PP+DP trial later fails, add `pipeline_parallel_size == 1 ||
   data_parallel_size == 1`), and the sampler-warmup constant is right.
3. **Place the SSH key** at
   `/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/akamas/id_rsa`
   on the toolbox host — never committed (`.gitignore`).
4. **Nothing has been created on the Akamas instance yet**, and no YAML here has been
   validated with `akamas create`. It parses, every parameter/metric resolves against the
   installed vLLM 1.9.1 and GPU 1.2.0 packs, every domain is a subset of the pack's, and
   every preset satisfies all six constraints — all checked offline.
5. **Re-calibrate the ramp after the baseline**, from per-level `Waiting` and TTFT p95, as
   study 10 did for gpt-oss.

## Why this study exists: the OOM that stopped study 15 (2026-09-18)

Study 15 (id `4fcd1c0f-b8e6-49b3-9a61-9256c9526281`) got through these experiments:

| exp | step | topology | score (tok/s per GPU) | |
|---|---|---|---|---|
| 1 | baseline | TP4/DP1, gmu 0.85, vLLM defaults | **607.19** | reference |
| 2 | S1 | TP4/DP1, gmu 0.88, seqs 768 | **827.90** | **+36.35%** |
| 3 | S2 | TP2/DP2, gmu 0.88, seqs 768 | — | **ERROR: CUDA OOM**, 7m 14s |
| 4 | S3 | TP1/DP4, gmu 0.88, seqs 768 | — | ran ~19 min then imported |

**What confirmed the study's assumptions.** The baseline measured 7.4 GiB of weights per
GPU against the 7.26 calculated (2% off), `Available KV cache memory: 9.2 GiB` and
`GPU KV cache size: 401,776 tokens` — which works out to 96 KiB/token, exactly the
constant derived from the architecture and configured on the KV-cache exporter. The
`marlin_utils_fp8` log line confirmed weight-only FP8 on Ada: the weights stay 8-bit.

**What broke.** Experiment 3 died in `warmup_kernels` → `flashinfer_sample` →
`torch.softmax`: `tried to allocate 446.00 MiB`, `361 MiB free`, `21.67 GiB in use by the
process of which 17.54 GiB allocated by PyTorch`. 446 MiB is exactly
`max_num_seqs x vocab_size` in fp32 (768 x 151936 x 4 B).

The warmup constraint had **admitted** that point with 0.07 GiB of nominal margin, because
it assumed all 22.03 GiB are vLLM's to spend. They are not: **~2.3-2.4 GiB per GPU are
CUDA context and NCCL buffers, outside `gpu_memory_utilization`'s budget**. Measured on
experiment 4, which did start with the same settings: 22,261 MiB of 23,034 in use — 96.6%
— against a 19.4 GiB nominal budget. Every configuration on this node runs on a knife
edge; TP2/DP2 adds a third communication group (`tp` on top of `dp` and `ep`) and goes
over. See ROADMAP.md section C.

**The fix, in this study's manifest.** The lever is `max_num_seqs`, not
`gpu_memory_utilization`: the failing allocation scales linearly with the first and not at
all with the second, and lowering the second would shrink the KV cache on exactly the
2-GPU layouts this study exists to measure (their weights already take 14.5 of the ~18
usable GiB per card). So the presets drop to `max_num_seqs` 256 — softmax 148 MiB instead
of 446 — at the same `gpu_memory_utilization` 0.88, and the constraint is recalibrated on
the two measured points:

```
gpu_memory_utilization x 22.03 + max_num_seqs x 0.0006 <= 19.7
   (0.88, 768) = 19.85  rejected — the point that OOMed
   (0.88, 256) = 19.54  admitted — the new preset value
```

This study imports experiment 1 as its baseline (same score scale) and bootstraps the
other experiments that finished with a valid score, then runs the 13 presets not yet
executed plus the optimize step. Experiments 3 and 6 are not imported: they have no score.

**Experiment 6 also failed** and its log was never read — the AWS endpoint was unreachable
from the workstation at the time. It must be read before starting this study: if it is
another sampler OOM the fix above covers it, but if the 2-GPU layouts are failing on KV
cache instead, the ones at 2 GPUs need `kv_cache_dtype` pinned to `fp8_e4m3` (which halves
the 96 KiB/token cost) rather than a smaller batch.

**Caveat on the imported points:** they carry `max_num_seqs` 768, which the recalibrated
constraint no longer admits. They are valid measurements taken before the correction, but
the optimizer's best known point sits outside the feasible region — deliberate, not an
oversight.

**Still open on the 2-GPU layouts:** at TP x DP = 2 the weights take 14.5 GiB of the ~18
usable per card, so even when they start they will be KV-starved (~3 GiB of cache, roughly
30k tokens). `kv_cache_dtype: fp8_e4m3` doubles that, which is why S8 and S9 matter. If the
2-GPU cells lose, check whether they lost on collectives or simply on cache capacity
before concluding anything about interconnect.

## Results

**Run 2026-09-18 13:24 → 2026-09-20 10:37 UTC (21 h 13 min), FINISHED — stopped manually by the
team to cap the g6.12xlarge cost after 27 of the 100 planned optimizer experiments.** 43
experiments: 1 imported baseline + 3 bootstrapped from study 15, 12 presets, 27 optimizer; 40
scored, 3 failed. Full analysis in [`results/report.html`](results/report.html) (analysis of
2026-09-21; the Akamas export was incomplete on 3.7.x, so the metric series were rebuilt from
the cluster's Prometheus — the reconstruction reproduces all 40 scores to 0.00%).

| | exp | layout | tok/s per GPU | total tok/s | SLA-max concurrency | KV / GPU |
|---|---|---|---|---|---|---|
| Baseline (study 15 exp 1) | 1 | TP4 | 607 | 2 429 | 192 | 9.2 GiB |
| Best TP4 (imported, 768 seqs) | 2 | TP4 | 828 | 3 312 | 543 | 9.0 GiB |
| Only TP1/DP4 point (imported, 768 seqs) | 3 | DP4 | 1 416 | 5 662 | 768 | 6.0 GiB |
| DP3 preset S14 (untuned) | 15 | DP3 | 1 091 | 3 272 | 271 | 4.0 GiB |
| Best DP3, kv auto | 25 | DP3 | 1 814 | 5 441 | 768 | 7.4 GiB |
| **Study best** | **37** | **TP1/DP3** | **2 186 (+260%)** | **6 559** | **1 024 (ramp end)** | 7.5 GiB (fp8) |
| Best 2-GPU layout | 9 | TP2, fp8 KV | 942 | 1 884 | 192 | 2.1 GiB |
| Worst | 13 | PP4 | 445 | 1 781 | 192 | 9.8 GiB |

- **TP1/DP3 wins, consistently**: five DP3 configurations within 1.5% of exp 37 (fp8 KV,
  ~490 sequences, gmu 0.877, priority + async scheduling, EP off), every other layout's best is
  below 1 420. Tuned DP3 also has the highest absolute goodput while leaving one L4 idle. Caveat:
  26 of the 27 optimizer experiments went to DP3, so DP4 (one untuned point) and TP2/DP2 were
  never tuned.
- **The top configurations are capped by the ramp**: ten DP3 experiments are SLA-compliant at
  the last level (1 024 concurrent, ITL p95 284–291 ms); their true capacity is a lower bound and
  the ranking among them is noise (one trial each).
- **fp8 KV cache is the lever inside DP3** (+20% vs best auto): ~490 k vs ~245 k tokens of
  cache; all ten runs with ≥ 344 k tokens of KV (≥ 5.3 GiB per GPU, fp8 only) show zero
  preemptions, every run at ≤ 285 k tokens (all kv-auto, and fp8 pushed to gmu ≤ 0.84) peaks at
  10–23 preemptions/s. fp8 hurts TP4 (−4%), which is not KV-bound.
  `fp8` and `fp8_e4m3` are the same kernel (near-twin exps 31/34).
- **2-GPU layouts lose on KV capacity, not on the interconnect** (the open question above):
  TP2 keeps 1.7–2.2 GiB of KV, breaches the SLA beyond 96–192 concurrent requests, moves only
  1.5–2.8 GB/s over PCIe; fp8 KV gives it +32% (716 → 942) but it still trails DP3 by 2.3×.
  **TP1/DP2 cannot start at all** (exp 6 and 10): vLLM reports 0.13 GiB available for KV against
  0.75 needed — constraint 5's 4 GiB headroom does not hold for a DP replica, whose activation
  peak is full-width. This also corrects study 15's inference about its exp 6 (KV-fit failure,
  not sampler OOM).
- **Pipeline parallelism is the worst family** (PP2 571, PP3 557, PP4 445, TP2/PP2 578): stages
  busy (SM active 0.55–0.74) but serialised. TP2 beats PP2 by 25% and TP4 beats PP4 at identical
  settings.
- **The recalibrated warmup constraint worked**: no sampler OOM in 40 experiments (up to 980
  sequences), and TP2/DP2 at 256 sequences (exp 5, 754) starts where study 15's 768 died.
- **Failures**: exp 6/10 as above; exp 17 (TP1/DP3, a valid configuration) completed its
  workflow but Akamas' telemetry service did not answer at collection time (platform incident).
- Side effect at the top: queue time p95 of 2–9 s in the 1 024-concurrency windows while TTFT
  and ITL meet the SLA; GPU power 71–72 W per active L4; SM activity balanced across the three
  engines; PCIe 3.1–3.3 GB/s per GPU (expert weights are sharded over the DP world whether
  `enable_expert_parallel` is on or off — non-KV footprint ≈ 13–16 GiB per GPU in both cases).

## Conclusions

1. **On a PCIe-only 4× L4 node, an MoE model that does not fit one GPU is best served with
   data parallelism and expert sharding, not tensor or pipeline parallelism** — TP1/DP3 with fp8
   KV reaches 2 186 tok/s per GPU (6 559 total), 2.6× the TP4 baseline per GPU and ~2× the best
   TP4 in absolute terms. Phase 2 (NCCL tuning) should therefore target the layouts that move
   the most PCIe traffic (TP4 at 4–6.5 GB/s per GPU, DP3/DP4 at 3–4), not TP2 or PP.
2. **KV-cache capacity, not collectives, decides the 2-GPU cells**: with 14.5 GiB of weights per
   card, TP2 keeps ~2 GiB of KV and DP2 keeps none. The per-GPU objective's premise ("half the
   hardware might win per GPU") is answered negatively for this model on 24 GB cards; it would
   need fp8 weights *and* fp8 KV *and* a smaller `max_num_batched_tokens` to be revisited.
3. **The study did not measure what it set out to compare in full**: the optimizer spent 26 of 27
   asks on DP3, so the DP3-vs-DP4 and DP3-vs-TP2/DP2 comparisons are tuned-vs-untuned. The
   follow-up needs forced topologies (presets or one short study per layout) with fp8 KV and the
   ~500-sequence region, and a ramp that goes beyond 1 024 concurrent requests, otherwise the top
   configurations remain indistinguishable.
4. **Constraint fixes for the next manifest**: constraint 5 must exclude TP1/DP2 (or model the
   DP activation peak, ≈ 4–5 GiB at 8 192 batched tokens); `fp8_e4m3` can be dropped from the
   KV domain (same kernel as `fp8`); `kv_cache_dtype` should be pinned to `fp8` for KV-bound
   layouts; pipeline parallelism can be dropped for throughput goals on this node. Constraint 6
   (EP required at 4 ranks) is still unverified (`smoke_test.sh tp4-noep` never ran).
5. **Platform debt surfaced by the analysis**: `akamas export study` on 3.7.x omits
   `last-optimization.json`/`logs.json` and caps metric files at 22 (goal metrics missing);
   the log service rejected large INFO queries; the telemetry service failed exp 17. Raise with
   Akamas; keep Prometheus retention (10 d) in mind when analysing a study late.
