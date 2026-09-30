# 24-L4-PD-Kernels

**Status:** RUNNING (created and started 2026-09-29 18:24 UTC; study id 2f1a2b8f-8d17-4e75-a200-c03cf2b5ba2a)
**Start note:** the first `akamas start` failed: Airflow registered the new study DAG after
the campaign service timeout, so the study stayed RUNNING with no experiment, and a restart
is refused (RUNNING -> RUNNING). Fix: delete the study, create it again, start it again.
**Needs:** vLLM optimization pack **1.12.0** installed (`linear_backend`, `attention_backend`
value `auto`, `tuned_kernel_configs`, `time_to_first_token_p95_150s`,
`inter_token_latency_p95_150s`).

> Prefill/decode disaggregation of Qwen3-8B-FP8 on one g6.12xlarge (4x L4), same load and
> goal as studies 22-23, with the **kernel backends in the search space**. Own system,
> telemetry instance and workflow; the workflow runs this folder's `k8s/` on the toolbox.

## Why this study exists

Study 23 and the 2026-09-28/29 analysis showed that on L4 the kernel choice changes the
prefill step more than any scheduler parameter:

- vLLM selects Marlin (weight-only FP8, BF16 compute) for block-FP8 on sm89. It is right for
  decode and slow for prefill. `kernel-bench/` measured the alternatives directly: prefill
  step Marlin 1.750 s, humming 1.329 s, triton 1.114 s, triton with tuned configs 1.066 s.
- `--linear-backend` can choose the kernel per role, and the tuned Triton configs for L4 now
  exist (`k8s/tune/`, `k8s/tuned-configs/`).
- Study 22-23's latency constraints (average of six 30 s p95s) made scores near the limits
  depend on sample alignment by up to ~30%.

## What changes against study 23

| Item | Study 23 | Study 24 |
|---|---|---|
| Latency constraints | TTFT/ITL p95 over 30 s, averaged | p95 over 150 s, `:max` (`*_p95_150s`) |
| Prefill linear kernel | fixed triton default (`VLLM_DISABLED_KERNELS`) | parameter: marlin / humming / triton |
| Decode linear kernel | fixed Marlin | parameter: marlin / humming |
| Attention | fixed FLASHINFER | one parameter for both roles: FLASHINFER / FLASH_ATTN |
| KV cache dtype | two parameters + equality disjunction | one parameter for both roles |
| Tuned Triton configs | none | `tuned_kernel_configs` true / false |
| Connector | pull or push | pull only (push: TTFT > 10 s from concurrency 8) |
| Topology | 1-3 P, 1-3 D | 0-3 P, 1-2 D |
| Baseline | imported from study 20 | re-run in this study (same values as study 20) |
| Prefill OOM guard | none (experiments 12-13 crashed) | `gmu*22.03 + batched*0.00016 <= 22.03` |
| GPU telemetry scope | `pod: .*` (any node) | `exported_pod` of this study's serving pod |

Design notes are in the manifests: `akamas/24-L4-PD-Kernels.yaml`,
`k8s/01-deployment_template.yaml` ("Kernel backends"), `akamas/telemetry/prometheus.yaml`
("STUDY 24").

## Steps and expected results (written before the start)

~65 min per experiment: baseline + 11 presets ≈ 13 h, then 60 AKAMAS experiments.

The expectations assume the P1D1 is prefill-bound in this setup (study 21, study 23), so the
score scales with the prefill capacity (1 / prefill step) until another constraint decides.
Reference: study 22's P1D1 (triton default) recomputed with the 150 s p95 = 1344-1518.
Decode differences below ~8% and score differences below ~7% are inside the noise.

| Step | What changes | Expected score (tok/s/GPU) | Why |
|---|---|---|---|
| baseline aggregated | 0P2D, vLLM defaults, Marlin, FLASHINFER | ~1040-1060 | study 20's baseline, limited by ITL |
| P1D1 prefill marlin | reference kernels | ~850-1050 | prefill step 1.75 s vs 1.11 s: ~0.64x of the triton P1D1; can fall below the baseline |
| P1D1 prefill humming | prefill humming | ~1100-1300 | prefill step 1.33 s: ~0.84x of triton |
| P1D1 prefill triton default | = study 22 exp 2 | ~1350-1500 | repeatability check |
| P1D1 prefill triton tuned | tuned configs on | triton default +0-8% | prefill step −4%; may stay inside the noise |
| P1D1 decode humming | decode humming | ≈ triton tuned (±5%) | decode is not the bottleneck |
| P1D1 kv fp8 | fp8 KV | ≈ triton tuned (−4% to +5%) | prefill +4%, decode capacity and TPOT better |
| P1D1 flash attn | FLASH_ATTN | ≈ triton tuned (±3%) | same prefill step as FLASHINFER |
| P2D1 best kernels | 2 prefill | ~1000-1300 (below P1D1) | 3 GPUs; the single bf16 decode saturates at ~14 sequences |
| P1D2 best kernels | 2 decode | ~950-1100 (≈ 2/3 of P1D1) | still prefill-bound, 3 GPUs |
| P2D2 best kernels | 2 + 2 | ≈ P1D1 (−10% to 0%) | two P1D1 pairs behind one router |
| P3D1 best kernels | 3 prefill | ~700-950 | the single decode is the bottleneck |

If the P1D1 prefill-kernel order (marlin < humming < triton ≤ triton tuned) does not hold in
the presets, the microbenchmark does not predict the loaded system, and the kernel ranking
must come from the study, not from `kernel-bench/`.

### Expected effect of the scheduler parameters and of the P:D ratio (added 2026-09-30, while experiment 2 ran)

Roofline estimates, not measurements. L4: ~300 GB/s memory bandwidth, ~242 TFLOP/s FP8 and
~121 TFLOP/s BF16 dense peak, about half of that at the 72 W power limit (SM clock ~1050
MHz). A GEMM with FP8 weights and M tokens does ~2·M FLOP per weight byte, so it becomes
compute-bound above a few hundred tokens (~200-400 for Marlin, BF16 compute; ~400-800 for
Triton FP8). Qwen3-8B KV: 36 layers × 8 KV heads × 128 × 2 (K, V) × 2 bytes = ~147 KB per
token in bf16 (~74 KB in fp8), so ~0.63 GB (bf16) / ~0.32 GB (fp8) per 4.3k-token sequence;
weights ~8.8 GB.

- **`max_num_batched_tokens`, prefill.** One 4096-token prompt is already far above the
  compute-bound threshold, so more tokens per step do not raise the throughput. They make the
  step longer, and the held-step and input-wait costs grow with the step; they also raise the
  OOM risk. Expected optimum: about one prompt per step (~4160). Values of 8192 and above:
  no gain, possible loss. Below ~2048: more steps per prompt, small loss.
- **`max_num_batched_tokens`, decode.** A decode step holds one token per running sequence,
  plus the chunk of any prompt; in P/D the decode does no prefill, so values above
  `max_num_seqs` do not change anything. Expected: no effect.
- **`max_num_seqs`, decode.** Step time ≈ (weights + b × KV per sequence) / bandwidth. ITL
  p95 ≤ 75 ms allows b ≈ 14 sequences in bf16 and ≈ 29 in fp8; the KV capacity at gmu 0.9
  is ≈ 14 (bf16) / ≈ 31 (fp8) sequences. Both limits agree, so values above ~14 (bf16) /
  ~30 (fp8) do not bind (the presets' 128 does not bind). Values below them cap the decode
  concurrency and lower the score. Expected optimum: ~14-16 with bf16, ~30-32 with fp8.
- **`max_num_seqs`, prefill.** Only the requests in the current step run: little effect,
  except that very low values (< 2-3) can starve the prefill between steps.
- **`gpu_memory_utilization`, decode.** More KV room: +1 sequence per ~0.63 GB (bf16). Small
  gain, bounded by the ITL limit above.

If the optimizer does not move these parameters, a later preset set can test them directly
(e.g. decode `max_num_seqs` 16 / 32 / 64 with fp8, prefill `max_num_batched_tokens` 4160 /
8192).

**P:D ratio (the number of GPUs matters through it).** Each instance is one GPU (TP=1), so
the per-instance capacity does not depend on the GPU count; the system serves about
min(P × prefill capacity, D × decode capacity), and the score divides it by P + D. Estimated
capacities, 4096 in / 256 out:

| Instance | Capacity |
|---|---|
| prefill, triton tuned (~1.07 s per prompt) | ~0.9 req/s |
| decode, bf16 KV (~14 sequences, ~75 ms step, 256 tokens) | ~0.75 req/s |
| decode, fp8 KV (~29 sequences, ~75 ms step) | ~1.5 req/s |

Consequences, and a **revision of the table above** (the table keeps the first prediction):

- With the triton prefill, the bf16 decode becomes the P1D1 bottleneck (as in study 23: at
  concurrency 24+ the bf16 decode queue reached 11-22 s).
- **`P1D1 kv fp8` revised: +20-25% over `P1D1 prefill triton tuned`** (first prediction:
  ≈ equal), because fp8 doubles the decode capacity.
- With fp8 KV, 2P1D (min(1.8, 1.5) / 3 GPUs) and 2P2D (1.8 / 4) reach about the P1D1 fp8
  value per GPU; with bf16 KV (the topology presets) they stay below it, as in the table.
- The SLA and waiting constraints stop the load before full saturation, so these ratios are
  upper bounds on the differences.

## Before creating the study

1. Install vLLM pack 1.12.0 on Akamas.
2. On the toolbox: `cd /work/vllm-benchmark && git pull`. The kernel-bench driver left
   untracked copies of `studies/24-l4-pd-kernels/k8s/tuned-configs/*.json` there (same
   content): remove them first, or the pull stops.
3. From the toolbox (the workflow checks the SSH key path when it is created):

```bash
cd /work/vllm-benchmark/studies/24-l4-pd-kernels/akamas
akamas create system system.yaml
akamas create component components/ vLLM_Benchmark_24_L4_PD_Kernels
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_24_L4_PD_Kernels
akamas create workflow 24-L4-PD-Kernels-Workflow.yaml
akamas create study 24-L4-PD-Kernels.yaml
akamas start study 24-L4-PD-Kernels
```

(Check the exact `akamas create` forms against the files' `kind`/`system` keys; the files
carry `kind:` for `akamas create -f`.)

## Other users of the cluster

A second GPU node exists (`node-role=llm-serving-g7-4500`, RTX PRO 4500, namespace
`gpu-sharing`). This study's pods pin `node-role: llm-serving-l4`, and its telemetry scopes
the vLLM, container and GPU series to this study's pod names. Names that must stay unique
to this study: Deployment/Service `vllm-pd`, label `app=vllm-pd` and ConfigMaps `pd-config`,
`pd-scripts`, `pd-tuned-configs` in `llm-serving`; Job `aiperf-benchmark` in
`llm-benchmark`; served model names `qwen3-8b-*`.
