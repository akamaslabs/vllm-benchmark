# l40s-gemma4-w4a16-kernels (probe, not a study)

**Status:** DONE — a probe run outside Akamas, not a study (no study number; kept apart from studies/)
**Dates:** 2026-10-09

## Objective

Groundwork for an upcoming customer optimization: **which kernels, kernel variants and base
vLLM levers make the customer's Gemma 4 W4A16 deployment on one L40S faster**, measured on
the shapes of their load, before an Akamas study is designed. No Akamas study yet: the load
(dataset, SLA, injector) is defined by the customer and not available.

What we know of the customer (2026-10-09):

- **Deployment** (docker compose): `vllm-openai:v0.25.1` (to be upgraded to **0.29.0**, decided),
  `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --revision 0ef577a`, TP 1, `gpu_memory_utilization`
  0.90, `max_model_len` 96000, `max_num_seqs` 64, `--enable-auto-tool-choice
  --tool-call-parser gemma4 --reasoning-parser gemma4`, `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True`, prefix caching on (vLLM default).
- **Load** (performance test they will run): synthetic, anonymized dataset from ~1 month of real
  use (18-19k requests); ~1/4 streaming; real think time capped at 30 s; ~30 min with 40-100
  users; context median 3,000 tokens, p90 ~6,000, p99 15,000, max 95,000. Injector from the
  Akamas toolbox pod through Kong. Prefixes probably preserved, thinking and output lengths
  unknown (to ask).
- **Goal** (stated): more concurrent requests under an e2e latency limit of "2-3 s", Kong
  included. Considered very tight for these prompt lengths: see the chat of 2026-10-09.

## Stack & versions

- **Akamas:** not used yet.
- **Workload:** `vllm/vllm-openai:v0.29.0`, `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` at
  `0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`, served as `gemma4-w4a16`, StatefulSet
  `vllm-w4a16` (namespace `llm-serving`), the customer's flags (`k8s/01-statefulset_template.yaml`).
- **Cluster / hardware:** `vllm-bench-gatling` (us-east-2, kubectl context
  `lab-vllm-bench-gatling`, AWS profile `lab`), node group `llm-serving-l40s`: one g6e.xlarge
  (1x L40S 48 GB, Ada SM 8.9, 4 vCPU / 32 GiB). Borrowed from study 2 of
  `akamas-gatling-llms-optimization` (branch `feat/study-2-gemma4-l40s-kernels`), idle while it
  waits for Gatling credits; this study uses its own StatefulSet name, ConfigMap and hostPath
  caches (`/var/lib/*-w4a16probe`), and `apply_config.sh` refuses to start if study 2's pod runs.

### The checkpoint

`config.json` at the pinned revision: compressed-tensors `pack-quantized`, **int4 symmetric,
group 32, activations bf16 (W4A16)**; despite the name, not the AutoAWQ format. Quantized: the
128 experts and the attention projections. bf16 (`ignore`): the dense MLP of each layer, the
router, embeddings / `lm_head`, the vision tower. 16.0 GiB of files against 26.7 GiB for the
FP8 checkpoint of studies 30/31. The KV cache is unaffected by weight quantization (bf16 with
`kv_cache_dtype=auto`, ~220 KiB/token below the 1024-token sliding window, then ~20 KiB/token
from the 5 full-attention layers only: study 30).

### Kernels vLLM 0.29.0 can run for it on SM 8.9 (source, 2026-10-09)

- **MoE** (`fused_moe/oracle/int_wna16.py`): priority flashinfer_trtllm -> **Marlin** ->
  batched Marlin -> Triton -> Humming. `--moe-backend` accepts triton / marlin / humming /
  flashinfer_trtllm / emulation. Variants by environment: `VLLM_MARLIN_INPUT_DTYPE` fp8 / int8
  (W4A8: activations quantized per token; fp8 explicitly allowed on SM 8.9 and SM 12.x only),
  `VLLM_MARLIN_USE_ATOMIC_ADD`, `VLLM_HUMMING_MOE_GEMM_TYPE` indexed / grouped,
  `VLLM_HUMMING_INPUT_QUANT_CONFIG` (`{"dtype":"float8e4m3"}` / `{"dtype":"int8"}`),
  `VLLM_HUMMING_USE_F16_ACCUM`.
- **Linear** (`kernels/linear/__init__.py`, mixed precision): CutlassW4A8 -> Machete (Hopper
  only) -> AllSpark -> **Marlin** -> Conch -> Exllama (fp16 activations only) -> TritonW4A16 ->
  Humming. `--linear-backend` accepts marlin / humming / triton / conch (AllSpark has no value).
  `VLLM_MARLIN_INPUT_DTYPE` applies to the linear Marlin path too.
- **Tunable by file:** only the Triton WNA16 fused-MoE kernel
  (`E=128,N=704,device_name=NVIDIA_L40S,dtype=int4_w4a16.json` via `VLLM_TUNED_CONFIG_FOLDER`),
  and only for steps above 96 tokens: below, a CUDA kernel runs (`should_moe_wna16_use_cuda`).
  Default tiles above 96 tokens: 64 x 64 x 32. Marlin and Humming choose tiles by heuristics;
  TritonW4A16 (linear) has hard-coded tiles.
- **Attention:** TRITON_ATTN, forced by `Gemma4Config` on SM 8.9 (head dims 256 and 512); fp8
  KV works with it (study 30, study 2 probe).

## Plan

1. **Stage 1 — every kernel and variant** (`kernel-bench/probe.sh`, 19 combinations on the
   customer's flags): MoE marlin / triton / humming / flashinfer_trtllm and their environment
   variants, linear marlin / humming / triton / conch, Humming everywhere with and without fp8
   activations.
2. **Triton WNA16 MoE tuning** (`kernel-bench/tune_moe_int4.py`, `tune-pod.yaml`), M 128-16384.
3. **Stage 2 — base levers on the best kernels:** fp8 KV, `--language-model-only`, MTP K=1-3
   (drafter `google/gemma-4-26B-A4B-it-assistant`), `performance_mode` interactivity,
   `max_num_batched_tokens` / `long_prefill_token_threshold`, the tuned Triton MoE.

Per combination (`kernel-bench/bench_in_pod.py`): KV cache tokens; prefill of unique ~3k / 6k /
15k-token prompts (95k on B-default); ms per output token at 1 / 8 / 16 / 32 / 48 / 64
concurrent requests; a mixed closed loop (3k-token context + 150 output tokens, 8 / 16 / 32
workers: e2e p50/p95, req/s); a follow-up turn with its history in the prefix cache; accuracy
on 40 arithmetic problems (guards the numerics-changing variants).

## Results

GPU time 2026-10-09 13:16-20:01 UTC (~6.75 h of g6e.xlarge, ~13 USD); `AlwaysOn` tags removed and
the node group back at 0 afterwards. Raw data: `kernel-bench/results/stage1/` and `stage2/`
(stage 3 is in `stage2/`), `summary.txt` in each; tuning log `kernel-bench/results/tune_moe_int4.log`,
tuned file `kernel-bench/tuned-configs/`. Repeatability: the same configuration started three
times (B-default, X-marlin, B2-default) scored within ~1 % on every measure.

### Customer configuration on vLLM 0.29.0 (B-default)

Marlin for MoE and linear, TRITON_ATTN; bf16 KV cache 708,768 tokens (7.4 requests of 96k);
vLLM's own `max_num_batched_tokens` 2496.

| Measure | Value |
|---|---|
| Prefill 3k / 6k / 15k / 95k tokens | 0.18 / 0.38 / 1.13 / 19.1 s |
| ms per output token at 1 / 8 / 16 / 32 / 64 concurrent | 7.3 / 12.1 / 14.7 / 17.3 / 21.3 |
| Mixed 3k in + 150 out (fixed by the bench), e2e p50 at 8 / 16 / 32 workers | 3.5 / 5.6 / 9.1 s (2.3 / 2.8 / 3.5 req/s) |
| Follow-up turn with 3k of history cached | 0.047 s against 0.19 s cold (when it hits: see below) |

### Stage 1: kernels (vs B-default, positive = better)

| Combination | Prefill 3k | ms/token at 16 | Mixed c16 req/s | Quality |
|---|---|---|---|---|
| Marlin W4A8-fp8 (`VLLM_MARLIN_INPUT_DTYPE=fp8`) | +8 % | -2 % | +9 % | 40/40 |
| Marlin W4A8-int8 | +13 % | +3 % | +12 % | **0/40: broken output** |
| Marlin atomic add | 0 | 0 | 0 | 40/40 |
| Humming MoE (indexed, its default) | 0 | -5 % | 0 | 40/40 |
| Humming MoE grouped | -3 % | -6 % | 0 (KV -18 %) | 40/40 |
| Humming MoE, activations fp8 / int8 | +10 % / +9 % | -5 % | +9 % | 40/40 |
| Humming MoE + linear (+ act fp8) | +2 % (+10 %) | -5 % (-6 %) | 0 (+11 %) | 40/40 |
| Linear Humming | +2 % | 0 | 0 | 40/40 |
| Linear TritonW4A16 | -9 % | -49 % | -20 % | 40/40 |
| MoE Triton WNA16, default config | -105 % | -161 % | -50 % | 40/40 |
| Do not start | linear conch (no kernel for the layer), MoE flashinfer_trtllm (not on SM 8.9), Humming f16 accumulation (AssertionError) | | | |

### Triton WNA16 MoE tuning (`tune_moe_int4.py`, 2 h 2 min)

Kernel time against vLLM's default config (64 x 64 x 32 tiles): M=128 -40 %, 256 -38 %, 512
-38 %, 1024 -49 %, 2048 -52 %, 4096 -55 %, 8192 -57 %; the winner is 128 x 128 x 32 (64 rows
up to M=512). End to end (X-triton-tuned): prefill 3k -105 % -> -33 % against Marlin, decode
unchanged (-161 % at 16: steps of <= 96 tokens run the CUDA kernel the file does not touch),
mixed -41 %. **Marlin stays the MoE kernel.**

### Stage 2: base levers (vs B2-default)

| Combination | KV tokens | Prefill 3k | ms/token at 16 | Mixed c16 e2e p50 / req/s | Mixed c8 e2e p50 |
|---|---|---|---|---|---|
| B2-default | 709k | 0.181 s | 14.7 | 5.57 s / 2.84 | 3.51 s |
| fp8 KV | 1,417k | +10 % | +3 % | 4.87 s / 3.36 (+18 %) | 3.16 s |
| MTP K=1 / 2 / 3 (acceptance 0.71 / 0.60 / 0.50 on essays) | ~630k | -6 / -7 / -7 % | +29 / +36 / +38 % | +20 / +23 / +21 % | 2.99 / 2.87 / 2.85 s |
| Marlin W4A8-fp8 | 688k | +7 % | -2 % | +9 % | 3.43 s |
| `--language-model-only` | 758k | 0 | 0 | 0 | |
| `performance_mode` interactivity | 703k | 0 | 0 | 0 | |
| `max_num_batched_tokens` 8192 (+/- `long_prefill_token_threshold` 2048) | **395k** | +5 % | 0 | 0 | |
| `max_num_batched_tokens` 1024 | does not start: the vision encoder needs >= 2496 without `--language-model-only` | | | | |
| combo: fp8 KV + lmo + MTP K=2 | 1,412k | +7 % | +39 % | 4.04 s / 3.96 (+39 %) | 2.59 s |
| combo + Marlin W4A8-fp8 | 1,363k | +16 % | +37 % | 3.78 s / 4.22 (+48 %) | 2.47 s |

### Stage 3: prefix-cache hits on the hybrid KV cache

A follow-up turn (previous turn ~3,000 tokens + ~300 new) gets **all or nothing** from the
prefix cache (2976 tokens or 0, vLLM's `prefix_cache_hits` counter), depending on the exact
length of the previous turn:

| Combination | Turn-2 hits | KV tokens |
|---|---|---|
| default (hybrid KV cache manager) | 5/8 (stage 1: misses in 3 of 19 combinations) | 709k |
| `--disable-hybrid-kv-cache-manager` | **8/8** | **105k (-85 %)** |
| `--block-size` 32 / 64 | 7/8 / 4/8 | 709k |
| **any MTP setting** | **0/8 in every run (24/24)**: prefix caching enabled, "Prefix cache hit rate: 0.0%" | |
| MTP + `--disable-hybrid-kv-cache-manager` | does not start (`KeyError: 'draft_model.layers.0.self_attn.attn'`) | |

## Conclusions

- **Kernels:** vLLM's default (Marlin for MoE and linear) is the best or tied on every
  measure; the only kernel lever is **fp8 activations** (Marlin W4A8-fp8, or Humming with fp8
  input quant): +8-11 % prefill and mixed throughput, -2-6 % decode. Its quality check here is
  40 arithmetic problems only: needs a real eval (tool calls, gsm8k) before it enters a study.
  **int8 activations break the model.** Triton MoE, even tuned (-40/-57 % kernel time), stays
  behind Marlin; the tuned file is kept for the record.
- **Levers:** fp8 KV (2x KV, +18 % mixed) and MTP (+36 % decode at 16, +23 % mixed, K=2-3)
  are the large ones; together with fp8 activations +48 % req/s at the same concurrency, and
  the mixed 3k/150 e2e at 8 workers drops from 3.5 to 2.5 s. `--language-model-only` only adds
  7 % KV; interactivity mode and larger batched-token budgets do nothing (8192 halves the KV).
- **Prefix caching is the open issue, and it decides MTP:** on this model (hybrid sliding-window
  KV) a follow-up turn misses the cache ~1 time in 3 by default, always with MTP, and the fix
  that restores hits (no hybrid manager) costs 85 % of the KV and cannot run with MTP. On a
  multi-turn agentic load the 4x of a prefix hit may outweigh MTP's decode gain: **measure MTP
  on the customer's dataset with prefix reuse before choosing it.** Not yet checked: whether the
  misses depend on how the previous turn ends (here: a prompt only, 1 output token; in a real
  conversation turn 2 also carries turn 1's answer).
- **For the customer's "2-3 s e2e":** at 3k in / 150 out the default is at 3.5 s with 8
  concurrent requests and the best combination at 2.5 s; a 95k prompt alone takes 19 s of
  prefill.
