# 10-gpt-oss-20b-goodput-per-gpu

**Status:** TODO (scaffolded 2026-09-14; created on Akamas and smoke-tested 5/5 on 2026-09-15; not started yet)
**Dates:** Scaffolded 2026-09-14

## Objective

Same question as `9-goodput-per-gpu`, asked of a different model: which vLLM configuration
extracts the most **goodput per GPU actually used** from one `g6.12xlarge` (4x NVIDIA L4
24 GB), serving **`openai/gpt-oss-20b`** instead of `Qwen/Qwen2.5-7B-Instruct`:

```
maximize  (vLLM.prefill_token_throughput + vLLM.decode_token_throughput) / vLLM.active_gpus
subject to  vLLM.time_to_first_token_p95 <= 1500 ms,  vLLM.inter_token_latency_p95 <= 300 ms
```

gpt-oss-20b is a 21B-parameter mixture-of-experts model (32 experts, 4 active per token,
3.6B active parameters) whose expert weights ship quantized in MXFP4. It changes three
things compared with a dense 7B model in bf16: the weights are small (12.8 GiB) but the
kernel path on this GPU generation is different (Marlin), the attention has sinks and
alternating sliding-window layers (only one attention backend qualifies on the L4), and
the model reasons before answering (output length is governed by `reasoning_effort`). It
also makes **`enable_expert_parallel`** a meaningful knob for the first time in this repo.

## Does gpt-oss-20b fit on one L4? Yes — TP > 1 is a choice, not a requirement

The premise "it may not fit on a single 24 GB GPU" holds for the bf16 view of a 21B model
(~42 GB). It does not hold for the shipped checkpoint, as long as vLLM keeps the experts
4-bit. Verified against the vLLM **v0.29.0 source tree** on 2026-09-14 (not yet on the
hardware — see the smoke test):

| Fact | Evidence |
|---|---|
| Checkpoint size | `model-0000{0,1,2}-of-00002.safetensors` = **12.81 GiB** (HF API, `siblings[].size`); vLLM's default `--ignore-patterns original/**/*` skips the duplicate `original/` and `metal/` copies. Not gated, Apache-2.0. |
| MXFP4 stays 4-bit on SM 8.9 | `Mxfp4Config.get_min_capability()` = 80 (`layers/quantization/mxfp4.py`); backend priority for gpt-oss (`fused_moe/oracle/mxfp4.py`, `_get_priority_backends_for_gpt_oss`) is FlashInfer TRTLLM (SM100) → AITER (ROCm) → **TRITON** (OpenAI `triton_kernels`, gated to `(9,0) <= cap < (11,0)` in `experts/gpt_oss_triton_kernels_moe.py`, so skipped on 8.9) → FlashInfer CUTLASS (SM90+) → **MARLIN** (`experts/marlin_moe.py`: `has_device_capability((7, 5))`) → emulation. **On the L4 the Marlin MXFP4 MoE kernel is selected**; the bf16 dequantized fallback (`EMULATION`) is last. The vLLM gpt-oss recipe states the same for Ampere ("TRITON_ATTN attention backend and Marlin MXFP4 MoE") and lists Ada Lovelace as work in progress. |
| Per-GPU budget at TP1 | usable framebuffer 22.5 GiB (DCGM `FB_TOTAL` 22 563 MiB) x `gpu_memory_utilization` 0.85 = 19.1 GiB; minus 12.8 GiB weights = **~6 GiB for activations + KV cache** (5.2 GiB at 0.80, 7.4 GiB at 0.90). |
| KV cache per token | 24 layers x 8 KV heads x head_dim 64 x 2 (K,V) x 2 B = 48 KiB bf16 / 24 KiB fp8 if every layer were full attention; 12 of the 24 layers are sliding-window (128 tokens) and vLLM's hybrid KV-cache manager sizes them separately, so the effective cost is ~24 KiB/token bf16 — roughly **200k tokens of KV at TP1**, ample for ShareGPT-shaped requests. |
| Attention backend | gpt-oss uses attention sinks. On compute capability 8.9: FLASH_ATTN → "sink not supported on compute capability < 9.0" (`v1/attention/backends/flash_attn.py`); FLASHINFER → sinks only on SM100 trtllm-gen / SM12x XQA (`flashinfer.py`); **TRITON_ATTN → `supports_sink` True, `supports_sliding_window` True, fp8 KV accepted on capability >= 89** (`triton_attn.py`). Hence `--attention-backend TRITON_ATTN` is pinned in the template and `attention_backend` is not tuned. |
| Tensor-parallel degrees | 64 attention heads / 8 KV heads → TP ∈ {1, 2, 4} on this node; TP = 3 is rejected at engine init (same constraint shape as study 9's "TP != 3"). |
| Expert parallelism | `--enable-expert-parallel` shards the 32 experts over the TP x DP ranks; the Marlin MoE kernel takes an `expert_map` (`marlin_moe.py`), so EP works on this path. A no-op on a single rank → enforced by a `parameterConstraint`. |

The constraints that "let it run and then let Akamas explore" are therefore:
`TP x DP <= 4`, `TP != 3`, `max_num_batched_tokens >= max_num_seqs`, `enable_expert_parallel`
only when `TP x DP > 1`, plus a memory-fit rule (`0.85 x 22.5 - 12.8 / TP >= 4` GiB) that is
always satisfied inside the current domains and exists to guard future domain changes.
`gpu_memory_utilization` is capped at 0.90 because study 9's crash analysis
(`~/akamas/2026-09-14-study9-exp64-collasso-throughput.md`) found 17 of 19 runtime crashes
at >= 0.90, including the "best" experiment 64 at 0.949 with 353 MiB free.

## Stack & versions

- **Akamas:** 3.7.1 (`akamas.lab.akamas.io`, workspace `default`), CLI 3.0.1 in the `toolbox`
  pod (login expires roughly daily — `akamas login` before using it).
- **Optimization packs:** vLLM **>= 1.9.1** — the union release that carries both the
  1.7.0/1.8.0 metrics this telemetry needs (`active_gpus`, `active_dp_engines`,
  `gpu_memory_allocated_gb`, `kv_cache_*`) and the 1.9.0 agentic knobs / vLLM 0.29.0
  reference (branch `feature/agentic-scheduling-knobs-1.9.1`, built as `vLLM_1-9-1.json`).
  **1.9.0 alone is not enough**: installed on 2026-09-14 from `develop`, it dropped those
  seven metrics and `akamas create telemetry-instance` failed with "metric(s) are not
  present in System" (check with `akamas describe optimization-pack vLLM | grep -E
  'version|active_gpus'`). GPU **1.2.0**, Kubernetes **1.8.0-dev**. `stream_interval`
  (1.9.x) stays optional: commented-out blocks in the study manifest and the template.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` (CUDA 13.0.2, same base and driver
  requirement as the `v0.22.0` image studies 7-9 ran on this node's AL2023 NVIDIA AMI),
  model `openai/gpt-oss-20b` served as `gpt-oss-20b`, pinned flags `--attention-backend
  TRITON_ATTN --reasoning-parser openai_gptoss --max-model-len 32768
  --no-enable-prefix-caching --enable-mfu-metrics` (rationale in the template's header).
- **Cluster / hardware:** EKS `vllm-bench` (us-east-2), node group `llm-serving-l4`
  (1x `g6.12xlarge`, 4x L4 24 GB, no NVLink), see `infra/README.md` — including why the
  model cache is a **new PVC** (`vllm-model-cache-gptoss`) and what the current
  `InsufficientInstanceCapacity` situation means.
- **Load generator:** NVIDIA AIPerf **0.11.0**, closed-loop concurrency sweep
  `150,179,213,253,302,359,428,509,606,722,860,1024` x 300 s, streaming chat completions,
  ShareGPT replay from a **model-specific** cache file (`inputs-gpt-oss-20b.json`, derived
  from studies 1-9's `inputs.json` by rewriting the per-request `model` field — that legacy
  file embeds `qwen2.5-7b` and caused 14 339 x HTTP 404 on the first baseline, 2026-09-15),
  **`--extra-inputs reasoning_effort:low`**
  (`reasoning_effort` is a top-level field of vLLM 0.29.0's chat request and is forwarded to
  the harmony chat template; without it every reply carries a medium-effort chain of
  thought and the sweep saturates far earlier than study 9's calibration). The ramp is the
  one calibrated on 1x A10G for Qwen2.5-7B — recalibrate after the first trials if the SLA
  breaks in the first levels.
- **Telemetry:** Prometheus (`kube-prometheus-stack`, 30 s), instance
  `Prometheus_10_GPT_OSS_20B`, the same 108 metrics as study 9 (`vllm:*`, DCGM per GPU,
  kube-state-metrics, PSI). `kv_cache_capacity_*` from the sidecar are **approximate** for
  this model (hybrid sliding-window KV layout; sidecar constant 49 152 B/token bf16).

## Parameters tuned

| Parameter | Domain (⊂ vLLM pack) | Baseline | Note |
|---|---|---|---|
| `vLLM.gpu_memory_utilization` | [0.80, 0.90] | **0.85** (pinned; the recipe's benchmark value) | upper bound lowered from study 9's 0.95 (crash analysis) |
| `vLLM.max_num_seqs` | [16, 1024] | unrendered (vLLM default) | crash risk at high values transfers from study 9 |
| `vLLM.max_num_batched_tokens` | [256, 8192] | unrendered | must be >= `max_num_seqs` |
| `vLLM.kv_cache_dtype` | auto, fp8, fp8_e4m3 | unrendered (auto) | fp8 + sinks on TRITON_ATTN untested until the smoke test |
| `vLLM.performance_mode` | balanced, interactivity, throughput | unrendered | |
| `vLLM.optimization_level` | [0, 3] | unrendered (2) | |
| `vLLM.enforce_eager` | true, false | unrendered | |
| `vLLM.scheduling_policy` | fcfs, priority | unrendered | |
| `vLLM.async_scheduling` | true, false | unrendered | |
| `vLLM.max_cudagraph_capture_size` | [1, 1024] | unrendered | |
| `vLLM.block_size` | 16 … 128 (ordinal) | unrendered (16) | TRITON_ATTN accepts every multiple of 16 |
| **`vLLM.tensor_parallel_size`** | [1, 4], != 3, TP x DP <= 4 | unrendered (1) | |
| **`vLLM.data_parallel_size`** | [1, 4], TP x DP <= 4 | unrendered (1) | |
| **`vLLM.enable_expert_parallel`** | true, false | unrendered (false) | only when TP x DP > 1 |

Dropped from study 9's set: `attention_backend` (pinned), `disable_cascade_attn` (no-op on
TRITON_ATTN), `tokenizer_mode` (gpt-oss has a fast tokenizer only). Steps: `baseline`
(1 trial, everything but `gpu_memory_utilization` unrendered → vLLM defaults, TP1/DP1 so
`active_gpus = 1`), `optimize` 200 experiments / 40 failures (~10 days at ~75 min each).

## What is new in the workflow: the restart guard

`k8s/run_test_goodput.sh` reads the vLLM container's `restartCount` before and after the
AIPerf job and, if it grew, dumps `kubectl logs --previous` (the traceback study 9 never
preserved) and exits 1 → the trial is FAILED. Study 9's top 12 scores were all
configurations that crashed at a concurrency step and were still marked VALID because the
`stability` window landed before the crash. The windowing block itself is unchanged
(methodological continuity; no gpt-oss/L4 data to calibrate a real `maxStdDev` yet).

## Smoke test run on 2026-09-15 (node back in us-east-2c): 5/5 configurations passed

`k8s/smoke_test.sh`, run from the toolbox at 06:57-07:13 UTC right after the `g6.12xlarge`
node relaunched (full log: `results/smoke_test_2026-09-15.log`). Every rollout succeeded, every
chat completion with `reasoning_effort: low` returned an answer, and vLLM's own log confirms the
source-verified kernel path on the L4: `Using 'MARLIN' Mxfp4 MoE backend` / `Using MarlinExperts`
in all five runs (weights stay 4-bit; 1.9 s to load from the cache).

| Config | TP/DP/EP | KV dtype | eager / opt | KV cache (tokens, per engine) | GPU memory used (MiB, per active GPU) |
|---|---|---|---|---|---|
| tp1-auto | 1/1/no | auto | false / 2 | 82 819 | 18 604 (GPU 0 only) |
| tp1-fp8 | 1/1/no | fp8 | false / 3 | 130 986 | 20 056 |
| tp1-block96 | 1/1/no | auto, `block_size` 96 | false / 2 | 82 453 | 18 606 |
| tp2-ep | 2/1/yes | fp8_e4m3 | false / 2 | 1 125 105 | 20 985 on GPUs 0-1 |
| dp4 | 1/4/no | auto | **true** / 1 | 438 932 per engine | 20 041 on all 4 |

Three things the numbers say before any optimization ran:

- **Memory is the lever.** Same TP1/auto/0.85 settings give 82 819 KV tokens with CUDA graphs
  and torch.compile (`enforce_eager false`, level 2) but 438 932 per engine with
  `enforce_eager true`, level 1 (dp4 row): the compiled path reserves several GiB per GPU for
  graphs and compile workspaces on this MoE/Marlin model. The optimizer will see that trade-off
  (KV capacity vs. per-step speed) directly.
- **TP2 + EP** halves the weight shard per GPU and, with fp8 KV, reaches 1.1 M KV tokens —
  the 34x "maximum concurrency" figure is at 32 768-token requests.
- `block_size` 96 (does not divide the 128-token sliding window) is accepted by the hybrid
  KV-cache manager; fp8 and fp8_e4m3 KV work with attention sinks on TRITON_ATTN.

One warning per start, harmless for serving: `Auto-initialization of reasoning token IDs failed.
Please check whether your reasoning parser has implemented the reasoning_start_str and
reasoning_end_str` (the `openai_gptoss` parser does not expose those; `reasoning_content` is
still returned). The last configuration (dp4) is left deployed; the study's baseline replaces it.

## Prerequisites before this study can be started

1. **GPU node.** `llm-serving-l4` is `DEGRADED` since 2026-09-14 14:36 UTC:
   `InsufficientInstanceCapacity` for `g6.12xlarge` in us-east-2a/b/c; the ASG retries every
   ~2 min by itself. Alternatives are the user's call — see `infra/README.md`, "AZ binding,
   capacity and what a region move entails".
2. **Model cache in the node's AZ.** Apply `k8s/01-pvc-model-cache.yaml` (new claim
   `vllm-model-cache-gptoss`, `gp3-ephemeral`, WaitForFirstConsumer) — it binds where the
   pod first starts. Study 9's `vllm-model-cache` (us-east-2c) is not used and not deleted
   here.
3. **Packs:** vLLM >= **1.9.1** (1.9.0 lacks `active_gpus` and the `kv_cache_*` metrics),
   GPU 1.2.0, Kubernetes 1.8.0-dev installed —
   `akamas install -f optimization-pack /work/vLLM_1-9-1.json` in the toolbox, then
   `akamas describe optimization-pack vLLM | grep -E 'version|active_gpus'`.
4. **Smoke test first:** `bash k8s/smoke_test.sh` on the toolbox (five configurations:
   TP1 auto KV, TP1 fp8, TP1 block 96, TP2 + EP, DP4) — **done 2026-09-15, 5/5 passed**, see
   the section above. Re-run it after any change to the template or the domains.
5. **Toolbox checkout and key:** `git pull` in `/work/vllm-benchmark`; copy the SSH key to
   `/work/vllm-benchmark/studies/10-gpt-oss-20b-goodput-per-gpu/akamas/id_rsa`
   (never in git — see `.gitignore`).
6. **No other study running against `llm-serving`** (studies 7/8/9 share `Deployment/vllm`
   and `Job/aiperf-benchmark`).
7. **Rebuild the AIPerf tokenizer cache once**: the `hf-cache` PVC holds the Qwen tokenizer;
   AIPerf downloads `openai/gpt-oss-20b`'s on first run (no token needed).

## Setup & run

All commands run in the `toolbox` pod (`kubectl -n akamas exec -it deploy/toolbox -- bash`),
from `/work/vllm-benchmark`, after `akamas login`.

```bash
# 0. one-time Kubernetes objects (idempotent)
kubectl apply -f studies/10-gpt-oss-20b-goodput-per-gpu/k8s/01-pvc-model-cache.yaml
kubectl apply -f studies/10-gpt-oss-20b-goodput-per-gpu/k8s/00-pvc.yaml
kubectl apply -f studies/10-gpt-oss-20b-goodput-per-gpu/k8s/06-hf-cache-pvc.yaml
kubectl apply -f studies/10-gpt-oss-20b-goodput-per-gpu/k8s/02-service.yaml
kubectl apply -f studies/10-gpt-oss-20b-goodput-per-gpu/k8s/04-kv-cache-exporter-configmap.yaml
bash studies/10-gpt-oss-20b-goodput-per-gpu/k8s/smoke_test.sh        # must end with "0 failed configuration(s)"

# 1. Akamas resources, typed form, dependency order
akamas create system            studies/10-gpt-oss-20b-goodput-per-gpu/akamas/system.yaml
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/vllm.yaml               "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/container.yaml          "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/cluster.yaml            "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/gpu0.yaml               "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/gpu1.yaml               "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/gpu2.yaml               "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/gpu3.yaml               "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/container_loadtest.yaml "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create component         studies/10-gpt-oss-20b-goodput-per-gpu/akamas/components/cluster_loadtest.yaml   "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create telemetry-instance studies/10-gpt-oss-20b-goodput-per-gpu/akamas/telemetry/prometheus.yaml         "vLLM_Benchmark_10_GPT_OSS_20B"
akamas create workflow          studies/10-gpt-oss-20b-goodput-per-gpu/akamas/10-GPT-OSS-20B-Goodput-Per-GPU-Workflow.yaml
akamas create study             studies/10-gpt-oss-20b-goodput-per-gpu/akamas/10-GPT-OSS-20B-Goodput-Per-GPU.yaml

#    or, bulk form (every file self-describes kind:/system:; same dependency order applies):
akamas create -f studies/10-gpt-oss-20b-goodput-per-gpu/akamas/

# 2. check, then start
akamas describe study "10-GPT-OSS-20B-Goodput-Per-GPU"      # expect 14 parameters, 5 parameterConstraints, 2 steps
akamas start study "10-GPT-OSS-20B-Goodput-Per-GPU"
akamas list experiment "10-GPT-OSS-20B-Goodput-Per-GPU"
```

If the telemetry instance fails with "metric(s) are not present in System", the installed
vLLM pack is 1.9.0 or older (no `active_gpus`/`kv_cache_*`) — install 1.9.1 first (see
"Prerequisites"), then re-create only the telemetry instance, the workflow and the study:
the system and its components are created independently and can stay.

## Known caveats

- **No optimization trial has run yet.** Kernel/backend selection and the fp8/fp8_e4m3 KV +
  sinks + TRITON_ATTN combination are now observed on the L4 (smoke test 2026-09-15), but
  behaviour under the AIPerf sweep, and the crash behaviour at high `max_num_seqs`, are not.
- Reasoning tokens inflate `decode_token_throughput` relative to a non-reasoning model:
  the goal measures engine tokens/s per GPU, not "useful answer tokens". AIPerf's own
  goodput numbers treat `reasoning_content` deltas as reasoning tokens — read
  `profile_export_aiperf.json` after the smoke test before comparing client-side TTFT/ITL
  with study 9.
- `kv_cache_capacity_*`/`kv_cache_used_gb` are approximate (hybrid KV layout; fp8 trials
  read 2x high as in study 9).
- Shares `llm-serving/vllm`, `llm-benchmark/aiperf-benchmark`, the `hf-cache` and
  `aiperf-results` PVCs with studies 7/8/9 — one study at a time.
- The DCGM exporter release for this node group (`dcgm-exporter-l4`) and the
  `kube-prometheus-stack` are shared with studies 6-9 (`k8s/monitoring/` is the same
  copy).

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
