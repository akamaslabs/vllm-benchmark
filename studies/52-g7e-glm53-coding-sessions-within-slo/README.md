# 52-g7e-glm53-coding-sessions-within-slo

**Status:** RUNNING on akamas-41 since 2026-11-18 11:54 UTC (study `52-G7E-GLM53-Coding-Sessions`), on a
**4-bit checkpoint**; smoke passed (see "Smoke 2026-11-18"). The 2026-11-17 smoke said no-go
for the target's FP8 checkpoint on the g7e.24xlarge: it loads at TP4 but leaves no room for the KV
cache on 96 GB GPUs (see "Smoke 2026-11-17"). Decided with the user on 2026-11-18: run
`canada-quant/GLM-5.3-Flash-W4A16-MTP` at TP4 with the target's arguments (~44 GiB of weights per
GPU, ~40 GiB left for the KV cache, close to the target's H200 at TP4) to measure the margin for
improvement. Waiting for g7e.24xlarge spot capacity (score 1 in every us-east-2 AZ at 08:33 UTC);
node groups at 0. The smoke runs again on the new checkpoint before any Akamas resource.

- **Experiments 1-2 (walk from the bottom):** `baseline` held **24** sessions (score 24.379),
  `Offload 256` held **24** (23.242): levels 4-24 passed, 32 failed on client ITL p95 57-60 ms
  (TTFT p95 ~3.5 s, within the SLO). 24-32 sessions of at most ~100k tokens need 2.4-3.2M tokens of
  KV, under the 3.77M on the GPUs, so offload has little to do at these levels. From experiment 3 (16:18 UTC) the ramp bisects (see Method).
- **Design:** `docs/superpowers/specs/2026-11-17-study-52-glm53-coding-sessions-design.md`.
- **Plan:** `docs/superpowers/plans/2026-11-17-study-52-glm53-coding-sessions.md`.
- **Follows the spike:** `docs/spikes/2026-11-17-glm53-flash-phase0/README.md`.

## Objective

A target deployment serves `zai-org/GLM-5.3-Flash` to coding agents with long contexts: one vLLM
pod per replica, TP4 on H200, the official FP8 checkpoint, and

```
--kv-cache-dtype fp8 --max-model-len 262144 --gpu-memory-utilization 0.95
--speculative-config.method mtp --speculative-config.num_speculative_tokens 5
--tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --tensor-parallel-size 4
```

Study 52 asks **how many concurrent coding sessions one replica serves within client TTFT p95
<= 5 s and client ITL p95 <= 50 ms**, with those arguments and with the levers they leave unused
(CPU KV offload, MTP depth, scheduler sizing; decode context parallelism is unavailable on SM120).

It runs on **4x RTX PRO 6000 Blackwell (one g7e.24xlarge spot)** with the target's TP4 and arguments,
on a **4-bit checkpoint** (`canada-quant/GLM-5.3-Flash-W4A16-MTP`: routed experts 4-bit; attention,
KDA and the MTP head BF16): the target's FP8 checkpoint leaves no KV room at TP4 on 96 GB GPUs
(smoke 2026-11-17). The GPU architecture (SM120 vs SM90) and the MoE kernels (W4A16 Marlin-class vs
FP8) differ, so absolute numbers do not transfer; the KV room per GPU is close to the target's, so
which levers matter, and roughly how much, should. Results are sessions on this rig, never a
sessions number for the target.

## Method

- **Sessions (AIPerf 0.11.0, user-centric multi-turn, history re-sent):** a 20k-token system
  prompt shared by every session, 10k tokens of per-session context, 16 turns of 4k input and
  400 output tokens (`ignore_eos`), one turn per session every 30 s. A session grows from ~34k to
  ~100k tokens. Two AIPerf clients on a CPU node, half the sessions each.
- **Ramp task:** levels of 8, 12, 16, 24, 32, 48 and 64 sessions, each a fresh 960 s run + 60 s
  grace, judged on requests started >= 500 s after the first one. A level passes when client
  TTFT p95 <= 5000 ms and client ITL p95 <= 50 ms (each the worse of the two clients), errors
  < 1 %, >= 95 % of sent requests completed and >= 95 % of expected requests sent. From
  experiment 3 the ramp **bisects** the levels (24, then 48 or 12, then the level between: 3 runs
  at most) and keeps the highest passing one. Experiments 1-2 walked 4, 8, ... upwards and stopped
  at the first failing level (6 runs, 2 h 11 min each); the user capped an experiment at 90 min on
  2026-11-18. A configuration gets the same level either way as long as failing a level means
  failing every higher one (both stopped at 32, which the bisection reaches as 24 -> 48 -> 32).
  Each level also prints the worst session's TTFT p95 and the number of sessions above 5 s.
- **Experiment length:** warm Apply ~4 min + 3 levels x ~18 min (AIPerf start-up ~95 s, 960 s,
  grace, parsing) + Hold ~20 min = **~77 min**. Not bounded: a cold vLLM start (Apply waits up
  to `START_WAIT` = 4800 s) and a lost spot node (new node + 191 GB download).
- **Hold task:** the last passing level again, fresh, 1080 s; Akamas trim `[520s, 90s]`.
- **Score:** `30 x vllm.request_success_rate` = sessions held. No TTFT tie-break (the user's
  call): within a level, compare TTFT and ITL directly.
- **Constraints (Hold, safety net):** server TTFT p95 <= 6 s, server TPOT p95 <= 80 ms (per request; the per-step ITL runs far above the client ITL).
- **Steps:** `baseline` (the target's arguments), presets `Offload 256`, `MTP three`, `MTP one` (only
  those the smoke confirms), `Baseline repeat`, then `optimize` (10 experiments).

## Stack & versions

- **Hardware:** one g7e.24xlarge **spot** in us-east-2c (score 8, ~8.83 USD/h on 2026-11-17;
  on-demand 16.57), node group `llm-serving-g7e-24xl-spot` (`infra/eks/`), cluster `vllm-bench`
  (EKS, us-east-2).
- **Server:** `vllm/vllm-openai:v0.34.0`, `canada-quant/GLM-5.3-Flash-W4A16-MTP` (190.9 GB, revision
  5681f7c2) served as `zai-org/GLM-5.3-Flash`, TP4, `--language-model-only`; the target's container shape
  (`vllm serve /mnt/models ... ${VLLM_ADDITIONAL_ARGS}`); namespace `glm53-coding`.
- **Load:** AIPerf 0.11.0 (`k8s/workload.env`).
- **Akamas:** 4.1 (akamas-41); packs vLLM 1.15.0, GPU 1.4.0, Kubernetes 1.9.0 (checked 2026-11-17), checked again before
  the YAML is written.
- **Telemetry:** Prometheus; the full catalog (CLAUDE.md rule 9): vLLM, one GPU component per
  GPU, Kubernetes Node, Container (vLLM and AIPerf), Workload, Cluster.

## Parameters tuned

| Parameter | Domain | Baseline |
|---|---|---|
| `vllm.kv_offloading_size` | 0–512 GiB | 0 |
| `vllm.gpu_memory_utilization` | 0.85–0.97 | 0.95 |
| `vllm.max_num_seqs` | 8–128 | TODO (smoke: under the Mamba block count) |
| `vllm.max_num_batched_tokens` | 2048–16384 | default |
| `vllm.spec_tokens` (MTP) | 0–5 | 5 (0 if the smoke shows no acceptance) |
| `vllm.moe_backend` | auto, marlin, humming, triton (W4A16; narrowed by the smoke) | auto |
| `vllm.max_cudagraph_capture_size` | 16–512 | default |
| `vllm.async_scheduling` | true, false | default |
| `vllm.performance_mode` | as in study 38 | default |
| `vllm.optimization_level` | 1–3 | default |

Not levers here: `kv_cache_dtype` and `block_size` (forced by the SM120 sparse MLA backend and
the KDA state), `tensor_parallel_size` (the target's TP4).

## TODO

- Pack gaps (raise with whoever manages packs): `mamba_ssm_cache_dtype` (KDA state fp32/bf16),
  `kv_cache_dtype` value `fp8_ds_mla`, `language_model_only`, and no per-container or per-pod
  network metrics in the Kubernetes pack.
- Smoke on the node (`k8s/smoke.sh`) with the go / no-go rules of the design.
- Spot quota: raised to 192 vCPU on 2026-11-17 (case 179491438900392 closed); spot node up in us-east-2c.
- **After the smoke, before the study (agreed with the user 2026-11-17):** move the checkpoint and
  the compile caches from the node's hostPath to an EBS volume of their own (PVC, gp3 ~600 GB in
  us-east-2c, 1000 MB/s), copied from what the smoke downloaded, so a reclaimed spot node is
  replaced without re-downloading 191 GB or recompiling. Update 01-vllm_template.yaml and its tests;
  delete the volume at the end of the study.

## Smoke 2026-11-18 (4-bit checkpoint canada-quant W4A16-MTP 5681f7c2, TP4, target's arguments)

- **Baseline (gmu 0.95, MTP 5, everything else default) starts:** 45.76 GiB of weights per GPU;
  **GPU KV cache 3,611,552 tokens** (13.78x a 262144-token request); vLLM's default max_num_seqs fits
  (no pin); KV block 2304 tokens (mamba align). Cold start 755 s (weights 233 s from disk, TileLang
  JIT 227 s); later starts reuse the compile caches on the node.
- **Kernels:** attention `FLASHINFER_MLA_SPARSE_SM120` (`fp8_ds_mla`), routed experts **MARLIN WNA16**,
  the BF16 MTP head's MoE FlashInfer CUTLASS.
- **Answers:** a coding prompt (correct code) and a tool call (`read_file`) through the glm47 parsers.
- **16 sessions, 300 s:** client TTFT p95 1636 ms, ITL p95 19.5 ms (within 5 s / 50 ms), worst
  session TTFT p95 1672 ms; 73-109 output tokens/s per user on average.
- **MTP works on SM120 here:** 40.1 % of drafted tokens accepted, 2.0 per 5-token draft; per position
  74 / 51 / 35 / 24 / 17 %. The MTP presets stay.
- **Server latency vs client:** server TPOT p95 53.6 ms (per request, p50 16.1 ms) and ITL p95 72.1 ms
  (per engine step) against the client ITL p95 19.5 ms; the TPOT safety net is 80 ms, not 60.
- **Not run before the study (the user's call, 2026-11-18):** warm restart, offload 256 GiB and the
  domain edges; an offload or MoE candidate that does not start costs one experiment.
- **Download path:** `hf download` stalled mid-file and skipped the tokenizer; the checkpoint and the
  tokenizer now come down through `k8s/download_checkpoint.sh` (curl with resume, ~700 MB/s with a
  read-only token in the namespace Secret `hf-token`).

## Smoke 2026-11-17 (g7e.24xlarge spot, us-east-2c, vLLM v0.34.0, FP8 checkpoint 46a050be, TP4)

Logs in `k8s/smoke-run1..3/` and `k8s/smoke-*.log` (local, not committed).

- **Weights:** 78.61 GiB per GPU (of 94.97 GiB usable), text only. Load 136 s cold, 35 s warm (page
  cache); TileLang JIT warmup 18 s with the compile caches on the node (213 s cold on the 2-GPU spike).
- **Picked by vLLM:** attention `FLASHINFER_MLA_SPARSE_SM120` (only candidate, `fp8_ds_mla`), FP8
  MoE `DEEPGEMM` out of AITER, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, DEEPGEMM, TRITON, MARLIN,
  HUMMING, ...
- **Memory, the blocker:**

  | gmu | max_num_seqs | Outcome |
  |---|---|---|
  | 0.95 | default 1024 | refused: at most 494 Mamba (KDA state) blocks |
  | 0.95 | 128 | started; after warmup 5.4 GiB free on the device, 2.1 GiB kept for activations: **0 KV blocks** within 90.22 GiB |
  | 0.95, mnbt 2048 | 128 | 1.88 GiB of KV, below the 2.01 GiB one 262144-token request needs |
  | 0.98, mnbt 4096 | 128 | **CUDA driver OOM during warmup** (12.46 GiB of KV reserved before warmup) |
  | 1.0 | default 1024 / 128 | refused at 823 Mamba blocks / **driver OOM during warmup** (13.07 GiB reserved) |

  v0.34.0's extensible KV cache reserves the estimate before warmup from gmu 0.98 up, and finds
  nothing after warmup at 0.95: no gmu value starts the FP8 checkpoint at TP4 on 96 GB GPUs. Only an
  explicit `--kv-cache-memory-bytes` (~4-5 GiB per GPU) or `--no-enable-extensible-kv-cache` might,
  leaving a few GiB of KV per GPU (~0.9M tokens, ~10 coding sessions at 100k).
- **Why more GPUs help:** the MLA latent KV is replicated on every TP rank (DCP is unavailable, see
  below), so session capacity = free memory on one GPU. TP8 on 8 GPUs halves the weights per GPU
  (~40 GiB) and leaves ~45 GiB per GPU, close to the target's H200 at TP4 (~50 GiB).
- **Not available on SM120:** DCP 2 and 4 ("No valid attention backend found ... use_mla=True,
  use_sparse=True"); MoE `cutlass` ("disabled for this configuration") and `flashinfer_cutlass`
  (does not support the checkpoint's 128x128 block FP8 scales). `triton` starts but needs more
  memory than `deep_gemm` (auto).
- **Harness bug found and fixed:** Apply counted the previous experiment's crashing pod
  (`newest_pod_since`, commit 0a5c1d5).

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
