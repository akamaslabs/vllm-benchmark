# 36-l40s-gemma4-26b-awq-replay-users-v2

**Status:** DESIGNED (2026-10-14). Needs the vLLM pack 1.14.0 installed on the 4.1 server
(`wna16_activation_dtype`; built as `/tmp/vLLM_1-14-0.json` on the toolbox, branch
`feature/wna16-activation-dtype` of `optimization-packs/vllm`).
**Dates:** scaffolded 2026-10-14 as a copy of study 35, on the GPU freed by study 34.

## Objective

**Study 35 with its parameters fixed**, on the second L40S: how many users can one L40S serve with
the customer's Docker Compose setup of Gemma 4 26B-A4B while the typical request's e2e latency stays
within 3 s, once the search drops the lever that only lost and adds the one kernel lever that
gained? Everything else is study 35's (`../35-l40s-gemma4-26b-awq-replay-users/README.md`): their
checkpoint and compose as baseline (cyankiwi AWQ W4A16 rev 0ef577a, vLLM 0.29.0, thinking off by
default), their AIPerf replay unchanged on their template rendered with Italian texts calibrated on
their output lengths, users 1 -> 200 over 6000 s, **score = users** (`aiperf.aiperf_active_sessions`),
**constraint = client-side e2e p50 over 150 s <= 3 s**, AIPerf records -> CSV File telemetry.

## What changes from study 35

| Item | Study 35 | Study 36 |
|---|---|---|
| Cluster / node | vllm-bench-gatling, `llm-serving-l40s(-ab)`, namespace `llm-replay` | **vllm-bench** (Akamas's own), `llm-serving-l40s-1xl`, namespace **`llm-l40s`** (studies 30-34's), in-cluster Prometheus, `default` context, AIPerf on `system-m8a` (`infra/README.md`) |
| `long_prefill_token_threshold` | tuned [0, 8192] | **fixed off** (vLLM's default): every value lost in study 35 (2048 -6 % against fp8 alone, 121 -36 %, 407 below the MTP configurations) and its 0 = off at the domain's edge is hard to optimize |
| fp8 activations | not tuned | **`wna16_activation_dtype` {auto, fp8}** (vLLM pack 1.14.0): Marlin W4A8-FP8 and Humming fp8 input (`VLLM_MARLIN_INPUT_DTYPE`, `VLLM_HUMMING_INPUT_QUANT_CONFIG`), +8-11 % prefill and mixed load in the W4A16 probe; int8 out (Marlin int8 broke the output) |
| Kernels | `linear_backend`, `moe_backend` {auto, humming} | unchanged |
| Baseline | three runs (noise +-1.4 %) | **two** (this node's level; the noise is known) |
| Presets | generic | **study 35's best** (`kv fp8 mtp3` 103.5, `study 32 best` 106.5, the optimizer's exp. 10 106.5 users) and the same three-way set with fp8 activations |

16 tuned parameters: `gpu_memory_utilization` [0.80, 0.94], `max_num_seqs` [16, 512],
`max_num_batched_tokens` [2496, 16384], `kv_cache_dtype` {auto, fp8}, `performance_mode`,
`optimization_level` [1, 3], `async_scheduling`, `max_cudagraph_capture_size` [16, 512], `block_size`,
`linear_backend` {auto, humming}, `moe_backend` {auto, humming}, `spec_method` {none, mtp},
`spec_tokens` [0, 4], `watermark` [0, 0.10], `scheduler_reserve_full_isl`, `stream_interval` [1, 16],
`wna16_activation_dtype` {auto, fp8}. Constraints: batched >= seqs; MTP none <-> 0.

## Steps

`baseline` (the compose), `baseline repeat`, presets `kv fp8 mtp3`, `study 32 best`, `study 35 best`
(study 35 exp. 10: fp8 KV, MTP K=1, 179 seqs, 2496 batched, Humming linear, gmu 0.90, block 32),
`kv fp8 act fp8`, `study 32 best act fp8`, `study 35 best act fp8`, then `optimize` (AKAMAS, 0 init,
60 experiments, 20 failures).

## Risks

- **fp8 activations change the numerics** and their accuracy on this checkpoint is checked only on
  40 arithmetic problems (40/40): before proposing them to the customer, GSM8K + IFEval + tool calls
  (study 30's method). The same holds for fp8 KV on this checkpoint.
- Study 35's risks (natural outputs vary between runs, guessed prefix reuse, the 4-vCPU node).
- **Toolbox disk:** records are kept gzipped (~10 MB a trial); move the parsed ones off the toolbox
  regularly (study 35's running notes, 2026-10-14).

## Runbook

1. vLLM pack 1.14.0 installed (`akamas install -f optimization-pack /tmp/vLLM_1-14-0.json` on the
   toolbox, Administrator); `akamas describe optimization-pack vLLM` -> 1.14.0.
2. `infra/provision.sh` (idempotent; the namespace and its objects exist from study 34); no other vLLM
   running in `llm-l40s` (study 34 FINISHED).
3. The template on the toolbox: `/work/customer-data/36-l40s-gemma4-26b-awq-replay-users-v2/` (`load/README.md`).
4. Commit, push, `git pull` on the toolbox; create and start (`akamas/README.md`).

## Offline checks (2026-10-14)

`python3 akamas/check_offline.py` 0 failures (vLLM pack 1.14.0 and AIPerf 1.0.0 checkouts; every
baseline and preset rendered); `bash k8s/tests/test_*.sh` 7 suites 0 failures (the fp8 environment
rendered, `auto` / empty render none, other values refused); the StatefulSet with fp8 activations
accepted by the vllm-bench API server (dry run).

## Running notes

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
