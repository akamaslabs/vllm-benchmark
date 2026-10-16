# 36-l40s-gemma4-26b-awq-replay-users-v2

**Status:** STOPPED (2026-10-15 ~09:00 UTC, by the user, after 20 experiments; exp. 21 aborted). Continued as study 38 on Gatling.
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

- 2026-10-14: vLLM pack 1.14.0 (`wna16_activation_dtype`) merged into `develop` of
  `optimization-packs/vllm` (MR !11; its CI's `deploy` job fails on the provisioner registry's
  credentials, as MR !9's did, the build and the pack validation pass) and installed on the 4.1
  server; study 35 kept running. Created on Akamas 4.1 (every resource accepted) and started 07:45
  UTC on the vllm-bench L40S (study 34 FINISHED, its StatefulSet already at 0).
- 2026-10-14: baselines 68.5 and 71.5 users (study 35 on Gatling: 72.5 / 72.5 / 71.5); `kv fp8 mtp3`
  104.5 and `study 32 best` 104.5 (study 35: 103.5, 106.5).
- **Load-test issue (2026-10-15): the score is inflated by the load's periodic valleys** (details
  and data in study 35's running notes, same date). The 10-session template replayed as k mod 10
  under a deterministic ramp makes the offered load drop 5-10 % at ~102 and ~117 users in every
  run, and the `when: max` windowing scores the valid window with the most users anywhere in the
  ramp, also after the e2e median has passed 3 s. Experiment 5 (`study 35 best`) scored 119.5 users
  while over 3 s from ~93 to 114 users: ~93 on the first breach. Re-scored offline on the first
  breach: baselines 60.5 / 71.5, `kv fp8 mtp3` 89.5, `study 32 best` 104.5, `study 35 best` 92.5;
  with fp8 activations 89.5 / 93.5 / 93.5 (no clear gain); optimizer experiments 9-18 60-95 (still
  exploring the domain's edges: `max_num_seqs` 16-30 in four, `gpu_memory_utilization` 0.80 in
  four). Read this study's results on the first-breach score.
- 2026-10-15: `render_statefulset.sh` fixed as study 35's (a tiny `watermark` written in
  scientific notation failed Apply config there, experiment 23); no experiment of this study hit it.
- **Stopped 2026-10-15 ~09:00 UTC** with study 35 (experiment 21 aborted, AIPerf Job deleted,
  StatefulSet at 0; the vllm-bench L40S left to Quartuccio): continued as study 38 on Gatling.

## Results

Raw data: `results/export.tar.gz`, `results/trials.csv` (as study 35's: Akamas score, recomputed
score, first-breach score; recomputed = Akamas on all 20), `results/aiperf-records/` (local only).
Read on the first-breach score (study 35's results).

| Experiment | Akamas | First breach |
|---|---|---|
| baseline x2 (1-2) | 68.5 / 71.5 | 60.5 / 71.5 |
| `kv fp8 mtp3` (3) | 104.5 | 89.5 |
| `study 32 best` (4) | 104.5 | **104.5 (+58 % on 66)** |
| `study 35 best` (5) | 119.5 | 92.5 |
| `kv fp8 act fp8` (6) | 89.5 | 89.5 |
| `study 32 best act fp8` (7) | 109.5 | 93.5 |
| `study 35 best act fp8` (8) | 103.5 | 93.5 |
| optimizer 9-20 | 60.5-104.5 | 60.5-95.5 (best: 11, 95.5; 19, 93.5) |

- Exp. 5's 119.5 came from a valley at 117-122 users after ~20 users over 3 s.
- The optimizer spent 4 of its 12 experiments at `max_num_seqs` 16-30 (60-77 users) and 4 at
  `gpu_memory_utilization` 0.80.

## Conclusions

- **fp8 activations (W4A8-FP8 via `wna16_activation_dtype`) gave no visible gain**: 89-94 first
  breach with them against 90-104 without, in the same presets.
- The node's level (baselines 66 on average) was close to study 35's on Gatling (72); the same
  presets ranked differently on the two nodes (`study 35 best` 106.5 there, 92.5 here), in line
  with the +-7 % spread of one run.
- 12 optimizer experiments were not enough to leave the domain's edges; the study was stopped for
  study 38 rather than for convergence.
