# 34-l40s-gemma4-26b-awq-customer-replay

**Status:** STOPPED (2026-10-14 07:23 UTC, by the user, to free the GPU for study 36) after 8 of 69
experiments: 3 baselines and 1 preset valid, 3 presets lost to the toolbox's disk, 1 aborted by the
stop. The optimize step never started. Results below.
**Dates:** scaffolded 2026-10-12 from study 33 (thinking on, Shakespeare texts); redesigned
2026-10-13 as **study 35 with every output capped**, on studies 30-33's node.

## Objective

**Study 35's question (how many users can one L40S serve with the customer's Docker Compose setup
of Gemma 4 26B-A4B while the typical request's end-to-end latency stays within 3 s, and which vLLM
configuration serves the most?) under a load that repeats exactly from trial to trial.**

Study 35 (`../35-l40s-gemma4-26b-awq-replay-users/`, run on the vllm-bench-gatling cluster) keeps
the outputs natural: the model decides each reply's length, so two runs of the same configuration
ask for different work (this study's load determinism test, "Load": 18 % fewer output tokens in a
second run). Here every request asks for **the output length the compose produced for it in a
calibration run** (`max_output_tokens`) **with `ignore_eos`**: the same requests, the same tokens,
in every trial; the only thing that changes between trials is the configuration. Run side by side,
the two studies say whether capping changes the ranking and how much of study 35's spread was the
load's.

Everything else is study 35's: their model and compose flags (thinking off by default), their
`run_benchmark.sh` unchanged on their template rendered with Italian texts, users ramping in a
closed loop, **score = users** (`aiperf.aiperf_active_sessions`) with **e2e median over 150 s <= 3
s** (`aiperf.aiperf_e2e_request_latency_p50_150s`), both client side from AIPerf's records, 17
parameters, three baselines, six presets, 60 AKAMAS experiments.

## What changes from study 35

| Item | Study 35 | Study 34 |
|---|---|---|
| Output length | natural (the model stops at its EOS) | **`max_output_tokens` = the request's calibrated length, `ignore_eos`** (`k8s/caps.py apply` in the Job, after the render; requests the calibration did not reach: their task's median) |
| Calibration | (the probe's, for the texts) | + **`k8s/calibrate_caps.sh`**, once before the study: vLLM with the compose's numerics, every request the ramp can reach sent uncapped, output tokens per request -> `caps.json.gz` on the toolbox |
| Cluster | `vllm-bench-gatling`, node groups `llm-serving-l40s` / `-ab`, namespace `llm-replay`, context `gatling`, Prometheus through an NLB | **`vllm-bench`**, node group **`llm-serving-l40s-1xl`** (studies 30-33's g6e.xlarge, checkpoint cached in `/var/lib/hf-cache-study30`), namespace **`llm-l40s`**, context `default`, Prometheus in-cluster; AIPerf on `system-m8a` |
| Akamas names | `..._35_...`, `35-...` | `..._34_...`, `34-...`; toolbox folders `/work/aiperf-{exports,csv}/34-l40s-gemma4-26b-awq-customer-replay` |

Goal, constraint, windowing, KPIs, the 17 parameters and their constraints, the steps and the
watchdog are study 35's, unchanged (`akamas/34-L40S-Gemma4-AWQ-Replay-Users.yaml`; rationale in
study 35's README, "Parameters tuned" and "Goal and scoring").

## Stack & versions

- **Akamas:** Studio 4.1.0 (`akamas41.lab.akamas.io`, namespace `akamas-41`, toolbox there).
- **Packs (installed on the 4.1 server, checked 2026-10-13):** vLLM **1.13.0**, **AIPerf 1.0.0**,
  GPU 1.4.0, Kubernetes 1.9.0. Providers: Prometheus, CSV File (READY).
- **Workload:** `vllm/vllm-openai:v0.29.0` serving `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev
  `0ef577a` as `aiservices_gemma`, StatefulSet `vllm` (pod `vllm-0`), namespace `llm-l40s`; the
  compose's flags (`max_model_len` 96000, tool calling, `gemma4` reasoning parser, thinking off by
  default, vision tower loaded). Pod: 3500m CPU request, no limit, 28 GiB.
- **Hardware:** one g6e.xlarge (1x L40S 48 GB, 4 vCPU / 32 GiB), node group `llm-serving-l40s-1xl`
  of the shared EKS cluster `vllm-bench` (us-east-2, `infra/`, study 30's); AIPerf 0.11.0 on the
  `system-m8a` node.
- **Telemetry:** Prometheus (server side; `akamas/telemetry/prometheus.yaml`, ServiceMonitor
  `vllm-l40s`, dcgm-exporter) and CSV File (client side, AIPerf; `akamas/telemetry/aiperf-csv.yaml`).

## Load

Study 35's ("Load" in its README): their `run_benchmark.sh` byte for byte (AIPerf 0.11.0,
`/v1/responses`, `mooncake_trace` payloads sent verbatim, `--concurrency` users ramping 1 -> 200
over 6000 s, sticky user sessions, think times capped at 60 s, seed 42); their template rendered
with Italian texts (`k8s/render_dataset_it.py`, corpus from `k8s/build_corpus.py`, task mix
calibrated on their production output lengths: p50 43 against their 38, p90 224 against 242, p99
2518 against 2035); 240 users rendered for 200 (68,352 requests).

**The output caps (this study's change).** AIPerf sends the payloads verbatim, so the caps are
written into them: after the render, the Job runs `k8s/caps.py apply`, which adds
`max_output_tokens` (the request's calibrated length) and `ignore_eos: true` to every payload and
prints a `CAPS:` line. The customer's script is untouched (sha256 still checked); the payloads are
theirs plus these two fields.

- **Where the lengths come from:** `k8s/calibrate_caps.sh`, once, by hand, before the study (~1.5-3
  h of GPU). It starts vLLM with `k8s/calibration.params.env` (the compose's numerics: bf16 KV, no
  MTP, vLLM's kernels; `max_num_seqs` 128 and `max_num_batched_tokens` 8192 instead of 64 / 2496 to
  go faster: batch shapes only), renders the dataset exactly as the Job does (same corpus on the
  results volume, 240 users, seed 42, mix), and `caps.py measure` sends **every request the ramp can
  reach** (user k starts at 30k s, its requests at their delays, latency left out: 47,794 of
  68,352; best effort, it assumes AIPerf opens the k-th slot no earlier than a linear ramp would) uncapped and not streamed, 128 at a time, recording `usage.output_tokens` per
  `<session>/<turn>`. Output: `/work/customer-data/34-l40s-gemma4-26b-awq-customer-replay/caps.json.gz`
  (~200 KB), with each task's median for the requests it did not reach.
- **Why calibrated lengths and not one fixed cap:** the test below capped every reply at 128 tokens:
  the load came out ~3x lighter than the natural one (e2e ~1.2 s against ~5 s), and `ignore_eos`
  made the model write past its EOS, which MTP drafts badly (acceptance 0.41 against 0.60-0.71). A
  cap equal to the length the model chose leaves few tokens after an EOS; the length distribution is
  the calibrated one (the customer's).
- **Guard:** `caps.json.gz` holds the sha256 of the dataset it was measured on; `caps.py apply`
  refuses (exit 2, the trial fails) a render that differs (another template, corpus, mix, seed or
  user count): recalibrate then. The render is deterministic (two renders of 240 users: same
  sha256, 2026-10-13) and the corpus is built once on the results volume and kept.
- **What still varies between trials:** the server's own timing (the closed loop sends a user's
  next request when the reply ends) and, slightly, whether a configuration that numerically differs
  from the compose (fp8 KV, MTP) would have stopped earlier: it now writes past its EOS up to the
  same count. Measured with natural outputs: ~3-5 % on e2e mean and p90 between two runs ("Load
  determinism test", below).

### Load determinism test (2026-10-13, manual, outside Akamas)

**Question:** how much does the same load vary between two runs with the same seed, and what makes
it repeat? AIPerf 0.11.0 source: `--random-seed` already fixes everything AIPerf draws itself
(`common/random_generator.py`, one derived RNG per component: dataset, samplers, intervals, ramps),
and the render is seeded too. What still varies comes from the server: the closed loop (a user's
next request waits for the whole reply), the cut at `--benchmark-duration`, and the reply lengths
(the payloads carry no output cap).

**Setup:** a test pod in `llm-l40s` (`python:3.12-slim`, AIPerf 0.11.0, the Job's node); vLLM
0.29.0 still on study 33's last trial (thinking **off**, MTP 4 tokens, served name
`gemma4-26b-awq-nothink`), not this study's configuration. Their template rendered with
`k8s/render_dataset.py` (20 users, seed 42), kept to a 300 s window (318 requests; user k starts at
3k s, each next request 3 s + its `delay` later), the model in the payloads rewritten to the
served name. Their script's flags (`/v1/responses`, sticky sessions, server token counts,
streaming, seed 42); closed loop `--concurrency 20 --concurrency-ramp-duration 60
--benchmark-duration 300`. Three variants, two runs each:

| Variant | Requests (run 1 / 2) | Same output tokens | Send-time diff p90 / max | e2e mean (ms) | e2e p90 (ms) | MTP acceptance | Running seqs (avg) |
|---|---|---|---|---|---|---|---|
| natural (their payloads) | 257 / 231, 227 shared | 9 of 227 | 4.4 s / 20.6 s | 5111 / 4893 | 8917 / 8858 | 0.60 / 0.71 | 6.3 / 7.7 |
| capped: `max_output_tokens` 128 + `ignore_eos`, closed loop | 388 / 387 | 387 of 387 | 1.3 s / 2.3 s | 1191 / 1174 | 1667 / 1553 | 0.41 / 0.42 | 1.4 / 1.4 |
| capped + `timestamp` per record (`--fixed-schedule`) | 318 / 318 | 318 of 318 | 0.00 s / 0.00 s | 1132 / 1169 | 1525 / 1596 | not read | not read |

(Capped closed-loop runs wrapped the dataset: the 300 s window's sessions ended at ~160 s and
AIPerf's sequential sampler started them again, ~70 requests a run; the full sessions of the
study, ~1 h each, do not wrap.)

**Findings:**
- **Their replay as is does not repeat its load:** run 2 sent 11 % fewer requests and 18 % fewer
  output tokens; 9 replies in 227 had the same length (temperature 0.1: the model stops at
  different points), and MTP acceptance moved from 0.60 to 0.71. The e2e looks stable (~4 %) over
  two different workloads.
- **`max_output_tokens` + `ignore_eos` work on `/v1/responses`** (vLLM 0.29.0; checked by hand
  too: `status: incomplete`, `incomplete_details.reason: max_output_tokens`, exactly N output
  tokens, also at temperature 0.7). `max_output_tokens` alone is only a ceiling (a 10-token reply
  stays 10). Not checked on their 0.25.1.
- **With the cap the requests and their tokens repeat exactly; with `--fixed-schedule` the send
  times repeat too** (0.00 s apart). What is left is the server's own noise: ~3-5 % on e2e mean and
  p90.
- **`ignore_eos` lowers MTP acceptance** (0.41 against 0.60-0.71): the tokens after the EOS are
  hard to draft. The capped runs' lower ITL (~8 ms against ~12.6 ms) comes from the lighter load
  (1.4 sequences running against 6-8), not from MTP.
- **A fixed cap of 128 is not their load:** natural replies here were ~420 tokens on average
  (p50 ~440, p90 ~730), so the capped load was ~3x lighter (e2e ~1.2 s against ~5 s).
- **`--fixed-schedule` is not a fully open loop** (`timing/strategies/fixed_schedule.py`): first
  turns go at their timestamps, a later turn at its timestamp only once the previous reply is
  back. A server slower than the gap between timestamps delays the session again.
- AIPerf refuses `--benchmark-grace-period` without `--benchmark-duration` (so not with a fixed
  schedule alone).

**What the study adopted (2026-10-13, "Load" above):** per-request `max_output_tokens` equal to
the length the compose produced for that request in a calibration run, with `ignore_eos`, so that
few tokens follow an EOS and MTP acceptance stays close to real text's; the closed loop and the
users score kept (no fixed schedule: it would change the question to the latency at a fixed load).
Lengths from the template's history were the first idea, dropped: study 35's Italian render makes
the model's lengths follow the customer's distribution, and half the requests (the task shape) have
no history. `VLLM_BATCH_INVARIANT=1` was ruled out: it changes the system being tuned (slower
kernels the customer will not run) and fixes only the reply contents, not the closed loop or the
duration cut.

## Goal and scoring

Study 35's: maximize `aiperf.aiperf_active_sessions` with
`aiperf.aiperf_e2e_request_latency_p50_150s <= 3000`, `stability` windowing on the users, 6 samples,
`when: max` (the last valid window before the median passes 3 s); 8 KPIs (users, completed req/s,
e2e p50 and p95 over 150 s, client TTFT p95, prefix cache hit rate, MTP acceptance, KV cache usage).
With the caps the window's mix of reply lengths no longer changes between trials, which removes the
part of study 35's p95 noise that came from which replies fell in the window; the median stays the
constraint, for comparability.

## Risks

- **The caps are the compose's lengths:** a configuration whose numerics differ (fp8 KV, MTP,
  another kernel) would have written slightly different replies; it now writes the compose's count,
  past its EOS when it would have stopped earlier. Fine for comparing configurations on the same
  work; MTP's acceptance on those extra tokens is lower than on real text (the test: 0.41 on
  128-token caps), so MTP presets may score a little worse here than in study 35.
- **Coverage of the calibration is best effort:** a request it did not reach (if AIPerf's ramp
  opens slots earlier than a linear one) or that failed in it gets its task's median. The
  guarantee is the fixed file: every trial asks for the same tokens whatever the coverage.
- **Tokens past an EOS and the reasoning parser:** with `ignore_eos` the model may write turn or
  thinking markers after its EOS, which `--reasoning-parser=gemma4` could count as reasoning: a
  non-zero `aiperf_reasoning_token_count_avg` or an odd time to the first output token is not, by
  itself, thinking switched on (study 35's "no reasoning tokens" check does not apply as is).
- **`/work` on the toolbox is shared with study 35** (356 MB free on 2026-10-13, 1 GiB volume):
  each trial of either study saves 25-60 MB of records there, and AIPerf to CSV (critical) fails
  when it fills. Move the records off before experiment ~4 and then every few experiments
  (`akamas/README.md`, "Room on the toolbox").
- **A recalibration is needed** whenever the template, `render_dataset_it.py`, `build_corpus.py`,
  `mix.json`, the seed or `RT_USERS` change (the guard stops the trial otherwise), and if the
  results volume loses the corpus (rebuilt from Hugging Face, it may differ).
- **The 4-vCPU g6e.xlarge** may bind before the GPU (API server, tokenization): watch the
  `container` component's CPU.
- **Prefix-cache reuse is guessed** from the template's lengths, as in study 35.

## Steps

Study 35's: `baseline` (the compose: two values rendered, the rest not), `baseline repeat 1` and `2`
(values written out), presets `kv fp8`, `mtp3`, `kv fp8 mtp3`, `kv fp8 long prefill cap`, `kv fp8
large batch`, `study 32 best`, then `optimize` (AKAMAS, 0 init, 60 experiments, 20 failures). With
the load fixed, the three baselines measure the server's own noise only.

## Runbook

1. GPU node free: no other study on `llm-serving-l40s-1xl`, no AIPerf Job in `llm-l40s`; `AlwaysOn`
   tags on the node and its ASG if a run crosses 17:00 UTC.
2. Template on the toolbox (`load/README.md`).
3. Commit, push, `git pull` in `/work/vllm-benchmark` on the toolbox.
4. **Calibrate:** `bash studies/34-l40s-gemma4-26b-awq-customer-replay/k8s/calibrate_caps.sh` on the
   toolbox, **inside `tmux`** (1.5-3 h; its log: the render line, progress every minute, the
   measured lengths' percentiles, `0 with reasoning tokens`; then `caps: N measured of 47794
   reachable`). Tag the GPU node `AlwaysOn` if it crosses 17:00 UTC. If the session drops, the pod
   keeps going and writes `/benchmarks/caps/caps.json.gz` on the `aiperf-results` volume, then
   sleeps 1 h after `CALIBRATION DONE`: copy it with `kubectl -n llm-l40s cp
   aiperf-calibration:/benchmarks/caps/caps.json.gz <D>/caps.json.gz -c calibration` while it
   sleeps, or later from any pod that mounts the volume.
5. Create and start: `akamas/README.md`, "Setup & run".
6. Experiment 1 checks: the `CAPS:` line in RunTest's log, the `LENGTHS output_sequence_length`
   equal to the caps' distribution, `/v1/responses` 200s, the CSV rows inside the trial window, the
   knee inside the ramp (study 35's checks otherwise).

## Offline checks (2026-10-13)

`python3 akamas/check_offline.py --packs <packs' develop branches>` 0 failures (vLLM 1.13.0, AIPerf
1.0.0); `bash k8s/tests/test_*.sh` 8 suites 0 failures, `test_caps.sh` new (measure on the
reachable set against a non-streaming mock, apply, refusal of another dataset) and `test_run_test.sh`
extended (caps in the ConfigMap, exit 2 without them); `DRY_CORPUS=<corpus> bash
k8s/tests/dry_run_replay.sh` 0 failures (the Job's script with real AIPerf 0.11.0 against the mock:
the Job's render accepted by `caps.py apply`, every request with `max_output_tokens` and
`ignore_eos`); `caps.py apply` on the real 240-user render: 3 s, 1.3 GB. Not run here: the
calibration itself (needs the GPU).

## Running notes

- 2026-10-12: scaffolded from study 33 the evening the customer's script and template arrived;
  thinking on (the user's choice after their email); `smoke/` and the synthetic dry run dropped.
  First draft rewrote their AIPerf command inside the Job; the user asked to use their script
  as is: the Job now wraps `run_benchmark.sh` unchanged.
- 2026-10-13: study 33 stopped by the user; its AIPerf Job was still running (rate ramp to 6
  req/s on `vllm-0`) and was deleted. Load determinism test by hand on the free GPU ("Load
  determinism test" above); the test pod `aiperf-determinism-test` deleted afterwards, its records
  with it (the numbers above are all that was kept).
- 2026-10-13: redesigned as study 35 (Cereda's, pushed the same day: thinking off, Italian
  texts, users under an e2e median SLA, client-side AIPerf metrics) with per-request output caps
  calibrated on the compose plus `ignore_eos` (the user's choice after the determinism test), on
  this cluster and node while study 35 runs on vllm-bench-gatling. The first design's files
  (`render_dataset.py`, its test, the old study and workflow YAML) removed; none had been created
  on Akamas.
- 2026-10-13: calibration (`k8s/calibrate_caps.sh`, 17:09-20:37 UTC, ~3 h 30 on the GPU): 47,717
  of 47,794 reachable requests measured, 77 client-side timeouts, 0 with reasoning tokens; output
  tokens p50 43, p90 177, p99 2314, max 18,947; task medians from 3 (`category`) to 3193
  (`translate`). The study's start (20:38) failed at once in RunTest: the client-side `kubectl
  apply` of the replay ConfigMap stored a copy of it in the `last-applied-configuration`
  annotation, over Kubernetes's 256 KiB limit with `caps.json.gz` (~190 KB, ~255 KB in base64);
  fixed with a server-side apply (`1c07940`), study restarted at 21:03 (Akamas deleted the failed
  experiment). First trial: `CAPS: 68352 requests, 47717 capped at their calibrated length, 20635
  at their task median`, load started 21:09.
- 2026-10-14: experiments 4-6 lost to the toolbox's disk ("Results"); records gzipped from
  experiment 7 on. Stopped by the user at 07:23 UTC during experiment 8 to free the GPU for study
  36; the AIPerf Job and vLLM (StatefulSet scaled to 0) removed by hand.

## Results

Akamas study `34-L40S-Gemma4-AWQ-Replay-Users` (id `a0ff5e97`), 2026-10-13 21:03 -> 2026-10-14
07:23 UTC. Raw data: `results/export.tar.gz` (Akamas export), `results/trials.csv` (one row per
trial: step, score, scored window, main parameters and metrics). Every scored trial: 0 request
errors; scored window = the last 150 s window before the e2e p50 passed 3 s.

| Exp | Step | Users | e2e p50 (150 s) | TTFT p50 | Output tok/s | Output len (avg) | KV usage |
|---|---|---|---|---|---|---|---|
| 1 | baseline (the compose) | **93.5** | 2967 ms | 311 ms | 516 | 120.5 | 0.39 |
| 2 | baseline repeat 1 | **94.5** | 2990 ms | 333 ms | 532 | 121.5 | 0.43 |
| 3 | baseline repeat 2 | **93.5** | 2964 ms | 322 ms | 519 | 120.6 | 0.42 |
| 4 | kv fp8 | lost | | | | | |
| 5 | mtp3 | lost | | | | | |
| 6 | kv fp8 mtp3 | lost | | | | | |
| 7 | kv fp8 long prefill cap (fp8 KV, `max_num_batched_tokens` 8192, `long_prefill_token_threshold` 2048) | **104.5 (+11.4 % on the baselines' mean)** | 2932 ms | 289 ms | 607 | 120.2 | 0.25 |
| 8 | kv fp8 large batch | aborted by the stop | | | | | |

- **Best:** experiment 7, 104.5 users against 93.8 (the three baselines' mean): +11.4 % (Akamas
  shows +11.76 % against experiment 1 alone). One run, no repeat.
- **Baseline noise: 93.5 / 94.5 / 93.5 users, 1.1 % between the extremes** (study 35, natural
  outputs: 72.5 / 72.5 / 71.5, 1.4 %). The mean output length in the scored window stays within
  1 % (120.2-121.5 tokens) across the four valid trials, against 104.9-113.7 (8 %) in study 35's
  three baselines: the caps fix the work per request, as designed.
- **Experiments 4-6 lost (records, not runs):** RunTest succeeded in each (the full ramp ran);
  Stop AIPerf then refused to copy the records because the copy would have left less than 150 MB
  free on the toolbox's shared 1 GiB `/work` (this study saved them uncompressed, ~40-60 MB a
  trial, alongside studies 33 and 35's), so AIPerf to CSV found nothing and the trial failed. Study
  35 lost its experiment 11 the same way the same night. The AIPerf PVC keeps only the last run,
  so these records cannot be recovered. Fixed in `34f03ce` (records gzipped, as study 35's
  `8d595bb`); experiment 7 saved its records (16 MB gzipped) and scored normally.
- **Not deleted:** experiments 4-6 stay in Akamas as ERROR (the study was stopped before the
  optimizer could use them).
- **Side by side with study 35 (same steps, natural outputs, vllm-bench-gatling):**

  | Step | Study 34 (capped) | Study 35 (natural) |
  |---|---|---|
  | baselines (mean) | 93.8 | 72.2 |
  | kv fp8 long prefill cap | 104.5 (+11.4 %) | 84.5 (+17.1 %) |

  Study 34 serves ~30 % more users at the same constraint, with the same input lengths (~2.2-2.3k
  tokens average in the window) and similar output lengths (120 against 105-114 tokens). Two
  things differ between the studies, not one: the caps and the node (another L40S, on another
  cluster). Study 36 (study 35's load, uncapped, on this study's node `llm-serving-l40s-1xl`)
  separates them: its baseline near 94 users would put the gap on the node, near 72 on the caps.

## Conclusions

- **The capped load does what it was built for:** identical work per trial (output length within
  1 % across trials) and baselines within 1.1 %. But study 35's natural load is barely noisier at
  the score (1.4 %): at this goal (users at the knee of a ramp) the natural load's variation
  averages out, so capping buys little repeatability here. It matters more for metrics read at a
  fixed load (tokens/s, latency at N users), where the determinism test saw 18 % differences.
- **Too few experiments for the main question** (does capping change the ranking of
  configurations?): one preset valid. On that one the ranking agrees with study 35 (fp8 KV with a
  2048-token long-prefill cap beats the baseline, by +11 % here and +17 % there); the levers that
  mattered most in study 35 (fp8 KV alone +24 %, fp8 KV + MTP +43 %) were the three experiments
  lost. MTP under capped outputs is still unmeasured: the determinism test saw lower acceptance on
  tokens generated past the EOS (0.41 against 0.60-0.71), a bias against MTP this design carries.
- **The absolute gap with study 35 (+30 % users) is not explained yet:** caps or node. Read study
  36's baseline before using either study's absolute numbers for the customer.
- **Operational:** a shared 1 GiB toolbox volume cannot hold several studies' uncompressed
  per-request records overnight; Stop AIPerf failing (critical: false) still fails the trial
  through AIPerf to CSV, so a full disk silently turns valid runs into failed experiments.
  Records are now gzipped (~4x smaller) in studies 33-36; moving them off the toolbox remains a
  manual step.
- **Next:** if the ranking question still matters, rerun the capped presets (`kv fp8`, `mtp3`,
  `kv fp8 mtp3`) as a short preset-only study on the same node, after study 36 and with study
  36's baseline as the uncapped reference on the same hardware.
