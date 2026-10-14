# 35-l40s-gemma4-26b-awq-replay-users

**Status:** RUNNING on Akamas 4.1 (`35-L40S-Gemma4-AWQ-Replay-Users`, started 2026-10-13 14:18 UTC).
**Dates:** scaffolded 2026-10-13 from study 34 (Quartuccio's design, left untouched).

## Objective

**The rehearsal of the customer PoC, as close as possible to what will run on site: how many users
can one L40S serve with the customer's Docker Compose setup of Gemma 4 26B-A4B while the typical
request's end-to-end latency stays within 3 s, and which vLLM configuration serves the most?**

- **Their model and baseline:** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a` (W4A16) with
  their compose's flags exactly (vLLM 0.29.0, to which they move): `gpu_memory_utilization` 0.90,
  `max_model_len` 96000, `max_num_seqs` 64, tool calling, `gemma4` reasoning parser, vision tower
  loaded, prefix caching at vLLM's default, **thinking off by default** (a client asks for it per
  request; their replay does not), their two environment variables.
- **Their load:** their `run_benchmark.sh` unchanged (AIPerf 0.11.0, `/v1/responses`,
  `mooncake_trace` payloads sent verbatim, sticky user sessions, think times capped at 60 s, seed
  42) on their dataset template, users ramping in a closed loop.
- **Their goal:** maximize the users with the latency unchanged. **Score = users**
  (`aiperf.aiperf_active_sessions`), **constraint = e2e median over 150 s <= 3 s**
  (`aiperf.aiperf_e2e_request_latency_p50_150s`), both client side, exact from AIPerf's
  per-request records. Kong (in their path) is left out: their configuration is not available.

## What changes from study 34

| Item | Study 34 | Study 35 |
|---|---|---|
| Cluster | `vllm-bench`, node group `llm-serving-l40s-1xl`, namespace `llm-l40s` | **`vllm-bench-gatling`** (freed by Gatling study 2), node groups `llm-serving-l40s` / `-ab` (`gpu=l40s`), namespace **`llm-replay`**; scripts use the toolbox's `gatling` context, Akamas reads Prometheus through its NLB (`infra/README.md`) |
| Thinking | on server-wide (`enable_thinking` true) | **off by default**, as their compose (2026-10-13: the "thinking on" compose was not theirs) |
| Dataset texts | AIPerf's Shakespeare, lengths only | **Italian texts calibrated on their output lengths** (`render_dataset_it.py`, "Load") |
| Goal | vLLM completed req/s, e2e p95 <= 30 s (Prometheus histograms) | **AIPerf users, e2e p50 (150 s) <= 3 s, exact** (CSV File telemetry) |
| Workflow | 4 tasks | **5**: + "AIPerf to CSV" (`k8s/aiperf_csv.sh`) |
| Components | vllm, gpu0, container(s), cluster(s) | **+ `aiperf`** (AIPerf pack 1.0.0) |
| Parameters | 13 | **17**: + `moe_backend`, `long_prefill_token_threshold`, `watermark`, `scheduler_reserve_full_isl`, `stream_interval`; `scheduling_policy` dropped (no priorities in the replay); `linear_backend` {auto, humming} (triton out) |
| Baseline | once + a repeat | **three times** (baseline + two repeats with the values written out) |
| Watchdog | vLLM TTFT p95 > 3 s or ITL p95 > 600 ms | **vLLM e2e p50 (150 s) > 6 s** for 120 s |
| Ramp | 1 -> 100 users over 6000 s | **1 -> 200** over 6000 s (provisional) |

## Stack & versions

- **Akamas:** Studio 4.1.0 (`akamas41.lab.akamas.io`, namespace `akamas-41`, toolbox there).
- **Packs:** vLLM **1.13.0** (`moe_backend`), **AIPerf 1.0.0** (to install:
  `~/workspace/akamas/optimization-packs/aiperf`, gitlab `optimization-packs/aiperf`), GPU 1.4.0,
  Kubernetes 1.9.0. Providers: Prometheus, **CSV File** (installed, READY on 2026-10-13).
- **Workload:** `vllm/vllm-openai:v0.29.0`, StatefulSet `vllm` (pod `vllm-0`), namespace
  `llm-replay`, served as `aiservices_gemma` (their payloads' model name; Kong maps it in
  production). Pod: 3500m CPU request, no limit, 28 GiB.
- **Hardware:** one g6e.xlarge (1x L40S 48 GB, 4 vCPU / 32 GiB) on `vllm-bench-gatling`; AIPerf on
  the `system` node (m6i.2xlarge). `infra/`.
- **Telemetry:** Prometheus (server side: vLLM, GPU, Kubernetes; `akamas/telemetry/prometheus.yaml`)
  and CSV File (client side: AIPerf; `akamas/telemetry/aiperf-csv.yaml`).
- **Model and KV (measured 2026-10-13 on this cluster):** model 16.39 GiB, bf16 KV cache 708,768
  tokens at 0.90; Marlin for the int4 linear layers and experts; TRITON_ATTN; vLLM raises
  `max_num_batched_tokens` to 2496 (one image) when it is not set.

## Parameters tuned

17, with their domains in `akamas/35-L40S-Gemma4-AWQ-Replay-Users.yaml` and the reasons in
`k8s/params.env.template`. From the W4A16 kernel probe (`../../probes/l40s-gemma4-w4a16-kernels/`,
2026-10-09) on this checkpoint and GPU: Marlin is the best or tied kernel everywhere (auto); Humming
stays in the domains (within ~5 %); triton (linear -49 %, MoE -41 % even tuned) is out; fp8 KV
(+18 % on a mixed load) and MTP (+36 % decode) are the large levers, but **with MTP on the prefix
cache made 0 hits in 24 follow-up turns**, and without MTP ~1 turn in 3 misses (hybrid KV cache
manager): on this load (82 % of the requests share text with another user's, conversation history)
that trade-off is the main question. `long_prefill_token_threshold` is new: their prompts reach
~13k tokens (p99) and a long chunk stalls every decode for a step. **Not tuned:** attention backend
(forced), `scheduling_policy` (no priorities), `enforce_eager` (false), `max_model_len`, prefix
caching, the vision tower (their production has a few image requests), fp8 activations
(`VLLM_MARLIN_INPUT_DTYPE`: not in the pack, quality unverified).

**parameterConstraints:** `max_num_batched_tokens >= max_num_seqs`; `spec_method` none <-> `spec_tokens` 0.

## Load

- **Their command:** `k8s/run_benchmark.sh` byte for byte (sha256 checked); the Job gives it its
  `benchmark.env` (USERS, RAMP = DURATION, CAP 60, GRACE 60) and AIPerf 0.11.0. `--concurrency`
  counts sessions: each user sends a request, waits for the whole reply, waits the trace's delay,
  sends the next (closed loop); a finished session hands its slot to the next.
- **Their template, Italian texts** (`k8s/render_dataset_it.py`, corpus from `k8s/build_corpus.py`,
  calibration in `../../probes/gemma4-customer-dataset-it/`): the template has two request shapes.
  *Chat* (54 %): `instructions` (median ~9.5k characters, 892 lengths) + history + a short question
  -> SQuAD-it paragraphs as documents, SQuAD-it questions about them, passages as the history's
  replies. *Task* (46 %): one string -> an Italian news article (Il Post, Fanpage) with a task
  (title, category, keywords, one sentence, paragraph, detailed summary, rewrite, translation of a
  long text). Same placeholder -> same text, so prefixes repeat as in the template; no output cap.
  The Job builds the corpus once on the results volume (Hugging Face downloads, ~1 min).
- **Calibration** (2026-10-13, the compose on this cluster, 900 rendered requests, the customer's
  production numbers from their `usage_stats.png`):

  | Output tokens | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
  |---|---|---|---|---|---|---|---|---|
  | Customer (production) | 13 | 19 | 38 | 100 | 242 | 502 | 2035 | 132 |
  | Rendered | 14 | 21 | 43 | 91 | 224 | 445 | 2518 | 146 |

  Input tokens p50 / p90 / p99: rendered 1569 / 6042 / 12419, their synthetic dataset 1354 / 5933
  / 13186 (their real one 1464 / 8064 / 17732: the template has 10 of their 100 users).
- **Ramp, PROVISIONAL:** users 1 -> 200 over 6000 s (one every 30 s); the watchdog ends the run
  once vLLM's e2e p50 stays above 6 s for 120 s. Experiment 1 checks that the knee falls inside.

## Goal and scoring

- **Why the median:** their outputs span 13 to 2035 tokens (p10-p99). With a perfectly repeatable
  server, a 3-minute window's e2e p95 moves by ~31 % (300 requests) from which replies fall in it,
  the median by ~8 % (simulation on their distribution, 2026-10-13). A request of thousands of
  tokens can never take 3 s; the median is the typical request. TTFT/TPOT would be the clean
  constraint, but the customer measures e2e.
- **Why users, not req/s:** at a given number of users, req/s depends on which sessions are active
  (212-416 requests per session in the template) and the think time dominates (~90 % of a
  session): req/s barely moves between configurations below the knee. The users are what the
  customer wants to maximize.
- **Windowing:** `stability` on `aiperf_active_sessions`, 6 samples (3 min), `when: max`: the last
  valid window before the e2e median passes 3 s.
- **KPIs (8):** users, completed req/s, e2e p50 and p95 (150 s), TTFT p95 (150 s, client), prefix
  cache hit rate, MTP acceptance, KV cache usage.

## Risks

- **Natural outputs do not repeat between runs** (study 34's load determinism test, Quartuccio,
  2026-10-13: their replay twice with the same seed sent 11 % fewer requests and 18 % fewer output
  tokens in run 2; 9 of 227 replies had the same length at temperature 0.1). This study keeps the
  outputs natural on purpose (MTP's acceptance on text after an EOS is not real: 0.41 against
  0.60-0.71 there), so the three baselines measure how much the score moves; the median and a
  users score are the robust choices, but a difference within that spread is not a result.
  Fallback if it is too wide: per-request `max_output_tokens` + `ignore_eos` near each request's
  natural length (study 34's proposal; vLLM 0.29.0's `/v1/responses` honours both).
- **Prefix-cache reuse is guessed** from the template's lengths (same length -> same text):
  production may reuse more or less.
- **The 4-vCPU g6e.xlarge** may bind before the GPU (API server, tokenization of ~2.4k-token
  prompts at several req/s): watch the `container` component's CPU.
- **Capacity:** g6e is often short in us-east-2 (`infra/README.md`).

## Steps

`baseline` (the compose: two values rendered, the rest not), `baseline repeat 1` and `2` (the
compose with vLLM's resolved defaults written out: the noise of the setup, three points),
presets `kv fp8`, `mtp3`, `kv fp8 mtp3`, `kv fp8 long prefill cap`, `kv fp8 large batch`,
`study 32 best`, then `optimize` (AKAMAS, 0 init, 60 experiments, 20 failures).

## Runbook

1. **AIPerf pack 1.0.0 installed** on the 4.1 server (Administrator; `akamas install -f
   optimization-pack <json>` from the pack's `dist/`).
2. **Cluster layer and GPU node:** `infra/README.md` (node group to 1, `provision.sh`); `AlwaysOn`
   tags if a run crosses 17:00 UTC.
3. **Template on the toolbox:** `load/README.md`.
4. **Commit, push, `git pull`** in `/work/vllm-benchmark` on the toolbox.
5. **Create and start:** `akamas/README.md`, "Setup & run".
6. **Experiment 1 checks:** the corpus build and the render line in the Job log; `/v1/responses`
   200s; no reasoning tokens; `aiperf_csv.sh`'s row count; the `aiperf` component's samples
   inside the trial window (CSV time zone, README of the AIPerf pack "Known limits"); the knee
   inside the ramp.

## Offline checks (2026-10-13)

`python3 akamas/check_offline.py` 0 failures (every baseline and preset rendered through the real
`render_statefulset.sh`, metrics against both packs, CSV telemetry included);
`bash k8s/tests/test_*.sh` 7 suites 0 failures (yq v4 needed); `bash k8s/tests/dry_run_replay.sh`
0 failures (the Job's script under a POSIX shell with real AIPerf 0.11.0 against the mock: their
script, the Italian render, 198 requests, no placeholder, no cap, no reasoning); `k8s/aiperf_csv.sh`
on the AIPerf pack's fixture (columns = the telemetry's metrics, stale records refused).

## Running notes

- 2026-10-13 14:18 UTC: created on Akamas 4.1 (every resource accepted; vLLM pack 1.13.0, AIPerf
  pack 1.0.0) and started. `AlwaysOn` set on the ASGs of `llm-serving-l40s-ab` and `system` and
  their instances (to remove at the end). Experiment 1 checked: the compose's flags only, vLLM raised
  `max_num_batched_tokens` to 2496, Marlin, 708,768 KV tokens; the Job built the corpus (16.8 MB, as
  locally) and rendered 240 users (68,352 requests); the watchdog ended the run at ~106 users; the
  CSV reached Akamas inside the trial window (the score recomputed from the CSV matches Akamas's).
- **Noise (three baselines, the compose): 72.5, 72.5, 71.5 users** (+-1.4 %). The points where the
  client-side e2e median passes 3 / 4 / 6 s differ by 2-3 users between runs, completed req/s in
  the scored window by ~3 % (3.23 / 3.32), although natural outputs make every run's load different.
- **Experiment 4, `kv fp8`: 90 users (+24.1 %)**: twice the KV tokens (1,417,301); the baseline's
  KV filled near ~100 users, and long requests waited for memory before that.
- Experiment 5, `mtp3` (running): MTP acceptance 0.65, server-side e2e median ~20 % lower than the
  baseline at ~52 users; prefix-cache hit rate 2.9 % against 7-9 % without MTP (the W4A16 probe's
  finding, cheap here: the template's chat requests change their `instructions` almost every turn,
  so there is little to reuse: 94 of 1544 chat requests extend an earlier one exactly).
- 2026-10-13: Stop AIPerf now keeps the records gzipped (`<epoch>.jsonl.gz`, ~9-12 MB a trial
  against ~38-40 MB): the toolbox's `/work` is a shared 1 GiB volume. The first three trials'
  records were gzipped in place.
- **Overnight 2026-10-13/14:** experiments 6-10 VALID: `kv fp8 mtp3` 103.5 (+42.8 %), `kv fp8 long
  prefill cap` 84.5 (+16.6 %: the 2048-token cap costs against fp8 alone), `kv fp8 large batch` 88.5
  (+22.1 %), `study 32 best` 106.5 (+46.9 %), the optimizer's first 106.5 (+46.9 %). **Experiment
  11 lost to the toolbox's disk:** RunTest succeeded, but Stop AIPerf refused to save the records
  with ~160 MB left on the shared 1 GiB `/work` (study 34's uncompressed records and study 33's
  filled it), so AIPerf to CSV failed too; the experiment was **deleted** (2026-10-14, user's
  call) so the optimizer does not take its configuration as a failure (experiment 12 had already
  been proposed with it in the history). Fix: study 33's records gzipped, study 34's scripts
  compress too, and every parsed record moved off the toolbox to `results/aiperf-records/`
  (local only, not in git; sha256-verified before deletion): `/work` back to 742 MB free.
- **Decided for the next study (2026-10-14): drop `long_prefill_token_threshold` from the
  parameters**, leaving vLLM's default (0, off); at most one preset at 8192 to see whether a high cap
  helps the long tail of their prompts (p99 ~13k, max 95k tokens). This study is the first in the
  repo to tune it (studies 0-17 pinned it at 0 after study 0's partial-prefill incident, studies
  30-31 left it out on short ShareGPT prompts), and every value measured so far loses: the 2048
  preset 84.5 users against `kv fp8` alone 90 (-6 %), experiment 12 (121, with `max_num_seqs` 16)
  46.5 (-36 %); experiment 13 (407) running. On this load (prompts ~1.5k median, ~6k p90, short
  replies) splitting each prefill over more steps only adds per-step cost. The domain is also hard
  to optimize: its "off" is a single point (0) at the edge of [0, 8192], and low values behave the
  opposite way, so the optimizer spends experiments learning that.
- **Decided for the next study (the customer PoC, 2026-10-13): add fp8 activations** (Marlin
  W4A8-fp8, `VLLM_MARLIN_INPUT_DTYPE=fp8`, which applies to the int4 linear layers and experts on SM
  8.9): the only kernel lever with a real gain in the W4A16 probe (+8-11 % prefill and mixed load,
  -2 % decode; this load is prefill-heavy). Not in this study: it is an environment variable, not a
  vLLM pack parameter, and it changes the numerics (only a 40-problem check so far). Before it:
  (1) a vLLM pack parameter rendering the variable (as `tuned_kernel_configs`), (2) an accuracy
  check on this checkpoint (GSM8K + IFEval as study 30's, plus tool calls) with and without it, and
  with fp8 KV (also unchecked on the AWQ checkpoint). int8 activations are excluded (0/40, broken
  output).

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
