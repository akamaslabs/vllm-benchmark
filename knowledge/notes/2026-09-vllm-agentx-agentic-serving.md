# vLLM x AgentX: Optimizing for Real-World Agentic Serving

**Source:** [vLLM x AgentX: Optimizing for Real-World Agentic Serving](https://vllm-project.github.io/2026/09/08/vllm-agentx.html)
(vLLM Team and Inferact, 2026-09-08) — vLLM's engineering write-up for SemiAnalysis'
AgentX benchmark (InferenceX v3), results at <https://inferencex.semianalysis.com>.
**Date distilled:** 2026-09-14

## Problem addressed

Agentic coding traffic looks nothing like chat: AgentX sessions have a median of 43
turns, 142K median input tokens, 444 median output tokens, a >96% prefix-cache hit rate
and 44% of sessions spawn subagents. Every turn re-sends the whole context plus a short
new tool result, so the server must (1) keep or quickly restore the session's KV cache
between turns, (2) execute very long contexts under a tight per-user interactivity SLO
(P90 above 50 tokens/s per user), and (3) size prefill vs. decode capacity for a
workload whose prefill/decode ratio shifts with concurrency and cache hit rate. The post
describes the scheduler, parallelism, KV-management, routing and kernel changes vLLM
made to lead the public leaderboard, mostly on DeepSeek V4 Pro, Kimi K3 and MiniMax M3
on B300/GB300 NVL72 systems.

## Levers / parameters touched

- **Scheduler**: `--long-prefill-token-threshold` (cap on prompt tokens a single request
  may schedule per engine step; used at 512) and `--prefill-schedule-interval` (admit
  prefill only every N engine steps, aligned across data-parallel ranks; used at 4 on a
  DEP8 group). Default chunked-prefill scheduling is FIFO.
- **Parallelism per model architecture**: TP vs. DCP (decode context parallel) vs. PCP
  (prefill context parallel) vs. DEP (wide expert + data parallel) vs. PP/CPP; sizes
  TP8, DCP4/8, DEP8/16, PCP8 appear.
- **KV cache management**: hybrid KV cache manager (one page size, shared block pool),
  packed KV layout for DeepSeek V4 (~10% KV memory saved with FP4 indexer), Mooncake
  Store hierarchical offloading (`standalone-store` mode, CPU + disk tiers), session-
  aware retention checkpoints for hybrid (sliding-window/Mamba) models.
- **Prefill/decode disaggregation rate-matching**: two-phase method — profile
  prefill-only and decode-only saturation (max req/s per parallelism and GPU count),
  derive the P/D ratio, then sweep concurrency on the combined deployment.
- **Routing**: session-sticky routing vs. load-balancing policies (queue depth, running
  tokens, KV utilization) in Dynamo / llm-d.
- Kernel work (indexer, top-k, MoE fusions) — not tunable, listed for completeness.

## Key results

- **Per-step prefill cap breaks head-of-line blocking**: with `--long-prefill-token-
  threshold 512`, DeepSeek V4 Pro on B300 gains up to **+93% total tokens per GPU-second
  (TPGS)** and **~2.3x better P90 interactivity**, because short cached turns join every
  step instead of waiting behind one 100K+ fresh prefill. Cost: higher TTFT for the long
  request itself — "TTFT-sensitive deployments should use a larger threshold".
- **Prefill cadence alignment under DEP**: MoE all-to-all forces DP ranks into lockstep,
  so a prefill on any rank slows the whole group; `--prefill-schedule-interval 4`
  coalesces prefills onto the same steps (qualitative, Figure 10, no gain figure).
- **Parallelism follows architecture**: Kimi K3 (MLA + Kimi Delta Attention) — DCP8 beats
  TP8 on P50 TPOT and scales to higher concurrency; DEP16 beats DCP8 on NVL72 once
  per-rank batch > 3. DeepSeek V4 — PCP8 gives a **2.65x prefill speedup vs TP8 on a 32K
  prompt** but replicates decode state, so it fits dedicated prefill workers; DEP is the
  default. PP/CPP suit cold compute-heavy prefills, not warm prefix-heavy turns.
- **Sticky routing beat every load-balancing policy** on AgentX: inter-turn gaps are
  short enough that the prefix is still resident; moving a session forces KV retrieval
  whose prefetched blocks eat the destination GPU's KV capacity.
- Leaderboard configs (all P90 interactivity > 50 tok/s/user): DSV4 Pro 1.6T on 12
  GB300 at concurrency 256 — 83K TPGS (up to 130K elsewhere on its Pareto curve);
  MiniMax M3 428B on 2 B300 at concurrency 24 — 70K TPGS; Kimi K3 2.8T on 16 GB300 at
  concurrency 48 — 11.8K TPGS. Metrics used throughout: TPGS (input + output + cached
  tokens per GPU-second), P90 per-user tokens/s, P50 TPOT, TTFT, prefix-cache hit rate.
- No vLLM version is named; PR numbers run up to #53152. Source check (2026-09-14):
  `--prefill-schedule-interval` and `--watermark` exist from **v0.24.0**, the queued-
  request/token admission caps from **v0.29.0**, and **v0.28.0 removed** `--max-num-
  partial-prefills`/`--max-long-partial-prefills` while keeping `--long-prefill-token-
  threshold` as a pure per-step cap (must be <= `max_model_len`, else `ValueError`).

## Implications for vLLM/k8s tuning

- The single most transferable lever is `long_prefill_token_threshold`: it is a
  goodput/interactivity vs. TTFT trade-off, exactly the shape of this repo's goodput goal
  (throughput under a P95 TTFT/ITL SLA). On a dense 7B model on L4/A10G the absolute
  numbers will differ, but the mechanism (HOL blocking of decodes by a long prefill
  inside a shared `max_num_batched_tokens` budget) is model-agnostic. It only bites when
  prompts are long relative to `max_num_batched_tokens` — ShareGPT-style short prompts
  will show little; a long-context/multi-turn load shape is needed to see it.
- `prefill_schedule_interval` is meaningful **only with `data_parallel_size > 1`** (studies
  #6/#8/#9 territory), and the post's rationale (MoE all-to-all lockstep) is weaker for
  dense models: expect a smaller effect than on DeepSeek/Kimi.
- The post's headline metric, TPGS, counts cached tokens; this repo's
  `prefill_token_throughput + decode_token_throughput` goal does not, so a prefix-heavy
  load shape would under-report relative to the leaderboard. Per-GPU normalisation is the
  same idea as study #9's `active_gpus` division.
- Parallelism-per-architecture results (DCP/PCP/DEP) are for MLA / MoE trillion-parameter
  models on NVL72 fabrics and do not transfer to the dense single-node models used here.
- P/D disaggregation and sticky routing are deployment-topology / router choices made
  before a study starts (see `ROADMAP.md` Q6 on llm-d) — N/A for single-replica studies.
- Load generation: the AgentX harness (github SemiAnalysisAI/agentx-harness) replays real
  1M-context agentic traces — a fourth load-shape option next to GuideLLM/AIPerf/
  inference-perf if a future study targets agentic traffic (`ROADMAP.md` Q2).

## Which Akamas parameters to explore

- **Already modeled**: `vLLM.long_prefill_token_threshold` (integer `[0, 8192]`, default
  0; add the constraint `<= vLLM.max_model_len`), `vLLM.max_num_batched_tokens`,
  `vLLM.max_num_seqs`, `vLLM.data_parallel_size`, `vLLM.tensor_parallel_size`,
  `vLLM.decode_context_parallel_size`, `vLLM.prefill_context_parallel_size`,
  `vLLM.prefix_cache_hit_rate` (metric).
- **Added to the vLLM pack 1.9.0 from this source** (MR
  <https://gitlab.com/akamas/optimization-packs/vllm/-/merge_requests/5>, pending review):
  `vLLM.prefill_schedule_interval`, `vLLM.watermark`, `vLLM.enable_chunked_prefill`,
  `vLLM.enable_prefix_caching`, `vLLM.stream_interval`, `vLLM.scheduler_reserve_full_isl`,
  `vLLM.max_num_queued_reqs`/`max_num_queued_tokens` (0 = drop-the-flag sentinel); metrics
  `time_to_first_token_p90`, `inter_token_latency_p90`, `time_per_output_token_{p95,avg}`,
  `request_queue_time_{p95,avg}` (the direct HOL-blocking signal), `request_prefill_time_p95`,
  `request_decode_time_p95`, `total_token_throughput` (TPGS numerator). The pack README
  lists the `parameterConstraints` to copy into a study.
- **Requires vLLM >= 0.24.0 (0.29.0 for the admission caps)** — this repo's studies pin
  `vllm/vllm-openai:v0.22.0`; see `ROADMAP.md` Q3 for the upgrade note and the two flags a
  0.28+ deploy script must drop.
- **Not modeled, deliberately**: KV offloading tiers (Mooncake `standalone-store`),
  session-aware retention policies, P/D ratio, router policy — deployment/topology
  choices, not per-instance vLLM flags.
