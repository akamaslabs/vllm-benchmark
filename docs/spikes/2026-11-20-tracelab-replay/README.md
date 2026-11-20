# Spike 2026-11-20 — TraceLab coding-agent traces as study 54's load (no GPU)

Question: can study 54 (GLM-5.3-Flash, maximize the coding sessions held with CPU KV offload)
replay real coding-agent sessions with AIPerf, ramping the number of users in one run?

Why not study 53's synthetic shape: there each session opens with ~100k uncached tokens (~11 s
of prefill on the g7e rig). In a continuous ramp to ~200 sessions inside 90 minutes the GPU spends
25-60 % of its time on these cold prefills, and resumed turns queue behind them: the TTFT SLO would
break at 20-30 sessions in every configuration, offload or not (vLLM serves a running long prefill
before a waiting request). Real coding sessions start small and grow turn by turn.

## Dataset

TraceLab (SyFI Lab, University of Washington, <https://tracelab.cs.washington.edu>,
<https://github.com/uw-syfi/TraceLab>), release **v0.0.2** (2026-07-24),
`syfi_coding_trace.jsonl.gz` (96 MB, 665,453 rounds), dataset licence **CC BY 4.0** (credit
TraceLab and link the site; the code is Apache-2.0). Real Claude Code and Codex sessions of
developers, sanitized: token counts, timings and tool events, no text. Not committed here.

Shape (`analyze_tracelab.py`):

| | p50 | p75 | p90 | p99 |
|---|---|---|---|---|
| sessions: 8,058 (5,319 Claude, 2,739 Codex); rounds per session | 16 | 43 | 146 | 1,138 |
| first-round input (tokens) | 17,719 | 24,921 | 36,052 | 169,170 |
| max context per session (tokens) | 68,503 | 125,858 | 227,654 | 522,815 |
| output per round (tokens) | 249 | 583 | 1,332 | 5,542 |
| gap before a tool-triggered round (s) | 1 | 8 | 24 | 218 |
| gap before a developer-triggered round (s) | 122 | 319 | 1,058 | 44,712 |

14.1 % of rounds exceed 262,144 input tokens (GLM-5.3-Flash's max model length).

## Finding: the recorded prefix carries the original provider's cache evictions

AIPerf's `tracelab` loader (AIPerf 0.13.0) keeps the first `prefix_tokens // block` block ids of
the previous round and mints fresh ids for the rest. `prefix_tokens` is what the original
provider's cache served, so after a pause it reflects that provider's eviction (`prefix_vs_gap.py`;
recorded prefix / history the agent re-sent):

| pause before the round | Claude | Codex |
|---|---|---|
| < 5 min | 98.8-99.3 % | 93.5-98.4 % |
| 5-10 min | 47.0 % | 86.1 % |
| 10-30 min | 58.2 % | 73.4 % |
| 30-60 min | 65.6 % | 46.4 % |
| > 1 h | 4.7 % | 14.1 % |

Replayed as is, those rounds reach the server with new content and miss whatever the server kept:
exactly the rounds a CPU KV offload is meant to serve. `prepare_tracelab.py` rewrites
`prefix_tokens` as the history re-sent, `min(input_i, input_{i-1} + output_{i-1})`, except where the
context shrank below 70 % of the previous round (compaction / reset); it also ends each session
before its first round above `--max-input` (250,000) and samples N sessions with a fixed seed.
After the rewrite 4.3 % of rounds append more than 10k new tokens, 0.25 % more than 40k (~4 s of
prefill on the g7e rig): a 5 s TTFT p95 is reachable when the cache holds.

## Replay with AIPerf 0.13.0 against a stub

`stub_trace.py` (OpenAI chat stub, logs the session header and the share of the session's previous
prompt that is an exact prefix of the new one); `run_trace.sh` runs
`aiperf profile --custom-dataset-type tracelab --no-fixed-schedule --concurrency N
--concurrency-ramp-duration R --inter-turn-delay-cap-seconds C --dataset-sampling-strategy shuffle`.
AIPerf 0.11.0 has no `tracelab` loader; 0.13.0 was installed in a scratch venv.

- **60 sessions, concurrency 30, ramp 60 s, delays capped at 20 s, 150 s:** 1,348 requests, 0
  errors. Users enter gradually (13 in the first 15 s); finished sessions are replaced by new ones
  (83 sessions seen). Each round re-sends the previous prompt: exact prefix median 99.9 %, p5
  99.7 %, 12 of 1,265 rounds under 90 % (compactions). Output lengths follow the trace
  (max_tokens median 212). Every request carries `X-Session-Affinity`.
- **600 sessions (3.3 billion input tokens summed over rounds), concurrency 100:** 3,938 requests,
  213 sessions consumed in 150 s. Loading (~80 s) is the memory peak: AIPerf materializes every
  round's prompt into a memory-mapped `dataset.dat` shared by its workers; one process reached
  5.1 GB (9.3 GB with `AIPERF_DATASET_WEKA_PARALLEL_WORKERS=2`), the RSS summed over processes
  25-62 GB (shared pages counted once per process). During the run ~2.5-3.5 GB.
- `AGENTIC_REPLAY` (warm start mid-trajectory) is tied to the InferenceX AgentX scenario and the
  SemiAnalysis Claude Code traces, not to `tracelab`.

## Open for study 54

- Client memory on the real AIPerf pods (Linux, m8a node 32 GiB, two clients): size the session
  pool per client and the pod limits; `AIPERF_DATASET_MMAP_CACHE_ENABLED` reuses the built dataset
  across experiments.
- Developer pauses: p90 18 min, p99 12 h; the cap (`--inter-turn-delay-cap-seconds`) decides how
  long idle sessions must stay cached inside a 90-minute experiment.
- Disjoint session pools for the two clients (a session replayed twice shares its prefix).
- The tokenizer: GLM-5.3-Flash's own, so the replayed token counts match on the server.
