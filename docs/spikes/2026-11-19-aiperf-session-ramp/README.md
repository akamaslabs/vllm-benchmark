# Spike 2026-11-19 — AIPerf session ramp for study 54 (no GPU)

Question: can one AIPerf 0.11.0 run per experiment open coding sessions gradually and keep them open
through their think time, so that a single run ramps the number of open sessions (study 54: maximize
the sessions held, KV cache and CPU offload saturated)? Study 52's bisection rebuilds the sessions at
every level: at ~11 s of prefill per 100k-token session, one level of 160 sessions costs ~30 min.

## Setup

- `stub_server.py`: a local OpenAI chat stub (fixed TTFT and reply) that logs each request with a
  session key (hash of the session's first user message) and its turn index.
- `run_ramp.sh`: AIPerf in session-concurrency mode, scaled down (seconds instead of minutes):
  `--concurrency N --concurrency-ramp-duration R --conversation-turn-delay-mean D` with sessions
  long enough never to end (`--conversation-turn-mean 200`), the study 52/53 prompt shape
  (shared system prompt + per-session context).
- `analyze.py`: sessions opened per 10 s, gap between a session's turns.
- Local AIPerf 0.11.0, tokenizer `Qwen/Qwen3-0.6B` from the local HF cache (`HF_HUB_OFFLINE=1`).

## Results

- **One ramp (N 30, R 90 s, D 20 s, 150 s):** sessions open linearly, 3-4 per 10 s, 30 at 90 s,
  then **no new session**: a session holds its concurrency slot through its think time until its
  last turn (as the docstring of `ConcurrencyManager.release_session_slot` says). Turn gaps 20.1 s
  (min 20.1, max 20.6): the delay is counted from the end of the previous reply. Requests settle at
  N / D per second (15 per 10 s).
- **Two slopes (two AIPerf processes on one server, seeds 11 and 12):** A ramps 1 → 10 over 60 s,
  B starts at 60 s and ramps 1 → 20 over 45 s: 9 sessions open at 60 s, 30 at ~110 s, 30 distinct
  sessions in total (different seeds give different sessions, no prefix shared by accident).

Go: the study 54 load is one AIPerf run (or two staggered ones, slow then fast) per client, with
`--concurrency` = the ramp's top, `--concurrency-ramp-duration` = its length, and the think time as
`--conversation-turn-delay-mean`. Open sessions at time t = the ramp's value at t.

## Scoring in Akamas (found, not built)

Studies 36 and 38 already scored "users held" this way: the AIPerf component type (AIPerf pack
1.0.0, metric `aiperf_active_sessions`) fed by a **CSV File** telemetry instance that a workflow task
writes from AIPerf's export, goal `maximize aiperf.aiperf_active_sessions`, `stability` windowing
`when: max`, and a client-side latency constraint over 150 s
(`studies/38-l40s-gemma4-26b-awq-replay-users-v3/akamas/`). The CSV File provider is READY on
akamas-41 (checked 2026-11-19). Open for study 54: the SLO is on resumed turns only (a cold
100k-token first turn never meets 5 s on these rigs), which the pack's TTFT metrics do not separate.
