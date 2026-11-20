# load/ — study 54's TraceLab pools

**Source:** TraceLab (SyFI Lab, University of Washington, <https://tracelab.cs.washington.edu>,
<https://github.com/uw-syfi/TraceLab>), release **v0.0.2**, `syfi_coding_trace.jsonl.gz` (96 MB,
8,058 Claude Code / Codex sessions, 665,453 rounds; token counts and timings, no text). Dataset
licence **CC BY 4.0**: credit TraceLab and link the site. Never committed: `prepare_load.sh` downloads
it (sha256 in `../k8s/workload.env`).

**Files**

- `prepare_tracelab.py` — the spike's rewrite (`docs/spikes/2026-11-20-tracelab-replay/`): history
  continuity (`prefix_tokens` = the history the agent re-sent, unless the context shrank below 70 %),
  sessions end before their first round above 250,000 input tokens.
- `prepare_pool.py` — the pools: horizon cut (rounds reachable within 80 min of replay, pauses capped
  at 1,200 s), context cut (input x 1.003 + 100 + output <= 262,144), K = 8,000 subtracted from every
  round at write time (sessions with a kept round under K + 200 dropped: 444 of 7,433), a seed-54
  shuffle, two disjoint pools of 600 sessions (one per client).
- `sys8k.txt` — the shared system prompt: 8,000 GLM tokens (47,942 characters) of filler text, one
  identical system message in every session (`aiperf --system-prompt-file`).
- `prepare_load.sh [OUT_DIR]` — downloads, builds, and checks the pools against the calibration's
  (sha256 of the decompressed pools, md5 of the prompt). `OUT_DIR` defaults to `/work/study54-load` on
  the toolbox. Takes ~30 s.
- `test_prepare_pool.sh` — local test on a synthetic fixture.

| Pool | Sessions | Rounds | Recorded input tokens | Smallest written input |
|---|---|---|---|---|
| `pool_0.jsonl.gz` (client 0) | 600 | 20,788 | 1.90 B | 343 |
| `pool_1.jsonl.gz` (client 1) | 600 | 21,795 | 1.94 B | 257 |

AIPerf 0.13's loader splits subagent chains into extra conversations (605 / 615 per pool); the CSV
counts those as sessions too.
