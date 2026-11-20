# Study 54 — GLM-5.3-Flash: coding users held per replica, CPU KV offload, TraceLab load (design)

Date: 2026-11-20. Status: approved by the user (2026-11-20); build started 2026-11-20 with the
calls in "Calls for the build", confirmed at the gate before `akamas create`.
Follows studies 52 and 53 (`studies/52-g7e-glm53-coding-sessions-within-slo/`,
`studies/53-g7e-glm53-kv-offload-idle-sessions/`) and two spikes
(`docs/spikes/2026-11-19-aiperf-session-ramp/`, `docs/spikes/2026-11-20-tracelab-replay/`).

## Intent and the why

A target deployment serves GLM-5.3-Flash to **coding agents with long contexts**: official FP8
checkpoint, TP4 on H200, vLLM 0.32.0 (0.34.0 planned), one vLLM pod per replica behind KServe's
`LLMInferenceService` and the llm-d router, no CPU KV offload. Study 53 showed on our rig that CPU KV
offload cuts the TTFT of a resumed session from 22.9 s to 1.73 s (p95) from 128 GiB up, at a fixed
40 sessions; it did not measure **how many users a replica holds**, which is the number the target
sizes its replicas with.

Study 54 asks **how many concurrent coding users one replica serves within the SLO**: with the
target's arguments, with CPU KV offload presets, and with an Akamas-optimized configuration. It is
the study we will run at the target, on their H200; here it runs end to end on the g7e as a dress
rehearsal (user, 2026-11-19).

## Goals

1. Users held within the SLO by the target's configuration (baseline) on this rig.
2. The same with CPU KV offload 128, 256, 512 GiB (native) and 512 GiB (LMCache).
3. An Akamas-optimized configuration (30 experiments) and its gain over the baseline, as an
   argument line.
4. A harness ready for the target: every rig-specific value a parameter (see "Portability"). The
   KServe `LLMInferenceService` apply mode is added when the target's details are known (my call:
   the user deferred them on 2026-11-19; nothing here runs KServe, so it cannot be tested on this
   cluster).

## Non-goals

- Multi-replica serving and the llm-d router (prefix-aware routing across replicas).
- Accuracy beyond a smoke check.
- Reporting rig numbers as the target's: absolute users differ (GPU, checkpoint precision).

## Anti-goals (a failure even if it runs)

- A load whose sessions open with a 100k-token cold prompt (study 53's shape): in a ramp, resumed
  rounds queue behind those prefills and the score measures arrivals, not the cache (spike
  2026-11-19, reasoning; vLLM serves a running long prefill before a waiting request).
- Replaying TraceLab's recorded prefixes as they are: they carry the original provider's cache
  evictions, so rounds after a pause would reach the server as new content and offload could
  never serve them (spike 2026-11-20).
- A baseline that is not the target's argument line beyond the listed rig differences.
- An experiment run with other arguments than Akamas chose (study 49): the apply script checks the
  pod's real arguments and `vllm:cache_config_info`.

## Decided with the user

- **Steps** (2026-11-19): baseline = the target's arguments, no offload; presets native offload
  128, 256, 512 GiB; preset LMCache 512 GiB; optimize, **30 experiments**, offload size **free**
  in [0, 512] together with `gpu_memory_utilization`, `max_num_batched_tokens`, `max_num_seqs` and
  the MTP tokens. DCP stays out on SM120 (no backend, study 52); at the target it is a candidate
  if their smoke passes.
- **Full run on the g7e as a dress rehearsal** (2026-11-19); the target's values (RAM per replica
  for offload, access to their `LLMInferenceService`, Prometheus / DCGM) come later.
- **Load = TraceLab** real coding sessions (2026-11-20), replayed with **AIPerf 0.13.0** (upgrade
  from 0.11.0, 2026-11-20).
- **Continuous ramp of users; the run stops as soon as the SLO breaks**, as in the Zucchetti
  studies (36-38): some experiments end in ~40 min; **at most ~1 h 30 - 1 h 40** each
  (2026-11-20).
- **SLO:** client TTFT p95 <= 5 s and ITL p95 <= 50 ms (the user's, study 52, 2026-11-17).
- **Telemetry:** the full catalog (`CLAUDE.md` rule 9).
- The GPU node stays up between the design and the study (2026-11-20).

## Approaches considered

| | Approach | Trade-off | |
|---|---|---|---|
| A | Study 53's synthetic sessions (20k shared + 100k own context) in a continuous ramp | Simple; cold 100k prefills (~11 s each) block resumed rounds: the SLO breaks at 20-30 sessions in every configuration | rejected |
| B | Steps with holds (add sessions, wait for their first turn, measure 5 min) | Measures cache capacity cleanly; unrealistic pacing, ~8 levels in 90 min | dropped for C |
| C | **TraceLab** real sessions in a continuous ramp | Realistic: sessions start small (p50 18k) and grow, agent rounds seconds apart, developer pauses of minutes; needs AIPerf 0.13 and a prefix rewrite | **chosen** |
| D | `aiperf synthesize agentic-code` (synthetic, calibrated) | Already in AIPerf 0.11; distributions to calibrate by hand | fallback |

## Rig and stack (dress rehearsal)

- **Node:** g7e.24xlarge spot (4x RTX PRO 6000 Blackwell 96 GB, 96 vCPU, 1 TiB), node group
  `llm-serving-g7e-24xl-spot`, us-east-2c, cluster `vllm-bench`; up since 2026-11-19 16:07 UTC.
- **Server:** `vllm/vllm-openai:v0.34.0` (LMCache upgraded to 0.5.8rc1 at start for the lmcache
  backend, see "LMCache on vLLM 0.34.0"); checkpoint `canada-quant/GLM-5.3-Flash-W4A16-MTP`
  revision 5681f7c2 (the target's FP8 leaves no KV on 96 GB GPUs, study 52 smoke).
- **Fixed arguments** (studies 52/53): `--served-model-name zai-org/GLM-5.3-Flash
  --kv-cache-dtype fp8 --max-model-len 262144 --tool-call-parser glm47 --reasoning-parser glm47
  --enable-auto-tool-choice --tensor-parallel-size 4 --language-model-only`.
- **Measured on this rig (study 53):** GPU KV 3.77M tokens at gmu 0.95; ~28.7 KB per token in
  RAM (MLA latent replicated on 4 ranks): 512 GiB ~ 19.1M tokens; prefill ~9.6k tokens/s; the
  offload cache is inclusive (RAM alone must hold the sessions' history).
- **Load generator:** AIPerf 0.13.0 in the two `aiperf-client` pods (m8a.2xlarge node, 3.5 CPU and
  13 GiB each today; resized if the calibration says so).
- **Differences from the target:** GPU (SM120 vs SM90), checkpoint (4-bit W4A16 vs FP8), vLLM
  0.34.0 vs 0.32.0, `--language-model-only`. What transfers: which levers move the users held and
  roughly how much; the harness.

## Load

- **Source:** TraceLab v0.0.2 (`syfi_coding_trace.jsonl.gz`, 8,058 Claude Code / Codex sessions,
  665k rounds), CC BY 4.0: credit "TraceLab (SyFI Lab, University of Washington)" and
  <https://tracelab.cs.washington.edu> in the README. Downloaded by the workflow (or staged on the
  toolbox), never committed.
- **Preparation** (`prepare_tracelab.py`, from the spike):
  - `prefix_tokens` = the history re-sent, `min(input_i, input_{i-1} + output_{i-1})`, except
    where the context shrank below 70 % of the previous round (compaction);
  - each session ends before its first round above 250,000 input tokens (max model length
    262,144 minus output);
  - sessions with >= 2 rounds; a fixed-seed shuffle; **two disjoint pools**, one per client (a
    session replayed twice would share its prefix across users).
- **Replay:** `aiperf profile --custom-dataset-type tracelab --no-fixed-schedule --concurrency N
  --concurrency-ramp-duration R --inter-turn-delay-cap-seconds 1200 --dataset-sampling-strategy
  shuffle --random-seed <fixed>`, streaming chat, `ignore_eos` (output lengths as recorded), the
  model's own tokenizer (from the checkpoint).
- **Users** = concurrency lanes. A lane replays sessions one after the other: when a session
  ends, the same user starts a new one (a new task). Users enter linearly during the ramp and
  never leave. The CSV counts a lane as active from its first request on, **including while it
  sits in a developer pause** (up to 20 min): study 38's span-based count would drop exactly those
  lanes, in the windows Akamas scores.
- **Pauses:** tool gaps as recorded; **developer pauses capped at 20 min** (my call: keeps 90 %
  of the recorded pauses, p90 18 min, inside a 90-minute experiment).
- **Same pools, seed and ramp in every experiment**: every configuration sees the same sequence
  of sessions.

## Score (Akamas)

- **Pattern of study 38:** an `aiperf` component (AIPerf pack) fed by a **CSV File** telemetry
  instance; the workflow converts AIPerf's per-request records into 30 s buckets
  (`aiperf_to_csv.py`, adapted to AIPerf 0.13's export), TS / COMPONENT / one column per metric.
- **Goal:** maximize `aiperf.aiperf_active_sessions` (users holding a lane in the bucket).
- **Constraints:** `aiperf.aiperf_time_to_first_token_p95_150s <= 5000` and
  `aiperf.aiperf_inter_token_latency_p95_150s <= 50` (all rounds, client side).
- **Windowing:** `stability` on `aiperf_active_sessions`, `when: max` (study 38): the best valid
  window is the last one before the SLO breaks.
- **Watchdog** (study 38's `run_test.sh` + `lib_watchdog.sh`): the load ends with success once the
  server TTFT p95 over 150 s (vLLM histogram, Prometheus) stays above 10 s (2x the SLO) for
  120 s, armed a few minutes after the ramp starts; plus a stall check and a deadline. Past the
  knee latency only grows with users, so every later window is invalid anyway.
- **KPIs** (<= 8): users, TTFT p95 150 s, ITL p95 150 s, completed requests/s, prefix cache hit
  rate incl. offload, CPU offload hit rate, offload read-back GB/s, GPU KV usage.

## Steps

| Step | Values (on top of the target's line) |
|---|---|
| baseline | gmu 0.95, MTP 5, no offload, everything else vLLM's default (not rendered) |
| Native 128 / 256 / 512 | + `--kv-offloading-size 128/256/512 --kv-offloading-backend native` |
| LMCache 512 | + `--kv-offloading-backend lmcache`, LMCache 0.5.8rc1 server with 512 GB in the container (below) |
| optimize | 30 experiments, AKAMAS optimizer |

- **Optimize domains** (inside the vLLM pack's; the pack installed on akamas-41 is checked before
  the build, studies 52/53 ran on 1.15.0): `kv_offloading_size` [0, 512],
  `gpu_memory_utilization` [0.85, 0.97], `max_num_batched_tokens` [2048, 32768], `max_num_seqs`
  [16, 1024] (the baseline runs vLLM's default 1024, which fits the 4-bit checkpoint at gmu 0.95,
  study 52), `spec_tokens` [0, 5].
- **`kv_offloading_backend`:** the presets set it in their values. It enters the optimize step
  only if the LMCache calibration ramp (below) holds at least as many users as Native 512, decided
  before the study is created (`parametersSelection` is frozen once a study runs); otherwise the
  optimize step renders `native` (my call).
- **Parameter constraints:** `max_num_batched_tokens >= max_num_seqs`. `max_num_seqs` above vLLM's
  Mamba (KDA state) block count at a low gmu cannot be written as a formula: vLLM refuses to start,
  the apply script exits 4 with the reason, the experiment fails and the optimizer learns it.
- `runOnFailure: true` on every preset (study 53: a reclaimed spot node must not stop the sweep).

## Time per experiment (target <= 1 h 40)

vLLM restart with warm caches ~6-8 min; AIPerf start and dataset build ~2-3 min; ramp <= 75 min;
grace, CSV, log dump ~3 min. The watchdog ends most runs at the knee. Study: 35 experiments, ~25-40
h of GPU (~9 USD/h spot). Not bounded: a cold start on a new node (checkpoint download ~5 min,
JIT warmup), the loss of the spot node.

## LMCache on vLLM 0.34.0 (smoke 2026-11-20, done)

How the LMCache preset runs, found on the node (study 53's Deployment, rendered by hand):

- `--kv-offloading-backend lmcache` makes vLLM use `LMCacheMPConnector`, which needs a separate
  **LMCache server** (default `tcp://localhost:5555`); vLLM ignores `--kv-offloading-size` for this
  backend: the RAM is the server's `--l1-size-gb`. The apply script starts
  `lmcache server --port 5555 --l1-size-gb <kv_offloading_size> --eviction-policy LRU` in the vLLM
  container before `vllm serve` (same GPUs, same `/dev/shm`), its log prefixed `[lmcache]`.
- **LMCache 0.5.8rc1** (`pip install --no-deps` at start): the image's 0.5.7.post1 crashes vLLM
  on the first request (`'CachedRequestData' object has no attribute 'resumed_req_ids'`, fixed by
  LMCache PR #6457, in 0.5.8rc1).
- With 0.5.8rc1 the **extensible KV cache stays on** (vLLM 0.34's default) and gmu 0.95 / MTP 5
  as in the baseline: Ready in 260 s, **GPU KV 3,670,016 tokens** (native presets 3.77M, study
  53), two 26k-token requests answered (3.85 s, then 0.29 s with LMCache retrieving chunks).
- Dead ends, for the target's 0.34 upgrade: with the image's 0.5.7.post1 the extensible cache is
  refused (legacy CUDA IPC); disabling it makes vLLM size the cache from an estimate that does not
  count the LMCache server's ~0.8 GiB per GPU (CUDA OOM in warmup at gmu 0.95 and 0.92, start at
  0.87); 0.5.7.post1's VMM mode fails on GLM's mixed cache ("tensor does not appear to be
  VMM-backed").
- **A worker that dies during warmup leaves vLLM hanging** (no restart, "No available shared
  memory broadcast block" every 60 s): the apply script must fail on `WorkerProc hit an exception`
  / `OutOfMemoryError` in the log, not only on restarts.

## Calibration before the build (on the node, no Akamas)

1. ~~LMCache 512 with GLM-5.3-Flash~~: starts and serves (above).
2. ~~AIPerf 0.13 on the client pods~~: done 2026-11-20, below (CPU under hundreds of users comes
   with the ramps).
3. ~~Three calibration ramps~~ (baseline, Native 512 and, if it starts, LMCache 512; users
   1 -> 300 over ~40 min, the watchdog off): where each breaks, and on what (TTFT or ITL). They
   fix the ramp's top N and length R so the best configuration's knee falls in the first two
   thirds of the ramp, and whether `kv_offloading_backend` enters the optimize step. If the ITL
   breaks first at similar users in all, the SLO, not the cache, binds on this rig: back to the
   user. Done 2026-11-20, below: **the ITL breaks first in all three** (back to the user).

### Calibration ramps (2026-11-20)

2 -> 300 users over 40 min (two clients of 150), 45 min, shared 8k system prompt, fresh vLLM pod
each, watchdog off. Users held = the last 30 s bucket before the metric stays above the threshold
for 4 buckets (p95 over a trailing 150 s, client side; one bucket = ~4 users at this slope).

| Users held | Baseline | Native 512 | LMCache 512 |
|---|---|---|---|
| ITL p95 <= 50 ms | 8 | 8 | 12 |
| ITL p95 <= 100 ms | 27 | 30 | 34 |
| TTFT p95 <= 5 s | 46 | 50 | 38 |
| TTFT p95 <= 10 s | 53 | 56 | 53 |
| Requests completed in 45 min | 1,219 | 2,797 | 2,137 |
| TTFT p95 at 100 users | 121 s | 37 s | 40 s |
| ITL p95 at 100 users | 366 ms | 130 ms | 110 ms |
| Errors | 9 | 40 | 136 |

- **The ITL SLO binds first, at ~8-12 users, in every configuration**, with the GPU KV at
  11-20 %. The exact figure is noise-limited (a p95 over 150 s this early rests on a few
  requests: baseline and Native 512 are the same server before the KV fills, yet 88 vs 42 ms at
  8 users); what is solid is that the client ITL p95 sits at 40-130 ms from ~8 users on, in all
  three ramps. Not a short-output artifact: in the first 5 min, requests with >= 200 output
  tokens have ITL p50 37-40 ms and p95 84-92 ms. Cause, supported but not yet tested: prefill
  interference. vLLM's server-side per-token ITL p95 is 0.5-0.8 s from minute 2 (baseline), the
  length of a step that carries a prefill chunk of up to `max_num_batched_tokens` 8192 (the
  default) at ~10k tokens/s next to the running decodes; this load prefills all the time. Levers:
  `max_num_batched_tokens`, MTP tokens (a ramp at 2048 would test it).
- **The TTFT knee is where the GPU KV fills** (~90 % at 50 users, minute 6.5): within the SLO,
  offload moved it by at most one bucket at this slope (46 -> 50 users native, 38 LMCache; single
  runs); its gain is past the knee, which the watchdog cuts off: 2.3x the completed requests,
  TTFT p95 at 100 users 37 s instead of 121 s, 80 % of the tokens queried served from RAM.
- **The slope is too steep for these knees**: they fall in minutes 1.5-7 of 45, with young
  sessions (small contexts). A ramp to ~100 users over 60-75 min puts the knee in its first two
  thirds with sessions that have grown; whether that makes the cache matter at the knee is
  untested (a baseline / Native 512 pair at that slope would say).
- **LMCache 512 vs Native 512 is within one to three buckets either way** (TTFT <= 5 s: 38 vs
  50; ITL <= 50 ms: 12 vs 8; single runs), so the design's rule cannot decide until the SLO is
  settled. The firm argument is another: **LMCache fails requests once its RAM fills** (~280
  users, L1 at its 80 % eviction watermark: 516 "Some keys not found during retrieve", 129
  requests failed with an internal error, LMCache 0.5.8rc1). My call, for the user to confirm:
  `kv_offloading_backend` stays out of the optimize step (native).
- **AIPerf's system message is the one shared**: on the baseline pod after its ramp, a new
  request with the prompt file hits 6,912 of 8,034 tokens at once (blocks kept hot by the ramp's
  first rounds through 45 min of a full KV).
- Errors otherwise: "no responses with actual content" (8-38 per ramp, ~1-1.4 % of requests)
  and 1-2 disconnects. Pools: at most 203 distinct sessions per client of 600, no repeat. Client
  CPU ~1.1 of 3.5 cores, no throttling during the ramps; memory up to 10.7 of 13 GiB (page
  cache).

### Calls for the build (2026-11-20)

The user asked to build the study ("fai lo study") and to keep the GPU node up throughout,
without answering the three calibration questions; they approved three slow-slope ramps run while
the study is built. My calls, each confirmed or changed at one gate before toolbox sync,
`akamas create` and start (constraints, `parametersSelection` and windowing freeze there):

- **SLO unchanged:** TTFT p95 <= 5 s and ITL p95 <= 50 ms (the user's; I do not relax it). Both
  thresholds are single values in the study and the scripts, so a change at the gate is one edit.
- **`kv_offloading_backend` stays out of the optimize step** (native): LMCache fails requests
  when its RAM fills. LMCache 512 stays a preset.
- **Ramp N and R pending the slow-slope ramps** (2 -> 100 users over 70 min, 75 min, watchdog off,
  fresh pod each): baseline, baseline + `max_num_batched_tokens` 2048, Native 512. Candidate:
  100 users over 70 min. They say whether the slower slope makes offload move the knee, and
  whether a smaller prefill chunk moves the ITL knee (read for trend: at ~10k tokens/s a 2,048-token
  chunk is still ~0.2 s per step). vLLM 0.34 accepts a chunk below the 2,304-token attention block
  in Mamba align mode (intermediate chunks keep private state until the next block boundary,
  `_mamba_block_aligned_split`).
- **The watchdog stays TTFT-only** (server TTFT p95 > 10 s for 120 s): every run goes past the ITL
  knee, so the CSV holds both knees whatever the ITL threshold; the threshold changes what the
  optimizer maximizes, not whether the preset sweep is informative.
- **Scoring check before the gate:** the calibration exports are replayed through the CSV converter
  and Akamas' windowing rule; the score must match the knee found by `ramp_buckets.py`, and
  baseline vs Native 512 before the KV fills (the same server) must score within the preset
  differences, or 30 experiments chase noise.

### AIPerf 0.13 on the client pods (calibration 2026-11-20)

In `aiperf-client-0` (3.5 CPU, 13 GiB, AIPerf 0.13.0 in a venv), the GLM tokenizer, against a stub
and then against vLLM (LMCache 512 up, 10 users, 3 min):

- **AIPerf stores every round's full message list** (`dataset.dat`, ~3.7 bytes per input token
  summed over rounds), so a session's size grows with the square of its rounds: one Codex session
  of 11,805 rounds took 6 GB, and 1,500 uncut sessions did not build in 5 min (> 20 GB).
- **Horizon cut** (lossless): each session ends at the rounds reachable within 80 min of replay
  (think time capped at 20 min, plus a fast service bound: new tokens at 20k tok/s, output at
  200 tok/s). **Context cut:** before the first round whose input x 1.003 + 100 + output exceeds
  262,144 (vLLM rejects prompt + max_tokens above the max model length; input alone under 250k is
  not enough).
- **Pool per client: 600 sessions** (two disjoint pools from one fixed-seed shuffle). A Monte
  Carlo of lanes drawing sessions until the horizon gives, for both clients together, p99 836
  sessions for 300 users over 40 min, 1,042 over 75 min, 1,386 for 400 users over 75 min.
  `shuffle` re-serves the pool once exhausted (two users would replay the same session: false
  cache hits), so the harness fails an experiment if the distinct sessions sent reach the pool
  size; the ramp's top N sets the final pool.
- **Build:** 600 sessions = 7.8 GiB, built in 197 s, peak anon memory 4.4 GiB, the 13 GiB limit
  filled by page cache (reclaimable), no OOM. The content-addressed mmap cache
  (`AIPERF_DATASET_MMAP_CACHE_DIR`; key: pool file, tokenizer, model name, prompt settings) then
  serves later runs by hardlink: start in 47 s, peak 3.7 GiB. The cache and the run's mmap
  (`AIPERF_DATASET_MMAP_BASE_PATH`) must live on one filesystem outside the 8 GiB `/work`
  emptyDir (the first build holds two copies, ~16 GiB); the node has 43 GiB free for both pods.
- **Settings:** `AIPERF_DATASET_MMAP_PREFAULT=false`,
  `AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800` with `AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800`
  (the first build exceeds the 300 s default); **no `HF_HUB_OFFLINE`** with `--tokenizer
  /work/tokenizer` (0.13 then looks the path up as a Hub repo and fails).
- **Against vLLM:** server prompt tokens / trace input 0.999-1.002 (chat template +26 tokens
  median), completion tokens = the recorded output in 145 of 145 requests (`--extra-inputs
  ignore_eos:true`), 0 errors.
- **Shared system prompt:** `--shared-system-prompt-length` is synthetic-only; `--system-prompt-file`
  works with `tracelab` (one identical system message in every session, stub-verified) and adds
  to the recorded input. The first-round floor in TraceLab (p1-p5: Claude 7.0-9.2k, Codex
  2.7-4.1k tokens) bounds the agents' common prompt.
- **Shared system prompt, decided (user, 2026-11-20): modeled, K = 8,000 tokens**, one file for
  every user of both clients (the target's users run one agent; K changes to the target's real
  prompt length once known). The preparation subtracts K from every round's recorded input (the
  first round's new tokens included; later rounds' appended tokens unchanged), so each round's
  size on the server stays the recorded one and only the share common to all users changes.
  Sessions with a kept round under K + 200 tokens are dropped before the shuffle (my call: 444 of
  7,433, mostly Codex; the pools draw 1,200), and K is subtracted after every cut ran on the
  recorded totals. The prompt file is cut to exactly 8,000 GLM tokens (47,942 characters).
- **Token check with the system prompt on (2026-11-20, both clients, 10 users each, 3 min):**
  server prompt tokens - (pool input + K) p5 -8, p50 +10, p95 +27 tokens over 146 requests, all
  within 0.5 %; one request of 147 failed with "no responses with actual content".
- **Sharing of the prompt across sessions depends on the KV backend.** GLM-5.3-Flash is hybrid
  (attention + KDA recurrent state): vLLM 0.34 runs prefix caching in Mamba `align` mode with
  2,304-token attention blocks. Three requests with the same 8k system prompt and different user
  messages, on a fresh pod: **Native 512** hits 0, then 6,912 and 6,912 of 8,035 tokens (three
  whole blocks: the prompt is shared from the second session on); **LMCache 512** showed 0 and 0
  local hits of 8,034 (pod already used by the token check; the test reads only the local
  counters, so the hits may have gone to LMCache's external counter: not settled).
  `--enable-mamba-shared-prefix-checkpoint` (off by default) would add a recurrent-state
  checkpoint where sibling requests diverge; mostly moot on native, which already shares 6,912 of
  the 8,000 tokens. The ramps run the target's configuration (flag off).

## Portability to the target

- Built and tested here: the **Deployment** apply mode. Everything rig-specific is a parameter in
  one env file: node role, namespace, image, model path and served name, maximum offload (RAM per
  replica), Prometheus address, toolbox paths, kube context. Target details (hostnames, names)
  never enter the repo.
- Added at the target, once their details are known (deferred by the user): the
  **LLMInferenceService** apply mode (patch `VLLM_ADDITIONAL_ARGS`, memory and `/dev/shm` of their
  resource, wait for the rollout, check the pod's arguments; their `pip install torchcodec` at
  start stays); whether AIPerf targets the pod or the llm-d router; whether our Akamas or theirs
  drives the study; `decode_context_parallel_size` in the optimize step if their smoke passes.

## Risks and open points

- **The ITL SLO may bind before the cache on this rig** (study 52: 50 ms broke at 24-32 sessions
  with a turn every 30 s; MTP accepts 40.1 % of drafted tokens on this checkpoint, study 52's
  smoke). The calibration says; then offload shows through fewer recomputations (study 53: TPOT
  p95 47.5 -> 21-24 ms with offload).
- **Prefill-heavy rounds:** 0.25 % of rounds append > 40k new tokens (> 4 s of prefill); they sit
  inside the p95.
- **Shared system prompt across sessions:** modeled with K = 8,000 tokens (calibration above).
  If the target's prompt is much shorter, the test here is optimistic by up to ~8k tokens of KV
  per user; K is one parameter in the env file.
- **Client sizing:** settled by the calibration above (600 sessions, 7.8 GiB per client, cached).
- **AIPerf 0.13 export format** vs study 38's CSV converter: adapt and test on the spike's records.
- **Spot reclaim** mid-study: apply waits for the node (study 53's `apply_config.sh`).

## Left to the builder

File layout from `studies/_TEMPLATE/`; the scripts adapted from study 53 (apply, telemetry
catalog) and study 38 (run_test watchdog, stop AIPerf, CSV); the Akamas YAML through the
akamas-study-manager plugin; tests in the style of studies 52/53 (`test_*.sh`).
