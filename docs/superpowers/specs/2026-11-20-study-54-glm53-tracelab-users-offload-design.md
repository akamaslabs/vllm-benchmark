# Study 54 — GLM-5.3-Flash: coding users held per replica, CPU KV offload, TraceLab load (design)

Date: 2026-11-20. Status: draft for the user's approval.
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
- **Server:** `vllm/vllm-openai:v0.34.0` (LMCache importable in the image, `--kv-offloading-backend
  {lmcache,native}`, checked 2026-11-20); checkpoint `canada-quant/GLM-5.3-Flash-W4A16-MTP`
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
| LMCache 512 | + `--kv-offloading-size 512 --kv-offloading-backend lmcache` (only if it starts, below) |
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

## Calibration before the build (on the node, no Akamas)

1. **LMCache 512 with GLM-5.3-Flash** (hybrid sparse MLA + KDA): does vLLM start, does it store
   and read back? If not, the LMCache preset is dropped and noted. (Running on 2026-11-20.)
2. **AIPerf 0.13 on the client pods:** memory and CPU while loading the session pool and during a
   ramp of a few hundred users; pod limits and pool size per client from it.
3. **Three calibration ramps** (baseline, Native 512 and, if it starts, LMCache 512; users
   1 -> 300 over ~40 min, the watchdog off): where each breaks, and on what (TTFT or ITL). They
   fix the ramp's top N and length R so the best configuration's knee falls in the first two
   thirds of the ramp, and whether `kv_offloading_backend` enters the optimize step. If the ITL
   breaks first at similar users in all, the SLO, not the cache, binds on this rig: back to the
   user.

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
- **No shared system prompt across sessions:** AIPerf mints block ids per session, so the agents'
  common tools/system prompt (~15-20k) is not shared; check whether a shared prefix can be added
  (e.g. `--shared-system-prompt-length`) in calibration step 2; otherwise conservative.
- **Client sizing:** loading 600 sessions peaked at 5-9 GB in one AIPerf process (spike);
  `AIPERF_DATASET_MMAP_CACHE_ENABLED` reuses the built dataset across experiments.
- **AIPerf 0.13 export format** vs study 38's CSV converter: adapt and test on the spike's records.
- **Spot reclaim** mid-study: apply waits for the node (study 53's `apply_config.sh`).

## Left to the builder

File layout from `studies/_TEMPLATE/`; the scripts adapted from study 53 (apply, telemetry
catalog) and study 38 (run_test watchdog, stop AIPerf, CSV); the Akamas YAML through the
akamas-study-manager plugin; tests in the style of studies 52/53 (`test_*.sh`).
