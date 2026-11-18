# Study 53 — CPU KV offload for long, mostly idle coding sessions (design)

Status: draft 2026-11-18, written while study 52 is paused (`akamas finish` after its experiment 3).
The user asked for it on 2026-11-18 ("facciamo un nuovo studio 53", then "finisce il preset 3 ... e
faremo partire il 53", leaving the session running unattended). The feasibility spike is
`docs/spikes/2026-11-18-glm53-offload-reload/`. Workload numbers are **my assumptions** (the target
deployment's session counts, context sizes and think times are unknown).

## Intent and the why

Study 52 runs coding sessions that send a turn every 30 s: the generation rate breaks the ITL SLO at
24-32 sessions while the KV cache is at most 71 % full, so CPU KV offload never reads anything back
(experiment 2: 1.3 TB written to RAM, 0 bytes read). Real coding agents keep many long sessions open
and mostly idle (tools running, a developer reading); between two turns a session's history is
evicted from the GPU, and without offload the next turn recomputes it. The target runs
`--max-model-len 262144` and no offload.

Study 53 asks: **when the open sessions' history exceeds the GPU KV cache, how much does CPU KV offload
cut the time to the first token of a resumed session, and from which offload size?** Same rig, model,
image and baseline arguments as study 52.

## What the spike measured (2026-11-18)

GLM-5.3-Flash W4A16 at TP4 on the g7e.24xlarge, study 52's baseline (gmu 0.95, MTP 5, FP8 KV,
`max_num_batched_tokens` 8192 by default), GPU KV cache 3.77M tokens.

- **Reload works on this hybrid model** (KDA + sparse MLA + MTP head): with 512 GiB, 157 GB read back
  from RAM in 25 min, 71 % of the prompt tokens of the steady window served from RAM.
- **Cold prefill is ~9.6k tokens/s** (computed tokens, constant with or without offload): a 124k-token
  first turn takes ~11 s alone, ~21 s when two overlap. **A first turn of a 100k-token session can
  never meet a 5 s TTFT on this rig**; the SLO question is about resumed sessions.
- **Paced pair** (40 sessions of ~125-130k tokens, one turn per session every 600 s, ~5.2M tokens of
  history vs 3.77M on the GPU):
  - offload 512 GiB: resumed turns TTFT **~1.0-1.7 s**; cold first turns 11-21 s; computed prefill in
    the steady window ~460 tokens/s; ITL p95 33 ms.
  - offload 0: resumed turns TTFT **~11-12 s (client 1) and ~21.6 s (client 0)**, the same as a cold
    first turn: every resumed turn recomputes its ~125k-token history (CPU hits 0, GPU hits 14 %,
    the shared prompt). Both runs completed all 100 requests (the load keeps up), ITL p95 9-35 ms.
  - **Go (19:52 UTC):** offload makes a resumed session ~12x faster to its first token (11-22 s ->
    1-1.7 s), inside a 5 s TTFT; without it no resumed turn is. Client 0 is consistently ~2x client
    1 in both runs (its turns land while the other client's prefill runs); the comparison is per client.
  - Per-request exports could not be kept: AIPerf flushes `profile_export.jsonl` in batches (0 lines
    at 1490 s with ~26 requests per client) and `run_level` deletes the run directory at the end. The
    study's parser splits first turns from resumed ones before that deletion.
- Unpaced runs (64 sessions created within 120 s) never leave the cold start: TTFT 450-650 s. Session
  creation must stay under the prefill rate.
- AIPerf 0.11.0 writes `--num-dataset-entries` whole conversations to `inputs.json`: 2000 entries of
  ~1M tokens = 8.6 GB, which evicted the clients (8Gi emptyDir). Entries = conversations: ~200 MB.
- Bytes per offloaded token: ~28.7 KB (157 GB / 5.47M CPU-hit tokens), so 512 GiB holds ~19M tokens;
  `kv_offload_cpu_cache_usage_perc` stays near 0 and is not a fill level.

## Goals

- TTFT of resumed sessions with offload 0, 32, 64, 128, 256, 512 GiB at one fixed load whose history
  is ~1.4x the GPU cache: the step where offload starts to pay, and what it is worth.
- Whether a larger prefill batch (`max_num_batched_tokens` 16384, 32768) shortens cold first turns
  and recomputation, with and without offload.
- At most 90 minutes per experiment (the user's cap, 2026-11-18): ~31 min here.

## Non-goals

- No optimizer step: the outcome is a step in one parameter, a preset sweep shows it directly.
- Not the most sessions within an SLO: building 100k-token sessions costs ~13 s of whole-rig prefill
  each, so a session ramp does not fit 90 minutes.
- No change to study 52's files on the toolbox (`/work/study52`); study 52 resumes after study 53.

## Workload (my calls)

- 40 sessions, two AIPerf clients with 20 each (study 52's client pods and image).
- Each session: the 20k-token shared system prompt, 100k tokens of its own context, then turns of
  4000 input and 400 output tokens (`ignore_eos`), history re-sent; one turn every 600 s; 10 turns
  (a session outlives the run, apart from AIPerf's shortened first sessions).
- Creation: 40 x ~100k uncached tokens over the first 600 s = ~6.7-8k tokens/s, under the ~9.6k
  prefill rate. History at the end: 40 x ~130k = ~5.2M tokens, 1.4x the GPU cache.
- `--num-dataset-entries` = `--conversation-num` = 40 per client.
- One run of 1500 s (+60 s grace) per experiment, the same seed in every experiment (vLLM restarts
  between experiments, so no cache carries over; the session schedule, including AIPerf's shortened
  first sessions, is identical).

## Measurement

- **Akamas goal:** minimize the server TTFT average (`vllm.time_to_first_token_avg`) over the window
  (the pack has no TTFT p50). The window holds the resumed turns and the same few cold first turns
  of restarted sessions in every experiment.
- **Window:** trim `[710s, 60s]` on the Load task (AIPerf start-up ~5 s, creation phase ~700 s).
- **Constraints:** `vllm.request_success_rate` >= 95 % of the offered 0.0667 turns/s (a configuration
  that falls behind is infeasible, not "fast"); server TPOT p95 <= 80 ms (study 52's safety net).
- **KPIs (8):** TTFT avg, p90, p95; prefix hit rate (GPU), CPU offload hit rate, hit rate with
  offload; CPU-to-GPU offload throughput; TPOT p95.
- **Client side (README numbers):** the load script splits TTFT by turn: first turns (turn_index 0)
  and resumed turns (>= 1), each count / p50 / p95 over the window, and resumed-turn TTFT per 5
  minutes. It fails the task only on AIPerf crash or kill (6), vLLM restart (5) or errors >= 1 %
  (6); never on latency.

## Steps (presets, ~31 min each, ~5.2 h)

1. `baseline`: study 52's baseline (gmu 0.95, MTP 5), offload 0.
2. `Offload 32`, 3. `Offload 64`, 4. `Offload 128`, 5. `Offload 256`, 6. `Offload 512`.
7. `Prefill 16k`: offload 512, `max_num_batched_tokens` 16384.
8. `Prefill 32k`: offload 512, `max_num_batched_tokens` 32768.
9. `Prefill 32k no offload`: offload 0, `max_num_batched_tokens` 32768.
10. `Baseline repeat`: offload 0 (run-to-run noise).

## Infrastructure (my calls)

- Own folder `studies/53-g7e-glm53-kv-offload-idle-sessions/`, own Akamas names (system, telemetry,
  workflow, study), own toolbox directory `/work/study53`.
- Same namespace `glm53-coding`, Deployment `vllm`, AIPerf StatefulSet and node hostPath caches as
  study 52: the two studies run one at a time on the one g7e.24xlarge, and sharing the checkpoint
  cache avoids a 191 GB download. Each study's Apply re-renders the Deployment, so switching back is
  just study 52's next Apply.
- Components and telemetry: study 52's full catalog (CLAUDE.md rule 9), renamed.
- Akamas 4.1.3 (akamas-41).
