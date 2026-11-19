# 53-g7e-glm53-kv-offload-idle-sessions

**Status:** DONE (Akamas 4.1.3, akamas-41, study `53-G7E-GLM53-Offload-Idle-Sessions`; 2026-11-18
20:20 → 2026-11-19 01:07 UTC, 9 of 10 experiments valid, experiment 8 lost to an infrastructure error).
Study 52 was stopped after its experiment 3 to free the node and is not resumed (the user's call).
GPU node group scaled to 0 at 01:08 UTC; `hf-token` Secret deleted. Created 2026-11-18.

- **Design:** `docs/superpowers/specs/2026-11-18-study-53-glm53-kv-offload-idle-sessions-design.md`.
- **Feasibility spike:** `docs/spikes/2026-11-18-glm53-offload-reload/` (go at 19:52 UTC).
- **Follows study 52:** `studies/52-g7e-glm53-coding-sessions-within-slo/README.md`.

## Objective

In study 52 coding sessions send a turn every 30 s: generation breaks the ITL SLO at 24-32 sessions
while the KV cache is at most 71 % full, so CPU KV offload never reads anything back. Coding agents
also keep many long sessions open and mostly idle; between two turns a session's history leaves the
GPU, and without offload the next turn recomputes it.

Study 53 asks **how much CPU KV offload cuts the time to the first token of a resumed session when
the open sessions' history exceeds the GPU KV cache, and from which offload size**. Same rig, model,
image and baseline arguments as study 52 (the target deployment's arguments, no offload).

## Method

- **Load (one per experiment, AIPerf 0.11.0, `k8s/workload.env`):** 40 sessions on two clients; each
  session = a 20k-token system prompt shared by all, 100k tokens of its own context, then turns of
  4k input and 400 output tokens (`ignore_eos`), history re-sent, **one turn per session every
  600 s**, for 1500 s. History: ~5.2M tokens vs 3.77M GPU KV tokens. Session creation (40 x ~100k
  cold tokens over the first 600 s) stays under the ~9.6k tokens/s prefill rate. Same seed in every
  experiment; vLLM restarts between experiments, so no cache carries over.
- **Score (Akamas, server side):** minimize the TTFT average over the window (trim `[830s, 60s]` on
  `Load`, telemetry `duration` 120 s); constraints: completed turns >= 0.06/s (the offered rate is
  0.0667/s) and TPOT p95 <= 80 ms. The window holds resumed turns plus the same few cold first
  turns of AIPerf's restarted sessions in every experiment.
- **Client view (`k8s/run_load.sh`, for the results):** first turns (turn_index 0) apart from
  resumed turns, each count / TTFT p50 / p95 over requests started >= 700 s (`TURNS`), resumed-turn
  TTFT per 5 minutes, TTFT per minute, vLLM counters (GPU prefix hits, CPU hits, offload bytes, MTP).
  A slow or lagging configuration exits 0 and scores badly; exit 6 only for an AIPerf failure or
  nothing completed, 5 for a vLLM restart.
- **Steps (presets only, ~31 min each, ~5.2 h):** `baseline` (offload 0), `Offload 32`, `64`,
  `128`, `256`, `512`, `Prefill 16k` and `Prefill 32k` (offload 512, `max_num_batched_tokens`
  16384 / 32768), `Prefill 32k no offload`, `Baseline repeat`. Every preset has `runOnFailure:
  true` (a failed step does not stop the sweep). **The `Prefill` presets are confounded:** at 32768
  the GPU KV cache halves (1.82M tokens, smoke) and cold first turns were no faster, so read them as
  "larger batch and smaller GPU cache", not as a pure prefill-batch effect.
- **Window checked before the start** (Prometheus replay of the two paced spike runs, the study's own
  expressions, `[120s]`, start + 830 s to + 1460 s): TTFT average 2.68 s (offload 512) vs 16.5 s (none);
  completed 0.069 / 0.070 turns/s (constraint >= 0.06); TPOT p95 23 / 47.5 ms (<= 80); prompt from
  cache incl. offload 88 % / 14 %. TTFT p95 stays high with offload (14.4 s, the cold first turns of
  restarted sessions): hence the average as the goal.
- **Read per client:** client 0's TTFT is consistently ~2x client 1's in the spike (its turns land
  while the other client's prefill runs); compare configurations client by client.

## Stack & versions

- **Hardware:** one g7e.24xlarge spot in us-east-2c (4x RTX PRO 6000 Blackwell 96 GB, 1 TiB RAM),
  node group `llm-serving-g7e-24xl-spot` shared with study 52; AIPerf clients on the `m8a-2xlarge`
  load node.
- **Server:** `vllm/vllm-openai:v0.34.0`, `canada-quant/GLM-5.3-Flash-W4A16-MTP` (revision 5681f7c2)
  served as `zai-org/GLM-5.3-Flash`, TP4, `--language-model-only`, the target's arguments (FP8 KV,
  `--max-model-len 262144`, gmu 0.95, MTP 5); offload with the native backend (`/dev/shm` =
  offload + 1 GiB, pod memory = offload + 96 GiB). Namespace `glm53-coding`, Deployment `vllm`,
  checkpoint and JIT caches on the node's hostPaths: shared with study 52 by design.
- **Load:** AIPerf 0.11.0 (`k8s/workload.env`), `--num-dataset-entries` = conversations (2000 entries
  of 100k-token sessions made an 8.6 GB `inputs.json` and evicted the clients in the spike).
- **Akamas:** 4.1.3 (akamas-41); packs vLLM 1.15.0, GPU 1.4.0, Kubernetes 1.9.0.
- **Telemetry:** Prometheus, study 52's full catalog (CLAUDE.md rule 9), `duration` 120 s.

## Parameters tuned

Presets only: `vllm.kv_offloading_size` (0, 32, 64, 128, 256, 512 GiB) and
`vllm.max_num_batched_tokens` (vLLM's 8192, 16384, 32768); everything else at the baseline (gmu
0.95, MTP 5 tokens, vLLM defaults).

## Spike 2026-11-18 (what the design rests on)

| Paced pair (40 sessions, one turn every 600 s) | Resumed turns TTFT, client 1 / client 0 | Source of the history |
|---|---|---|
| Offload 512 GiB | **~0.95 s / ~1.7 s** | RAM: 71 % of the prompt tokens, 157 GB read back |
| No offload | **~11-12 s / ~21.6 s** | recomputed (same as a cold first turn) |

Cold prefill ~9.6k tokens/s; a 124k-token first turn takes ~11 s alone (never within 5 s on this rig).

## Smoke 2026-11-18 (passed, 20:00-20:11 UTC)

On the toolbox (`k8s/smoke53.sh` there, not in the repo): the heaviest preset, `Prefill 32k`
(offload 512 GiB + `max_num_batched_tokens` 32768), then `run_load.sh` for 360 s.

- Apply: exit 0 in ~4 min; arguments and `cache_config_info` checked on the pod (offload 512,
  native backend, gmu 0.95). **GPU KV cache 1,816,148 tokens** (963 blocks of 2304), half of the
  3.77M at vLLM's 8192: the larger prefill batch takes activation memory from the KV cache, so the
  `Prefill` presets trade GPU cache for prefill batch.
- Load: exit 0; `RESULT`, `TURNS` (16 first turns, no resumed turn in 6 min: expected with one turn
  per 600 s) and `RESULT load` printed. Cold first turns 15-20 s, not faster than at 8192 (11-21 s).

## Results

Raw data: `results/export.tar.gz` (Akamas export: parameters, aggregated metrics, timeseries) and
`results/experiments.csv` (per experiment: the Akamas goal and KPIs, plus the client `TURNS` split and
the vLLM offload counters read from the task logs). Goal = server TTFT average over the window
(lower is better); "resumed" = turns >= 1 of a session, after 600 s of idleness; "first" = a
session's cold first turn (~124k tokens). Clients 0 and 1 are listed apart because one client's
turns land while the other's prefill runs (~2x apart, which client is slower changes per run).

| # | Step | Offload GiB | Batched tokens | TTFT avg (goal) | Resumed TTFT p50, clients 0 / 1 | Resumed p95 (worse client) | Read back from RAM | GPU KV tokens |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 0 | 8192 | 16.67 s | 21.9 / 12.3 s | 22.9 s | — | 3.77M |
| 2 | Offload 32 | 32 | 8192 | 16.64 s | 11.7 / 22.0 s | 22.9 s | 0 (309 GB written) | 3.69M |
| 3 | Offload 64 | 64 | 8192 | 16.63 s | 11.0 / 22.0 s | 22.9 s | 0 (309 GB written) | 3.69M |
| 4 | **Offload 128** | 128 | 8192 | **2.77 s** | **0.95 / 1.71 s** | **1.73 s** | 163 GB (5.67M tokens) | 3.69M |
| 5 | Offload 256 | 256 | 8192 | 3.29 s | 1.68 / 0.94 s | 1.70 s | 157 GB | 3.77M |
| 6 | Offload 512 | 512 | 8192 | 3.31 s | 1.73 / 0.95 s | 1.74 s | 157 GB | 3.77M |
| 7 | Prefill 16k | 512 | 16384 | 2.52 s | 1.70 / 0.95 s | 1.72 s | 163 GB | 3.21M |
| 8 | Prefill 32k | 512 | 32768 | ERROR | — | — | (6 GB in the first 10 min) | 2.11M |
| 9 | Prefill 32k no offload | 0 | 32768 | 16.64 s | 12.2 / 21.0 s | 21.9 s | — | 1.82M |
| 10 | Baseline repeat | 0 | 8192 | 16.64 s | 11.7 / 21.9 s | 22.8 s | — | 3.77M |

- **Best:** every configuration with >= 128 GiB of offload; the goal ranks `Prefill 16k` first
  (2.52 s, −84.9 % vs the baseline's 16.67 s) and `Offload 128` second (2.77 s, −83.4 %), but the
  client view of experiments 4-7 is the same (resumed p50 0.95 / 1.7 s, p95 1.70-1.74 s, cold
  first turns 20.7-21.4 s). The 2.5-3.3 s spread of the goal among them is where the few cold first
  turns of restarted sessions fall in the 120 s rate windows, not a configuration effect; read the
  goal by level (~16.6 s vs ~2.5-3.3 s). Resumed turns: p95 22.9 s → 1.73 s (−92 %), every one of
  them within a 5 s TTFT, none without offload.
- **Constraints:** none binding. Completed 0.063-0.070 turns/s (>= 0.06) and server TPOT p95 21-48 ms
  (<= 80) everywhere; TPOT p95 halves with offload (47.5 → 21-24 ms: fewer recomputations interleaved
  with decode).
- **The step lies between 64 and 128 GiB, and the RAM buffer alone must hold the history.** With 32
  and 64 GiB vLLM wrote 309 GB to RAM and read back nothing: the native offload stores every computed
  block (the CPU tier is inclusive of the GPU's), so with sessions resumed in turn the RAM buffer,
  like the GPU cache, evicts each session before it comes back. GPU + RAM did not add up: 64 GiB
  (~2.4M tokens at the measured ~28.7 KB per token across the four ranks) + 3.69M on the GPU exceed
  the ~4.4-4.8M tokens actually stored, and still nothing was read back; 128 GiB (~4.8M tokens) held
  it. Prefix hit rate incl. offload: 12-14 % (the shared system prompt) below the step, 84-89 % above.
- **Prefill batch:** 16384 and 32768 changed nothing measurable (cold first turns 20.1-20.7 s vs
  21.0-21.4 s at 8192; without offload the resumed turns stay at 21-22 s) and cost GPU KV cache
  (3.77M → 3.21M at 16384, 1.8-2.1M at 32768).
- **Noise:** baseline and its repeat differ by 0.1 % on the goal (16.67 / 16.64 s).
- **Cold first turns** (a new 124k-token session) took 11-22 s in every configuration (~9.6k
  tokens/s of prefill on this rig): no configuration here brings them within 5 s.

### Data limitations

- **Experiment 8 (`Prefill 32k`) has no score:** at 00:06 UTC both `kubectl exec` streams from the
  toolbox to the AIPerf clients closed at once ("websocket: close 1006 (abnormal closure)") ~10 min
  into the load; vLLM was healthy (42 requests completed, no restart) and `run_load.sh` exited 6 on
  an empty export. Not re-run; experiments 7 and 9 bracket it.
- Per-request exports were not kept (AIPerf flushes `profile_export.jsonl` in batches and the run
  directory is deleted after parsing); the client numbers are the parser's summaries in the logs.
- **`num_requests_running` / `num_requests_waiting` in Akamas are aliased:** the load sends two turns
  together every 30 s (20 sessions per client, one turn each per 600 s, both clients in phase) and
  Akamas samples these gauges every 30 s, so each experiment's line is flat at whatever phase it
  hits (experiment 2 reads 0 throughout; Prometheus at 5 s shows a 0-1-2-2-2-0 sawtooth). True
  in-flight averages (Prometheus, 5-minute `avg_over_time`): baseline ~1.35 for the whole run;
  experiments 4 and 7 ~1.1-1.3 during session creation (first ~10 min, cold first turns) and
  ~0.1-0.3 after, when resumed turns come back from RAM. The arrival rate is the same everywhere:
  the gauge tracks how long a turn stays in the server, not how many sessions are open.

## Conclusions

- **For long, mostly idle coding sessions, CPU KV offload decides the time to resume a session:**
  on this rig a resumed 125-130k-token session answers in ~1-1.7 s from RAM against ~11-22 s
  recomputed, a ~12x cut that brings every resumed turn inside a 5 s TTFT. In study 52's always-active
  sessions the same offload did nothing; whether it matters depends on how much history the open
  sessions hold, not on the request rate.
- **Size the RAM buffer for the whole history of the open sessions, not for the part that does not
  fit on the GPU** (vLLM 0.34.0 native backend): the effect is a step at "RAM buffer >= stored
  history"; below it the offload writes hundreds of GB and reads none back. Estimate sessions ×
  context × ~28.7 KB per token (this model, FP8 KV, TP4) and add margin; on this rig 128 GiB
  covered 40 sessions of ~110k unique tokens. Above the step more RAM buys nothing at this load.
- **Generalizable vs specific:** the step shape and the "RAM alone must hold it" rule follow from
  how the native backend stores blocks and should transfer to other models and GPUs (study 47's
  numbers fit it too: 21 GiB of FP8 Qwen3-8B KV held its ~250k live tokens, 18 GiB did not). The
  ~28.7 KB per token, the ~9.6k tokens/s prefill and the absolute TTFTs are this model (W4A16
  GLM-5.3-Flash, MLA latent replicated on four ranks), SM120 and vLLM 0.34.0; on the target's H200
  with the FP8 checkpoint the per-token size and the prefill rate differ.
- **For the target deployment** (no offload today, `--max-model-len 262144`): if its replicas hold
  more open-session history than their GPU KV cache, `--kv-offloading-size` sized to that history
  (and `/dev/shm` and pod memory to match) is the lever; the rest of the arguments can stay. What
  this study cannot say: the target's real session count, context sizes and idle times (assumed
  here), and the effect on the H200 rig.
- **Next:** a load whose stored history crosses the RAM buffer during the run (sessions growing past
  the step) to see whether the LRU degrades gradually or falls off the step; LMCache as an
  alternative backend; the first-turn cost (prefill of a new long session) as its own target
  (prefix-sharing of repository context across sessions, or prefill/decode separation); the
  load-generator fix (AIPerf started detached in the pod, polled, instead of a 25-minute
  `kubectl exec` stream).
