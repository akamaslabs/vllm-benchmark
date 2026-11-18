# 53-g7e-glm53-kv-offload-idle-sessions

**Status:** BUILT 2026-11-18, smoke pending; runs while study 52 is paused (finished after its
experiment 3, resumed after this study). Created 2026-11-18.

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
  16384 / 32768), `Prefill 32k no offload`, `Baseline repeat`.
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

## Smoke

Pending: one start with the heaviest preset (`Prefill 32k`: offload 512 GiB + `max_num_batched_tokens`
32768) and a short `run_load.sh` (the TURNS line, exit 0).
