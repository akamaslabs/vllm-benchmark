# Study 52 — GLM-5.3-Flash coding sessions within a TTFT and ITL SLO (design)

Status: draft for approval, 2026-11-17. Follows the phase 0 spike
(`docs/spikes/2026-11-17-glm53-flash-phase0/README.md`).

## Intent and the why

A target deployment serves `zai-org/GLM-5.3-Flash` to **coding agents with long contexts**. Each
replica is one vLLM pod with **TP4 on H200**, the **official FP8 checkpoint**, upstream vLLM
0.32.0 (an upgrade to the latest release is possible at once), and these arguments:

```
--kv-cache-dtype fp8 --max-model-len 262144 --gpu-memory-utilization 0.95
--speculative-config.method mtp --speculative-config.num_speculative_tokens 5
--tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --tensor-parallel-size 4
```

Nothing else is set: no CPU KV offload, no decode context parallelism, scheduler defaults.

The study asks **how many concurrent coding sessions one replica serves within the SLO**, with that
configuration and with the levers it leaves unused, so the result can be handed back as a new
argument line.

H200 capacity is not available on demand (8-GPU instances only, through Capacity Blocks), so the
study runs on **4x RTX PRO 6000 Blackwell (g7e.24xlarge, spot)**. With four 96 GB GPUs the
**target's own FP8 checkpoint fits with the target's TP4**, which the 2-GPU spike could not do.
The GPU architecture differs (SM120 vs SM90), so kernels and absolute numbers do not transfer.
What transfers is which levers matter and roughly how much. The whole harness can be re-run on
H200 later.

## Fairness and what transfers

- Same checkpoint, same TP, same arguments as the target for the baseline; only the GPU and the
  vLLM version differ, plus `--language-model-only` (text-only coding; the target loads the vision
  tower).
- On SM120 the sparse MLA backend forces `fp8_ds_mla` KV and the KV block size; on SM90 they are
  choices. So `kv_cache_dtype` and `block_size` are not levers here, and their effect on the
  target stays untested.
- KV room per GPU is smaller than on H200 (96 vs 141 GB per GPU for the same ~76 GiB of weights
  per GPU), so this rig hits the cache wall at fewer sessions than the target would. Results are
  reported as sessions on this rig, never as a sessions number for the target.

## Prior results this builds on

- **Studies 47 and 48:** users within a TTFT SLO scale with total KV capacity (GPU + RAM); FP8 KV
  and CPU offload multiply; `/dev/shm` must exceed the offload size; with offload, compare the
  combined prefix hit rate. Ramp + Hold harness, score `rate x period - TTFT/1000`.
- **Study 51:** same Ramp + Hold harness on g7e, two AIPerf clients, full GPU / node / container
  telemetry (113 metrics), NCCL checks in `apply_config.sh`.
- **Spike 2026-11-17 (2x RTX PRO 6000, W4A16 checkpoint, v0.34.0):**
  - upstream picks `FLASHINFER_MLA_SPARSE_SM120` (only candidate) with `fp8_ds_mla`, vLLM's custom
    all-reduce first, Marlin for W4A16 MoE;
  - the KDA recurrent state sets the KV block to 4352 tokens (`mamba_cache_mode align`), so prefix
    hits land on 4352-token boundaries;
  - v0.34.0's extensible KV cache sizes the cache after warmup; `max_num_seqs` must not exceed the
    number of Mamba blocks (default 1024 failed);
  - start-up: weights ~30 s for 178 GiB, TileLang JIT warmup 213 s (cacheable on disk), FlashInfer
    autotune ~70 s;
  - MTP: open upstream bug vllm#59724 (0 % acceptance with the SM120 sparse MLA backend, reported on
    FlashInfer 0.7.0; v0.34.0 ships 0.7.3);
  - GPU-to-GPU on the 12xlarge: PIX, P2P on, NCCL all-reduce ~21 us up to 128 KiB.

## Goals

1. The baseline: sessions within SLO with the target's arguments on this rig.
2. Which unused levers move that number, one at a time: CPU KV offload, decode context
   parallelism, MTP depth, scheduler sizing.
3. An Akamas-optimized configuration and its gain over the baseline, as an argument line.
4. A harness (load, telemetry, workflow) that can be re-run on H200 unchanged except for the node.

## Non-goals

- Multi-replica serving, KServe, the llm-d router (prefix-aware routing across replicas): a later
  study.
- Accuracy evaluation beyond a smoke check (one coding prompt, one tool call per configuration in
  the smoke).
- Vision inputs.
- vLLM 0.32.0 on this rig: the upgrade gain measured on SM120 would not be the target's.

## Anti-goals (a failure even if it runs)

- A baseline that is not the target's argument line (beyond the listed differences) without
  saying so in the README.
- A score that rewards configurations by cache-starving a few sessions: every level must hold the
  SLO for all its sessions, client-side.
- Reporting rig numbers as the target's expected numbers.
- An experiment silently run with other arguments than Akamas chose (study 49): the apply script
  checks the pod's real arguments and `vllm:cache_config_info`.

## Decided (with the user)

- **Rig:** one g7e.24xlarge **spot** in us-east-2c (score 8, ~8.83 USD/h on 2026-11-17), cluster
  `vllm-bench`, node group `llm-serving-g7e-24xl-spot`. Waits for the spot G vCPU quota in
  us-east-2 (64 -> 192 requested on 2026-11-17, AWS case 179491438900392).
- **vLLM v0.34.0** (`vllm/vllm-openai:v0.34.0`), the target's planned upgrade.
- **Workload:** coding-agent sessions with long contexts (shape below is my call).
- **SLO:** client **TTFT p95 <= 5 s** and client **ITL p95 <= 50 ms** (the target gave none; the
  user's choice, 2026-11-17).
- **Goal:** maximize concurrent sessions within the SLO.
- **Telemetry:** the full catalog (working rule 9 in `CLAUDE.md`).
- No KServe: a plain pod that keeps the target's container shape (below).

## My calls (the user can override)

- **Deployment shape:** one fresh pod per experiment, same command as the target
  (`vllm serve /mnt/models --served-model-name ... --port 8000
  --disable-access-log-for-endpoints /health,/metrics,/ping --shutdown-timeout 40
  ${VLLM_ADDITIONAL_ARGS}`); Akamas renders `VLLM_ADDITIONAL_ARGS`. The checkpoint sits on a node
  hostPath; the pod's init step downloads it if missing (a reclaimed spot node comes back empty).
  Compile caches (`/root/.cache`, `/root/.tilelang`, `/root/.triton`) on a hostPath.
- **Session shape (AIPerf 0.11.0, user-centric multi-turn, history re-sent):**
  - shared system prompt **20k tokens**, identical across sessions
    (`--shared-system-prompt-length`; the agent's instructions and tool schemas);
  - per-session context **10k tokens** (`--user-context-prompt-length`; files read at start);
  - **16 turns** per session; each turn adds **4k input tokens** (tool output, user message) and
    produces **400 output tokens** (`ignore_eos`; reasoning included);
  - so a session grows from ~34k to ~100k tokens;
  - one turn per session every **30 s** (`--user-centric-rate SESSIONS/30`): agent tool time plus
    response time. Sessions = 30 x completed requests/s whenever the server keeps up.
  - The period must exceed the slowest answer the SLO allows: 400 tokens x 50 ms + 5 s TTFT =
    25 s < 30 s. AIPerf sends a session's next turn only after the previous answer is back, so a
    shorter period (the first draft's 20 s with 500 tokens: up to 30 s per answer) would let
    sessions slip, trip the sent check or under-count live sessions (builder check, 2026-11-17).
- **Ramp levels: 4, 8, 12, 16, 24, 32, 48, 64 sessions.** Each level is a fresh AIPerf run of
  **960 s + 60 s grace**, judged on requests started >= **500 s** after its first request (a
  session lives 16 x 30 s = 480 s, so the session mix is steady by then). A level passes when:
  client TTFT p95 <= 5 s, client ITL p95 <= 50 ms (each the worse of the two AIPerf clients),
  errors < 1 %, >= 95 % of sent requests completed, >= 95 % of expected requests sent. The ramp
  stops at the first failing level.
- **The SLO is pooled over a level's requests** (p95, as in studies 48 and 51), not per session:
  a per-session p95 over ~15 requests is close to a per-session max. Each level also prints the
  worst session's TTFT p95 and how many sessions exceed 5 s, so a configuration that starves a few
  sessions shows in the logs and the README.
- **First turns are part of the tail on purpose:** 1 turn in 16 is a session's cold start (~14k
  new tokens after the shared system prompt), about the size of the p95 tail. A coding agent's
  first answer counts, so it stays in.
- **Hold:** the last passing level, fresh, 1080 s; Akamas trim `[520s, 90s]`.
- **Score:** `30 x vllm.request_success_rate` (sessions held). No TTFT tie-break (the user's call,
  2026-11-17): study 48 found the rate term noisy by about ±0.5 session over the window, as large
  as any tie-break. Configurations at the same level are compared on TTFT and ITL directly in the
  README.
- **Constraints (Hold window, safety net):** server TTFT p95 <= 6 s and server ITL p95 <= 60 ms.
  With MTP the server-side ITL histogram may count engine steps rather than tokens; the smoke
  compares it with the client ITL before the constraint is written.
- **Steps:**
  1. `baseline`: the target's arguments, TP4, `--language-model-only`. MTP 5 if the smoke shows
     non-zero acceptance; otherwise MTP off, written in the README as a rig difference.
  2. Presets, one lever each on top of the baseline:
     - `Offload 256 GiB` (native backend; `/dev/shm` sized to match);
     - `DCP four` (`decode_context_parallel_size 4`), if the smoke starts it on SM120;
     - `MTP three` and `MTP one` (`spec_tokens`), if MTP works;
     - `Baseline repeat`.
  3. `optimize`: Akamas, **10 experiments**, over the parameters below.
- **Parameters (vLLM pack; domains to check against the installed pack before writing):**

  | Parameter | Domain | Note |
  |---|---|---|
  | `kv_offloading_size` | 0-512 GiB | pack max 512; node RAM 1 TiB |
  | `gpu_memory_utilization` | 0.85-0.97 | extensible KV cache; 0.95 is the target's |
  | `max_num_seqs` | 8-128 | must stay under the Mamba block count |
  | `max_num_batched_tokens` | 2048-16384 | >= max_num_seqs |
  | `spec_tokens` | 0-5 | with `spec_method` mtp; 0 = off; only if MTP works |
  | `decode_context_parallel_size` | 1, 2, 4 | only values the smoke starts |
  | `moe_backend` | auto + the FP8 candidates the smoke log lists | |
  | `max_cudagraph_capture_size` | 16-512 | |
  | `async_scheduling` | true, false | |
  | `performance_mode` | as in study 38 | |
  | `optimization_level` | 1-3 | as in study 38 |

  Not levers here: `kv_cache_dtype` and `block_size` (forced on SM120), `tensor_parallel_size`
  (the target's TP4). **Pack gaps (TODO in the README):** `mamba_ssm_cache_dtype` (KDA state
  fp32/bf16), `kv_cache_dtype` `fp8_ds_mla`, `language_model_only`.
- **AIPerf placement:** two clients on a separate CPU node (study 51's `m8a-2xlarge` pattern):
  100k-token prompts make client-side tokenization heavy. Results PVC sized for long contexts
  and cleaned per level (study 41: ~355 MB per run).
- **KPIs (<= 8):** sessions, output tokens/s, TTFT p95, ITL p95, combined prefix hit rate, MTP
  acceptance rate, KV cache usage, GPU 0 utilization. Everything else stays telemetry.
- **Cost estimate:** a configuration that passes ~6 levels: 6 x 17 min + Hold 18 min + restart
  ~10 min ≈ 2.2 h; one that fails early ~45 min. About 16 experiments ≈ 25-30 h ≈ **220-270 USD**
  at the spot price, plus a ~1 h smoke.

## Prerequisites

1. Spot G quota in us-east-2 >= 96 vCPU; node group `llm-serving-g7e-24xl-spot` up
   (`docs/spikes/2026-11-17-glm53-flash-phase0/infra/nodegroup-g7e-24xl-spot.yaml`); dcgm-exporter
   already covers its role (Helm revision 30).
2. **Smoke on the node, before any Akamas resource is created:**
   - the FP8 checkpoint downloads (328 GB) and starts with the target's arguments at TP4; KV
     tokens and Mamba blocks read from the log;
   - MTP acceptance > 0 (`vllm:spec_decode_*`), else MTP off in the baseline;
   - DCP 4 and the MoE backend candidates start or fail;
   - offload 256 GiB starts and the offload counters move under a multi-turn load;
   - start-up time with warm compile caches.
   Go / no-go after the smoke (my calls, builder check 2026-11-17):
   - **FP8 at TP4:** go if it starts with `--max-model-len` >= 131072 (sessions reach ~100k)
     and `--gpu-memory-utilization` <= 1.0, writing any change from the target's arguments in the
     README as a rig difference. Below 131072, stop and bring it back to the user (fallback: the
     W4A16 checkpoint at TP4).
   - **Offload:** if it does not start, or its counters do not move under the multi-turn load (the
     KDA state may not be offloaded on this hybrid model), stop and bring it back to the user
     before creating the study: studies 47 and 48 put the whole gain on cache capacity, and the
     other levers alone may not be worth ~25-30 h.
3. Akamas: vLLM pack version on akamas-41 checked (`akamas describe optimization-pack`), Akamas
   login renewed.

## Left to the builder

- Study folder from `studies/_TEMPLATE`, k8s manifests and scripts adapted from studies 48 and 51
  (`apply_config.sh`, `run_ramp.sh`, `run_hold.sh`, `lib_load.sh`, reaper, checks).
- The two-criteria pass test (TTFT and ITL) in `run_ramp.sh`.
- The exact AIPerf flags for the session shape and how the shared system prompt interacts with
  `--user-centric-rate`.
- Spot reclaim handling in `apply_config.sh` (wait for the node, re-download the checkpoint).
- Telemetry queries per component, the role / GPU-index joins.
