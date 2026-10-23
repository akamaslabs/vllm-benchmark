# 41-l4-dsv2-lite-context-parallel

**Status:** TODO — scaffolded 2026-10-22; nothing created on Akamas yet (waiting for the user's go).
**Dates:** —

## Objective

**Does decode context parallelism (DCP) turn into more long-context throughput (more concurrent
users) at parity of GPUs, and does prefill context parallelism (PCP) help under load?**

On an MLA model (one latent KV head) tensor parallelism replicates the whole latent KV cache on
every TP rank: TP4 pays four copies. DCP shards it across the TP ranks instead (KV x2 at DCP2,
x4 at DCP4, measured: 374,192 -> 724,800 -> 1,524,864 tokens). PCP splits a long prefill across
ranks. The smoke test (`smoke/`) shows that neither helps a *single* request (DCP: TTFT +14-25 %
at 16-31k, TPOT slightly better; PCP: TTFT and TPOT worse); the open question is whether the
capacity DCP buys, or the prefill parallelism PCP buys, wins under a multi-user long-context load
where TP4's KV cache saturates.

- **Goal:** maximize vLLM's **generation token throughput** (`vllm.decode_token_throughput`,
  generated tokens/s, summed over DP engines). Not total tokens/s (decided with the user
  2026-10-22): every turn resends its whole history (~14k tokens, mostly prefix-cache hits) for 256
  generated tokens, and `vllm:prompt_tokens` counts cached tokens too (`v1/metrics/stats.py`), so
  total tokens/s would be ~98 % resent history. With `ignore_eos` and a fixed 256-token reply,
  generation tokens/s = completed turns/s x 256, the closest measure of users served within the
  SLA without the AIPerf pack. Counting only computed prompt tokens would reward configurations
  that miss the prefix cache, so it is not used either.
- **Constraints (SLA):** `vllm.time_to_first_token_p95:max <= 10000` ms and
  `vllm.time_per_output_token_p95:max <= 100` ms over the scored window (150 s p95s; the pack
  metrics are unitless, the telemetry queries return ms). TPOT was 50 ms at design time; the
  calibration (Smoke study, below) showed TPOT > 50 ms breaking every layout at 23-31 users, before
  the TP4 KV cache filled, so the study would have measured decode speed only. 100 ms (10
  tokens/s per user, the usual chat SLA, as study 39) was agreed with the user on 2026-10-23.
- **Parity:** every configuration uses all four L4s (`TP x PCP x DP = 4`; DCP reuses TP ranks).
- **Not the goal:** concurrent users. Akamas 4.1 has no AIPerf pack installed, so the goal reads
  vLLM's own counters; the users ramp is the load, and the best valid window of tokens/s is the
  score (users at the knee are in the AIPerf logs).
- No ROADMAP hypothesis named DCP/PCP before; it continues the parallelism line of studies 7-16
  (H5/H6) on a new axis, and uses the DCP/PCP parameters added to the vLLM pack in 1.4.0
  (`knowledge/notes/2026-07-vllm-pack-v1.4.0-dcp-pcp-parameterconstraints.md`, whose note that
  "dcp > 1 is only useful on MLA" holds: this model is MLA).

## Stack & versions

- **Akamas version:** Studio 4.1.0 (<https://akamas41.lab.akamas.io>, namespace `akamas-41`).
- **Optimization packs** (installed on 4.1, `akamas describe optimization-pack`, 2026-10-22):
  vLLM **1.14.0** (component type `vLLM`, reference engine 0.29.0; has `tensor_parallel_size`,
  `data_parallel_size`, `decode_context_parallel_size`, `prefill_context_parallel_size`,
  `enable_expert_parallel`), Kubernetes **1.9.0**, GPU **1.4.0** (component type `GPU`). No AIPerf
  pack.
- **Workload under test:** `vllm/vllm-openai:v0.32.0` serving
  **`deepseek-ai/DeepSeek-V2-Lite-Chat`** as `dsv2-lite`, StatefulSet `vllm-dsv2` (one pod,
  `vllm-dsv2-0`, all four GPUs) in namespace `llm-serving`. Fixed flags: `--trust-remote-code`,
  `--max-model-len 32768`, `--enable-prompt-tokens-details`; attention backend automatic
  (resolves to `TRITON_MLA` decode + FlashAttention MLA prefill on SM 8.9, measured); prefix
  caching at vLLM's default (on).
  - **Why v0.32.0:** 0.29 rejects PCP + DP and full CUDA graphs with PCP; 0.31 lacks the
    MoE-padding fix. Every flag the study renders was checked against the v0.32.0 source (the
    pack was written for 0.29.0).
- **The model:** DeepSeek-V2-Lite-Chat, 15.7B MoE (64 routed + 2 shared experts, top-6, ~2.4B
  active), MLA with `kv_lora_rank` 512 + rope 64 (one latent KV head), 27 layers, bf16 31.4 GB
  (does not fit one 24 GB L4), YaRN context 163,840, ungated (no HF token).
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), existing node group
  **`llm-serving-l4`**: one g6.12xlarge, 4x NVIDIA L4 24 GB on PCIe (no NVLink, no P2P), 48 vCPU,
  192 GiB, On-Demand, label `node-role=llm-serving-l4`, ASG tagged `AlwaysOn=true`. Node up since
  2026-10-22 09:06 UTC in **us-east-2a** (`ip-192-168-64-57`). The model cache PVC
  `vllm-model-cache-dsv2lite` (60Gi gp3-ephemeral) already exists, bound in us-east-2a, model
  downloaded. Provisioning and scaling: `infra/`.
- **Load generator:** AIPerf **0.11.0**, Kubernetes Job `aiperf-dsv2` on the system node
  (`node-role=system-m8a`, m8a.xlarge, us-east-2b), streaming chat endpoint, closed-loop users
  ramp `--concurrency N --concurrency-ramp-duration RAMP` (as study 38's `run_benchmark.sh`),
  synthetic multi-turn long-context sessions (`k8s/lib_workload.sh`, flags checked against the
  0.11.0 wheel). See "Design".
- **Telemetry:** Prometheus only (in-cluster kube-prometheus-stack): vLLM `/metrics` through
  ServiceMonitor `vllm-dsv2`, dcgm-exporter (already on the node) for one component per physical
  GPU (`gpu0`..`gpu3`, study 16's modelling: per-GPU series, never label splitting), cAdvisor and
  kube-state-metrics for the containers and nodes.

## Parameters tuned

Component `vllm` (type `vLLM`, pack 1.14.0). Every value goes through `k8s/params.env.template`
-> `k8s/render_statefulset.sh` (booleans as `--x` / `--no-x`; an empty value would omit the flag,
used only for the fixed prefix-caching key).

| Parameter | Domain (pack domain) | Baseline | Why |
|---|---|---|---|
| `vllm.tensor_parallel_size` | {1, 2, 4} = [1, 4] minus 3 ([1, 16]) | 4 | the usual layout for a model that does not fit one L4 |
| `vllm.data_parallel_size` | {1, 2, 4} ([1, 8]) | 1 | replicas instead of shards |
| `vllm.decode_context_parallel_size` | {1, 2, 4} ([1, 16]) | 1 | the study's main lever: shards the latent KV across TP ranks |
| `vllm.prefill_context_parallel_size` | {1, 2, 4} ([1, 8]) | 1 | splits long prefills across ranks |
| `vllm.enable_expert_parallel` | true, false | false | whole experts per rank vs TP-sharded experts |
| `vllm.max_num_seqs` | 16-512 (1-4096) | 256 (vLLM 0.32.0 default on a < 70 GiB GPU) | admission cap per engine |
| `vllm.max_num_batched_tokens` | 2048-32768 (1-65536) | 2048 (vLLM 0.32.0 default on a < 70 GiB GPU) | prefill budget per step: TTFT of 8-30k prompts |
| `vllm.gpu_memory_utilization` | 0.80-0.94 (0-1) | 0.90 (repo convention; vLLM default 0.92) | the KV budget per GPU |

Defaults from `vllm/engine/arg_utils.py` `get_batch_defaults` (OpenAI server, device memory < 70
GiB: 2048 / 256, confirmed in the smoke log "Chunked prefill is enabled with
max_num_batched_tokens=2048").

**parameterConstraints** (`akamas/gen_study.py`; every rule is also enforced by
`render_statefulset.sh`, and `akamas/check_offline.py` checks that both accept exactly the same 13
layouts):

| Rule | Formula | Source |
|---|---|---|
| sizes in {1, 2, 4} | `<size> != 3` for TP, DP, DCP, PCP | domain [1, 4] (study 16's precedent) |
| parity: 4 GPUs | `tp * pcp * dp == 4` | design (always all four GPUs) |
| DCP divides TP | `dcp <= tp` (equivalent to `tp % dcp == 0` for sizes in {1, 2, 4}; no modulo operator is documented for Akamas formulas) | `vllm/config/parallel.py` |
| PCP needs DCP 1 | `pcp == 1 \|\| dcp == 1` | `v1/worker/gpu/pcp_manager.py`: "MRV2 PCP + DCP supports sparse MLA models only" (this model is dense MLA) |
| PCP needs EP | `pcp == 1 \|\| ep == "true"` | smoke run 1: TP2xPCP2 without EP crashed (`topk_softmax ... is_padding size mismatch`) |
| PCP without DP | `pcp == 1 \|\| dp == 1` | PCP + DP unverified (vLLM PR #49246 open) |
| TP1 only with EP | `tp > 1 \|\| ep == "true"` | agreed design; see "Deviations from the spec" for the reason |
| batched >= seqs | `max_num_batched_tokens >= max_num_seqs` | `vllm/config/scheduler.py` |

Feasible layouts (TP, DP, DCP, PCP, EP): TP4 / TP4+DCP2 / TP4+DCP4 (EP on or off), TP2xDP2 /
TP2xDP2+DCP2 (EP on or off), TP2xPCP2+EP, TP1xPCP4+EP, DP4+EP.

**Fixed, with the reason:**

| Setting | Value | Why |
|---|---|---|
| `--max-model-len` | 32768 | covers the load (worst case ~28k); YaRN native 163840 |
| prefix caching | vLLM default (on), `ENABLE_PREFIX_CACHING=` | realistic for multi-turn; **DCP/PCP + prefix caching unverified** (the smoke ran with it off): checked by the calibration run, pinned off with a note if a layout fails |
| attention backend | automatic | `TRITON_MLA` decode + FA MLA prefill on SM 8.9 (measured), the only MLA decode backend there |
| `--trust-remote-code` | on | as the smoke run (the architecture is native in vLLM) |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 1800 | starts took 350-535 s, the default is 600 s |
| `VLLM_CACHE_ROOT` | on the model PVC | a layout's torch.compile cache survives the next start |
| pipeline parallelism, speculative decoding, block size, CUDA graph settings | vLLM defaults | out of scope: the question is the KV layout |

## Design

### Smoke-test evidence (2026-10-22, `smoke/`)

One request at a time, prefix caching off, gmu 0.90, max_model_len 32768, vLLM 0.32.0:

| Layout | KV tokens | TTFT 4k / 16k / 31k (ms) | TPOT@128 4k / 31k (ms) | Needle |
|---|---|---|---|---|
| TP4 | 374,192 | 455 / 1991 / 4439 | 8.3 / 12.5 | 9/9 |
| TP4+DCP2 | 724,800 | 462 / 2268 / 5563 | 9.0 / 10.9 | 9/9 |
| TP4+DCP4 | 1,524,864 | 460 / 2271 / 5583 | 9.2 / 10.7 | 9/9 |
| TP1xPCP4+EP | 292,896 (replicated) | 377 / 2214 / 6240 | 17.5 / 22.0 | 9/9 |
| TP2xPCP2+EP | 319,776 | 466 / 2273 / 5567 | 13.9 / 18.2 | 9/9 |
| DP4+EP | 292,960 per engine | 590 / 2979 / 7543 | 16.8 / 21.3 | 9/9 |
| TP2xPCP2 (no EP) | — | fails: `dispatch_topk_launch ... topk_softmax_kernels.cu:785, is_padding size mismatch, expected: 4096` | | |
| TP2+EP (run 1) | 119,936 | 941 / 4627 (6k/26k prompts) | 11.8 / 13.2 (6-token replies) | 6/6 |

Backends: `TRITON_MLA` (decode), `FLASH_ATTN MLA prefill backend`. Startup 350-535 s per layout.

### Load and why it is sized this way

- **Synthetic multi-turn long-context chat** (`k8s/lib_workload.sh`): 32 document-like shared
  prefixes of 8192 tokens (`--prefix-prompt-pool-size 32 --prefix-prompt-length 8192`), per turn
  2048±512 input / 256±64 output tokens (`ignore_eos`), 5±1 turns (AIPerf rounds up: ~5.5 on
  average), 10±3 s think time, 4000 unique sessions, fixed seed 40, 60 s warm-up on other
  sessions (seed 7).
- **Context per sequence** ≈ 8k + k x 2.3k: ~19.7k at turn 5, ~26.4k at turn 8 (+2.5 sigma), below
  32768.
- **Closed loop, users ramp:** `--concurrency N --concurrency-ramp-duration RAMP` (AIPerf 0.11.0:
  a concurrency slot is a *session*, held across the think time, so N = concurrent users).
  **N = 60, RAMP = 3600 s** (`RT_USERS` / `RT_RAMP_S` in the workflow): one user a minute, set from
  the calibration run (TP4 broke the SLA at ~27 users, DCP4 at ~31, on a 96-user/1800 s ramp).
- **Why it saturates where it should:** TP4 holds 374,192 KV tokens, i.e. ~19-26 in-flight
  sequences of 14-20k; with ~10 s think time out of a ~15-25 s turn cycle, roughly half the users
  have a request in flight, so TP4's KV fills around 40-50 users — mid-ramp — and preemption /
  queueing then drive TTFT up. DCP4 holds 1,524,864 tokens (~75-100 sequences), so if its decode
  and prefill speed hold, its knee should sit much later in the same ramp. The ramp must be long
  and high enough that DCP4's knee is not capped by its end (hence the calibration on both).
- **Watchdog** (`k8s/run_test.sh`, study 39's pattern): ends the run with success once vLLM's TTFT
  p95 > 20000 ms or TPOT p95 > 200 ms (2 x the SLA) holds for 120 s; the trial fails on a Job
  failure, a vLLM restart, a stall or the deadline (RAMP + 2400 s).
- **Scoring:** `stability` windowing on `vllm.decode_token_throughput`, width 6 samples (150-180 s
  at 25-30 s scrape), `maxStdDev` 300000000 (filter disabled), `when: max`: the valid window with
  the most tokens/s; constraints read the window's `:max` p95s (no "window after breach"
  artefact).

### Steps

| # | Step | Values | Why |
|---|---|---|---|
| 1 | baseline (2 trials) | TP4, vLLM 0.32.0 defaults, gmu 0.90 | reference + first noise measure |
| 2 | tp4 dcp2 | baseline + DCP2 | KV x2 |
| 3 | tp4 dcp4 | baseline + DCP4 | KV x4 |
| 4 | tp1 pcp4 ep | TP1 x PCP4 + EP | PCP at its widest |
| 5 | tp2 pcp2 ep | TP2 x PCP2 + EP | PCP with sharded KV |
| 6 | dp4 ep | TP1 x DP4 + EP | replicas, the classic throughput layout |
| 7 | optimize | AKAMAS, 0 init, 60 experiments, `maxFailedExperiments` 20 | as studies 27-39 |

`kv_cache_dtype` is not tuned: fp8 crashed TRITON_MLA on the L4 (calibration, below), so it is
pinned to vLLM's default (`auto`, bf16) and `render_statefulset.sh` refuses fp8.

### Calibration (Smoke study `41-L4-DSV2-Lite-Context-Parallel-Smoke`, 2026-10-22)

Workflow `...-Smoke-Workflow`, users 1 -> 96 over 1800 s, SLA then TTFT p95 <= 10 s / TPOT p95
<= 50 ms (the manifest in git is the one that ran). Each trial lasted ~15-20 min: the watchdog
(then at TPOT 100 ms) ended every run 11-14 min into the ramp. Values read from vLLM's Prometheus
series at the first watchdog breach:

| exp | layout | users at 1st breach | KV usage | waiting | TPOT p95 | TTFT p95 | prefix-cache hit rate | score (gen tok/s) |
|---|---|---|---|---|---|---|---|---|
| 1 | TP4 baseline | ~27 | 89 % | 4 | 119 ms | 3.3 s | 69 % | 94.1 (VALID) |
| 2 | TP4 + DCP4 | ~31 | 25 % | 0 | 100 ms | 3.7 s | **82 %** | 115.2 (+22 %, VALID) |
| 3 | TP4 + DCP4 + fp8 KV | — | — | — | — | — | — | ERROR: `OutOfResources: shared memory` in a Triton MLA kernel (worker `TP0_DCP0`) at the first requests |
| 4 | TP1 x PCP4 + EP | ~23 | 80 % | 0 | 137 ms | 6.3 s | 62 % | 149.2 (CONSTRAINTS_VIOLATED) |

- The load hits what it was sized for: TP4's KV cache was at 89 % with requests queueing at ~27
  users, DCP4's at 25 %.
- Prefix caching works with DCP and PCP; DCP4 keeps more history in cache (82 % vs 69 % hit rate).
- TPOT <= 50 ms broke first in every layout (decode of ~15k-token contexts with 20-30 running
  sequences on the L4), before the KV cache mattered: hence the 100 ms SLA.
- Context measured over the whole Smoke run (vLLM `request_prompt_tokens`): median 15.3k, mean
  14.9k, p10 10.1k prompt tokens (p90 above the 20k histogram bucket; ~28k max by design), 249
  generated tokens per request, 73 % of prompt tokens served from the prefix cache.
- PCP was the worst layout, as in the single-request smoke test.
- The first start needed `akamas finish study` + `start` (Akamas 4.1 timed out waiting for the
  Airflow DAG), and the first trial failed on a `render_statefulset.sh` bug (a commented `${...}`
  read as a leftover token), fixed with a regression test before the trials above.

**KPIs (8):** Generation throughput, TTFT P95 150s, TPOT P95 150s, Completed requests, KV cache
usage (max), Preemptions, Waiting requests (max), Prefix cache hit rate. Total tokens/s and the
per-GPU series (`gpu0`..`gpu3`) are still collected as metrics.

## Before starting

State on 2026-10-23: system, components, telemetry and the Smoke study exist on Akamas 4.1; the
Kubernetes layer is applied; the node group `llm-serving-l4` was scaled to 0 after the Smoke study
(the model PVC stays in us-east-2a). Steps 1-5 below were done for the Smoke study; for the main
study: scale up (1-2), sync the changed files to the toolbox, create the main workflow and study,
start it.

1. **Node in us-east-2a.** The model PVC is an EBS volume in us-east-2a; the ASG spans 2a/2b/2c.
   After any scale-up: `kubectl get nodes -l node-role=llm-serving-l4 -L
   topology.kubernetes.io/zone` must say us-east-2a (`infra/provision.sh --scale 1` checks it;
   remedies in `infra/README.md`, "Wrong AZ").
2. **GPUs free.** No other pod on the node may request `nvidia.com/gpu` (the smoke pod
   `dsv2-smoke`, other studies' `vllm` / `vllm-pd` in `llm-serving`): `apply_config.sh` refuses to
   start otherwise (exit 5).
3. **Kubernetes layer:** `AWS_PROFILE=lab infra/provision.sh` (creates the AIPerf PVCs, the
   Services and the ServiceMonitor if missing; never touches the model PVC).
4. **Sync the repo to the toolbox** (`/work/vllm-benchmark`), then create the system, components,
   telemetry and the Smoke workflow/study (`akamas/README.md`).
5. **Calibration run** — done 2026-10-22, results above (Smoke study: TP4 baseline, TP4+DCP4,
   TP4+DCP4 fp8 KV, TP1xPCP4+EP, 1 -> 96 users over 1800 s). It read:
   - the users at each knee (AIPerf log, `kv_cache_usage`, `num_requests_waiting`): set
     `RT_USERS` / `RT_RAMP_S` in the main workflow so TP4 breaks the SLA around mid-ramp and
     TP4+DCP4 still has headroom before the end (raise N if DCP4 never breaks);
   - the AIPerf dataset-generation time (4000 long sessions; the deadline budgets ~15 min);
   - **prefix caching with DCP:** the startup log line `enable_prefix_caching=True` and a
     `prefix_cache_hit_rate` > 0 on the DCP4 trial. If DCP or PCP fails with prefix caching on,
     set `ENABLE_PREFIX_CACHING=false` in `k8s/params.env.template`, document it here, and
     regenerate nothing else (it is a fixed key). PCP layouts are not in the calibration: their
     first run is the main study's presets 5-6 — watch them;
   - that `gpu0`..`gpu3` show four distinct series in Akamas and the vLLM metrics sum over DP
     engines.
6. Delete the Smoke study, create the main workflow and study, start it.
7. **Nightly stop:** the ASG is already `AlwaysOn=true`; check the instance tag after any
   replacement.

## Deviations from the spec

- **No `tp == 1 -> enable_expert_parallel` rule (dropped with the user, 2026-10-22).** The spec's
  reason ("31.4 GB bf16 does not fit 24 GB without sharding experts") does not hold in vLLM 0.32.0:
  without EP the experts are TP-sharded across all `dp x pcp x tp` ranks
  (`model_executor/layers/fused_moe/config.py`, `flatten_tp_across_dp_and_pcp`), so DP4 without EP
  fits. It is a feasible, not yet measured layout for the optimizer.
- **Goal on generation tokens/s, not total tokens/s** (decided with the user, 2026-10-22; see
  Objective).
- **`tp % dcp == 0` written as `dcp <= tp`:** equivalent for sizes in {1, 2, 4}; Akamas formulas
  have no documented modulo operator.
- **Watchdog on TPOT too:** the spec named TTFT p95 > 2 x SLA; the watchdog also stops at TPOT p95
  > 2 x SLA (200 ms since the 100 ms SLA; 100 ms in the Smoke study), as study 39's did, since past either breach every later window is invalid.
- **AIPerf `--tokenizer-trust-remote-code`** added to the workload (the checkpoint ships remote
  code; vLLM is started with `--trust-remote-code`). The local dry run loaded the tokenizer fine.
- **Calibration as an Akamas Smoke study** (baseline + `tp4 dcp4` on a 1800 s ramp) rather than
  ad-hoc runs, as study 39 did: it also tests the whole pipeline and the telemetry.
- **Names `vllm-dsv2*` / `aiperf-dsv2`** instead of study 39's `vllm` / `aiperf-cpu`: namespace
  `llm-serving` (forced by the model PVC) already has another study's Deployment, Service and
  ServiceMonitor named `vllm` selecting `app: vllm`.
- **DCGM queries without a pod filter:** study 16 filtered DCGM on `pod` / `exported_pod`; the live
  exporter (2026-10-22) has no `exported_pod` label. The `gpuN` components filter on the GPU index
  and on the node role (join on the exporter's `hostname` label), verified live.
- **Additions not in the spec:** `VLLM_CACHE_ROOT` on the model PVC (compile cache reuse),
  `VLLM_ENGINE_READY_TIMEOUT_S=1800`, 4000 AIPerf sessions (sizing in `k8s/lib_workload.sh`).

## Pack TODOs

For whoever manages the vLLM pack (outside this repo, <https://gitlab.com/akamas/optimization-packs/vllm>).
None blocks the study.

- **Reference engine 0.29.0 vs 0.32.0:** every flag this study renders was checked against
  0.32.0; the pack's other parameters were not.
- **DCP/PCP descriptions:** worth stating in the pack that DCP is only useful on MLA models
  (KV shards across TP ranks; dcp <= tp / num_kv_heads), and that in 0.32.0 PCP + DCP is
  sparse-MLA-only and PCP runs only on the V2 model runner.
- **No AIPerf pack** on 4.1: a users-based goal (users at the knee) would need one.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
