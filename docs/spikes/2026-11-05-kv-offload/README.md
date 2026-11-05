# KV cache offload spike (study 47 preparation), 2026-11-05

Manual A/B to check whether vLLM's CPU KV offload helps a multi-turn chat workload before
building study 47. Throwaway: no Akamas study, run by hand. Times are UTC.

## Setup

- Node: node group `llm-serving-g7-kv` (created for study 47, label `node-role=llm-serving-g7-kv`,
  taint `nvidia.com/gpu=present:NoSchedule`), one g7.4xlarge: RTX PRO 4500 Blackwell 32 GB,
  16 vCPU, 64 GiB (57 GiB allocatable), us-east-2b. GPU exposed by the kube-system
  nvidia-device-plugin (no DRA, no gpu-sharing plugin). Up 08:15–10:14, scaled back to 0.
  A new node group was needed because the old StatefulSet `gpu-sharing/vllm` takes the GPU of
  `llm-serving-g7-4500` whenever that node group comes up (see the study 46 night log).
- vLLM 0.29.0 (`vllm/vllm-openai:v0.29.0`), Qwen/Qwen3-8B-FP8, max-model-len 16384, prefix
  caching on, gmu 0.90, max_num_seqs 128; pod 10 CPU / 44 GiB. GPU KV cache: 16.53 GiB =
  **120,368 tokens** (~144 KiB per token).
- Offload (run B): `--kv-offloading-size=24 --kv-offloading-backend=native` → 25.77 GB buffer
  (~175k tokens). Manifests: `vllm-A.yaml` (no offload), `vllm-B.yaml` (offload).
- Load: AIPerf 0.11.0 in a pod on the same node (4 CPU), `run.sh`: synthetic chat, 600-token user
  turns, 200-token answers (`ignore_eos`), 10 turns per conversation, history re-sent every turn
  (ISL 600, 1402, 2204, …, ~7.8k), `--user-centric-rate R --num-users N` (each user sends a turn
  every N/R seconds), 900 s + 60 s grace, 60 s time slices. `summ.sh` prints counter deltas,
  the headline metrics and TTFT per minute. Raw CSVs and /metrics snapshots in `raw/`.

## Results

| Run | Load | Offload | TTFT p50 / p95 (whole run) | TTFT p50 / p95, minutes 6–11 | GPU prefix hits | CPU (offload) hits | Preemptions |
|---|---|---|---|---|---|---|---|
| A64 | 64 users, 2 req/s | no | 34.3 s / 54.2 s | 25–42 s | 1.9 % | – | 35 |
| B64 | 64 users, 2 req/s | 24 GiB | 34.3 s / 54.2 s | 24–44 s | 1.9 % | 4.1 % | 524 |
| B64r1 | 64 users, 1 req/s | 24 GiB | 262 ms / 788 ms | 375–502 ms / 578–894 ms | 3.3 % | 7.2 % | 0 |
| **A40** | 40 users, 0.625 req/s | no | 236 ms / 734 ms | **380–480 ms / 650–844 ms** | 12.7 % | – | 0 |
| **B40** | 40 users, 0.625 req/s | 24 GiB | 175 ms / 671 ms | **181–196 ms / 196–225 ms** | 12.7 % | 42.2 % (48 % of what the GPU missed) | 0 |

Hits are % of prompt tokens. Output throughput is the same within each pair (A64/B64 313 tok/s,
A40/B40 125 tok/s): the load is open-loop, so offload changes latency, not throughput.

1. **Offload works when the active context fits between GPU and GPU+CPU.** With 40 users
   (~175k tokens of live conversations, more than the GPU's 120k, less than GPU + CPU ~295k) the
   offload served 42 % of the prompt tokens from CPU RAM. From minute 6 to 11, TTFT p50 was halved
   (~190 ms vs ~385 ms) and p95 was 3× lower (~210 ms vs ~700 ms); ITL avg 14.9 vs 16.1 ms.
   Minutes 12–14 B40 rose to 350–480 ms as the conversations grew to ~4.5k tokens and the
   working set approached the CPU buffer size.
2. **Above GPU+CPU capacity the cache thrashes and offload gives nothing.** With 64 users
   (~250k tokens active, cyclic access: each user returns every 32–64 s) LRU evicted every
   conversation just before its user returned: offload hits 4–7 %. At 2 req/s both runs
   collapsed from minute 3–4 (TTFT p50 up to 54 s): recomputing whole histories saturated the GPU.
   With offload the preemptions went from 35 to 524 in that overloaded regime (not investigated).
3. **AIPerf multi-turn is correct.** One user, 4 turns: 4,832 of 7,292 prompt tokens hit (66 %).
4. **Qwen3 chat template:** the latest assistant answer is never a cache hit: the template
   re-renders past assistant turns (it drops the empty `<think></think>` that generation started
   with), so the prefix match stops where the previous answer begins. Tested with two turns:
   thinking off → hit exactly the first user turn (784 of 789 tokens, block 16). Everything before
   that answer is reused. Costs ~200 tokens per turn here.

## Gotchas for study 47

- **/dev/shm must hold the whole offload buffer.** The native backend mmaps the buffer in
  /dev/shm; with the usual 1–2 GiB emptyDir vLLM fails at startup: `RuntimeError: Insufficient
  space in /dev/shm: 24574 MiB required, 2048 MiB free`. The shm emptyDir (medium Memory) counts
  against the pod memory limit: size shm ≥ offload GiB + ~1 and the pod limit ≥ offload + ~20 GiB.
- Node RAM bounds the offload size: 57 GiB allocatable on g7.4xlarge → offload ≤ ~32 GiB with
  room for vLLM itself and the load generator.
- Metrics present in vLLM 0.29 with offload: `vllm:external_prefix_cache_hits_total` /
  `_queries_total` (CPU hits), `vllm:kv_offload_total_bytes_total{transfer_type=GPU_to_CPU|CPU_to_GPU}`,
  `vllm:kv_offload_total_time_total`, `vllm:kv_offload_cpu_cache_usage_perc`, store/load size
  histograms. None of these are in the vLLM optimization pack 1.14.0 yet.
- The benefit is in TTFT at a given load, and only in the band where the working set fits
  GPU+CPU but not GPU. An SLA of 1500 ms TTFT is met by both A40 and B40, so "users within SLA"
  needs a tighter TTFT target, longer contexts, or a load sweep to show a difference.
- Cost: ~2 h of g7.4xlarge (~6 USD).
