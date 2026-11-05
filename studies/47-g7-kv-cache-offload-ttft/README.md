# 47-g7-kv-cache-offload-ttft

**Status:** RUNNING (started 2026-11-05 11:53 UTC on akamas-41; restarted from scratch 12:26 UTC after the ITL constraint fix, see below). **Design:**
`docs/superpowers/specs/2026-11-05-study-47-kv-cache-offload-design.md`. **Plan:**
`docs/superpowers/plans/2026-11-05-study-47-kv-cache-offload.md`. **Spike:**
`docs/spikes/2026-11-05-kv-offload/README.md`.

## Objective

A chat assistant whose users hold conversations of many turns and come back after reading the
answer. Each turn re-sends the whole history. When the live conversations do not fit in GPU
memory, vLLM either recomputes evicted history (slow prefill, high time to first token) or, with
CPU KV offload (`--kv-offloading-size`), reloads it from RAM.

**Goal:** minimize **TTFT p95** of a fixed load in steady state. The levers against KV memory
pressure are tuned together:

- CPU offload size (0 = off);
- FP8 KV cache;
- `gpu_memory_utilization`;
- batching;
- block size.

Presets measure each lever alone. The workload was fixed from the scenario before any offload
run (design, "Fairness"), so offload is not favoured by construction.

**Constraints:**

- ITL p95 ≤ 50 ms (Akamas constraint).
- Errors < 1 % and ≥ 95 % of the sent requests completed. `run_load.sh` fails the trial
  otherwise (the load is open-loop).

## Stack & versions

- **Hardware:** one g7.4xlarge, node group `llm-serving-g7-kv`:
  - NVIDIA RTX PRO 4500 Blackwell 32 GB;
  - 16 vCPU, 64 GiB (57 GiB allocatable);
  - 3.04 USD/h.

  EKS `vllm-bench` 1.36.5, us-east-2. AlwaysOn tag while the study runs. See `infra/README.md`.
- **Server:**
  - `vllm/vllm-openai:v0.29.0`, model `Qwen/Qwen3-8B-FP8` served as `qwen3-8b`;
  - `--max-model-len 16384`, prefix caching on;
  - 10 CPU, memory = offload + 20 GiB, `/dev/shm` = offload + 1 GiB.
- **Load:**
  - AIPerf 0.11.0, synthetic multi-turn chat, `--user-centric-rate 1 --num-users 64`, i.e. each
    user sends one turn every 64 s;
  - 10 turns per conversation, 600-token user turns, 200-token answers (`ignore_eos`), history
    re-sent;
  - 1200 s + 60 s grace per experiment.

  Runs in a pod on the GPU node (4 CPU / 6 GiB). Frozen in `k8s/workload.env`.
- **Akamas:** 4.1.0 (`akamas-41`), packs vLLM **1.15.0** (offload parameters and metrics, MR
  https://gitlab.com/akamas/optimization-packs/vllm/-/merge_requests/12) and Kubernetes
  **1.9.0**. Resources and setup in `akamas/README.md`.
- **Telemetry:** Prometheus (kube-prometheus-stack). The ServiceMonitor `vllm-kv-offload`
  scrapes vLLM every 5 s.

## Calibration (2026-11-05, before freezing the workload)

The rule was fixed in the design before any data:

- the baseline keeps up if ≥ 95 % of the sent requests complete and the client TTFT p95 over
  minutes 6–20 is < 10 s;
- otherwise drop to 56 users, then 48.

Baseline (vLLM defaults, no offload) at 64 users, 11:19–11:40 UTC:

| Metric | Value |
|---|---|
| Requests completed | 839 of 839 |
| Errors | 0 |
| Client TTFT p50 / p95, minutes 6–20 | 408 / 884 ms |
| Server TTFT p95, Prometheus | 0.94 s |
| ITL p95 | 33.1 ms |
| GPU prefix-cache hits | 2.5 % of prompt tokens |

The GPU KV cache holds 123,440 tokens; the live context is ~250k, so the cache thrashes. The rule
passed and the workload stays at 64 users.

vLLM 0.29.0's defaults on this GPU, read from the log, `vllm:cache_config_info` and
`EngineArgs.get_batch_defaults` for < 70 GB:

- `gpu_memory_utilization` 0.92;
- `max_num_seqs` 256;
- `max_num_batched_tokens` 2048;
- block size 16;
- KV dtype auto.

The baseline step renders them as explicit flags, except block size, because Akamas 4.1 refuses
a parameter that is in both `values` and `doNotRenderParameters`.

**Offload smoke (11:41–11:50):** offload 24 GiB + FP8 KV, 5 min of load:

- `/dev/shm` mmap 25.77 GB, vLLM ready in 168 s;
- GPU KV cache 241,184 tokens with FP8;
- TTFT p95 96 ms over the first 5 min, before the contexts grow;
- all four offload PromQL queries return values.

## ITL constraint fix (2026-11-05 12:26 UTC)

The first threshold, `vllm.inter_token_latency_p95 <= 50`, flagged the baseline itself
(experiment 1: CONSTRAINTS_VIOLATED, TTFT p95 872 ms server-side / 885 ms client-side, 840/840
requests). The 50 ms came from AIPerf's client-side figure, a p95 of per-request average ITL
(33 ms). The server-side vLLM histogram instead measures every gap between two tokens,
including decode steps stalled by the prefill of recomputed histories: ~96 ms on average over
the window, 30 s values up to 164 ms.

The user chose **120 ms** (~baseline + 25 %). Akamas 4.1 refuses a constraint change on a
RUNNING study, so the study was finished, updated (experiment 1 re-scored VALID) and started
again. The start reran the study from experiment 1 and wiped the history: ~33 min lost.

## Parameters tuned

| Parameter | Domain | Note |
|---|---|---|
| `vllm.kv_offloading_size` | 0–28 GiB | 0 = no flag. 28 + 20 (vLLM) + 6 (AIPerf) ≤ 57 GiB allocatable |
| `vllm.kv_cache_dtype` | auto, fp8 | |
| `vllm.gpu_memory_utilization` | 0.80–0.94 | |
| `vllm.max_num_seqs` | 16–256 | |
| `vllm.max_num_batched_tokens` | 1024–16384 | ≥ `max_num_seqs` (parameterConstraint) |
| `vllm.block_size` | 16, 32, 64, 128 | |

Steps:

1. `baseline` (defaults).
2. `Offload only` (24 GiB).
3. `FP8 KV only`.
4. `Offload and FP8`.
5. `Baseline repeat` (drift check).
6. `optimize`: AKAMAS, 4 init, 25 experiments, 10 failures max.

Each experiment takes ~27 min (vLLM restart + 21 min of load).

## Results

TODO (study-recap at the end).

## Conclusions

TODO.
