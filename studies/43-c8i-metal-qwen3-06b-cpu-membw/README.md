# 43-c8i-metal-qwen3-06b-cpu-membw

**Status:** TODO (created 2026-10-27, not started)
**Dates:** 2026-10-27 –

## Objective

**Find what bounds vLLM CPU inference** — memory bandwidth or compute — by running study 39/42's
optimization on a **bare-metal** `c8i.metal-48xl` while `membw.py` reads the memory controllers'
(uncore IMC) counters. The c8i.12xlarge of studies 39/42 is a VM and exposes no uncore PMUs (study
42's README), so its bandwidth cannot be measured there.

The study is **meant to run long and be stopped by hand** once enough experiments cover the
configuration space; the result is the bandwidth (GB/s, % of the NUMA node's share) recorded
alongside each experiment's tokens/s, not the optimum itself.

Everything else is study 42's (= study 39's): `Qwen/Qwen3-0.6B` bf16 (thinking off), vLLM 0.31.0
CPU image, goal (total tokens/s under TTFT p95 <= 3000 ms and TPOT p95 <= 100 ms, 150 s p95s,
`:max` over the window), windowing, parameters, constraints, steps and the same AIPerf load and
seed (`WORKLOAD_SEED=39`, 0 -> 4 req/s over 4000 s).

## Differences from study 42

| | study 42 | study 43 |
|---|---|---|
| Node | node group `llm-serving-c8i`, one c8i.12xlarge **spot** (VM, 48 vCPU, 1 NUMA node) | node group **`llm-serving-c8i-metal`**, one **c8i.metal-48xl on demand** (~9 $/h), us-east-2c only (spot: no capacity, placement score 1/10; on demand: no capacity in 2b on 2026-10-27) |
| CPU / memory | Xeon 6975P-C VM slice | the whole Xeon 6975P-C: 96 cores / 192 threads, 1 socket, **3 NUMA nodes (SNC3, 32 cores each)**, 384 GiB, **12 x DDR5-7200 = 691.2 GB/s** theoretical (SMBIOS) |
| vLLM pod | 44 CPUs = 22 cores of 24 | the same 44 CPUs = 22 cores, placed by the kubelet's single-numa-node policy on one NUMA node (first run: NUMA 0, CPUs 2-23,98-119; vLLM binds memory to it), so the per-core load matches studies 39/42 |
| Pod placement | `node-role: llm-serving-cpu` | + `node.kubernetes.io/instance-type: c8i.metal-48xl` (the template and `apply_config.sh`), so the 12xlarge node group can never take the pod |
| Akamas resources | `42-C8i-Qwen3-06B-CPU-TPS-Repeat*` | `43-C8i-Metal-Qwen3-06B-CPU-Membw*`, system `vLLM_Benchmark_43_C8i_Metal_Qwen3_06B_CPU_Membw`, telemetry `Prometheus_43_C8i_Metal_Qwen3_06B_CPU_Membw` |
| AIPerf results volume | `aiperf-results-42` | `aiperf-results-43` (new, 10 GiB; the Job runs on the system node in us-east-2b) |
| Caches | hostPath `/var/lib/{hf,vllm}-cache-study42` | `/var/lib/{hf,vllm}-cache-study43` |

## Memory bandwidth (membw.py)

`membw.py` (the user's vmstat-style IMC monitor) got a **Granite Rapids profile (model 173)** on
2026-10-27: per IMC it reads `cas_count_{read,write}_sch{0,1}` (each DDR5 channel's two
sub-channels, summed; 64 B per count), allows 12 channels and a 768-bit bus. On this node: 12 PMUs
`uncore_imc_0..11` (all `cpumask=0`), idle ~0.7 GB/s; checked against `perf bench mem memcpy`
(13.9 GB/s single thread -> membw 15.2 GB/s read + 15.2 GB/s write). The patched script is not in
this repo (the user's tool).

It runs outside Akamas, in a privileged pod on the node (`hostPID`, Amazon Linux 2023 with
`perf`, `python3`, `dmidecode`; script from a ConfigMap), 1 s samples, for the whole study. With
SNC3 the pod's NUMA node is served by about 4 of the 12 channels, so its practical ceiling is about
1/3 of 691 GB/s (~230 GB/s); membw's util% is against the full socket.

## Stack & versions

- **Akamas version:** Studio 4.1.0 (<https://akamas41.lab.akamas.io>, namespace
  `akamas-41` on `vllm-bench`).
- **Optimization packs:** vLLM **1.14.0** (component type `vLLM`) and Kubernetes **1.9.0**
  (component types `Kubernetes Container`, `Kubernetes Cluster`), both INSTALLED on the
  4.1 instance (`akamas describe optimization-pack`, 2026-10-16). The vLLM pack's reference engine is 0.29.0, this study runs 0.31.0 (see
  "Pack TODOs").
- **Workload under test:** `vllm/vllm-openai-cpu:v0.31.0` (x86_64) serving
  **`Qwen/Qwen3-0.6B`** (Apache-2.0, ungated, no HF token) as `qwen3-06b`, StatefulSet
  `vllm` (one pod, `vllm-0`) in namespace `llm-cpu`. Fixed flags: `--dtype bfloat16`,
  `--max-model-len 16384`, `--default-chat-template-kwargs '{"enable_thinking": false}'`
  (the flag exists in v0.31.0; Qwen3 thinks by default).
  - Pod: Guaranteed QoS, CPU request = limit = 44 (22 physical cores, both SMT siblings),
    memory request = limit = 48Gi; `seccompProfile: Unconfined` and `SYS_NICE` (without
    them vLLM logs `get_mempolicy: Operation not permitted` and loses its NUMA
    optimizations); `/dev/shm` as a 4Gi memory-backed emptyDir; HF cache on a volume.
  - Fixed env: `VLLM_CPU_OMP_THREADS_BIND=auto`, `VLLM_CPU_NUM_OF_RESERVED_CPU=2`,
    `VLLM_ENGINE_READY_TIMEOUT_S=1800` (CPU compile at cold start can exceed the default
    600 s; the startup probe is sized to match). `VLLM_CPU_KVCACHE_SPACE` is **never set**:
    when set, `vllm/platforms/cpu.py` overwrites `kv_cache_memory_bytes` with it, which
    would silently disable the tuned `gpu_memory_utilization`.
- **The model:** Qwen3-0.6B, dense GQA, 28 layers, 16 query / 8 KV heads, head dim 128,
  native context 32768. KV per token in bf16: 2 x 28 x 8 x 128 x 2 B = 114,688 B (~112
  KiB, ~9.4k tokens per GiB); half that in fp8. Not on vLLM's CPU validated-models list
  (Qwen3-1.7B is). Weights: about 1.5 GB of bf16 safetensors (to confirm in the probe log).
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), managed node group
  **`llm-serving-c8i-metal`**: one **c8i.metal-48xl on demand** (Intel Xeon 6975P-C Granite
  Rapids, AMX, 96 cores / 192 threads, 1 socket, 3 NUMA nodes, 384 GiB, 12 x DDR5-7200; ~9 USD/h),
  private subnet us-east-2c only, AL2023 kernel 6.12, min 0 / desired 0 / max 1, label
  `node-role=llm-serving-cpu`, taint `workload=cpu-inference:NoSchedule`, kubelet
  `cpuManagerPolicy: static` and `topologyManagerPolicy: single-numa-node` (study 39's
  NodeConfig; its reserved-CPU arithmetic was made for 48 vCPUs), AlwaysOn tagging for the
  nightly EC2 stop. Provisioning: `infra/eks/cpu-nodegroup.sh` (defaults set for this node).
  AIPerf and Prometheus run on the existing system node (`node-role: system-m8a`).
- **Load generator:** AIPerf **0.11.0**, Kubernetes Job `aiperf-cpu` on the system node,
  streaming chat endpoint, `--request-rate R --request-rate-ramp-duration D`, synthetic
  multi-turn sessions with a shared prefix pool. The workload is defined once in
  `k8s/lib_workload.sh` (flags checked against the AIPerf 0.11.0 wheel) and rendered by
  `k8s/render_job.sh` into `k8s/05-job_template.yaml`; R/D are still to be calibrated (see
  "Load").
- **Telemetry:** Prometheus only (in-cluster kube-prometheus-stack, as studies 30/36),
  vLLM `/metrics` through a ServiceMonitor. No DCGM, no GPU component. Placeholder keys
  alphanumeric only (keys with underscores, such as `$NODE_ROLE$`, are left unsubstituted and the query returns nothing, without an error).

### Why AMX and bf16

vLLM 0.31.0's CPU attention backend takes the AMX path when the CPU has AMX tiles, the
dtype is bf16 and `block_size % 32 == 0` (`vllm/v1/attention/backends/cpu_attn.py`).
fp16 on torch CPU is documented as unstable. fp8 KV needs AVX-512 or AMX and a head size
that is a multiple of 32 (128 here): both hold on c8i.

### KV budget on CPU

On CPU, `gpu_memory_utilization` (default 0.92) is the fraction of the NUMA node's RAM the
worker requests, and the KV cache gets that minus the process's RSS
(`vllm/v1/worker/cpu_worker.py`). In v0.31.0 the RAM total is clamped to the pod's cgroup
memory limit (`vllm/utils/cpu_resource_utils.py`), so here it is a fraction of 48Gi.
Estimate before the probe (RSS unmeasured, assumed ~2-4 GiB with the weights): 0.15 ->
~3-5 GiB of KV (~30-50k bf16 tokens), 0.90 (baseline, domain top) -> ~38-40 GiB (~360-375k
tokens). The study uses 0.90, not vLLM's default 0.92: on CPU the pod runs three vLLM
processes (API server, EngineCore, worker) and vLLM's KV-allocation check subtracts only the
worker's RSS, so 0.92 leaves about 0-0.5 GiB of the 48Gi limit there (`k8s/params.env.template`). Whether the lower
bound is small enough to cause preemption at the target load is what the probe and the
smoke run check.

## Parameters tuned

Component `vllm` (type `vLLM`, pack 1.14.0). Every value goes through
`k8s/params.env.template` -> the StatefulSet renderer; booleans render as `--x` / `--no-x`.
Every flag name is checked against the v0.31.0 source (the pack was written for 0.29.0).
Baselines are vLLM 0.31.0's CPU online-serving defaults (`vllm/engine/arg_utils.py`, CPU
branch: 2048 batched tokens and 128 sequences per world-size unit) except
`gpu_memory_utilization` 0.90.

| Parameter | env key | Domain | Baseline (vLLM 0.31.0 CPU default) | Why |
|---|---|---|---|---|
| `vllm.gpu_memory_utilization` | `GPU_MEMORY_UTILIZATION` | 0.15-0.90 (decided 2026-10-16; every edge starts in the probe) | 0.90 (not the vLLM default 0.92) | the KV budget on CPU (fraction of the 48Gi limit, minus RSS); the lower bound must make preemption possible at target load. 0.92 leaves only ~0-0.5 GiB of the 48Gi limit at the KV-allocation check (`k8s/params.env.template`) |
| `vllm.max_num_seqs` | `MAX_NUM_SEQS` | 8-256 | 128 | admission cap; on CPU it mainly drives output throughput |
| `vllm.max_num_batched_tokens` | `MAX_NUM_BATCHED_TOKENS` | 512-8192 | 2048 | prefill budget per step; on CPU it mainly drives TTFT |
| `vllm.block_size` | `BLOCK_SIZE` | categorical 32, 64, 96, 128 | 128 | the AMX attention path needs a multiple of 32 |
| `vllm.kv_cache_dtype` | `KV_CACHE_DTYPE` | auto, fp8 | auto | twice the KV tokens per GiB; supported on CPU with AMX/AVX-512 |
| `vllm.enable_prefix_caching` | `ENABLE_PREFIX_CACHING` | true, false | true | the load shares prefixes, so the hit rate and the KV budget interact |
| `vllm.long_prefill_token_threshold` | `LONG_PREFILL_TOKEN_THRESHOLD` | 0-4096 | 0 | caps per-step prefill of one request: head-of-line blocking by the 1-2k shared prefixes |

**parameterConstraints:** `vllm.max_num_batched_tokens >= vllm.max_num_seqs` (vLLM raises
otherwise).

**Fixed, with the reason:**

| Setting | Value | Why |
|---|---|---|
| `--dtype` | bfloat16 | AMX path; fp16 unstable on torch CPU |
| `--max-model-len` | 16384 | explicit cap; covers the multi-turn sessions (prefix + 3-6 turns) |
| `--enable-chunked-prefill` | on (decided 2026-10-16) | off cannot start: vLLM raises when chunked prefill is off and `max_num_batched_tokens` (max 8192) < `max_model_len` 16384 |
| `--stream-interval` | 1 (decided 2026-10-16) | a larger value batches streamed tokens and hides latency from the server-side TTFT/TPOT histograms the constraints read |
| thinking | off (`enable_thinking: false`) | Qwen3 thinks by default; long outputs would distort TTFT/TPOT |
| `VLLM_CPU_OMP_THREADS_BIND` | auto | not in the pack (Pack TODOs); `auto` follows the pod's exclusive cpuset under the static CPU manager |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | 2 | not in the pack; CPUs left to the API server and tokenizer |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 1800 | slow CPU compile at cold start |
| `VLLM_CPU_KVCACHE_SPACE` | unset | would override `gpu_memory_utilization` |
| `async_scheduling` | not rendered | forced off on CPU (`vllm/platforms/cpu.py`) |
| `max_cudagraph_capture_size`, `enforce_eager` | not rendered | CUDA graphs are cleared on CPU; compile mode is rewritten to `DYNAMO_TRACE_ONCE` |
| attention / linear / MoE backends, wna16 | not rendered | GPU-only choices; the CPU attention backend is the only one |
| tensor / pipeline / data parallelism | 1 | a 0.6B model on one NUMA node (to confirm: if the node has 2+ NUMA nodes, TP/DP changes the design) |
| speculative decoding | off | out of scope |
| `compilation_mode`, `optimization_level`, `performance_mode` | not rendered | meaning on CPU unclear; **TODO:** add only if the smoke run shows an effect |

## Load

- **Open loop, linear rate ramp (as study 30):** `--request-rate R
  --request-rate-ramp-duration D` on the streaming chat endpoint. R and D are the
  workflow's `RT_RATE` / `RT_RAMP_S` env vars: **0 -> 4 req/s over 4000 s** (from the probe).
- **Synthetic multi-turn with shared prefixes:** the workload is defined once in
  `k8s/lib_workload.sh` (flags checked against the AIPerf 0.11.0 wheel) and rendered by
  `k8s/render_job.sh` into `k8s/05-job_template.yaml`: 16 prefixes of 1536 tokens prepended
  to the first user message, per-turn ISL 384±96 / OSL 192±48 with `ignore_eos`, 4±1 turns,
  5±2 s think time, 8000 unique sessions.
- **Watchdog** (`k8s/run_test.sh`, as study 30): stops the run with success once TTFT p95
  or TPOT p95 stays past its SLO for 120 s: past the knee of an open loop every later
  window is invalid anyway.
- **Scoring:** `stability` windowing on `vllm.total_token_throughput`, width 6 samples,
  `maxStdDev` 300000000 (filter disabled), `when: max`: the valid window with the most
  tokens/s.

## Steps

| # | Step | Values | Why |
|---|---|---|---|
| 1 | baseline | vLLM 0.31.0 CPU defaults except `gpu_memory_utilization` 0.90, every parameter written out | the reference point |
| 2 | baseline repeat | the same | first measure of the noise on this node |
| 3 | kv fp8 | baseline + `kv_cache_dtype` fp8 | twice the KV tokens per GiB |
| 4 | no prefix cache | baseline + `enable_prefix_caching` false | how much the shared prefixes are worth |
| 5 | small kv budget | baseline + `gpu_memory_utilization` 0.15 (the domain's lower bound) | the KV-pressure corner |
| 6 | optimize | AKAMAS, 0 init experiments, 60 experiments, `maxFailedExperiments` 20 | as studies 27-30 |

Workflow (`43-C8i-Metal-Qwen3-06B-CPU-Membw-Workflow`, identical to study 39's): Write config
(FileConfigurator) -> Apply config (`k8s/apply_config.sh`: render, roll out, fail at the first
container restart, warm-up, log dump) -> RunTest (`k8s/run_test.sh`). The ramp is study 39's:
0 -> 4 req/s over 4000 s (from study 39's probe; no Smoke study in this repeat).

**KPIs (8):** Total throughput, Completed requests, TTFT P95 150s, TPOT P95 150s, Prefix
cache hit rate, KV cache usage, Preemptions, Running requests (`akamas/`).

## Setup & run

From the toolbox (`/work/vllm-benchmark`, after a git sync), node group up first:

```bash
AWS_PROFILE=lab ./studies/43-c8i-metal-qwen3-06b-cpu-membw/infra/eks/cpu-nodegroup.sh --up --always-on   # local machine
kubectl apply -f studies/43-c8i-metal-qwen3-06b-cpu-membw/k8s/00-pvc.yaml
cd studies/43-c8i-metal-qwen3-06b-cpu-membw/akamas
S=vLLM_Benchmark_43_C8i_Metal_Qwen3_06B_CPU_Membw
akamas create system system.yaml
for c in components/*.yaml; do akamas create component "$c" "$S"; done
akamas create telemetry-instance telemetry/prometheus.yaml "$S"
akamas create workflow 43-C8i-Metal-Qwen3-06B-CPU-Membw-Workflow.yaml
akamas create study    43-C8i-Metal-Qwen3-06B-CPU-Membw.yaml
akamas start study "43-C8i-Metal-Qwen3-06B-CPU-Membw"
```

Bulk alternative, same order: `akamas create -f system.yaml`, `akamas create -f components/`,
`akamas create -f telemetry/`, then the workflow and the study. To stop it:
`akamas finish study "43-C8i-Metal-Qwen3-06B-CPU-Membw"`, then
`cpu-nodegroup.sh --down` (on demand: ~9 $/h while up).

## Results

<Filled in by the study-recap skill once the study is stopped. Study 39's numbers for comparison:
`../39-c8i-qwen3-06b-cpu-tps/README.md`, Results.>

### Manual run before the study (2026-10-27)

Study 39's best configuration (exp 50: fp8 KV, block 96, max_num_seqs 176, batched 512, gmu
0.68, prefix caching on) run by hand from the toolbox with study 42's scripts on this node, ramp
0 -> 4 req/s over 4000 s from 11:14 UTC, membw.py recording. The watchdog ended it at 11:56
(~2500 s, ~2.5 req/s offered; study 39's knee for this configuration: ~2.4 req/s).

| 5 min from (UTC) | tokens/s (prompt + gen) | running (max) | DRAM read avg / p95 (GB/s) | write avg (GB/s) |
|---|---|---|---|---|
| 11:22 | 916 | 2 | 87.7 / 99.6 | 0.6 |
| 11:27 | 2052 | 4 | 90.0 / 100.1 | 0.7 |
| 11:32 | 2801 | 7 | 90.5 / 101.2 | 0.8 |
| 11:37 | 3843 | 12 | 89.7 / 101.7 | 1.0 |
| 11:42 | 4829 | 18 | 87.1 / 100.7 | 1.3 |
| 11:47 | 5767 | 24 | 85.1 / 97.7 | 1.6 |
| 11:52 | 6517 | 32 | 80.8 / 93.7 | 2.0 |
| 11:57 | 7558 | 57 | — | — |

(tokens/s and running from Prometheus, 5 min rates at the window end; bandwidth from membw, whole
socket, but the pod's memory is bound to NUMA 0.) The pod's CPU usage was ~21 cores from 11:27 on
(the OMP threads spin, so it reads as saturated at any load).

Ceilings measured on NUMA 0 right after (22 processes pinned with `numactl --cpunodebind=0
--membind=0`, membw reading): **~152 GB/s read-only** (numpy sum of 1 GiB arrays, one thread each)
and **~159 GB/s read + write** (`perf bench mem memcpy`, ~80 + 80).

First reading: DRAM traffic is almost all reads and **flat at ~90 GB/s (~100 at p95) from the
lowest load to the SLO break**, while tokens/s grow 7x — consistent with every engine step
streaming the weights once whatever the batch size, so batching amortizes it. At the SLO break
vLLM uses ~60 % of the read bandwidth the same 22 cores can pull from their NUMA node: this
configuration does not look memory-bandwidth bound; the limit is elsewhere (compute per step /
scheduling). The study's other configurations will show whether any of them moves the
bandwidth.

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
