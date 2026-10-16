# 39-c8i-qwen3-06b-cpu-tps

**Status:** RUNNING (Akamas 4.1, main study started 2026-10-16 ~14:53 UTC). Smoke study
(baseline only, 0 -> 4 req/s over 1200 s): 1 experiment, VALID, **5486 tokens/s**, 17 min;
then deleted.
**Dates:** 2026-10-16 → (running)

## Objective

**How many tokens per second can vLLM's CPU backend serve with a sub-1B model on one Xeon 6
node within the vLLM Xeon SLOs, and which vLLM settings get there?** The first study in this
repo on a CPU node and on a vLLM CPU image: every earlier study ran on NVIDIA GPUs.

- **Goal:** maximize vLLM's **total token throughput** (`vllm.total_token_throughput`,
  prompt + generation tokens/s), as studies 11/12/25/30.
- **Constraints (vLLM Xeon 6 recipe SLOs, `tools/recipes/RUNTIME_TUNING.md`):**
  `vllm.time_to_first_token_p95:max <= 3000` ms and `vllm.time_per_output_token_p95:max <=
  100` ms over the scored window.
- **Load:** AIPerf open-loop request-rate ramp with synthetic multi-turn sessions that share
  a pool of prompt prefixes, so the prefix cache and the KV budget matter (see "Load").
- **What is not in it:** KV cache offload to RAM was asked for and dropped (user,
  2026-10-16): vLLM 0.31.0's native `OffloadingConnector` refuses the CPU backend
  (`vllm/v1/kv_offload/cpu/spec.py`, `get_worker()` raises "CPU Offloading is currently only
  supported on CUDA-alike and XPU GPUs"), and on CPU the KV cache already lives in RAM. The
  KV-capacity question is studied instead through the KV budget (`gpu_memory_utilization`),
  fp8 KV and prefix caching.
- No ROADMAP hypothesis covered CPU inference before this study; it is backlog row #39.

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
- **Cluster / hardware:** shared EKS cluster `vllm-bench` (us-east-2), new managed node
  group **`llm-serving-c8i`**: one **c8i.12xlarge** (Intel Xeon 6 Granite Rapids, AMX, 48
  vCPU / 24 cores, 96 GiB; **spot**, 0.68-0.81 USD/h on 2026-10-16 vs 2.249 on demand, private subnets 2b + 2c), AL2023, min 0 / desired 0 / max 1,
  label `node-role=llm-serving-cpu`, taint `workload=cpu-inference:NoSchedule`, kubelet
  `cpuManagerPolicy: static` and `topologyManagerPolicy: single-numa-node`, system CPUs
  reserved, AlwaysOn tagging for the nightly EC2 stop. The NUMA layout is not exposed by
  the EC2 API: the probe records it. Provisioning: `infra/`. AIPerf and Prometheus run on
  the existing system node (`node-role: system-m8a`).
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
  workflow's `RT_RATE` / `RT_RAMP_S` env vars, **TODO values until the Smoke calibration**.
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

Workflow (`39-C8i-Qwen3-06B-CPU-TPS-Workflow`): Write config (FileConfigurator) -> Apply
config (`k8s/apply_config.sh`: render, roll out, fail at the first container restart, warm-up,
log dump) -> RunTest (`k8s/run_test.sh`). **Smoke study** `39-C8i-Qwen3-06B-CPU-TPS-Smoke`
(workflow `...-Smoke-Workflow`): the baseline only, a steeper ramp (0 -> 4 req/s over 1200 s) that checks the pipeline end to
end. The main workflow ramps 0 -> 4 req/s over 4000 s (from the probe, see "Probe results").

**KPIs (8):** Total throughput, Completed requests, TTFT P95 150s, TPOT P95 150s, Prefix
cache hit rate, KV cache usage, Preemptions, Running requests (`akamas/`).

## Pack TODOs

For whoever manages the vLLM pack (outside this repo, <https://gitlab.com/akamas/optimization-packs/vllm>).
None of these blocks the study; each is fixed in the StatefulSet for now.

- **Missing CPU parameters:** `VLLM_CPU_OMP_THREADS_BIND` (env; `auto`, `nobind` or core
  lists), `VLLM_CPU_NUM_OF_RESERVED_CPU` (env; only with `auto`), `kv_cache_memory_bytes`
  (`--kv-cache-memory-bytes`, the explicit KV size the vLLM docs recommend on CPU),
  `VLLM_CPU_ATTN_SPLIT_KV` (env, default 1).
- **`gpu_memory_utilization` semantics on CPU:** the pack describes it as a GPU memory
  fraction; on the CPU backend it is the fraction of the NUMA node's RAM (clamped to the
  cgroup limit in 0.31) requested by the worker, KV = that minus RSS. The description and
  domain should say so, or a CPU-specific parameter should replace it.
- **Reference engine 0.29.0 vs 0.31.0:** this study runs vLLM 0.31.0 (the CPU image and the
  CPU fixes); every flag it renders was checked against the 0.31.0 source, but the pack's
  other parameters were not.
- **Not applicable on CPU** (should be marked so in the pack): `async_scheduling`,
  `max_cudagraph_capture_size`, `enforce_eager`, the attention/linear/MoE backends.

## Next steps

1. **Confirm the packs:** `akamas login` in the `akamas-41` toolbox, then `akamas list
   optimization-pack` (vLLM 1.14.0, Kubernetes version).
2. **Provision the node group** `llm-serving-c8i` (`infra/`) — **DONE 2026-10-16**: spot
   node in us-east-2b/2c, kubelet static CPU manager + full-pcpus-only + single-numa-node,
   2 reserved CPUs via `systemReserved` (core 0; `reservedSystemCPUs` is refused next to
   EKS's reserved cgroups), allocatable 46 CPU / ~86 GiB, 1 NUMA node, siblings n / n+24,
   AlwaysOn tagged (`infra/README.md`).
3. **Run the probe** (`probe/`) — **DONE 2026-10-16**, see "Probe results".
4. **Smoke study** (`39-C8i-Qwen3-06B-CPU-TPS-Smoke`, baseline only, 0 -> 4 req/s over
   1200 s) — **DONE 2026-10-16**: VALID, 5486 tokens/s (between the probe's 3117 at 1 req/s
   and 5904 at 1.94 req/s, which broke TPOT); the pipeline works end to end in Akamas.
5. **Run the main study** `39-C8i-Qwen3-06B-CPU-TPS` — **STARTED 2026-10-16**.

## Probe results (2026-10-16, `probe/results/summary.txt`)

Every startup edge starts (~100-140 s each, the first included: image and model pulled in
~2 min). OMP: 20 threads on 20 cores (`reserve_cpu_num=2`, 2 of the pod's 22 cores left to the
API server), 1 NUMA node, no LD_PRELOAD / mempolicy warning.

| Combination | KV tokens | KV GiB | non-KV GiB | peak GiB |
|---|---|---|---|---|
| baseline (gmu 0.90) | 377,088 | 40.29 | 2.91 | 46.5 |
| gmu 0.92 (vLLM default) | 386,432 | 41.28 | 2.88 | 46.1 |
| gmu 0.15, 256 seqs, 8192 batched | 37,632 | 4.02 | 3.18 | 8.8 |
| gmu 0.15, fp8 KV | 80,768 | 4.32 | 2.88 | 9.0 |
| fp8 KV | 754,944 | 40.32 | 2.88 | 45.1 |
| block 32 / 96 | 377,440 / 376,640 | 40.32 | 2.88 | 45.1 |
| 256 seqs, 8192 batched, long prefill 4096, prefix off | 374,656 | 40.02 | 3.18 | 45.1 |

Chunked prefill off is refused by `render_statefulset.sh`, as designed. The domain's lower
bound holds (non-KV RSS <= 3.18 GiB -> f >= 0.103 fits one 16k sequence).

Baseline constant-rate sweep (AIPerf client p95s; 240 s per step):

| req/s | tokens/s | TTFT p95 | TPOT p95 | prefix hit | KV used | running | SLO |
|---|---|---|---|---|---|---|---|
| 1.04 | 3117 | 209 ms | 34 ms | 75 % | 6 % | 10 | ok |
| 1.94 | 5904 | 320 ms | 132 ms | 78 % | 30 % | 55 | TPOT over |
| 2.45 | 6513 | 38.5 s | 266 ms | 77 % | 54 % | 128 (= max_num_seqs) | over |

The binding constraint is TPOT (decode on 20 cores), the knee is between 1 and 2 req/s, and the
node saturates at ~2.5 req/s. With the baseline's 40 GiB the KV cache is never short (54 % at
saturation); KV pressure comes only from the lower part of the `gpu_memory_utilization` domain
(0.15 = 37.6k tokens, ~10 running sequences of this workload). The probe proposed
`RT_RATE=2`; the main workflow uses 4 (over 4000 s) so a tuned configuration's knee is not
capped by the ramp.

## Risks

- **SMT and OMP binding:** auto-binding uses one logical CPU per physical core. With the
  static CPU manager, 44 CPUs are 22 cores' worth of sibling pairs, so vLLM may run ~22
  OMP threads (minus the reserved CPUs), not 44. The probe's binding log decides.
  `full-pcpus-only` is already on (`infra/eks/nodeconfig.yaml`), so the cpuset is 22 whole
  cores; if ~20 OMP threads underuse the node, the alternatives are
  `VLLM_CPU_OMP_THREADS_BIND` with an explicit list of all 44 logical CPUs, or SMT off.
- **NUMA:** if c8i.12xlarge exposes 2+ NUMA nodes (sub-NUMA clustering), the
  `single-numa-node` policy may not place a 44-CPU pod, and vLLM's one-rank-per-NUMA-node
  rule changes the design.
- **Memory:** the 4Gi memory-backed `/dev/shm` counts against the pod's memory limit when
  written; weights + KV over the node's memory kills the worker with exit code 9.
- **Throughput unknown:** no vLLM-CPU numbers are published for models under 1B; the
  estimate of several hundred to ~1000+ output tokens/s on 24 c8i cores is extrapolated.
- **Model not validated on CPU** by vLLM (Qwen3-1.7B is).

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
