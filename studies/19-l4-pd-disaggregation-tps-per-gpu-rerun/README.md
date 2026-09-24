# 19-L4-PD-Disaggregation-TPS-Per-GPU-Rerun

**Status:** TODO (created 2026-09-24 as the rerun of study 18)
**Dates:** created 2026-09-24 – not started

## Why this study exists (rerun of study 18)

Study 18 (`../18-l4-pd-disaggregation-tps-per-gpu/`) is the same study, same node, same
presets, same goal, same SLA. It was stopped after its first experiments because of a
**measurement defect in the router**, not in the design. Study 18's `pd_router.py` added
all 4096 + 256 tokens of a request to its counters when the request **ended**. In
Akamas' 30 s telemetry windows the router throughput became a sawtooth: 0 in a window
with no completion, then a spike. The means were right (engine-side vLLM agreed), but
Akamas scores the best window, and the noise leaked into the score.

**The one change:** `k8s/pd_router.py` counts tokens **as they stream**. It asks the
decode instance for `stream_options.continuous_usage_stats`, counts prompt tokens once at
the first chunk and output tokens by the per-chunk delta, and strips the usage field
again where the client did not ask for it. Everything else is identical to study 18 at
commit 9d955c1, apart from the renamed Akamas resources and paths. Study 18's
experiments are NOT bootstrapped: their scores were computed with the old counting.

Study 18's own findings (phase B, calibration, first experiments) are in its README and
`results/` folder. They are summarized below, with links, so this study keeps them without
copying the raw logs.


## Objective

**Does prefill/decode disaggregation beat aggregated serving on a PCIe-only 4x L4 node,
and if so, in which topology and on which metric?**

Every study so far (0-17) ran aggregated vLLM: each instance does both prefill and decode.
`ROADMAP.md` (Q6 and the vLLM-pack backlog) kept disaggregation out of scope because it is
a deployment-topology choice, not a vLLM flag. Its adoption thresholds (100M-1B tokens/day,
a model of at least ~100B parameters, prefill-heavy traffic) fail here by construction.
`knowledge/notes/2026-07-distributed-inference-scaling-dimensions.md` states the opposite
criterion: what decides is the *measured* prefill/decode imbalance against the KV-transfer
cost. This study measures that on this hardware.

**Expected result, stated up front (falsification framing, like study 17):** vLLM's own
docs say "Disaggregated prefill DOES NOT improve throughput". The mechanism that can win
is **tail inter-token latency**: prefill chunks leave the decode batch. So the likely
outcome is that disaggregation loses on tokens/s per GPU at equal load, and possibly wins
where the ITL SLA binds. The KV transfer crosses PCIe, not NVLink, which works against it.
The study's value is where exactly that line falls.

- **Goal:** maximize `(pd_topology.prefill_token_throughput + pd_topology.decode_token_throughput) / pd_topology.active_gpus`.
  This is deployment-wide tokens/s per GPU holding weights, the same shape as studies
  9-16. It is measured **at the router**: tokens delivered to clients, each counted once.
- **SLA (calibrated 2026-09-24):** TTFT p95 <= 5000 ms and ITL p95 <= 75 ms, both measured at
  the router. The study manifest has the reasoning.

## Stack & versions

- **Akamas version:** 3.7.x (repo target, `CLAUDE.md`).
- **Optimization packs:**
  - vLLM **1.11.0**: branch `feature/pd-disaggregation-topology`, GitLab MR
    akamas/optimization-packs/vllm!7, **not yet merged or installed**. It adds the
    `vLLM_PD_Topology` component type (`pd_prefill_instances`, `pd_decode_instances`,
    `pd_kv_connector`, `pd_kv_buffer_device`, plus deployment-wide throughput/latency/
    `active_gpus`) and 6 NIXL metrics on `vLLM`. It was branched from 1.10.1 (MR !6,
    unmerged), so it carries the speculative-decoding parameters too.
  - GPU pack 1.2.0 and Kubernetes pack: same as study 16.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` (CUDA 13 build; ships NIXL 1.3.2 per
  vLLM's release pipeline), model **`Qwen/Qwen3-8B-FP8`**. It is dense, has 36 layers and 8
  KV heads, uses full attention on every layer, has 8.79 GiB of FP8 weights, and is
  Apache-2.0 and not gated. The KV cost is 144 KiB/token in bf16, so each 4096-token prompt
  moves ~0.56 GiB prefill -> decode.
  Why this model: it fits one L4 with ~9.5 GiB left for KV, so every topology up to 4
  instances is possible. It has no sliding window, sinks or MoE, which is the plainest
  case for NixlConnector. Its per-prompt KV is large enough to make the transfer cost
  visible. Rejected alternatives:
  - gpt-oss-20b: TRITON_ATTN only on SM 8.9, hybrid SWA, NIXL unverified.
  - Qwen3-14B-FP8: ~3 GiB of KV left.
  - Qwen3-30B-A3B: needs TP2, so only one topology would be possible.
  - Qwen2.5-7B: bf16 weights leave ~4 GiB of KV.
- **KV transfer:** `NixlConnector` (pull), `kv_load_failure_policy=fail`. `P2pNcclConnector`
  no longer exists in 0.29.0 (vllm PR #44854).
- **Cluster / hardware:** `vllm-bench`, us-east-2, node group `llm-serving-l4`: 1x
  g6.12xlarge, 4x NVIDIA L4 24 GB, SM 8.9, 72 W cap, PCIe only, 4.60 USD/h. See
  `infra/README.md`.
- **Topology:** one pod. The `engine` container sees all 4 GPUs and runs P prefill + D
  decode `vllm serve` processes, one GPU each (the layout vLLM's own NIXL CI uses on 4x L4;
  one PID namespace keeps cuda_ipc possible). The `router` container is the front door for
  every preset.
- **Load generator:** AIPerf 0.11.0, chat endpoint, streaming, synthetic prompts of
  **4096 tokens in / 256 out** (stddev 0, `ignore_eos`), 1000 prompts, seed 18. Concurrency
  ramp of **6 levels x 600 s = 60 min** per experiment, levels `2,4,8,16,24,32`
  (calibrated: the 2-GPU aggregated baseline saturates from ~12. The ramp goes to 32 so that
  the 4-GPU layouts, which have ~2x the capacity, also reach their peak and their tokens/s per
  GPU is not under-measured). The long levels are deliberate: queue build-up, KV fill and preemption,
  transfer back-pressure and L4 power throttling are steady-state effects.
  **Methodology break vs studies 1-17**, which all replayed ShareGPT (short-input chat, on
  which disaggregation loses by construction).
- **Telemetry:** Prometheus (kube-prometheus-stack), 5 s scrape. The 116-metric catalog is
  study 16's plus NIXL. Role separation is done with `model_name`:
  - `qwen3-8b-router`: the router, feeding the `pd_topology` component;
  - `qwen3-8b-prefill`: the prefill instances;
  - `qwen3-8b-decode`: the decode instances.

  Engine-side TTFT under-reports and engine-side prompt tokens double-count in
  disaggregated presets. That is why the goal and SLA read from the router. The
  kv-cache-exporter of studies 9-16 is not deployed.

## Parameters tuned

10 presets, then the repo's usual optimize step (AKAMAS optimizer, 100 experiments, 20
failures max; the node is stopped by hand). `parametersSelection` declares the 12
parameters explicitly, as subsets of the pack 1.11.0 domains.

| Parameter | Domain | Baseline |
|---|---|---|
| `pd_topology.pd_prefill_instances` | [0, 3] (0 = aggregated) | 0 |
| `pd_topology.pd_decode_instances` | [1, 4] | 2 |
| `pd_topology.pd_kv_connector` | NixlConnector, NixlPushConnector (**pinned to NixlConnector**) | NixlConnector |
| `pd_topology.pd_kv_buffer_device` | cuda, cpu (**pinned to cpu**) | cpu |
| `vllm_prefill.gpu_memory_utilization` / `vllm_decode.…` | [0.8, 0.92] | 0.9 / 0.9 |
| `vllm_prefill.max_num_seqs` / `vllm_decode.…` | [8, 512] | 128 / 128 |
| `vllm_prefill.max_num_batched_tokens` / `vllm_decode.…` | [512, 16384] | 8192 / 8192 |
| `vllm_prefill.kv_cache_dtype` / `vllm_decode.…` | auto, fp8 | auto / auto |

Pinned for every instance: `--max-model-len 8192`, `--attention-backend FLASHINFER`
(prefill and decode must match. It is FLASHINFER, not FLASH_ATTN, because FlashAttention runs
as v2 on SM 8.9 and rejects fp8 KV, which would stop preset S5. Decided 2026-09-24), `--no-enable-prefix-caching`, `--enable-mfu-metrics`,
`--default-chat-template-kwargs '{"enable_thinking": false}'` (reasoning off, as in earlier
studies; no effect on the load, since output is fixed at 256 tokens), and chunked prefill at
vLLM's default (on).

`parameterConstraints`:
- P + D <= 4;
- `pd_kv_buffer_device == "cpu"` and `pd_kv_connector == "NixlConnector"`, so the optimizer
  spends no hour-long experiments on a transport the smoke test measured 23x slower, or on
  the untested push mode;
- `max_num_batched_tokens >= max_num_seqs` for each role;
- the same `kv_cache_dtype` on both roles when disaggregated (NIXL compatibility hash).

### The 10 presets (revised after the smoke test)

All disaggregated presets stage the KV through host memory (`kv_buffer_device=cpu`). The
GPU path costs ~1.4 s per transfer on this node, which has no P2P. The prefill role batches
2 prompts per step (8192 tokens) instead of 4: prefill throughput is the same and each step
takes half the time. 1P3D was dropped (a decode-skewed ratio on a prefill-bound load).
2P1D and 3P1D + fp8 were added.

| # | Step | GPUs | P/D | Prefill (seqs / batched tok / KV) | Decode (seqs / batched tok / KV) | Isolates |
|---|---|---|---|---|---|---|
| 0 | `baseline` | 2 | 0/2 | — | 128 / 8192 / auto | aggregated, typical settings |
| 1 | `S1 agg2 small chunks` | 2 | 0/2 | — | 128 / 1024 / auto | the strong aggregated control (4 chunks per prompt) |
| 2 | `S2 1P1D same settings` | 2 | 1/1 | 128 / 8192 / auto | 128 / 8192 / auto | topology only |
| 3 | `S3 1P1D specialized` | 2 | 1/1 | 16 / 8192 / auto | 256 / 2048 / auto | role specialization |
| 4 | `S4 1P1D fp8 KV` | 2 | 1/1 | 16 / 8192 / fp8 | 256 / 2048 / fp8 | half the transfer bytes, 2x KV capacity |
| 5 | `S5 2P1D` | 3 | 2/1 | as S3 | as S3 | prefill-skewed ratio |
| 6 | `S6 agg4 small chunks` | 4 | 0/4 | — | 128 / 1024 / auto | 4-GPU aggregated control |
| 7 | `S7 2P2D` | 4 | 2/2 | as S3 | as S3 | balanced ratio |
| 8 | `S8 3P1D` | 4 | 3/1 | as S3 | as S3 | ratio matching the measured ~70:30 prefill:decode work |
| 9 | `S9 3P1D fp8 KV` | 4 | 3/1 | 16 / 8192 / fp8 | 256 / 2048 / fp8 | best-guess combination |

About 65-70 min each: ~11-12 h for the presets (~55 USD of GPU node), then the optimizer.

## Components, telemetry, workflow

- **System** `vLLM_Benchmark_19_L4_PD_Disaggregation`, 11 components:
  - `pd_topology` (`vLLM_PD_Topology`);
  - `vllm_prefill` and `vllm_decode` (`vLLM`);
  - `gpu0`-`gpu3` (`GPU`);
  - `cluster` and `cluster_loadtest` (`Kubernetes Cluster`);
  - `container` and `container_loadtest` (`Kubernetes Container`).
- **Telemetry instance** `Prometheus_19_L4_PD_Disaggregation`, `akamas/telemetry/prometheus.yaml`.
  The `container` component (`pod: ^vllm-pd-.*`) covers both containers of the serving
  pod, so its CPU/memory include the router's. Don't read router CPU as vLLM CPU.
  `active_gpus` counts DCGM series whose `exported_pod` matches `^vllm-pd-.*` and whose
  framebuffer is above 1 GiB. It was checked on 2026-09-24 that dcgm-exporter fills
  `exported_pod` with the workload pod's name (series from the study-17 `vllm-*` pods).
- **Workflow** `19-L4-PD-Disaggregation-Workflow`, 3 tasks on the `toolbox` host with key
  `/home/akamas/.ssh/id_rsa` (the toolbox's own key, from the `toolbox-keys` secret):
  1. FileConfigurator renders `k8s/01-deployment_template.yaml`.
  2. `k8s/apply_config.sh` (40 min timeout).
  3. `k8s/run_test_tps.sh` (95 min timeout).

  Both scripts dump the full workload logs to stdout.
- **Windowing:** stability, 6 samples, at the router's prefill-throughput peak (as
  studies 6-16). Per-level SLA analysis comes afterwards, from AIPerf per-level output
  and Prometheus.

## Preconditions and placeholders (before `akamas create`)

1. **Install vLLM pack 1.11.0** once MR !7 is merged (or build it from the branch):
   `akamas build optimization-pack <pack dir>` then
   `akamas install -f optimization-pack vLLM_1-11-0.json`. This is an instance-wide admin
   operation: the toolbox CLI user got "Access forbidden" on `akamas list
   optimization-packs` (2026-09-23), so an Akamas admin (CLI or UI) has to do it. Until
   then, `akamas create` fails for every study-19 resource that references
   `vLLM_PD_Topology` or the `pd_*` parameters. Pack 1.11.0 is installed (2026-09-24), and study 18's
   identical resources were accepted by the instance that day. Study 19's own resources
   (system `vLLM_Benchmark_19_L4_PD_Disaggregation`, its components, telemetry, workflow
   and study) are created with the commands in "Setup & run" below.
2. **Ramp levels and SLA:** calibrated on 2026-09-24 (below). They are set in
   `k8s/05-job.yaml` (`CONCURRENCY_LIST`, `--goodput`) and in the study's
   `goal.constraints`.
3. **Pull the repo on the toolbox** (`/work/vllm-benchmark`). It must include commit
   8f5044f (workflow key path) and this study.
4. **One-time Kubernetes setup and DCGM re-point:** `k8s/README.md`.

### Phase B, smoke test: what can only be checked on the node

**Run `bash k8s/smoke_test.sh diag` first**, while the 4 GPUs are free (~5 min). It
answers, with numbers, whether GPU-to-GPU traffic on this node goes peer-to-peer over PCIe
or through host memory:
- `nvidia-smi topo -m` / `-p2p r|w` and the CUDA peer-access matrix;
- copy bandwidth direct vs staged through host, for every GPU pair;
- the NCCL transport NCCL_DEBUG=INFO reports (`via P2P/...` vs `via SHM/...`), and
  all-reduce bus bandwidth on 2 and 4 GPUs with and without `NCCL_P2P_DISABLE=1`.

This study does not use NCCL (every instance is TP1). But the same peer access decides
whether NIXL can use cuda_ipc here. It also tests the open hypothesis from the study-16 TP
analysis: inter-GPU traffic there costs ~10x what PCIe Gen4 bandwidth predicts, and the
container CPU grows ~2 cores per extra rank, which is what host-staged copies look like.
Study 18's log is in `../18-l4-pd-disaggregation-tps-per-gpu/results/`. NCCL tuning itself belongs to a TP >= 2 study (ROADMAP
"PACK REQUEST — NCCL interconnect tunables"), not to this one.

### Phase B results (2026-09-24, node in us-east-2c)

**diag** (`../18-l4-pd-disaggregation-tps-per-gpu/results/diag-2026-09-24.log`):

| Check | Result |
|---|---|
| `nvidia-smi topo -p2p r/w` | **NS (not supported)** for every pair; `can_device_access_peer` NO; topology `NODE` |
| PCIe link | **x8** of x16 max (Gen1 at idle, Gen4 under load): **~13 GB/s** per direction measured D2H/H2D |
| GPU-to-GPU copy | 12 GB/s (driver-staged through host); explicit via-host 6.7 GB/s |
| NCCL all-reduce | 4.0 GB/s (2 GPUs), 5.5 GB/s (4 GPUs), **the same with and without `NCCL_P2P_DISABLE=1`**: NCCL logs "P2P is disabled between connected GPUs" and uses `via SHM/direct` |

So on g6.12xlarge GPU-to-GPU traffic always goes through host memory. This is a platform
limit, not an NCCL setting. It confirms the study-16 TP hypothesis (inter-GPU traffic
costing ~10x what PCIe bandwidth predicts).

**Smoke test** (`smoke_test.sh up/probe`):
- Every checked preset starts in 196-212 s (S3 cuda, S4 cpu, S5 fp8, S9 3P1D) with
  FLASHINFER. FP8 weights run through Marlin weight-only, since the L4 has no FP8 compute.
- NIXL is present in the image, the compatibility check passes, and there were 0 failed
  transfers. The 40,080 prompt tokens show as `local_compute` on prefill and
  `external_kv_transfer` on decode (564 MB per 4096-token request).
- The `model_name` role labels, the router series, `active_gpus` (= P + D) and every
  endpoint scrape work as designed.
- **KV transfer:** `kv_buffer_device=cuda` takes **~1.4 s** per transfer (UCX `cuda_copy`,
  no `cuda_ipc`, ~0.4 GB/s). `cpu` takes **~0.06 s**: 23x faster. Router TTFT at idle: 3.0 s
  on the GPU path vs 2.07 s on the host path. Prefill alone is 1.5 s.
- fp8 KV works (scale 1.0 warning, harmless for performance): the decode KV capacity doubles
  from 63k to 126k tokens.

**Calibration** (`smoke_test.sh calibrate`, 60 s levels, so high levels undercount completions):

| Concurrency | baseline agg-2: TTFT p90 / ITL p99 / req/s | 1P1D cpu (old S4): TTFT p90 / ITL p99 / req/s |
|---|---|---|
| 2 | 1.6 s / 36 ms / 0.18 | 4.3 s / 39 ms / 0.14 |
| 4 | 3.2 s / 50 ms / 0.30 | 8.4 s / 44 ms / 0.21 |
| 8 | 6.3 s / 74 ms / 0.41 | 16.6 s / 53 ms / 0.19 |
| 12 | 9.5 s / 100 ms / 0.45 | 24.9 s / 67 ms / 0.28 |
| 16 | 13.0 s / 120 ms / 0.46 | 25.2 s / 99 ms / 0.16 |
| 24+ | ~20 s / 150-164 ms / ~0.45 (saturated) | 30-44 s / 54-65 ms / collapsing in 60 s windows |

Reading it: the load is **prefill-bound**. One prefill costs ~1.5 s of GPU, and 256 batched
decode tokens cost ~0.6 s, so the work splits ~70:30. With 1P1D the single prefill GPU is always
full (4-5 running, up to 147 waiting) while decode idles half the time. Disaggregation does
keep ITL low under load, as expected. But the ratio has to lean to prefill, hence the
revised presets (2P1D, 3P1D).

## Setup & run

From the toolbox (the workflow's key path must exist where the CLI runs):
`kubectl -n akamas exec -it deploy/toolbox -c toolbox -- bash`, then
`cd /work/vllm-benchmark/studies/19-l4-pd-disaggregation-tps-per-gpu-rerun/akamas`.

Typed form, in dependency order:
```bash
akamas create system system.yaml
akamas create component components/pd_topology.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/vllm_prefill.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/vllm_decode.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/gpu0.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/gpu1.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/gpu2.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/gpu3.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/cluster.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/cluster_loadtest.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/container.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create component components/container_loadtest.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create telemetry-instance telemetry/prometheus.yaml vLLM_Benchmark_19_L4_PD_Disaggregation
akamas create workflow 19-L4-PD-Disaggregation-Workflow.yaml
akamas create study 19-L4-PD-Disaggregation-TPS-Per-GPU-Rerun.yaml

akamas start study "19-L4-PD-Disaggregation-TPS-Per-GPU-Rerun"
```

Bulk alternative (every file carries `kind:` and, where needed, `system:`; the same
dependency order still applies, so create the system first):
```bash
akamas create -f system.yaml
akamas create -f components/
akamas create -f telemetry/
akamas create -f 19-L4-PD-Disaggregation-Workflow.yaml
akamas create -f 19-L4-PD-Disaggregation-TPS-Per-GPU-Rerun.yaml
```

After the run, export it straight away, because Prometheus keeps only 10 days:
`akamas export study "19-L4-PD-Disaggregation-TPS-Per-GPU-Rerun" studies/19-l4-pd-disaggregation-tps-per-gpu-rerun/results/export.tar.gz`.

Editing a created study: on 3.7.x only the goal can be updated in place
(`akamas update study "19-L4-PD-Disaggregation-TPS-Per-GPU-Rerun" 19-L4-PD-Disaggregation-TPS-Per-GPU-Rerun.yaml`).
Steps or parameter changes need a new study.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
