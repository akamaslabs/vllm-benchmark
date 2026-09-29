# 25-g7-4500-gpu-sharing-goodput

**Status:** RUNNING — created and started on Akamas 2026-09-29 21:25 UTC (GPU pack 1.3.0
installed, dcgm-exporter covering both GPU nodes).
**Dates:** 2026-09-29 –

## Objective

Does splitting one GPU between replicas of a small model beat serving it with a single
replica on the whole GPU — and if it does, with which technique? One categorical
parameter, `gpu0.sharing_mode`, picks how the single RTX PRO 4500 is shared, and the
optimizer searches it together with the vLLM parameters that interact with it:

| `sharing_mode` | vLLM replicas | What each replica gets |
|---|---|---|
| `exclusive` | 1 | the whole GPU (84 SMs, 31.38 GiB) |
| `mig` | 2 | one `1g.16gb` MIG slice (42 SMs, 15.66 GiB, isolated) |
| `time_slicing` | 2 | the whole GPU, CUDA contexts time-sliced, no memory isolation |
| `mps` | 2 | one CUDA MPS client each: kernels of both run concurrently, but the device plugin's MPS daemon caps each client at 50 % of the SMs (default active thread percentage 100/replicas) and half the memory (pinned-memory limit; vLLM sees 15.79 GiB free) — a soft slice, not the whole GPU |

**Goal:** maximize aggregate `vllm.prefill_token_throughput + vllm.decode_token_throughput`
(tokens/s summed over the replicas), subject to TTFT p95 <= 1500 ms and ITL p95 <= 300 ms,
`stability` windowing — the same goodput goal and interactive-chat SLA as studies
1/9/10/17. One GPU, so per-GPU and aggregate goodput coincide.

`exclusive` is in the domain as the real opponent, not as a formality: phase 0 (below)
found it ahead of every split mode at vLLM defaults. The study is a fair test of the
hypothesis "there is room on this GPU, so splitting pays", and it can come out negative.

Related: ROADMAP H4 (the "GPU fraction per replica" gap, `knowledge/notes/2026-07-gpu-
fractioning-nvidia-runai.md`) and Section D study #4 (MIG right-sizing, a different
question: the smallest slice meeting a target).

## Stack & versions

- **Akamas:** 3.7.x.
- **Optimization packs:**
  - **GPU 1.3.0** — adds `sharing_mode` to the `GPU` component type. Built for this study
    on branch `feature/gpu-sharing-mode` of the `nvidia-gpu` pack repo (commit `74d0fd2`,
    local only). **Not installed** — the server runs GPU 1.2.0 (see "Before starting").
  - **vLLM 1.12.0** (installed) — `gpu_memory_utilization`, `max_num_seqs`,
    `max_num_batched_tokens`, `stream_interval`.
  - **Kubernetes** (installed) — node and container components.
- **Workload:** `vllm/vllm-openai:v0.29.0` serving `Qwen/Qwen3-4B-Instruct-2507-FP8`
  (4.83 GiB of weights, non-thinking instruct, 36 layers, 8 KV heads, 144 KiB/token of
  bf16 KV) as `qwen3-4b`, `--max-model-len 4096`, prefix caching off. FP8 GEMMs run on
  `DeepGemmFp8BlockScaledMMKernel` (W8A8) on SM 12.0; attention on `FLASH_ATTN`.
  StatefulSet `vllm` in namespace `gpu-sharing`, pods `vllm-0` / `vllm-1`.
- **Cluster / hardware:** EKS `vllm-bench` 1.35, us-east-2. Node group
  `llm-serving-g7-4500`: 1x `g7.4xlarge` (16 vCPU, 64 GiB), 1x **NVIDIA RTX PRO 4500
  Blackwell Server Edition** — 32623 MiB, 165 W power limit, compute capability 12.0,
  driver 595.91.07, CUDA 13.2, EKS AMI release 1.35.8-20260923. MIG supports exactly two
  layouts: 2x `1g.16gb` (profile 5) or 1x `2g.32gb`. See `infra/README.md`.
- **Load generator:** AIPerf 0.11.0 on `system-m8a` (m8a.xlarge), ShareGPT replay
  (`inputs_json` cache generated once per served model), closed loop, 60 s warm-up at 64
  users (discarded), then 12 levels `16,32,64,96,128,192,256,320,384,512,640,768` x 300 s.
  `k8s/05-job.yaml`.
- **Telemetry:** Prometheus (kube-prometheus-stack), 117 metrics
  (`akamas/telemetry/prometheus.yaml`); vLLM scraped every 5 s through ServiceMonitor
  `vllm-gpu-sharing`; DCGM through the shared `dcgm-exporter` release (see "Before
  starting"); cAdvisor / kube-state-metrics for CPU.

## Parameters tuned

| Parameter | Domain | Baseline / presets | Note |
|---|---|---|---|
| `gpu0.sharing_mode` | `exclusive`, `mig`, `time_slicing`, `mps` | `exclusive` (baseline), then one preset each | new in GPU pack 1.3.0 |
| `vllm.gpu_memory_utilization` | [0.80, 0.90] | 0.90 | fraction of the memory the **replica** owns; `apply_config.sh` halves it under `time_slicing` / `mps` |
| `vllm.max_num_seqs` | [64, 768] | 256 | per replica; 256 is vLLM's default on this GPU and was the ceiling phase 0 hit |
| `vllm.max_num_batched_tokens` | [1024, 8192] | 2048 | vLLM's default here; always >= `max_num_seqs`, so no constraint is needed |
| `vllm.stream_interval` | [1, 16] | 1 | host-side streaming cost |

Presets and baseline share the same vLLM values (0.90 / 256 / 2048 / 1), which are
exactly what phase 0 ran: the first four experiments are the head-to-head comparison of
the four modes, before the optimizer (30 experiments) searches mode x vLLM settings.

Domain bounds come from phase 0 memory measurements, not from the pack: at 0.90 a MIG
slice ends up at 13.77 of 16.03 GB used; the halved 0.45 leaves MPS clients inside the
15.79 GiB MPS limit (an un-halved 0.90 failed at startup). `max_num_seqs` stops at 768
to keep the sampler-warmup logits (fp32, 151936-token vocabulary, ~0.6 MiB per sequence)
inside the ~2 GiB left over on a slice.

Deliberately **not** tuned: `api_server_count`. It was planned (and approved) to give
`exclusive` the extra host CPU a split would get, but phase 0 measured each vLLM process
at ~10 % of one core at 128 users: the confounder does not exist here, so the pack change
it needed was dropped.

## Design

### How an experiment switches mode (`k8s/apply_config.sh`)

1. FileConfigurator renders `k8s/params.env.template` -> `params.env` (all five
   parameters). The script refuses to run with an unsubstituted or empty value.
2. Scale the StatefulSet to 0 and delete its pods explicitly (with `OrderedReady` a
   scale-down stalls while a replica is unready); refuse to continue if any is left.
3. Set the device-plugin config to the neutral `exclusive` (stops an MPS daemon, which
   holds a GPU context and Exclusive_Process compute mode).
4. MIG: for `mig`, enable MIG and recreate 2x `1g.16gb` (`mig -cgi 5,5 -C`); otherwise
   destroy the slices and disable MIG. Reset compute mode to DEFAULT. Fail if
   `mig.mode.current,pending` does not match the mode.
5. Label the node `nvidia.com/device-plugin.config=<mode>`; wait until it advertises
   `nvidia.com/gpu` = 1 (exclusive) or 2; for `mps` wait for the MPS control daemon.
6. Restart the dcgm-exporter pod on the node (it re-reads the MIG layout).
7. Render the StatefulSet (`sed` — the toolbox has no `envsubst`) with 1 or 2 replicas
   and the effective memory fraction; `OrderedReady` starts `vllm-1` only after `vllm-0`
   is Ready. The wait fails fast when a replica restarts twice (crash loop) instead of
   sitting out the 40-minute budget.

`k8s/run_test_goodput.sh` then runs the AIPerf Job and fails the trial as soon as the
Job fails, a vLLM replica restarts or is replaced, or no request completes for 15 min
(guards ported from study 24): a replica that crashed mid-ramp must not score VALID,
which is what happened to study 9.

The GPU sharing layer — the study's own device plugin with four named configs and the
privileged `gpu-admin` DaemonSet for host `nvidia-smi` — is installed once by
`infra/gpu-sharing/install.sh`; `infra/README.md` explains the extra node taint that
keeps the cluster-wide device plugin off this node.

### Telemetry rules that phase 0 forced

- GPU queries filter on the GPU (node / model name), **never on `pod`**: under
  time-slicing and MPS the single DCGM series carries only one of the two pods.
- Under MIG DCGM reports per slice: `PROF_*` averaged over the two equal slices gives
  the GPU value; device-level values (power, temperature, clocks, throttle reasons) are
  duplicated on each slice and must be aggregated with `max`/`avg`, never `sum`;
  framebuffer used is summed.
- **Out of goal, constraints and KPIs:** `gpu_util` (absent under MIG, reads 0 under MPS)
  and `gpu_dram_active` (absent under MIG). Mode-agnostic GPU metrics:
  `gpu_gr_engine_active`, `gpu_sm_active`, `gpu_sm_occupancy`, `gpu_tensor_core_active`,
  `gpu_power_usage`, `gpu_sm_clock`, `gpu_clock_throttle_reasons`, `gpu_fb_used`.
- Goal, constraints, windowing and KPIs reference only the aggregate `vllm` component.
  `vllm_r0` / `vllm_r1` exist for per-replica views (load balance, which replica
  saturates first); `vllm_r1` is empty in `exclusive` experiments by construction.
- Server-side TTFT p95 near the 1500 ms SLA is interpolated inside vLLM 0.29's
  1.0-2.5 s histogram bucket, so a constraint verdict close to the limit is approximate
  (ITL's 300 ms is an exact bucket edge). Cross-check the four head-to-head presets
  against AIPerf's client-side per-level percentiles in the Job artifacts. The queue-time
  KPI uses the mean: the p95 floors at ~285 ms (first bucket 0.3 s).
- Power and clocks matter here: at 128 users in `exclusive` the GPU sat at 165 W (its
  limit) with the SM clock at ~1.8 GHz of 2.415 and the SW-power-cap throttle reason
  active. A power-bound GPU puts one ceiling on every mode.

## Phase 0 — manual validation, 2026-09-29

Done by hand on the live node before writing the study (throwaway manifests, same
names as the committed infra). Single 60 s runs per level at 64/128/256/512/1024/2048
users, vLLM defaults except `--gpu-memory-utilization` — indicative, not the study.

**Mechanism.** All four modes and every transition between them work: MIG on/off applies
immediately with dcgm-exporter and the device plugin running (no reboot), a plugin config
change re-registers in 15-40 s, the MPS daemon starts on the `mps.capable` label and
leaves Exclusive_Process compute mode behind (reset explicitly).

**The committed `apply_config.sh`, end to end** (same day, run from a workstation with
`STUDY_DIR` pointing at a copy): `mps` (with `stream_interval` 4, the first run passing
every template flag) -> `mig` -> `exclusive`, all rc=0, node left at MIG off / compute
mode Default / 0 MiB. The `mps` -> `mig` transition had never been exercised by hand and
exposed a latent bug, now fixed: the chart gives the MPS daemon's pods the same labels
as the plugin's, so the old label-based wait for the daemon to go returned immediately;
the script now waits by pod name and checks the host for `nvidia-cuda-mps` processes
before touching MIG.

**Inside the toolbox, after the audit** (2026-09-29 20:19-20:37 UTC): the committed
`apply_config.sh` (mode `mps`, `sed` rendering, crash-loop-aware rollout) rc=0 in 457 s,
then `run_test_goodput.sh` with a copy of the Job cut to 32/64 users x 60 s rc=0 in
508 s, no false trip of the restart/stall guards. That run generated the ShareGPT cache
on the study's `aiperf-results` volume, and with ServiceMonitor `vllm-gpu-sharing` live the
scored queries were evaluated on the real `vllm-0` / `vllm-1` series: 12 scrapes per
minute per pod, load split 52/48 by generated tokens, ~3300 tokens/s decode at 64 users
(phase 0 MPS: 3170). The ~30 s zero stretches between AIPerf levels are its inter-level
setup, not lost scrapes.

**Memory seen by vLLM.** MIG slice: 7.69 GiB KV = 56,016 tokens per replica at 0.90.
Exclusive: 21.84 GiB = 159,008 tokens at 0.90 (splitting costs 30 % of the KV pool:
the weights are loaded twice). Time-slicing at 0.45: 56,208 tokens per replica. MPS:
total 31.38 GiB, **free 15.79 GiB** — the limit lowers free memory, not total — so 0.90
fails and 0.45 gives 56,192 tokens.

**Workload.** ShareGPT via AIPerf averaged 75-103 prompt and ~212 output tokens per
request. KV never bound (exclusive peaked at 60 %, zero preemptions): the ceiling at 256
running requests is vLLM's default `max_num_seqs`, after which requests queue and TTFT
explodes. The first burst on a cold server had TTFT p95 33.6 s (every slow request
started at t=0); hence the warm-up.

**Throughput** (best level meeting TTFT p95 <= 1500 / ITL p95 <= 300):

| Mode | Best level | Output tokens/s | vs exclusive |
|---|---|---|---|
| exclusive | 128 | 4560 | — |
| mps | 128 | 4252 | -7 % |
| mig | 128 | 4152 | -9 % |
| time_slicing | 256 | 3661 | -20 % |

**vLLM data parallelism instead of two pods (tested the same day, MIG only).** One pod
requesting both slices, `--data-parallel-size 2`, each DP rank pinned to one slice by
exporting `CUDA_VISIBLE_DEVICES` from the MIG UUIDs `nvidia-smi -L` lists inside the pod
(`NVIDIA_VISIBLE_DEVICES` reads `void` there; one CUDA process sees one MIG device, so the
ranks must be split explicitly). It starts cleanly (vLLM also defaults `api_server_count`
to 2) and serves 4050 / 4125 / 3381 output tokens/s at 128 / 256 / 512 users against the
two-pod layout's 4152 / 4018 / 3437, TTFT p95 421 / 1489 ms vs 361 / 1288 ms: the same
within run-to-run noise. The frontend's least-loaded routing buys nothing over kube-proxy
spreading 128+ connections at random, because both engines are GPU-bound. So
`data_parallel_size` is not a parameter of this study; time-slicing and MPS with DP (two
ranks on one GPU UUID) were not tried.

**One MIG slice alone, the other idle (same day).** One replica on one `1g.16gb` slice:
2235 / 2378 / 2034 / 2099 output tokens/s at 64 / 128 / 256 / 512 users, SLA held up to
128 (TTFT p95 824 ms). The GPU drew 132-136 W with the SM clock at 2400 MHz and no throttle
reason, against 165 W / ~1.8 GHz / SW power cap for exclusive: an idle neighbour slice
leaves its power budget to the busy one. Per unit of GPU that is 2378 / 0.5 = 4756 tokens/s,
+4 % over exclusive's 4560 — while with both slices busy each delivers ~2076 (4152 / 2), 9 %
below. The half-GPU result therefore depends on what the other half is doing.

CPU was never the bottleneck (EngineCore and API server ~10 % of a core each, AIPerf
~1.3 of 4 cores). At defaults the split modes lose: they halve the KV pool and add
scheduling overhead without unlocking anything a single engine lacked. The optimizer can
still find settings (e.g. larger `max_num_seqs` per replica) where that changes.

## Before starting

Nothing below has been done — each is a deliberate step for the user to confirm.

1. **GPU pack 1.3.0 install.** Branch `feature/gpu-sharing-mode` of the `nvidia-gpu` pack
   repo, rebased 2026-09-29 onto `feature/nvlink-profiling-counters` (what the server runs
   as 1.2.0), so 1.3.0 is a strict superset: 1.2.0's 42 metrics unchanged plus
   `sharing_mode` (checked on the built JSON). Installing needs an Akamas login with the
   **Administrator** role in the toolbox; the toolbox session was not logged in / not
   admin on 2026-09-29 evening.
2. **dcgm-exporter** — DONE 2026-09-29 (helm revision 22 of the shared release): it now
   covers BOTH `llm-serving-l4` and `llm-serving-g7-4500` with a nodeAffinity
   (`k8s/monitoring/dcgm-exporter-values.yaml`). Checked in Prometheus right after: the
   RTX PRO 4500 series are there, the four L4s still are, and study 24's
   `exported_pod=~"vllm-pd.*"` filter sees only the L4s. Safe for studies 24 and 25; NOT
   for studies 0-17 (`pod: .*`) — pin the exporter back before resuming any of them.
3. **Do not overlap** with another study whose AIPerf Job runs on `system-m8a` (4 vCPU)
   unless both fit: this Job requests 1500m (study 24's requests 2). And studies 4-17
   must not be resumed while this one runs: their vLLM components use `model: .*` /
   `pod: .*` and would absorb `qwen3-4b`'s series (study 24 filters on its own models and
   pods, so it does not).
4. **Toolbox sync** (git pull there), then validate and create:
   `akamas create -f studies/25-g7-4500-gpu-sharing-goodput/akamas/` — the YAML is
   validated offline against the pack sources, not yet against the server.
5. **ServiceMonitor** `vllm-gpu-sharing` and the study PVCs/Services:
   `infra/eks/provision.sh` step 7 (additive, scoped to namespace `gpu-sharing`).
6. **AlwaysOn** is already on the node group's ASG (set 2026-09-29); remove it when the
   study ends and the node group goes to 0.

Budget: ~70-80 min per experiment (1-2 cold-ish replica starts, 60 s warm-up, 60 min
ramp) x 34 experiments ~= 42 h of node time, ~130 USD at 3.04 USD/h.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
