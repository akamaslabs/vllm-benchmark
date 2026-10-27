# k8s/ — 43-c8i-metal-qwen3-06b-cpu-membw

Study 30's single-vLLM machinery moved to the vLLM CPU backend: one StatefulSet replica
(`vllm-0`, `vllm/vllm-openai-cpu:v0.31.0`, `Qwen/Qwen3-0.6B` in bf16, thinking off) on the
c8i.12xlarge node (`node-role: llm-serving-cpu`, taint `workload=cpu-inference:NoSchedule`),
one AIPerf Job on the system node, everything in namespace `llm-cpu`. No GPU, no HF token.

| File | What |
|---|---|
| `params.env.template` | Rendered by the workflow's FileConfigurator (`${vllm.*}` tokens) into `params.env`. Its header holds the KV-budget arithmetic for `gpu_memory_utilization` on CPU. |
| `render_statefulset.sh` | Validates `params.env` against vLLM 0.31.0's own startup checks and writes the vLLM flags into `01-statefulset_template.yaml` (booleans as `--x` / `--no-x`). |
| `apply_config.sh` | Apply config task: validate, stop the old replica, start `vllm-0`, wait (fail fast on a crash loop or a pod the kubelet does not admit), 8 warm-up requests, health check, startup summary (OpenMP binding, KV size) and full logs. |
| `run_test.sh`, `render_job.sh`, `05-job_template.yaml`, `lib_workload.sh`, `lib_watchdog.sh` | RunTest task: AIPerf 0.11.0 synthetic multi-turn sessions with a shared-prefix pool (`lib_workload.sh`), study 27's open-loop linear rate ramp (`RT_RATE`, `RT_RAMP_S`), watchdog at the SLO (TTFT p95 > 3000 ms or TPOT p95 > 100 ms for 120 s), fail-fast guards, full logs. |
| `lib_health.sh` | The post-warm-up health decision. |
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml`, `02-service.yaml` | AIPerf results, AIPerf HF cache (tokenizer), Services (`vllm`, `vllm-headless`). |
| `monitoring/servicemonitor.yaml` | ServiceMonitor `vllm-cpu` (namespace `llm-cpu`). |
| `tests/` | `for t in tests/test_*.sh; do bash $t; done` (yq v4 needed): renderers, guards, watchdog, run_test and apply_config with a stub kubectl. `tests/dry_run_aiperf.sh` runs the real AIPerf 0.11.0 workload against `mock_openai.py` (network: pip, tokenizer). |

Apply once per cluster: `kubectl apply -f 00-pvc.yaml -f 06-hf-cache-pvc.yaml -f 02-service.yaml
-f monitoring/servicemonitor.yaml` (namespace `llm-cpu` from `../infra/k8s-bootstrap`).

## vLLM CPU pod

- **Guaranteed QoS**: `cpu: 44`, `memory: 48Gi`, requests = limits. With the node's static CPU
  manager (`full-pcpus-only`) the container gets an exclusive cpuset of 22 whole cores; vLLM's
  `VLLM_CPU_OMP_THREADS_BIND=auto` reads `sched_getaffinity`, takes one thread per core (the
  higher SMT sibling) and leaves `VLLM_CPU_NUM_OF_RESERVED_CPU=2` of them out: **20 OMP
  threads** expected (the probe checks the `OpenMP thread binding info` log lines).
- `securityContext`: seccomp `Unconfined` + `SYS_NICE` (vLLM CPU docs: otherwise
  `get_mempolicy: Operation not permitted` and no NUMA binding). `/dev/shm`: 4Gi memory
  emptyDir, counted against the 48Gi limit.
- Startup: `VLLM_ENGINE_READY_TIMEOUT_S=1800` (CPU inductor compile at cold start),
  startupProbe 40 min, `ROLLOUT_S` 45 min (below the workflow's 50 min Apply config timeout).
  The HF cache and vLLM's compile cache (`VLLM_CACHE_ROOT`) are hostPaths on the node.
- Fixed flags: `--dtype=bfloat16` (AMX path), `--max-model-len=16384`,
  `--default-chat-template-kwargs '{"enable_thinking": false}'`, `--enable-prompt-tokens-details`
  (AIPerf then reports cached prompt tokens). `VLLM_CPU_KVCACHE_SPACE` is never set: any
  non-empty value overrides `--gpu-memory-utilization` (`vllm/platforms/cpu.py`).
- No KV offload: vLLM 0.31.0's native offload (`OffloadingConnector` / `--kv-offloading-size`)
  fails at startup on the CPU backend (`vllm/v1/kv_offload/cpu/spec.py`).

## Known conflicts (vLLM 0.31.0 source)

- **`enable_chunked_prefill=false` cannot start with this study's domains.**
  `vllm/config/scheduler.py` `verify_max_model_len` raises when chunked prefill is off and
  `max_num_batched_tokens < max_model_len`; `--max-model-len` is 16384 and the
  `max_num_batched_tokens` domain tops at 8192. Chunked prefill is therefore
  fixed on in `params.env.template` (decided 2026-10-16) and not in the study;
  `render_statefulset.sh` still refuses `ENABLE_CHUNKED_PREFILL=false` (exit 2, before any
  cluster call) if it is set by hand. `long_prefill_token_threshold` only acts with chunked
  prefill on.
- **`gpu_memory_utilization` edges are computed, not measured** (see `params.env.template`):
  0.90 (the baseline and domain top) leaves about 1-1.5 GiB at the KV-allocation check, and
  vLLM's default 0.92 about 0-0.5 GiB (the EngineCore process's RSS counts in the cgroup but is
  not subtracted by vLLM); 0.15 fails if the engine RSS after warm-up exceeds ~5.4 GiB. The
  probe's `B-default` (0.90) / `E-gmu-high` (0.92) / `E-gmu-low` runs settle both.
- `vllm:request_time_per_output_token_seconds` (TPOT) is observed when a request completes,
  so the watchdog's and the constraint's TPOT p95 lag the load by about one request.
- `total_token_throughput` counts all prompt tokens, cached prefix tokens included: a
  configuration that hits the prefix cache more scores more for the same requests (the effect
  the study's prefix-caching / KV-budget axes are meant to expose).
