# k8s/ — 41-l4-dsv2-lite-context-parallel

Study 39's single-vLLM machinery moved to the 4x L4 GPU node: one StatefulSet replica
(`vllm-dsv2-0`, `vllm/vllm-openai:v0.32.0`, `deepseek-ai/DeepSeek-V2-Lite-Chat` in bf16, served as
`dsv2-lite`) holding all four L4s of the g6.12xlarge node (`node-role: llm-serving-l4`, taint
`nvidia.com/gpu`), one AIPerf Job on the system node, everything in namespace `llm-serving`.
No HF token (the model is ungated).

**Names are study-specific (`vllm-dsv2*`, `aiperf-dsv2`).** Namespace `llm-serving` is forced by
the model cache PVC, which already exists there, and it also holds another study's Deployment
`vllm`, Service `vllm` and ServiceMonitor `vllm` (selector `app: vllm`). Study 39's names
(`vllm`, `vllm-0`, `app: vllm`) would collide with all three.

| File | What |
|---|---|
| `params.env.template` | Rendered by the workflow's FileConfigurator (`${vllm.*}` tokens) into `params.env`. Empty value = flag omitted = vLLM's default; its header holds the topology rules. |
| `render_statefulset.sh` | Validates `params.env` against vLLM 0.32.0's own startup checks and the smoke run's crash, and writes the vLLM flags into `01-statefulset_template.yaml` (booleans as `--x` / `--no-x`). |
| `apply_config.sh` | Apply config task: validate, check the node (present, in us-east-2a where the model PVC lives), **GPU-busy guard** (exit 5 if any other pod on the node requests `nvidia.com/gpu`), stop the old replica, start `vllm-dsv2-0`, wait (fail fast on a crash, a failed pod or an unschedulable one), 8 warm-up requests (short and ~8k-token), health check, startup summary (backends, KV size) and full logs. |
| `run_test.sh`, `render_job.sh`, `05-job_template.yaml`, `lib_workload.sh`, `lib_watchdog.sh` | RunTest task: AIPerf 0.11.0 synthetic multi-turn long-context sessions with an 8k-token shared-prefix pool (`lib_workload.sh`), closed-loop users ramp `--concurrency N --concurrency-ramp-duration RAMP` (`RT_USERS`, `RT_RAMP_S`, as study 38), watchdog at 2 x the SLO (TTFT p95 > 20000 ms or TPOT p95 > 100 ms for 120 s), fail-fast guards, full logs. |
| `lib_health.sh` | Pure decisions: the post-warm-up health check and the GPU-busy filter. |
| `00-pvc.yaml` | The model cache PVC `vllm-model-cache-dsv2lite` (already exists, declared with its live spec) and the AIPerf results PVC `aiperf-results-dsv2`. |
| `06-hf-cache-pvc.yaml`, `02-service.yaml` | AIPerf HF cache (tokenizer), Services `vllm-dsv2` and `vllm-dsv2-headless`. |
| `monitoring/servicemonitor.yaml` | ServiceMonitor `vllm-dsv2` (namespace `monitoring`, selects `app: vllm-dsv2` in `llm-serving`). DCGM already runs on the node. |
| `tests/` | `for t in tests/test_*.sh; do bash $t; done` (yq v4 needed): renderers, guards, watchdog, run_test and apply_config with a stub kubectl. `tests/dry_run_aiperf.sh` runs the real AIPerf 0.11.0 workload against `mock_openai.py` (network: pip, tokenizer). |

Apply once (from `../infra/provision.sh`, or by hand): `kubectl apply -f 00-pvc.yaml -f
06-hf-cache-pvc.yaml -f 02-service.yaml -f monitoring/servicemonitor.yaml`.

## vLLM pod

- `resources`: 4 x `nvidia.com/gpu`, CPU 16 request / 40 limit, memory 64Gi / 160Gi (as the smoke
  pod); `/dev/shm` 32Gi memory emptyDir for NCCL between the ranks.
- Model cache PVC mounted at `/root/.cache/huggingface` (`HF_HOME`), as in the smoke run, so the
  downloaded weights are reused; vLLM's compile cache (`VLLM_CACHE_ROOT`) on the same volume, so a
  layout compiles once.
- `VLLM_ENGINE_READY_TIMEOUT_S=1800` (the slowest start in the smoke run took ~535 s, TP4 cold; the default is
  600 s), startupProbe 40 min, `ROLLOUT_S` 45 min (below the workflow's 75 min Apply config timeout).
- Fixed flags: `--trust-remote-code`, `--max-model-len=32768`, `--enable-prompt-tokens-details`.
  Attention backend automatic (`TRITON_MLA` decode, FlashAttention MLA prefill on SM 8.9,
  measured). Prefix caching at vLLM's default (on) via the fixed empty `ENABLE_PREFIX_CACHING=`.

## Known constraints (vLLM 0.32.0 source + smoke run)

- `tp x pcp x dp == 4` (the pod holds 4 GPUs); `tp % dcp == 0` (`vllm/config/parallel.py`).
- PCP > 1: DCP must be 1 on dense MLA (`v1/worker/gpu/pcp_manager.py`: "MRV2 PCP + DCP supports
  sparse MLA models only"), EP on (TP2xPCP2 without EP: `topk_softmax ... is_padding size
  mismatch`, smoke run 1), DP 1 (unverified upstream).
- TP 1 without EP (e.g. DP4): allowed. Without EP the experts are TP-sharded across all
  dp x pcp x tp ranks (`model_executor/layers/fused_moe/config.py`), so it fits; not yet measured.
- `max_num_batched_tokens >= max_num_seqs` (`vllm/config/scheduler.py`).
- `vllm:request_time_per_output_token_seconds` (TPOT, the constraint and the watchdog) exists in
  0.32.0 (`v1/metrics/loggers.py`); it is observed when a request completes, so it lags the load
  by about one request.
- `total_token_throughput` counts all prompt tokens, cached prefix tokens included: that is why
  the goal is `decode_token_throughput` (generation tokens/s).
