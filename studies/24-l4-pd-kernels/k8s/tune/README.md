# Triton FP8 block-GEMM tuning for Qwen3-8B-FP8 on L4 (study 24)

vLLM 0.29.0 ships no tuned `TritonFp8BlockScaledMMKernel` configs for `NVIDIA_L4`, so the
`triton` linear backend runs on the default config ("Using default W8A8 Block FP8 kernel
config"). This folder generates the tuned configs once. The study then ships them with
`tuned_kernel_configs=true` (see `../01-deployment_template.yaml`, "Kernel backends").

- `tune_fp8_block.py`: wrapper around vLLM's own
  `/vllm-workspace/benchmarks/kernels/benchmark_w8a8_block_fp8.py`, for the 4 Qwen3-8B TP=1
  shapes (QKV 6144x4096, O 4096x4096, gate/up 24576x4096, down 4096x12288) and M = 1 ...
  16384, output dtype bfloat16. See its docstring for why it does not call the script's
  own `main()`.
- `tune-pod.yaml`: one-off pod on the GPU node, 4 GPUs.

## Run (from the toolbox, with no study running)

```bash
cd /work/vllm-benchmark/studies/24-l4-pd-kernels/k8s/tune
kubectl -n llm-serving scale deployment/vllm-pd --replicas=0      # frees the 4 GPUs
kubectl -n llm-serving create configmap fp8-tune-script --from-file=tune_fp8_block.py \
  --dry-run=client -o yaml | kubectl apply -f -

# 1. Smoke run: 2 batch sizes, a few minutes. Set TUNE_ARGS in tune-pod.yaml to
#    "--batch-sizes 16,4096", then:
kubectl apply -f tune-pod.yaml
kubectl -n llm-serving logs -f fp8-tune          # ends with "done in ... s" / TUNING-DONE
kubectl -n llm-serving delete pod fp8-tune

# 2. Full run: TUNE_ARGS back to "", then apply again. Duration not measured yet: 88 jobs
#    (4 shapes x 22 M) x 1280 configs each; the large-M gate/up jobs dominate.
kubectl apply -f tune-pod.yaml
kubectl -n llm-serving logs -f fp8-tune

# 3. Copy the results next to the study, then free the node.
kubectl -n llm-serving cp fp8-tune:/out ../tuned-configs
kubectl -n llm-serving delete pod fp8-tune
```

The next trial's `apply_config.sh` packs `../tuned-configs/*.json` into ConfigMap
`pd-tuned-configs`; the launcher installs them only when `tuned_kernel_configs` is true.

## Before using them in the study

Measure the effect outside Akamas first, with the same direct requests used on 2026-09-28
(one 4090-token prompt, and two prompts 0.9 s apart, straight to prefill-0; decode TPOT via
the router), on `--linear-backend=triton` with and without the tuned files. The launcher log
must show `launcher: tuned kernel configs installed`, and vLLM must log
`Using configuration from ... for W8A8 Block FP8 kernel` instead of `Using default ...`.

Notes:
- Tuning runs at the L4's power limit like serving does (72 W, SM clock ~1050 MHz under
  sustained load), so configs are picked for throttled clocks, which is the serving regime.
- The configs depend on the GPU model, Triton version (vLLM image) and weight shapes. Re-tune
  after changing any of them.
