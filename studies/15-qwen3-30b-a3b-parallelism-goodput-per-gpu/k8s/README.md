# k8s/ — the files this study actually runs

Unlike study 14 (whose `k8s/` was a read-only snapshot, because it reused study 13's
workflow), **this study owns its workflow**: `15-Qwen3-30B-A3B-Parallelism-Goodput-Per-GPU-Workflow`
references these files by absolute path on the toolbox host.

```
/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/k8s/01-deployment_template.yaml
/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/k8s/apply_config.sh
/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/k8s/run_test_goodput.sh
```

So the repo must be synced to `/work/vllm-benchmark` on the toolbox before the study
starts, and editing a file here changes what the next trial runs.

## Order of application (once, before `akamas start study`)

```bash
kubectl apply -f 00-pvc.yaml                       # AIPerf results volume (llm-benchmark)
kubectl apply -f 01-pvc-model-cache.yaml           # 80Gi model cache — NEW claim, Qwen weights
kubectl apply -f 06-hf-cache-pvc.yaml              # AIPerf's HF cache (ShareGPT + tokenizer)
kubectl apply -f 02-service.yaml
kubectl apply -f 04-kv-cache-exporter-configmap.yaml
# 03-hf-secret.yaml is NOT needed: the model is not gated (HF API, 2026-09-17).
bash smoke_test.sh                                 # ~8 configs, see its own header
```

`01-deployment_template.yaml` is never applied by hand: the Akamas FileConfigurator
renders it into `01-deployment.yaml` per trial, and `apply_config.sh` applies that.

## What is model-specific here (vs studies 13/14)

| File | What changed for Qwen3-30B-A3B |
|---|---|
| `01-deployment_template.yaml` | model + served name, `--max-model-len 8192` (was 32768), no `--reasoning-parser`, `progressDeadlineSeconds` 1800 (was 1200), startupProbe 30 min, KV-exporter geometry 48 layers / 4 KV heads / 128 head_dim, model-cache claim |
| `01-pvc-model-cache.yaml` | own claim `vllm-model-cache-qwen3moe`, 80Gi (31.18 GB checkpoint) |
| `05-job.yaml` | ramp starts at 24 instead of 150, no `reasoning_effort`, own ShareGPT cache file + tokenizer |
| `apply_config.sh` | rollout poll deadline 2100s, kernel-selection grep looks for FP8/Marlin instead of MXFP4 |
| `smoke_test.sh` | 8 new configurations, including the two that test the study's riskiest assumptions (FP8 block-quant on Ada, expert parallelism required at 4 GPUs) |

Unchanged from study 13/14: `00-pvc.yaml`, `02-service.yaml`, `04-kv-cache-exporter-configmap.yaml`,
`06-hf-cache-pvc.yaml`, `run_test_goodput.sh` (paths aside) and everything under `monitoring/`.
