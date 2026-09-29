# Kernel microbenchmark (study 24 prep, 2026-09-29)

Direct measurements of the linear (GEMM) and attention kernel backends, outside Akamas, to
choose study 24's domains and presets. Qwen3-8B-FP8, vLLM 0.29.0, 1P1D on the g6.12xlarge
(L4), NixlConnector with host buffer, `gpu_memory_utilization` 0.9, `max_num_seqs` 128,
`max_num_batched_tokens` 8192, prompts of about 4090 tokens. One run per configuration.

- `driver.py`: runs on the toolbox pod (`setsid nohup python3 driver.py`). It waits for the
  Triton tuning pod (`../k8s/tune/`), copies its output to `../k8s/tuned-configs/`, then for
  each configuration renders `../k8s/01-deployment_template.yaml` with fixed values, applies
  it, waits for the rollout and runs `bench_in_pod.py` in the engine container. At the end it
  scales `vllm-pd` to 0. Paths: `KB_K8S` / `KB_OUT` environment variables.
- `bench_in_pod.py`: the measurements. Prefill step = one prompt straight to prefill-0 (mean
  of 4, after one warm-up). Overlap = two prompts 0.9 s apart. E2E = 5 streaming requests
  through the router (64 output tokens). Decode c16 = 16 concurrent requests, about 256 in /
  256 out tokens.
- `results/`: one `.json` (config, startup time, all measurements) and one `.log` (key vLLM
  log lines: selected kernel, attention backend, tuned-config load; then the log tail) per
  configuration, `summary.txt`, `driver.log`, and `tuning-jobs.log` (per-job lines of the
  Triton tuning run).

## Results

Linear backend, the same on both roles (FLASHINFER, KV bf16):

| Config | Linear backend | Prefill step | Overlap 1st / 2nd | TPOT, 1 req | Decode c16 |
|---|---|---|---|---|---|
| L1 | auto (Marlin) | 1.750 s | 3.50 / 2.60 s | 36.1 ms | 339 tok/s, 38.2 ms |
| L2 | humming | 1.329 s | 2.67 / 1.77 s | 35.9 ms | 356 tok/s, 37.7 ms |
| L3 | triton, default config | 1.114 s | 2.22 / 1.32 s | 41.0 ms | 325 tok/s, 42.8 ms |
| L4 | triton, tuned config | 1.066 s | 2.13 / 1.23 s | 39.0 ms | 340 tok/s, 40.8 ms |

Attention backend (prefill triton tuned, decode auto = Marlin):

| Config | Attention | KV | Prefill step | Overlap 1st / 2nd | Decode c16 |
|---|---|---|---|---|---|
| A1 | FLASHINFER | bf16 | 1.068 s | 2.13 / 1.23 s | 359 tok/s, 38.3 ms |
| A2 | TRITON_ATTN | bf16 | 1.221 s | 2.44 / 1.54 s | 337 tok/s, 38.3 ms |
| A3 | FLASH_ATTN | bf16 | 1.087 s | 2.17 / 1.27 s | 336 tok/s, 38.2 ms |
| A4 | auto (vLLM selected FLASH_ATTN) | bf16 | 1.086 s | 2.54 / 1.64 s | 359 tok/s, 38.2 ms |
| A5 | FLASHINFER | fp8 | 1.114 s | 2.21 / 1.31 s | 371 tok/s, 36.8 ms |
| A6 | TRITON_ATTN | fp8 | 1.427 s | 2.51 / 1.61 s | 371 tok/s, 36.8 ms |

Findings:
- Prefill: triton −36% against Marlin, humming −24%. The tuned configs add −4% over triton's
  default config.
- Decode: Marlin and humming are equal; triton is slower (+8% TPOT even with tuned configs).
- Attention: FLASHINFER ≈ FLASH_ATTN on prefill; TRITON_ATTN is 14% slower. With bf16 KV,
  `auto` selects FLASH_ATTN.
- fp8 KV: prefill +4%, decode TPOT −4%. TRITON_ATTN + fp8 KV starts on L4 (sm89).
- Noise: the prefill step is stable (within 2% of the 2026-09-28 run of L1). Decode c16 moves
  by about ±7% (A3 and A4 used the same backend), and the overlap result is also noisy.
