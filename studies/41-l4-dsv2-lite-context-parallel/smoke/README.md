# smoke/ — single-request layout smoke test (2026-10-22)

Before the study was designed, every candidate parallel layout of
`deepseek-ai/DeepSeek-V2-Lite-Chat` was started once on the 4x L4 node and probed with long
single requests, to learn which layouts vLLM 0.32.0 accepts, what KV capacity each one gets,
and whether long-context answers stay correct.

| File | What |
|---|---|
| `smoke.sh <name> <vllm args...>` | Creates the model PVC if missing, starts one bare pod `dsv2-smoke` (namespace `llm-serving`, `vllm/vllm-openai:v0.32.0`, `--max-model-len 32768`, `--gpu-memory-utilization 0.90`, **`--no-enable-prefix-caching`**, 4 GPUs), waits for Ready, greps the startup log, runs `bench.py` inside the pod, deletes the pod. |
| `bench.py` | One request at a time, greedy, streaming, on `/v1/completions`: a haystack of ~4k / 16k / 31k tokens with a hidden "secret code" (needle at 40 %), 3 repeats each; TTFT on the 64-token answer, TPOT on a 128-token `ignore_eos` reply, needle found or not. |
| `results/*.log` | Run 2 (12:25-13:10 UTC): status, startup-log excerpt, per-request JSON lines, `SUMMARY`. |
| `results-v1/*.log` | Run 1 (11:31-12:24 UTC, the same script writing to `results/`, renamed afterwards): same layouts plus `tp2-pcp2` (no EP, failed) and `tp2-ep`. |

The full vLLM logs (`*.vllm.log`, 36-174 KB each) were not copied; the lines that matter are in
the `*.log` files (startup excerpt) and in the table below.

## Results (run 2; prefix caching OFF, gmu 0.90, max_model_len 32768, one request at a time)

TTFT and TPOT are the median of 3 repeats; prompt lengths 3,984 / 16,223 / 30,844 tokens.

| Layout (flags) | KV tokens (vLLM) | TTFT 4k / 16k / 31k (ms) | TPOT@128 4k / 16k / 31k (ms) | Needle | Ready after |
|---|---|---|---|---|---|
| TP4 | 374,192 | 455 / 1991 / 4439 | 8.3 / 10.2 / 12.5 | 9/9 | 352 s |
| TP4 + DCP2 | 724,800 | 462 / 2268 / 5563 | 9.0 / 9.9 / 10.9 | 9/9 | 370 s |
| TP4 + DCP4 | 1,524,864 | 460 / 2271 / 5583 | 9.2 / 9.9 / 10.7 | 9/9 | 367 s |
| TP1 x PCP4 + EP | 292,896 (replicated per PCP rank) | 377 / 2214 / 6240 | 17.5 / 19.5 / 22.0 | 9/9 | 366 s |
| TP2 x PCP2 + EP | 319,776 | 466 / 2273 / 5567 | 13.9 / 15.9 / 18.2 | 9/9 | 351 s |
| DP4 + EP | 292,960 per engine (x4) | 590 / 2979 / 7543 | 16.8 / 18.8 / 21.3 | 9/9 | 350 s |
| TP2 x PCP2, no EP (run 1) | — | fails at startup: `RuntimeError: dispatch_topk_launch, /workspace/csrc/moe/topk_softmax_kernels.cu:785, is_padding size mismatch, expected: 4096` | | | failed after 325 s |
| TP2 + EP (run 1, 2 GPUs) | 119,936 | 941 / 4627 (6k / 26k prompts) | 11.8 / 13.2 (6-token replies) | 6/6 | 353 s |

Backends selected in every layout: `TRITON_MLA` (decode attention), `FLASH_ATTN MLA prefill
backend`, `TRITON Unquantized MoE backend`. Run 1 starts took 351-535 s (the first, TP4, was the
slowest: cold image).

## What it says, and what it does not

- **DCP does what it promises on capacity**: on an MLA model (one latent KV head) TP4 replicates
  the latent KV on every rank; DCP2 / DCP4 shard it, KV x1.94 / x4.07. Single-request TPOT at
  31k improves slightly with DCP (12.5 -> 10.7 ms), TTFT at 16-31k gets worse (+14-25 %): more
  capacity, not more single-request speed.
- **PCP does not help a single request here**: TTFT at 31k is 25-40 % worse than TP4, TPOT
  ~1.5-1.8x worse. Whether it helps under concurrent long prefills is what the study measures.
- **All layouts answer correctly** at 31k tokens (needle 9/9): no layout breaks the model's
  long-context quality.
- **Not tested here**: prefix caching (OFF in every run; the study keeps vLLM's default, ON,
  so DCP/PCP + prefix caching must be verified by the calibration run), fp8 KV with any layout,
  concurrency, and TP2 x DP2 layouts.
