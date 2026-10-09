# probe/ — study 32 startup probe

Study 30's probe, pointed at the customer's checkpoint: which configurations start, and how fast
the kernels are, for `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (rev `0ef577a`, W4A16 int4) on one
NVIDIA L40S (SM 8.9), vLLM 0.29.0, with the compose's fixed flags (`--max-model-len 96000`, tool
calling, reasoning parser, vision tower loaded, prefix caching at its default). It decides
`linear_backend`'s domain and checks the domain edges and the presets before any experiment
budget is spent (study README "Runbook").

- `probe.sh`: per combination, `../k8s/apply_config.sh` with the baseline (the compose's values:
  `gpu_memory_utilization` 0.90, `max_num_seqs` 64, vLLM 0.29.0 defaults otherwise) plus the
  combination's overrides, then `bench_in_pod.py` in `vllm-0`; ends with vLLM at 0 replicas.
  Combinations: `B-compose`, `K-fp8`, `L-triton`, `L-humming`,
  `E-memory` (gmu 0.94, 512 seqs, 16384 batched tokens, fp8), `E-o3` (O3, throughput, block 48,
  capture 16, gmu 0.80), `M-mtp2`, `M-mtp2-fp8`, `S30-best` (the "study 30 best" preset).
- `bench_in_pod.py`: study 30's benchmark with thinking turned off per request
  (`chat_template_kwargs`, assumed to override the server's `--default-chat-template-kwargs`; the
  first row confirms it: no reasoning tokens in its responses), so each row compares with study
  30's probe (FP8 checkpoint): prefill step (~2000-token prompt, mean of 4);
  essays (256 tokens) one at a time, 64 concurrent in English, 32 concurrent in Italian; MTP
  acceptance per phase from vLLM's counters. Client-side wall times.
- `summarize.py results/`: the table in `results/summary.txt` (rule: a linear backend enters the
  domain if it starts and is within 15 % of the best of the group, `B-compose` being `auto`).
- `results/` (written by the run): per combination `.json`, `.log` (full apply log),
  `.kernels.txt` (what vLLM reports it selected, KV cache size, and what it resolved: capture
  size, cudagraph mode, prefix caching, chunked prefill), `.params.env`,
  `.sts.yaml`.

Run with the node to itself (study 31 stopped and its load Job deleted), writing outside the
checkout, on the workstation from the study folder (`caffeinate -i` keeps macOS awake; only
kubectl is needed):
`mkdir -p /tmp/probe32 && KP_OUT=/tmp/probe32/results caffeinate -i nohup bash probe/probe.sh > /tmp/probe32/probe.log 2>&1 &`
9 starts (3 with MTP), ~50-60 min; the first one also downloads the checkpoint onto the node
(16 GiB, ~5 min).
