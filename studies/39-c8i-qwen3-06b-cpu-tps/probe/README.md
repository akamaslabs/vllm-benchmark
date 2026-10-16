# probe/ — study 39 calibration probe

What the vLLM CPU pod actually gets on the c8i.12xlarge node, whether the edges of the study's
domains start, and where the baseline's SLO knee is, before any experiment budget is spent.
Its proposal (`RT_RATE`, `RT_RAMP_S`) is the starting point of the Smoke study, which fixes the
main workflow's values.

- `probe.sh`: phase A, per combination, `../k8s/apply_config.sh` with the baseline (vLLM 0.31.0
  CPU defaults) plus the combination's overrides, then `node_facts.sh` inside `vllm-0` and the
  vLLM lines the decisions rest on; phase B, the baseline again, then one AIPerf Job
  (`sweep_job_template.yaml`, rendered by `render_sweep_job.sh` with the study's own workload
  from `../k8s/lib_workload.sh`) at constant rates `KP_RATES` (default `1 2 4 6 8 12 16 24`),
  `KP_STEP_S` (240 s) each, stopping once a step passes 2x the SLO. Ends with vLLM at 0 replicas.
- `node_facts.sh`: `lscpu`, SMT siblings, NUMA (numactl if present, else `/sys`), AMX/AVX-512
  flags, the container's cpuset, cgroup `memory.max` / `memory.current` / `memory.peak`,
  per-process RSS, the vLLM CPU env.
- `summarize.py`: `--parse` (one startup -> JSON: KV tokens and GiB, engine non-KV RSS,
  OMP threads / reserved, NUMA nodes, AMX, memory peak, LD_PRELOAD / mempolicy warnings),
  `--sweep` (AIPerf p95s per step plus vLLM's 150 s p95s, prefix-cache hit rate, KV usage,
  preemptions from Prometheus), and the summary: both tables, the `gpu_memory_utilization`
  lower bound implied by the measured RSS, the knee (highest rate meeting TTFT p95 <= 3000 ms
  and TPOT p95 <= 100 ms) and `RT_RATE = 2 x knee`, `RT_RAMP_S = 3000`.
- `results/` (written by the run, `KP_OUT`): per combination `.json`, `.log` (full apply log),
  `.vllm.txt`, `.facts.txt`, `.params.env`, `.sts.yaml`; `sweep-job.log`, `sweep.jsonl`,
  `summary.txt`.
- `tests/`: `test_summarize.sh` and `test_probe_stub.sh` (stub kubectl, no cluster);
  `dry_run_sweep.sh` runs the sweep Job's own shell against `../k8s/tests/mock_openai.py` with
  a local AIPerf 0.11.0 (network: pip, tokenizer).

Phase A combinations: `B-default` (the study's baseline: 0.90 / 128 / 2048 / 128 / auto /
prefix on / chunked on / 0 / 1), `E-gmu-high` (0.92, vLLM's default: whether the domain top
could extend to it), `E-gmu-low` (0.15 with 256 seqs and 8192 batched tokens: the
largest RSS at the smallest budget), `E-gmu-low-fp8`, `K-fp8`, `S-block32`, `S-block96`,
`E-corner` (256 / 8192 / long-prefill 4096 / prefix off), `C-nochunk` (must be
refused by `render_statefulset.sh`, see `../k8s/README.md` "Known conflicts").

Run with the node to itself (no study running), writing outside the toolbox checkout:
`mkdir -p /tmp/probe39 && KP_OUT=/tmp/probe39/results setsid nohup bash probe/probe.sh > /tmp/probe39/probe.log 2>&1 &`
from the study folder on the toolbox (`/work/vllm-benchmark/studies/39-c8i-qwen3-06b-cpu-tps`),
or on the workstation with `caffeinate -i` instead of `setsid` and `KP_PROM` pointing at a
Prometheus port-forward. 10 starts plus the sweep: roughly 2-3 h (CPU compile at each start;
the first one also pulls the image and downloads the model).
