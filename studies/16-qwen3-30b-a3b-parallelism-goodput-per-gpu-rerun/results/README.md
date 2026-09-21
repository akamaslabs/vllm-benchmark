# Results — study 16 (Qwen3-30B-A3B-FP8 parallelism, goodput per GPU)

Full analysis: **[report.html](report.html)** (self-contained, open from disk). Study design and
incident log: [../README.md](../README.md).

**Outcome (FINISHED 2026-09-20, stopped manually to cap cost after 27 of 100 optimizer
experiments; 43 experiments, 3 failed).** Best goodput per GPU **2 186 tok/s** (exp 37,
**TP1/DP3** on 3 of the 4 L4s, fp8 KV cache, ~490 sequences, gmu 0.877, priority + async
scheduling), **+260%** vs the TP4 baseline (607). Five DP3 configurations sit within 1.5% of it,
all scored at the ramp's last level (1 024 concurrent requests) with ITL p95 284–291 ms — the
load generator, not the server, caps them. Tuned DP3 also has the highest absolute goodput
(6 559 tok/s) vs the single untuned TP1/DP4 point (5 662) and the best TP4 (3 312). Inside DP3
the decisive lever is fp8 KV (+20%, doubles the cache to ~490 k tokens; zero preemptions in
every run with ≥ 344 k tokens of KV, 10–23/s peaks below ~285 k); 2-GPU layouts lose on KV capacity (TP2: 1.7–2.2 GiB of KV, SLA breached beyond
96–192 concurrent) rather than on the PCIe interconnect; pipeline parallelism is the worst family
(445–578 tok/s per GPU). TP1/DP2 cannot start on this node (vLLM finds 0.13 GiB left for KV),
which also corrects study 15's reading of its own experiment 6.

## Files

- `export.tar.gz` — `akamas export study` bundle (2026-09-21). **Incomplete on Akamas 3.7.x**:
  no `last-optimization.json`/`logs.json`, only 22 metric files (goal metrics missing). The
  report rebuilt the experiment table from `optimizations.json` + the CLI listing and the metric
  series from the cluster's Prometheus; the SLA-window reconstruction matches all 40 scores to
  0.00%.
- `report.html` — findings report (executive summary, per-topology results, parameter effects,
  failures, timeseries, caveats, recommendations, full experiment table).
