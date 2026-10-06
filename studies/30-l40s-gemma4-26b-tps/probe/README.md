# probe/ — study 30 startup probe

Which configurations start, and how fast the kernels are, for
`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` on one NVIDIA L40S (SM 8.9), vLLM 0.29.0.
Decides whether `attention_backend` / `linear_backend` enter the study and checks the edges of
the other domains before any experiment budget is spent (study README "Startup probe").

- `probe.sh`: per combination, `../k8s/apply_config.sh` with the baseline plus the
  combination's overrides, then `bench_in_pod.py` in `vllm-0`; ends with vLLM at 0 replicas.
- `bench_in_pod.py`: prefill step (~2000-token prompt, mean of 4) and decode TPOT at 64
  concurrent short requests. Client-side wall times.
- `summarize.py results/`: the table in `results/summary.txt`.
- `results/` (written by the run): per combination `.json`, `.log` (full apply log),
  `.kernels.txt` (what vLLM reports it selected, KV cache size), `.params.env`, `.sts.yaml`.

Run with the node to itself (no study running), writing outside the toolbox checkout:
`mkdir -p /tmp/probe30 && KP_OUT=/tmp/probe30/results setsid nohup bash probe/probe.sh > /tmp/probe30/probe.log 2>&1 &`
on the toolbox, or on the workstation (macOS has no `setsid`; `caffeinate -i` keeps it awake):
`mkdir -p /tmp/probe30 && KP_OUT=/tmp/probe30/results caffeinate -i nohup bash probe/probe.sh > /tmp/probe30/probe.log 2>&1 &`
from the study folder, on the toolbox (`/work/vllm-benchmark/studies/30-l40s-gemma4-26b-tps`)
or on the workstation: `probe.sh` points `apply_config.sh` at its own checkout, and only
kubectl is needed. ~12 starts, ~60-75
min; the first one also pulls the image and downloads the model onto the node (~10-15 min).
