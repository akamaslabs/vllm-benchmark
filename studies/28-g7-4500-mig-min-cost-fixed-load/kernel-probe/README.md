# kernel-probe/ — study 28

Which `linear_backend` / `attention_backend` / KV dtype combinations start and how fast
they are for Qwen3-8B-FP8 on one `1g.16gb` slice of the RTX PRO 4500 (SM 12.0), vLLM
0.29.0. Decides the two kernel domains of the study (README "Kernel probe").

- `probe.sh` (toolbox): per combination, `../k8s/apply_config.sh` with one replica, then
  `bench_in_pod.py` in `vllm-0`; ends with vLLM at 0 replicas.
- `bench_in_pod.py`: prefill step (~2048-token prompt, mean of 4) and decode TPOT at 30
  concurrent short requests (the study's regime). Client-side wall times.
- `summarize.py results/`: the table in `results/summary.txt`; `within 15 %` = keep in the
  domain.
- `results/` (written by the run): per combination `.json`, `.log` (full apply log),
  `.kernels.txt` (the kernel vLLM reports it selected), `.params.env`, `.sts.yaml`.

Run it with the node to itself, writing OUTSIDE the toolbox's git checkout (the results are
committed from the workstation afterwards, and a later `git pull` on the toolbox would refuse
to overwrite untracked files at the same paths):
`mkdir -p /tmp/kp28 && KP_OUT=/tmp/kp28/results setsid nohup bash kernel-probe/probe.sh > /tmp/kp28/probe.log 2>&1 &`
(~1-2 h; the first combination also downloads the model onto the node). Then copy
`/tmp/kp28/results` and `probe.log` back into this folder (plan Task 9, Step 3).
