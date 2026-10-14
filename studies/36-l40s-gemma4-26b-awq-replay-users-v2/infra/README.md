# infra/: 36-l40s-gemma4-26b-awq-replay-users-v2

The study runs on the **vllm-bench cluster** (us-east-2, Akamas 4.1's own cluster: kubectl context
`lab-vllm-bench` from a workstation, the in-cluster `default` context on the toolbox), on the node
group and namespace of studies 30-35 on this cluster:

| What | Where |
|---|---|
| GPU node | node group `llm-serving-l40s-1xl` (one g6e.xlarge, us-east-2b, label `node-role=llm-serving-l40s-1xl`); its ASG and instance already carry `AlwaysOn=true` |
| CPU node | `system-2b` (m8a.xlarge, label `node-role=system-m8a`): the AIPerf Job |
| Namespace | `llm-l40s` (studies 30-34's): StatefulSet `vllm`, Services `vllm` / `vllm-headless`, PVCs `aiperf-results` and `hf-cache`, ServiceMonitor `vllm-l40s` in `monitoring` |
| Model cache | hostPath `/var/lib/hf-cache-study30` on the node (the AWQ checkpoint and the MTP drafter are already there) |
| Monitoring | kube-prometheus-stack (in-cluster address for Akamas and the watchdog), dcgm-exporter covering `llm-serving-l40s-1xl` |

`provision.sh` re-applies the study's objects (idempotent: they exist from study 34). The cluster
and its node groups are provisioned by studies 30's `infra/` (`../../30-l40s-gemma4-26b-tps/infra/`).
