# k8s/ — study 44 serving layout (multi-node P/D)

```
AIPerf (system-2b) ──► Service vllm-pd-mn:8000 ──► router pod (on the decode node)
                                                    │ prefill leg ──► vllm-pd-mn-prefill pod   node A, 1x L4
                                                    │ decode leg  ──► vllm-pd-mn-decode pod    node B, 1x L4
                                       decode pulls the KV from the prefill pod IP:
                                       NIXL side channel :5600, data over UCX/TCP across the VPC
```

Why not study 27's files: there every vLLM process ran in ONE container (one PID namespace
for UCX cuda_ipc) and the router called 127.0.0.1. Here each role is its own Deployment, with
anti-affinity so prefill and decode never share a node, and the router reads its backends
from `PD_PREFILL_URLS` / `PD_DECODE_URLS`.

| File | Role |
|---|---|
| `20-pd-config.yaml` | ConfigMap `pd-mn-config`: study 27 exp 8 flags per role (P1D1 kv fp8) + the aggregated control's |
| `serve.sh` | Entrypoint: one `vllm serve` per pod; `VLLM_NIXL_SIDE_CHANNEL_HOST` = pod IP (the prefill returns it as `remote_host`, vllm v0.29.0 `nixl/pull_scheduler.py`) |
| `pd_router.py` | Study 27's router; backends from env, GPU index 0 per pod |
| `21-serving_template.yaml` | P1D1: prefill, decode, router Deployments (rendered by `run.sh`) |
| `22-aggregated_template.yaml` | P0D1 control: one aggregated vLLM + router on one node |
| `02-service.yaml`, `monitoring/servicemonitor.yaml` | Services and scrape, label `app=vllm-pd-mn` |
| `10-iperf3_template.yaml` | iperf3 server pod + client Job on the two nodes |
| `05-job_template.yaml` | Study 27's AIPerf job, pointed at `vllm-pd-mn`, Job `aiperf-pd-mn` |
| `score.py` | Study 27's goal recomputed from Prometheus for a time range |
| `run.sh` | Driver: `prep`, `iperf`, `up`, `smoke`, `ramp`, `score`, `logs`, `down` |
| `tuned-configs/` | Study 27's tuned Triton FP8 configs for L4 (prefill on triton) |

Model weights: emptyDir per pod (the RWO EBS model-cache volume attaches to one node only),
so each start downloads 8.8 GiB.

Checked offline (2026-10-29): `kubectl apply --dry-run=server` accepts every rendered
template for both node groups, the Services, the ConfigMap, the ServiceMonitor, the iperf3
pair and the AIPerf job; `bash -n` on the scripts; `score.py` runs against the live
Prometheus.
