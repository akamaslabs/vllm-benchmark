# infra/: 38-l40s-gemma4-26b-awq-replay-users-v3

The study runs on the **existing `vllm-bench-gatling` cluster** (us-east-2, AWS profile `lab`,
kubectl context `lab-vllm-bench-gatling` from a workstation, `gatling` from the Akamas 4.1
toolbox), freed by the end of Gatling study 2 on 2026-10-13. The cluster, its node groups and its
monitoring stack are defined in `akamas-gatling-llms-optimization` (`infra/eks/cluster.yaml`), not
here: this folder only adds the study's layer.

| What | Where |
|---|---|
| GPU node | node group `llm-serving-l40s` (g6e.xlarge, us-east-2c) or its twin `llm-serving-l40s-ab` (us-east-2a/2b), both labelled `node-role=llm-serving`, `gpu=l40s`, taint `nvidia.com/gpu`; min 0. The StatefulSet selects `gpu: l40s`, so either works. |
| CPU node | node group `system` (m6i.2xlarge, 8 vCPU / 32 GiB, us-east-2c, label `node-role=system`): Prometheus, Grafana and the AIPerf Job |
| Monitoring | kube-prometheus-stack (namespace `monitoring`, ServiceMonitors from every namespace), dcgm-exporter on `node-role=llm-serving`; Prometheus reachable by Akamas through the NLB `a2e152afd47944cd6a9e3b27dcc70180-de2ee87c41d70dfd.elb.us-east-2.amazonaws.com:9090` (Gatling study 2's telemetry) |
| Study layer | `provision.sh`: namespace `llm-replay`, PVCs `aiperf-results` (gp3) and `hf-cache` (gp3-ephemeral), Services `vllm` / `vllm-headless`, ServiceMonitor `vllm-replay` |

## Runbook

```bash
AWS_PROFILE=lab eksctl scale nodegroup --cluster vllm-bench-gatling --region us-east-2 \
  --name llm-serving-l40s --nodes 1 --nodes-min 0          # or llm-serving-l40s-ab
KUBE_CONTEXT=lab-vllm-bench-gatling bash infra/provision.sh
```

- **Capacity.** g6e is often short in us-east-2 (2026-10-13: every size, every zone, for ~10
  minutes; the `-ab` group got a node first). `InsufficientInstanceCapacity` shows in the ASG's
  scaling activities; scale the other group instead.
- **Nightly stop.** The account's Lambda stops every instance without the tag `AlwaysOn=true` at
  17:00 UTC, and the ASG replaces it: tag the GPU node group's ASG (PropagateAtLaunch) and the
  `system` ASG and their instances before a run that crosses 17:00 UTC; untag at the end.
- **Caches.** vLLM's model cache is a hostPath (`/var/lib/hf-cache-study35`): a new node
  downloads the checkpoint again (~16 GiB, ~4 min).
- **At the end:** node groups to 0, `AlwaysOn` tags removed.
