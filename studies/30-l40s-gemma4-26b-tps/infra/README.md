# infra/ — 30-l40s-gemma4-26b-tps

Takes the lab AWS account to a cluster ready for this study: the shared `vllm-bench` EKS
cluster (1.35, us-east-2), one GPU node group with a single NVIDIA L40S, namespace
`llm-l40s`, the study's PVCs, Services and ServiceMonitor. Every script is idempotent.

## Layout

| Path | What |
|---|---|
| `eks/cluster.yaml` | eksctl config: cluster + system (`system-2b`, AIPerf/Prometheus) and `akamas` node groups. **No GPU node group** — see below. |
| `eks/gpu-nodegroup.sh` | Creates/reconciles `llm-serving-l40s-1xl` (g6e.xlarge; `NG` / `INSTANCE_TYPE` override it) with `aws eks create-nodegroup --ami-type AL2023_x86_64_NVIDIA`, taint `nvidia.com/gpu`. `--up` scales it to 1 and waits for the node, `--always-on` tags the ASG and the instance for the 17:00 UTC stop Lambda. |
| `eks/provision.sh` | Runs everything below in order. |
| `eks/storageclass.yaml`, `k8s-bootstrap/` | Default StorageClasses; namespaces `llm-l40s`, `monitoring`. |

## The node group

- **`llm-serving-l40s-1xl`** (the study's node): one **g6e.xlarge** (1x L40S 48 GB, 4 vCPU /
  32 GiB, 1.861 USD/h), created 2026-10-06 ~07:20 UTC with `gpu-nodegroup.sh --up`, release
  `1.35.8-20260930`, subnets 2a/2b/2c, label `node-role=llm-serving-l40s-1xl`, disk 200 GB.
- **Why xlarge:** capacity-reservation probes (create a reservation, cancel it at once; study
  17's `gpu-capacity-fallback.sh probe`). 2026-10-05: only g6e.8xlarge had capacity in
  us-east-2, so `llm-serving-l40s-8xl` (32 vCPU / 256 GiB, 4.53 USD/h) was created at
  desiredSize 0. 2026-10-06 07:18 UTC: g6e.8xlarge empty everywhere, only g6e.xlarge had
  capacity (2a/2b/2c); the user chose to run on it. `llm-serving-l40s-8xl` stays at 0.
- The older `llm-serving-l40s` node group (g6e.4xlarge, from study 17's fallback script)
  is left as it is: the instance type of a managed node group is immutable, so every size
  has its own node group and label (`NG=... INSTANCE_TYPE=... ./eks/gpu-nodegroup.sh`).
- eksctl is not used for GPU node groups on this cluster (study 17/28: it picked a
  driver-less AMI for the g7 families); the EKS-managed NVIDIA AMI covers g6e.
- One taint only: the cluster-wide NVIDIA device plugin (kube-system, static v0.18.0,
  tolerates `nvidia.com/gpu`) serves the node. No GPU sharing in this study.

## Prerequisites

`aws` (profile `lab`, account 916205288457), `eksctl`, `kubectl`, `helm`. The workstation's
default AWS profile points at another account — always pass `--profile lab` /
`AWS_PROFILE=lab`.

## Usage

```bash
AWS_PROFILE=lab ./eks/gpu-nodegroup.sh --up --always-on   # node to 1 + AlwaysOn tags (llm-serving-l40s-1xl)
AWS_PROFILE=lab ./eks/provision.sh
```

If `--up` hangs on "waiting for a Ready node", read
`aws eks describe-nodegroup --cluster-name vllm-bench --region us-east-2 --nodegroup-name
llm-serving-l40s-1xl --query nodegroup.health.issues`: an `InsufficientInstanceCapacity`
means the pool emptied overnight (probe the other sizes with study 17's script).

## Teardown

- Pause GPU billing: `aws eks update-nodegroup-config --cluster-name vllm-bench --region
  us-east-2 --nodegroup-name llm-serving-l40s-1xl --scaling-config
  minSize=0,maxSize=1,desiredSize=0`, and remove the ASG's `AlwaysOn` tag so a forgotten
  scale-up is still stopped at night.
- Remove this study's layer: `kubectl delete namespace llm-l40s; kubectl -n monitoring
  delete servicemonitor vllm-l40s` (the `aiperf-results` volume has reclaim policy Retain).
- Delete the node group: `aws eks delete-nodegroup --cluster-name vllm-bench --region
  us-east-2 --nodegroup-name llm-serving-l40s-1xl` (and the unused `llm-serving-l40s-8xl`). The cluster is shared; do not delete it
  from here.

## What this does NOT cover

- The Akamas platform and the `toolbox` host its workflows SSH into (namespace `akamas`).
- kube-prometheus-stack installation (`k8s/monitoring/values-kube-prometheus.yaml`).
- The shared `dcgm-exporter` release: `k8s/monitoring/dcgm-exporter-values.yaml` adds this
  node role to it, applied by hand (study README, "Morning runbook").
- The optimization packs (vLLM 1.12.0, GPU, Kubernetes), managed outside this repo.
