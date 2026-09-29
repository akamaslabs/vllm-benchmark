# infra/ — 25-g7-4500-gpu-sharing-goodput

Takes the lab AWS account to a cluster ready for this study: the `vllm-bench` EKS
cluster (1.35, us-east-2), one GPU node group with a single RTX PRO 4500 Blackwell, and
the GPU sharing layer that lets each experiment switch between exclusive, MIG,
time-slicing and MPS.

## Layout

| Path | What |
|---|---|
| `eks/cluster.yaml` | eksctl config: cluster + system (`system-2b`, AIPerf/Prometheus) and `akamas` node groups. **No GPU node group** — see below. |
| `eks/gpu-nodegroup.sh` | Creates/reconciles `llm-serving-g7-4500` (g7.4xlarge) with `aws eks create-nodegroup --ami-type AL2023_x86_64_NVIDIA` and its two taints. `--always-on` tags the ASG for the 17:00 UTC stop Lambda. |
| `eks/provision.sh` | Runs everything below in order; idempotent. |
| `eks/storageclass.yaml`, `k8s-bootstrap/` | Default StorageClasses; namespaces `gpu-sharing`, `monitoring`. |
| `gpu-sharing/nvdp-values.yaml` | This study's own NVIDIA device plugin (chart 0.18.0, release `nvdp-g7`), 4 named configs = the 4 sharing modes. |
| `gpu-sharing/gpu-admin.yaml` | Privileged hostPID DaemonSet used by `k8s/apply_config.sh` to run host `nvidia-smi` (MIG on/off, slices, compute mode). |
| `gpu-sharing/install.sh` | Installs both, after removing the cluster-wide plugin's pod from this node (one-time). |

## Why the GPU node group is not in cluster.yaml

eksctl does not recognise the g7 family as a GPU instance and would pick the standard,
driver-less AL2023 AMI (the same trap study 17 hit with g7e). The EKS-managed
`AL2023_x86_64_NVIDIA` AMI supports g7 since release v20260917 (it ships driver 595 and
selects it by instance type); on 2026-09-29 the node came up at release
`1.35.8-20260923` with driver 595.91.07 and `nvidia-smi` working out of the box.

## Why two taints

`nvidia.com/gpu=present:NoSchedule` is the usual one. `akamas.io/gpu-sharing=managed:
NoSchedule` exists to keep the **cluster-wide** device plugin (kube-system, a static
manifest with no config) off this node without editing it: it tolerates only
`nvidia.com/gpu`. Two plugins cannot both register `nvidia.com/gpu` on one node, and the
cluster-wide one cannot switch sharing modes. Checked on 2026-09-29: `aws-node`,
`kube-proxy`, `ebs-csi-node` and `node-exporter` all tolerate any `NoSchedule` taint and
keep running on the node; `dcgm-exporter` does not, hence the extra toleration in
`k8s/monitoring/dcgm-exporter-values.yaml`.

## Prerequisites

`aws` (profile `lab`, account 916205288457), `eksctl`, `kubectl`, `helm`. The lab's
default AWS profile points at another account — always pass `--profile lab` /
`AWS_PROFILE=lab`.

## Usage

```bash
AWS_PROFILE=lab ./eks/provision.sh
AWS_PROFILE=lab ./eks/gpu-nodegroup.sh --always-on    # before runs past 17:00 UTC
```

## Teardown

- Pause GPU billing: `aws eks update-nodegroup-config --cluster-name vllm-bench --region
  us-east-2 --nodegroup-name llm-serving-g7-4500 --scaling-config
  minSize=0,maxSize=1,desiredSize=0`, and remove the ASG's `AlwaysOn` tag so a forgotten
  scale-up is still stopped at night.
- Remove this study's layer: `helm -n gpu-sharing uninstall nvdp-g7; kubectl delete
  namespace gpu-sharing; kubectl -n monitoring delete servicemonitor vllm-gpu-sharing`.
- Delete the node group: `aws eks delete-nodegroup --cluster-name vllm-bench --region
  us-east-2 --nodegroup-name llm-serving-g7-4500`. The cluster itself is shared by every
  study; do not delete it from here.

## What this does NOT cover

- The Akamas platform and the `toolbox` host its workflows SSH into (namespace `akamas`).
- kube-prometheus-stack installation (`k8s/monitoring/values-kube-prometheus.yaml`).
- Re-pointing the **shared** `dcgm-exporter` release at this node group — a deliberate,
  manual step because it takes GPU telemetry away from whichever study uses it now
  (study README, "Before starting").
- Installing the GPU optimization pack 1.3.0 (`GPU.sharing_mode`), managed outside this
  repo.
