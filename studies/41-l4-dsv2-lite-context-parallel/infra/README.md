# infra/: 41-l4-dsv2-lite-context-parallel

This study runs on the existing `vllm-bench` EKS cluster (1.35, us-east-2) and its existing
GPU node group. Nothing here creates a cluster or a node group. `provision.sh` checks what is
shared, scales the node group if asked, and creates this study's Kubernetes objects **only if
they are missing** (`kubectl create`, existing objects are left untouched):

- PVCs `aiperf-results-dsv2` (gp3) and `aiperf-hf-cache-dsv2` (gp3-ephemeral) for the AIPerf Job;
- Services `vllm-dsv2` / `vllm-dsv2-headless`, ServiceMonitor `vllm-dsv2`;
- the model cache PVC `vllm-model-cache-dsv2lite` is declared too, but it **already exists**
  (created by the smoke run on 2026-10-22, 60Gi gp3-ephemeral, bound to an EBS volume in
  **us-east-2a**, DeepSeek-V2-Lite-Chat already downloaded) and is never modified.

## The node group (shared, already exists)

| Setting | Value (read 2026-10-22) |
|---|---|
| Name | `llm-serving-l4` (eksctl, launch template `eksctl-vllm-bench-nodegroup-llm-serving-l4`) |
| Instance | g6.12xlarge: 4x NVIDIA L4 24 GB (PCIe, no NVLink / P2P), 48 vCPU, 192 GiB; On-Demand |
| AMI | `AL2023_x86_64_NVIDIA` |
| Scaling | min 0 / max 1; desired 1 since 2026-10-22 09:06 UTC (node `ip-192-168-64-57`, us-east-2a) |
| Label / taint | `node-role=llm-serving-l4` / `nvidia.com/gpu=present:NoSchedule` |
| Subnets | public subnets in us-east-2a, 2b and 2c (the ASG spans all three) |
| Nightly stop | ASG tagged `AlwaysOn=true` (propagated at launch), so the 17:00 UTC Lambda skips it |
| GPU stack | NVIDIA device plugin and dcgm-exporter DaemonSets already run on the node |

### Scaling 0 / 1

```bash
# Up (waits for the node, checks its AZ, then provisions what is missing):
AWS_PROFILE=lab ./provision.sh --scale 1
# Down (pause billing; nothing else is touched):
AWS_PROFILE=lab ./provision.sh --scale 0
# Same thing by hand:
aws eks update-nodegroup-config --cluster-name vllm-bench --region us-east-2 \
  --nodegroup-name llm-serving-l4 --scaling-config minSize=0,maxSize=1,desiredSize=1   # or 0
```

### Wrong AZ

The model PVC is an EBS volume in **us-east-2a**, but the ASG may start the node in 2b or 2c. A
node there can never mount it: the vLLM pod stays Pending with a volume node affinity
conflict. `provision.sh --scale 1` and `k8s/apply_config.sh` both refuse such a node (exit 1 /
exit 3). Remedies, in order of preference:

1. Replace the instance until it lands in 2a (the ASG keeps desired 1):
   ```bash
   aws ec2 describe-instances --region us-east-2 --filters Name=private-dns-name,Values=<node name> \
     --query 'Reservations[].Instances[].InstanceId' --output text
   aws autoscaling terminate-instance-in-auto-scaling-group --region us-east-2 \
     --instance-id <id> --no-should-decrement-desired-capacity
   ```
2. Restrict the ASG to the 2a subnet for the duration of the study (`aws autoscaling
   update-auto-scaling-group --vpc-zone-identifier subnet-04db61365d14dd296`), and restore all
   three afterwards. The node group is shared: note it in the study README if done.
3. Last resort: delete the PVC and let the next start download the model again in the node's
   AZ (~31 GB; the startup probe allows 40 min).

## Prerequisites

- Tools: `aws` (profile `lab`, account 916205288457), `kubectl`.
- The workstation's default AWS profile points at another account: always `AWS_PROFILE=lab`
  (the script defaults to it).
- Quota: g6.12xlarge (48 vCPU) counts against the G-instance quota (64 vCPU in us-east-2):
  no other G node may run at the same time beyond 16 vCPU.

## Teardown

- **Pause billing:** `AWS_PROFILE=lab ./provision.sh --scale 0`.
- **Remove this study's Kubernetes layer** (other studies use `llm-serving`: never delete the
  namespace):
  ```bash
  kubectl -n llm-serving delete sts vllm-dsv2
  kubectl -n llm-serving delete job -l app=aiperf-dsv2
  kubectl -n llm-serving delete svc vllm-dsv2 vllm-dsv2-headless
  kubectl -n llm-serving delete pvc aiperf-results-dsv2 aiperf-hf-cache-dsv2
  kubectl -n monitoring delete servicemonitor vllm-dsv2
  ``` `aiperf-results-dsv2` uses `gp3` (reclaim policy Retain):
  delete its PV and EBS volume by hand. Delete `vllm-model-cache-dsv2lite` only when nobody needs
  the downloaded model any more.
- The node group and the cluster are shared: never delete them from here.

## What this does NOT cover

- The cluster, the node groups, the StorageClasses, namespace `llm-serving`, the NVIDIA device
  plugin, dcgm-exporter and kube-prometheus-stack (shared, created by earlier studies).
- The Akamas 4.1 platform and its toolbox (namespace `akamas-41`).
- The optimization packs (vLLM 1.14.0, Kubernetes 1.9.0, GPU 1.4.0), managed outside this repo.
