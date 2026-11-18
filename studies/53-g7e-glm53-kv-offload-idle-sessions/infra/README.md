# infra/ — study 53's node (the same node group as study 52, copied: the two studies run one at a time on it)

Cluster `vllm-bench` (EKS, us-east-2) already exists; this study adds one GPU node group.

- `eks/nodegroup-g7e-24xl-spot.yaml`: eksctl managed node group `llm-serving-g7e-24xl-spot`, one
  g7e.24xlarge **spot** in us-east-2c, gp3 1200 GB, label `node-role=llm-serving-g7e-24xl-spot`,
  taint `nvidia.com/gpu`, created at desired 0.
- `eks/nodegroups.sh`: `quota`, `probe`, `create`, `up`, `down`, `delete`, `status`
  (`AWS_PROFILE=lab`).
- The spot G vCPU quota in us-east-2 must be >= 96 (`./nodegroups.sh quota`); `up` refuses
  otherwise.
- dcgm-exporter: `../k8s/monitoring/dcgm-exporter-values.yaml` (Helm revision 30 already covers the
  node role):
  `helm -n monitoring upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter --version 4.8.3 -f k8s/monitoring/dcgm-exporter-values.yaml`.

Bring-up: `./nodegroups.sh create` (once), `./nodegroups.sh up`; at the end `./nodegroups.sh down`.
The node is tagged AlwaysOn while up (17:00 UTC stop Lambda).
