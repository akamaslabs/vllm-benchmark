# infra/ — study 49 (copied from study 46)

Study 49 uses study 46's node groups on the existing EKS cluster `vllm-bench` (us-east-2,
Kubernetes 1.36.5). It does not create a cluster.

- **`eks/gpu-nodegroup.sh`:** `llm-serving-g7e-4xl`, one **g7e.4xlarge**.
  - Hardware: RTX PRO 6000 Blackwell 96 GB, 16 vCPU, 128 GiB, 4.00 USD/h.
  - Label `node-role=llm-serving-g7e-4xl`, taint `nvidia.com/gpu=present:NoSchedule`.
  - Flags: `--up --always-on` before the study, `--down` at the end. `--down` sets
    desiredSize 0 and removes the ASG AlwaysOn tag; it was added for study 49.
  - The g7e.2xlarge is not a fallback: study 46 saw 14–16 % CPU throttling there with four
    replicas.
- **`eks/load-node.sh`:** scales `m8a-2xlarge` between 0 and 1 (`up` / `down`). The AIPerf
  clients and the slice-roles exporter run on it.
- **`dra/`:** the NVIDIA DRA driver (`dra-driver-nvidia-gpu`, DynamicMIG) on the g7e node
  only, with the classic device plugin kept off it (study 46).
- **dcgm-exporter:** its node affinity must include `llm-serving-g7e-4xl`. See
  `../k8s/monitoring/dcgm-exporter-values.yaml`, as deployed for study 46.

**Before the build:**

1. Probe g7e.4xlarge capacity. This needs the user's go, because it creates a capacity
   reservation (paid). Study 46's ODCR has expired.
2. Run `gpu-nodegroup.sh --up --always-on` and `load-node.sh up`.
3. Check that the DRA driver and dcgm-exporter pods run on the node.

**Teardown:**

1. Run `gpu-nodegroup.sh --down` and `load-node.sh down`.
2. The DRA driver can stay: it only runs on the g7e node.
