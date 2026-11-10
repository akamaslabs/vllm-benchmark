# infra/ — study 50

Study 50 adds one node group to the existing EKS cluster `vllm-bench` (us-east-2, Kubernetes
1.36). It does not create a cluster. Everything here is torn down at the end of the study.

- **`eks/nodegroups.yaml` + `eks/nodegroups.sh`:** `llm-serving-g7e-8xl-efa`, two
  **g7e.8xlarge** on demand in one availability zone and one cluster placement group.
  - Hardware per node: RTX PRO 6000 Blackwell 96 GB, 32 vCPU, 256 GiB, 100 Gbit/s, EFA,
    5.27 USD/h. Root volume 300 GiB gp3 at 1000 MiB/s (the model is ~65 GB).
  - Created with **eksctl** (`efaEnabled: true`), unlike the AWS-CLI scripts of the other
    studies: eksctl builds the launch template with the EFA interface, the placement group and
    the security-group rules. Always with `--install-nvidia-plugin=false`: on the spike eksctl
    overwrote the cluster's shared NVIDIA device plugin DaemonSet
    (`docs/spikes/2026-11-10-efa-eks/README.md`).
  - Label `node-role=llm-serving-g7e-8xl-efa`, taint `nvidia.com/gpu=present:NoSchedule`.
    `up` labels the two nodes `study50/role=a` and `=b`, sorted by name.
  - Commands: `probe`, `create <az>`, `up`, `down`, `delete`, `status`, `wait-up [hours]`.
    `probe` asks EC2 for the pair in each zone with a capacity reservation cancelled at once.
    `wait-up` re-probes every 5 minutes and runs `create` and `up` at the first zone with room.
  - **Quota:** the lab's on-demand G quota is 64 vCPU, exactly this pair. No other G instance
    can run next to it.
- **`efa/efa-device-plugin.yaml`:** the study's own copy of the AWS EFA device plugin
  (v0.5.4), in namespace `moe-efa`, selected by node role. The DaemonSet eksctl installs in
  `kube-system` lists 121 instance types without any g7e and does not tolerate the GPU taint.
  Applied by `nodegroups.sh create`.
- **`eks/load-node.sh`:** scales `m8a-2xlarge` between 0 and 1 (`up` / `down`). The two AIPerf
  clients run on it.
- **dcgm-exporter:** its node affinity must include `llm-serving-g7e-8xl-efa`. See
  `../k8s/monitoring/dcgm-exporter-values.yaml`, applied on 2026-11-10 as Helm revision 27 of the
  shared release in `monitoring` (one more node role, nothing else changed).
- **AlwaysOn tags:** `up` tags the ASG and the instances `AlwaysOn=true`, or the lab's 17:00 UTC
  Lambda stops them. `down` removes the ASG tag.

**Before the study:**

1. `nodegroups.sh probe` (free), then `create <az>` and `up`, or `wait-up` for both. Paid from
   the moment the nodes start: it needs the user's go.
2. `load-node.sh up`.
3. `nodegroups.sh status`: each node must show `nvidia.com/gpu: 1`, `vpc.amazonaws.com/efa: 1`
   and at least 5120 Mi of `hugepages-2Mi`.
4. `kubectl apply -f ../k8s/00-base.yaml -f ../k8s/02-model-download.yaml` and wait for the two
   download Jobs (the model lands in `/var/lib/hf-cache-study50` on each node).
5. Check that a dcgm-exporter pod runs on each node.

**Teardown:**

1. `nodegroups.sh delete` (stack, launch template, placement group) and `load-node.sh down`.
2. `kubectl delete namespace moe-efa`.
3. Check that no G instance is left in EC2.
4. Left on the cluster: `kube-system/aws-efa-k8s-device-plugin`, created by eksctl on the spike
   (0 pods without matching nodes), and the extra node role in the dcgm-exporter values.
