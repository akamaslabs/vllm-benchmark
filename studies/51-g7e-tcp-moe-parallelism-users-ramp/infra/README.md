# infra/ — study 51

Study 51 adds one node group to the existing EKS cluster `vllm-bench` (us-east-2, Kubernetes
1.36). It does not create a cluster. Everything here is torn down at the end of the study.

- **`eks/nodegroups.yaml` + `eks/nodegroups.sh`:** `llm-serving-g7e-4xl-tcp`, two
  **g7e.4xlarge** on demand in one availability zone and one cluster placement group
  (`study51-g7e-4xl-tcp`).
  - Hardware per node: RTX PRO 6000 Blackwell 96 GB, 16 vCPU, 128 GiB, 50 Gbit/s, no EFA,
    4.00 USD/h. Root volume 300 GiB gp3 at 1000 MiB/s (the model is ~65 GB).
  - Created with **eksctl**, as in study 50, so the two studies differ in instance size and
    transport only. `nodegroups.sh create` creates the placement group with the AWS CLI (eksctl
    only references it). Always with `--install-nvidia-plugin=false`: on the EFA spike eksctl
    overwrote the cluster's shared NVIDIA device plugin DaemonSet
    (`docs/spikes/2026-11-10-efa-eks/README.md`).
  - `PLACEMENT=none nodegroups.sh create <az>` creates the group without a placement group, if
    the pair cannot start inside one.
  - Label `node-role=llm-serving-g7e-4xl-tcp`, taint `nvidia.com/gpu=present:NoSchedule`.
    `up` labels the two nodes `study51/role=a` and `=b`, sorted by name.
  - Commands: `probe`, `create <az>`, `up`, `down`, `delete`, `status`, `wait-up [hours]`.
    `probe` asks EC2 for the pair in each zone with a capacity reservation cancelled at once
    (it does not test the placement group). `wait-up` re-probes every 5 minutes and runs
    `create` and `up` at the first zone with room.
  - **Quota:** the lab's on-demand G quota is 64 vCPU and this pair takes 32. Study 50's
    g7e.8xlarge pair takes all 64: the two studies cannot run, or wait for capacity, at the
    same time.
- **`eks/load-node.sh`:** scales `m8a-2xlarge` between 0 and 1 (`up` / `down`). The two AIPerf
  clients run on it.
- **dcgm-exporter:** its node affinity must include `llm-serving-g7e-4xl-tcp`. See
  `../k8s/monitoring/dcgm-exporter-values.yaml` (the shared Helm release in `monitoring`: one
  more node role, nothing else changed).
- **AlwaysOn tags:** `up` tags the ASG and the instances `AlwaysOn=true`, or the lab's 17:00 UTC
  Lambda stops them. `down` removes the ASG tag.

**Before the study:**

1. `nodegroups.sh probe` (free), then `create <az>` and `up`, or `wait-up` for both. Paid from
   the moment the nodes start: it needs the user's go.
2. `load-node.sh up`.
3. `nodegroups.sh status`: each instance in the placement group, each node with
   `nvidia.com/gpu: 1`.
4. `kubectl apply -f ../k8s/00-base.yaml -f ../k8s/02-model-download.yaml` and wait for the two
   download Jobs (the model lands in `/var/lib/hf-cache-study51` on each node).
5. Check that a dcgm-exporter pod runs on each node.

**Teardown:**

1. `nodegroups.sh delete` (stack, launch template, then the placement group) and
   `load-node.sh down`.
2. `kubectl delete namespace moe-tcp`.
3. Check that no G instance is left in EC2.
4. Left on the cluster: the extra node role in the dcgm-exporter values.
