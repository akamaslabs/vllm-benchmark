# infra/ — study 46

This study adds node groups to the existing EKS cluster `vllm-bench` (us-east-2,
Kubernetes 1.36.5, upgraded on 2026-11-02/03), as studies 35 and later do. It does not
create a cluster. `studies/0-explorative/infra/` is the reference for building one from
scratch.

To be written before Phase 0:

- **`gpu-nodegroup.sh`**: probe g7e capacity first
  (`studies/17-g7e-speculative-decoding-goodput/infra/eks/gpu-capacity-fallback.sh probe
  g7e.2xlarge`). Then create `llm-serving-g7e-2xl` with `aws eks create-nodegroup`:
  - `g7e.2xlarge`, `AL2023_x86_64_NVIDIA` for 1.36;
  - subnets 2a/2b/2c;
  - label `node-role=llm-serving-g7e-2xl`, taint `nvidia.com/gpu=present:NoSchedule`;
  - min 0 / max 1.

  Flags: `--up` / `--down`, and `--always-on`, which tags the node and the ASG with
  `AlwaysOn=true`.
- **`load-node.sh`**: scale `m8a-2xlarge` between 0 and 1, and tag it.
- **`dra/`**: Helm values for `dra-driver-nvidia-gpu` (`featureGates.DynamicMIG=true`,
  `gpuResourcesEnabledOverride=true`, nodeSelector on the g7e node only), and patches
  that keep the classic device plugin (`kube-system` DaemonSet, `nvdp-g7`) off this
  node.

**Teardown:** scale both node groups to 0. Then uninstall the DRA driver release, or keep
it, since it only runs on the g7e node.
