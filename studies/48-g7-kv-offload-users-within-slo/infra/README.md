# infra/ — study 48 (same node group as study 47)

The study adds one node group to the existing EKS cluster `vllm-bench` (us-east-2, Kubernetes
1.36.5). It does not create a cluster (`studies/0-explorative/infra/` is the reference for that).

- **`eks/gpu-nodegroup.sh`**: node group `llm-serving-g7-kv`, one g7.4xlarge (RTX PRO 4500
  Blackwell 32 GB, 16 vCPU, 64 GiB), label `node-role=llm-serving-g7-kv`, taint
  `nvidia.com/gpu=present:NoSchedule`. The GPU comes from the kube-system nvidia-device-plugin
  (its node affinity excludes only the node groups it lists; this one is not listed). Created on
  2026-11-05. `--up`, `--always-on` before a study, `--down` after it.
- No load node: the AIPerf client runs on the GPU node (4 CPU / 6 GiB, vLLM keeps 10 CPU), as in
  the spike.
- Do not use `llm-serving-g7-4500` for this study: the StatefulSet `gpu-sharing/vllm` of an old
  study selects that node group and takes the GPU when it comes up.

Teardown: `AWS_PROFILE=lab ./eks/gpu-nodegroup.sh --down`.
