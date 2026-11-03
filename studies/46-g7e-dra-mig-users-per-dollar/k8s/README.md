# k8s/ — study 46

To be written:

- **vLLM StatefulSet template:** N replicas, one `ResourceClaimTemplate` per MIG profile
  (`mig.nvidia.com` DeviceClass with a CEL selector on `profile`, or `gpu.nvidia.com` for
  the whole GPU), CPU = ⌊allocatable / N⌋.
- **AIPerf Job template:** one Job per replica, scheduled on `m8a-2xlarge`.
- **`apply_config.sh` / `run_test.sh`:** same shape as study 26's, without the
  device-plugin steps.
- **Score computation:** N × min over replicas of the highest concurrency level that
  meets the SLA.
- **dcgm-exporter affinity patch:** add the new node.

Use a new namespace for this study.
