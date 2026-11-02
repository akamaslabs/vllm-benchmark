# infra/ — study 44's GPU nodes

Study 44 runs on the existing `vllm-bench` cluster (us-east-2), as studies 18-43 did: the
cluster itself, the system node (`system-2b`, Prometheus/Grafana, AIPerf), the `akamas` node
and the NVIDIA device plugin / dcgm-exporter are described in
`../../19-l4-pd-disaggregation-tps-per-gpu-rerun/infra/` and are not repeated here. What
this study adds is two node groups, created by `nodegroups.sh`:

| Node group | Instance | Per node | Network (baseline / burst) | $/h on-demand |
|---|---|---|---|---|
| `llm-serving-l4-pd` | g6.4xlarge | 1x L4, 16 vCPU, 64 GiB | 10 / 25 Gbps | 1.3232 |
| `llm-serving-l4-pd-xl` | g6.xlarge | 1x L4, 4 vCPU, 16 GiB | 2.5 / 10 Gbps | 0.8048 |

Both: min 0 / max 2, AL2023 NVIDIA AMI, 200 GB disk, taint `nvidia.com/gpu=present:NoSchedule`,
labels `node-role=<group>` and `study=44`, **one subnet: public us-east-2b**
(`subnet-0cfc3532e8268ea75`). Same AZ for both roles because inter-AZ traffic is billed in
both directions (~$0.02/GB) and the KV stream would cost more than the GPUs; 2b because the
system node, Prometheus and AIPerf are there. Capacity for 2 instances of each type was
probed free in 2a, 2b and 2c on 2026-10-29 (capacity reservation created and cancelled at
once, as in `../../17-g7e-speculative-decoding-goodput/infra/eks/gpu-capacity-fallback.sh`).
No instance here has EFA: the KV goes over TCP.

## Usage

```bash
export AWS_PROFILE=lab
./nodegroups.sh create        # both groups at 0 nodes (idempotent)
./nodegroups.sh up 4xl        # 2 nodes, waits Ready, tags ASG + instances AlwaysOn=true
./nodegroups.sh down 4xl      # pause GPU billing
./nodegroups.sh down all
./nodegroups.sh delete        # full teardown of this study's groups
./nodegroups.sh status
```

The nightly Lambda stops untagged instances at 17:00 UTC; `up` tags them. G-family quota in
us-east-2 is 64 vCPU: keep one group up at a time.

## Not covered

The cluster, the system/akamas node groups, the monitoring stack, the Akamas platform and
its toolbox (tappa A uses none of them except Prometheus).
