# infra/: 39-c8i-qwen3-06b-cpu-tps

These scripts prepare the existing `vllm-bench` EKS cluster (1.35, us-east-2) for this study. They do not create a new cluster. They add:

- one CPU node group with a single c8i.12xlarge;
- namespace `llm-cpu`;
- the study's PVCs, Services and ServiceMonitor.

Every script is idempotent.

## Layout

| Path | What it does |
|---|---|
| `eks/cpu-nodegroup.sh` | Creates or reconciles node group `llm-serving-c8i` and its launch template. `--up` scales it to 1, waits for the node, then checks the live kubelet config and the CPU topology. `--always-on` tags the ASG and the instance so the 17:00 UTC stop Lambda skips them. `--down` scales to 0 and removes the tag. |
| `eks/nodeconfig.yaml` | The nodeadm `NodeConfig` that holds the kubelet settings: static CPU manager, `single-numa-node` topology manager, `systemReserved` cpu 1810m (2 reserved CPUs with EKS's kubeReserved). The script wraps it into the launch template's user data. |
| `provision.sh` | Checks the shared cluster and its StorageClasses, then applies the namespace, the node group (via the script above) and `../k8s/` (`00-pvc.yaml`, `02-service.yaml`, optionally `06-hf-cache-pvc.yaml`, and `monitoring/servicemonitor.yaml`). |
| `k8s-bootstrap/00-namespaces.yaml` | Namespace `llm-cpu`. |

The cluster itself, its `system-2b` (label `node-role=system-m8a`, where the AIPerf Job and Prometheus run) and `akamas` node groups, the `gp3` / `gp3-ephemeral` StorageClasses and kube-prometheus-stack are shared. They are defined in `studies/30-l40s-gemma4-26b-tps/infra/`. This study only checks that they are there.

## The node group

| Setting | Value |
|---|---|
| Name | `llm-serving-c8i` |
| Instance | one c8i.12xlarge: Xeon 6 Granite Rapids with AMX, 48 vCPU = 24 cores x 2 threads, 96 GiB; **Spot** (decided 2026-10-16): 0.68-0.81 USD/h in us-east-2 on that day against 2.25 USD/h On-Demand |
| Scaling | min 0 / desired 0 / max 1 |
| AMI | AMI type `AL2023_x86_64_STANDARD` |
| Label | `node-role=llm-serving-cpu` (it differs from the node-group name, per the study contract) |
| Taint | `workload=cpu-inference:NoSchedule` |
| Disk | 100 GB gp3, encrypted, set in the launch template |
| Capacity | `SPOT` (`CAPACITY_TYPE=ON_DEMAND` to switch; a managed node group's capacity type is fixed at creation, so switching means deleting the node group) |
| Subnets | the private us-east-2b and us-east-2c subnets (`eksctl-vllm-bench-cluster/SubnetPrivateUSEAST2B` / `...2C`), egress via the cluster's NAT; override with `SUBNET_TAGS` |

**Why spot, and what an interruption costs.** On 2026-10-16 the spot placement score for one c8i.12xlarge was 3/10 in 2b and 2c and 1/10 in 2a (`aws ec2 get-spot-placement-scores`), so interruptions are expected now and then over a multi-day study. With 2 minutes' notice the node goes away: the running trial fails (it counts against `maxFailedExperiments` 20; a failed **baseline** puts the study in ERROR, delete and recreate it). The ASG keeps desiredSize 1 and starts a replacement node; `k8s/apply_config.sh` waits up to `UNSCHED_S` (900 s) for it before failing the next trial. If interruptions become frequent, recreate the node group On-Demand.

**Why two AZs.** Two spot capacity pools. The vLLM pod uses hostPath caches only, so no EBS volume ties it to an AZ (the PVCs in `k8s/00-pvc.yaml` / `06-hf-cache-pvc.yaml` belong to the AIPerf Job on the system node).

The instance type of a managed node group cannot be changed. Another size gets its own node group: `NG=llm-serving-c8i-8xl INSTANCE_TYPE=c8i.8xlarge ./eks/cpu-nodegroup.sh`. The label stays the same, so do not run two such node groups at the same time.

**Taint tolerations.** These DaemonSets tolerate the taint and still run on the node: aws-node, kube-proxy, ebs-csi-node and the kube-prometheus-stack node-exporter (which tolerates `NoSchedule`). The vLLM pod carries the matching toleration.

**Instance metadata.** The launch template sets the IMDS hop limit to 1, as `system-2b`'s `disablePodIMDS` does, so pods cannot read the node's IAM credentials. A custom launch template would otherwise default to 2.

### How the kubelet config reaches the node (AL2023 managed node group)

1. **Where the settings go.** AL2023 nodes start with `nodeadm`, which reads a `NodeConfig` YAML from the EC2 user data. On a managed node group, the only way to change kubelet settings such as `cpuManagerPolicy` is a custom launch template.
2. **User data format.** It must be a MIME multipart archive. Here it has one part with `Content-Type: application/node.eks.aws` containing `spec.kubelet.config`. Sources: [EKS launch templates](https://docs.aws.amazon.com/eks/latest/userguide/launch-templates.html), "Amazon Linux 2023 user data"; [nodeadm](https://awslabs.github.io/amazon-eks-ami/nodeadm/).
3. **No `ImageId`.** The launch template does not set an AMI, so EKS keeps picking the AMI from `--ami-type`. EKS also merges its own `NodeConfig` (cluster endpoint, CA, service CIDR, clusterDNS, maxPods, kubeReserved, labels, taints) into the user data, and nodeadm merges both parts. If the template did set an AMI ID, EKS would stop merging and the whole cluster block would have to be written by hand. Our part sets only keys that EKS does not set. When several parts set the same key, the last part wins ([nodeadm examples](https://awslabs.github.io/amazon-eks-ami/nodeadm/doc/examples/)).
4. **Settings that move into the launch template.** With a launch template, `create-nodegroup` refuses `--disk-size` and `--remote-access`, and the template must not set a subnet or an IAM instance profile. So the disk goes into the template's `BlockDeviceMappings`. The instance type stays on `--instance-types`, so the `INSTANCE_TYPE=` override still works.
5. **Why not eksctl.** The eksctl docs document a custom AL2023 `NodeConfig` through `overrideBootstrapCommand`, and show it only with a custom AMI; whether it works without one was not confirmed. We did not confirm whether `kubeletExtraConfig` works on AL2023 managed node groups, so nothing here depends on it. Using `aws eks create-nodegroup` also keeps this study consistent with how study 30 created its GPU node groups (`gpu-nodegroup.sh`).
6. **A malformed part fails silently.** nodeadm ignores it and the node joins with the default kubelet config. That is why `--up` reads the config back:

   ```bash
   kubectl get --raw /api/v1/nodes/<node>/proxy/configz \
     | jq '.kubeletconfig | {cpuManagerPolicy,topologyManagerPolicy,kubeReserved,systemReserved}'
   ```

   The script stops with an error if `cpuManagerPolicy` is not `static`. If configz cannot be read, it only warns: the allocatable CPU it prints is the fallback check (46 means the NodeConfig was applied, about `47810m` means it was not). The checks run after the `--always-on` tagging, so a failed check never leaves a running node untagged.
7. **How to change the config.** Edit `eks/nodeconfig.yaml` and re-run the script. It creates a new launch template version and runs `aws eks update-nodegroup-version`. At desiredSize 0 this only changes the template. At 1 it should replace the node, but on 2026-10-16 a node that had never joined the cluster kept running on the old version after a "Successful" update: check `eks.amazonaws.com/sourceLaunchTemplateVersion` on the node, and if it is stale replace the instance with `aws autoscaling terminate-instance-in-auto-scaling-group --instance-id <id> --no-should-decrement-desired-capacity`.

### Kubelet settings and why

- **`cpuManagerPolicy: static`.** The vLLM pod is Guaranteed: CPU request = limit = 44, an integer, so it gets an exclusive cpuset. vLLM's `VLLM_CPU_OMP_THREADS_BIND=auto` reads `sched_getaffinity`. Without the static policy, the pod would see all 48 vCPUs, bind an OMP thread to each, and then be throttled by the CFS quota.
- **`cpuManagerPolicyOptions: full-pcpus-only: "true"`.** This was not in the contract. It makes the kubelet hand out only whole physical cores: 44 CPUs = 22 cores with both threads. A request that is not a multiple of 2 is rejected with `SMTAlignmentError` instead of being given a half-used core. This is only a safeguard, since 44 is already whole cores.
  - Thread count: vLLM's x86 auto-binding uses one thread per physical core, the higher-numbered sibling. So the 44-CPU cpuset becomes about 22 OMP threads, minus `VLLM_CPU_NUM_OF_RESERVED_CPU`.
- **`topologyManagerPolicy: single-numa-node`.** All of a pod's CPUs come from one NUMA node. c8i.12xlarge is expected to have a single NUMA node, in which case this setting has no effect. The EC2 API does not report the NUMA layout, so `--up` prints it.
  - If the node shows 2 NUMA nodes (sub-NUMA clustering), the 44-CPU pod fails admission with `TopologyAffinityError`. Fix: lower the pod to at most one NUMA node's cores, or drop the topology policy in `nodeconfig.yaml` and re-run.
- **2 reserved CPUs = core 0 (cpus 0 and 24), via `systemReserved: {cpu: 1810m}`.** Kept for the kubelet, containerd and the DaemonSets, never given to pods.
  - **Why not `reservedSystemCPUs: "0,24"`** (the first design): EKS's AL2023 kubelet config sets `kubeReservedCgroup: /runtime` and `systemReservedCgroup: /system`, and the kubelet refuses to start with `reservedSystemCPUs` next to either ("can't use reservedSystemCPUs (--reserved-cpus) with systemReservedCgroup ... or kubeReservedCgroup"; the first node, 2026-10-16, never joined).
  - **How it works:** EKS computes `kubeReserved.cpu` = 190m on 48 vCPUs; 190m + 1810m = 2 CPUs exactly. The static policy reserves that many CPUs topology-aware, from the lowest whole core: core 0 = cpus 0 and 24.
  - **Verified on the node (2026-10-16):** configz shows static + full-pcpus-only + single-numa-node, kubeReserved cpu 190m, systemReserved cpu 1810m; allocatable cpu 46; cpu0 siblings `0,24`, cpu1 `1,25`; 1 NUMA node (`lscpu`: 1 socket, 24 cores x 2 threads, NUMA node0 = 0-47); allocatable memory ~86 GiB (kubeReserved memory 6745Mi).
  - **If EKS changes its 190m**, the sum is no longer an integer, the reservation rounds up to 3 CPUs and `--up` shows allocatable != 46 (the 44-CPU pod still fits at 45).

### The CPU manager state file

`/var/lib/kubelet/cpu_manager_state` only has to be deleted when the policy changes on a node that already ran the kubelet with another policy. Otherwise the kubelet refuses to start.

That never happens here:

- the node group is new;
- the policy is in the launch template from the first version;
- every `--up` boots a fresh instance (nodeadm writes the config before the kubelet's first start);
- a change to `nodeconfig.yaml` gives a new launch template version, which replaces the instance instead of reconfiguring it.

Never switch an existing node to this policy by hand.

## Prerequisites

- Tools: `aws` (profile `lab`, account 916205288457), `kubectl`, and `jq` (optional, makes the `--up` checks easier to read).
- The workstation's default AWS profile points at another account. Always use `AWS_PROFILE=lab` (the scripts default to it) or `--profile lab`.
- **Quota:** on spot, c8i counts against "All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests" (L-34B43A08, 640 vCPU in us-east-2 on 2026-10-16); On-Demand against L-1216C47A (640 vCPU). Both are separate from the G-instance quota of 64 vCPU that held back the GPU studies. To check:

  ```bash
  aws service-quotas get-service-quota --service-code ec2 --quota-code L-34B43A08 --region us-east-2 --profile lab
  ```

- **Nightly stop:** a Lambda stops every EC2 instance without the tag `AlwaysOn=true` at 17:00 UTC. Use `--always-on` before any run that goes past that time.

## Usage

```bash
cd studies/39-c8i-qwen3-06b-cpu-tps/infra
AWS_PROFILE=lab ./eks/cpu-nodegroup.sh --up --always-on   # node to 1, kubelet/topology checks, AlwaysOn tags
AWS_PROFILE=lab ./provision.sh
```

If `--up` hangs on "waiting for a Ready node", run:

```bash
aws eks describe-nodegroup --cluster-name vllm-bench --region us-east-2 \
  --nodegroup-name llm-serving-c8i --query nodegroup.health.issues
```

- `InsufficientInstanceCapacity`: use another AZ (see "The node group").
- A node that never registers usually means a bad user-data part. Read the instance's console output (`aws ec2 get-console-output`), which shows nodeadm's log.

## Teardown

- **Pause billing:** `AWS_PROFILE=lab ./eks/cpu-nodegroup.sh --down` scales to 0 and removes the ASG's `AlwaysOn` tag, so a forgotten scale-up is still stopped at night.
- **Remove this study's Kubernetes layer:** `kubectl delete namespace llm-cpu`, then `kubectl -n monitoring delete servicemonitor <name>`. Use the ServiceMonitor's name from `../k8s/monitoring/servicemonitor.yaml`, if it lives in `monitoring`. Volumes created with `gp3` have reclaim policy Retain: delete their PVs and EBS volumes by hand.
- **Delete the node group and its launch template:**

  ```bash
  aws eks delete-nodegroup --cluster-name vllm-bench --region us-east-2 --nodegroup-name llm-serving-c8i
  aws eks wait nodegroup-deleted --cluster-name vllm-bench --region us-east-2 --nodegroup-name llm-serving-c8i
  aws ec2 delete-launch-template --region us-east-2 --launch-template-name llm-serving-c8i
  ```

  The cluster is shared. Do not delete it from here.

## What this does NOT cover

- The cluster, its system and `akamas` node groups, and the StorageClasses (study 30's `infra/eks/`).
- The Akamas 4.1 platform and its toolbox (namespace `akamas-41`).
- kube-prometheus-stack itself.
- The optimization packs (vLLM 1.14.0, Kubernetes), which are managed outside this repo.
- No NVIDIA device plugin, DCGM exporter or GPU taint: this study uses no GPU.
