# infra/ — this study's cluster, from zero

> Copied from `9-goodput-per-gpu/infra/` on 2026-09-14 for `12-gpt-oss-20b-steady-throughput`
> (same cluster, same `llm-serving-l4` node group, no new node group needed). The
> historical notes below about creating `llm-serving-l4`/`system-m8a` are kept as the record
> of how this cluster got here; the new section right after describes the situation this
> study starts from.

This study is atomic: everything needed to go from an empty AWS account to a cluster
ready for the `akamas create`/`akamas start study` commands in this study's own
`README.md` lives here — nothing is assumed to already exist on a shared cluster.
Deliberately duplicated across studies rather than centralized (see the repo root
`README.md` on why studies are self-contained).

**This study needs a NEW GPU node group, `llm-serving-l4`** (g6.12xlarge, 4x NVIDIA L4
24GB) — **created 2026-09-07** (`eksctl create nodegroup --config-file=... --include=
"llm-serving-l4,system-m8a"`, profile `lab`; no manual `ami:` pin needed — eksctl
recognized `g6` and auto-selected the NVIDIA AMI, unlike the `g7e` gap
`2-larger-model-g7e` hit). Same cluster (`vllm-bench`), same region (`us-east-2`) as
every other study. `provision.sh` detects whether the cluster and this node group
already exist and only creates what's missing — it never touches the existing
`akamas`/`llm-serving`/`llm-serving-g7e` node groups (nor the pre-existing, untracked
`m7i-2xlarge`/`m8a-2xlarge` node groups found already on this cluster at provisioning
time — unrelated to this study, left alone).

**`system-m8a` — created AND migrated to, same day.** The candidate replacement for
the shared `system` node group (`m6i.xlarge` → `m8a.xlarge`, per explicit request) was
created alongside `llm-serving-l4`, then the entire shared monitoring/ingress stack was
migrated onto it: `cert-manager` (Helm, 3 sub-nodeSelectors), `kube-prometheus-stack`
(Helm, 5: prometheus/alertmanager/grafana/kube-state-metrics/operator),
`nginx-ingress`, `external-dns`, `open-webui` (the last two are raw Deployments, not
Helm releases — nodeSelector patched directly). One real gotcha hit during migration:
`system-m8a`'s ASG initially came up in `us-east-2a` while `system`'s persistent
volumes (Prometheus/Grafana/open-webui, `Retain` policy) are in `us-east-2b` — fixed
the same way the `system` node group's own AZ mismatch was fixed earlier (pin the
ASG's `VPCZoneIdentifier` to the single `us-east-2b` subnet, cycle the node), verified
live via Prometheus's own scrape continuity (`count(up)` over the full migration
window, no gap) that no historical data was lost. A second gotcha: Grafana/open-webui
(Deployments with an RWO EBS volume) deadlocked on `RollingUpdate` — the new pod
couldn't attach the volume while the old pod on the other node still held it, and the
old pod wouldn't terminate until the new one was Ready. Fixed by directly scaling the
OLD ReplicaSet to 0 (not just deleting its pod, which the Deployment controller just
recreates) — a one-time manual unblock, not something `provision.sh` automates.

**`system` node group is now empty** (only DaemonSets remain: `aws-node`,
`kube-proxy`, `ebs-csi-node`, node-exporter, the NVIDIA device plugin) and can be
scaled to 0 (`eksctl scale nodegroup --name system --nodes 0`) once you're satisfied
nothing regressed. Not decommissioned automatically by anything in this repo — that's
a deliberate manual step, do it when ready.

## AZ binding, capacity and what a region move entails (2026-09-14)

Three facts found while scaffolding this study, all live on the cluster:

1. **The model-cache EBS volume is pinned to one Availability Zone.** Study 9's
   `vllm-model-cache` PV (`gp3-ephemeral`, `WaitForFirstConsumer`) was created in
   **us-east-2c**; the `llm-serving-l4` node group spans us-east-2a/b/c. A node that comes
   back in another AZ leaves the vLLM pod `Pending` with a volume node-affinity conflict —
   this study therefore uses its **own claim, `vllm-model-cache-gptoss`**
   (`k8s/01-pvc-model-cache.yaml`), which binds wherever the GPU node lands on first start.
   The old claim can be deleted once study 9 is finished (not done by anything here).
2. **`g6.12xlarge` capacity.** The node group has been `DEGRADED` since 2026-09-14 14:36 UTC:
   `AsgInstanceLaunchFailures` / `InsufficientInstanceCapacity` in us-east-2a, b and c in
   turn, retried by the ASG every ~2 minutes. The previous instance was terminated on
   2026-09-13 19:07 UTC by a manual `UpdateNodegroupConfig desiredSize 0` (AWS CLI, role
   `akamas_lab_admin`), not by AWS. EC2 spot placement scores (1-10) read 1-3 for
   `g6.12xlarge` and `g6.24xlarge` in us-east-2 and 3 in us-east-1 / us-west-2; the first
   launch on 2026-09-10 already needed ~10 failed attempts. Options, in order of cost:
   - **wait**: keep `desiredSize: 1`, the ASG keeps trying (no action needed);
   - **`g6e.12xlarge` node group in the same cluster** (4x L40S 48 GB, also compute
     capability 8.9 → identical kernel path, placement score 3, ~2.3x the hourly price):
     a new managed node group (`instanceType` is immutable), a new DCGM exporter release
     and `node-role` label, and results that are not comparable with the L4 studies;
   - **change region**: EKS is regional and the Akamas platform itself runs in this cluster
     (namespace `akamas`: database, elasticsearch, keycloak, kong, license, optimizer,
     orchestrator, toolbox, ... 40 pods), so a move means a new cluster from
     `eks/cluster.yaml`, reinstalling Akamas and re-issuing its license, recreating every
     PVC (Akamas data, Prometheus/Grafana history, model and dataset caches), the
     monitoring stack, the toolbox checkout and keys, and the `kubectl` contexts of every
     study — days of work for the same capacity signal (score 3) elsewhere.
3. **Capacity reservations**: none exist in the account; an On-Demand Capacity
   Reservation can only be created when capacity is available, so it is a way to *keep*
   the node once it is obtained, not to obtain it.

## Layout

- **`eks/cluster.yaml`** — the full `eksctl` `ClusterConfig` for the `vllm-bench`
  cluster: `system` (existing, `m6i.xlarge`) + `system-m8a` (new, `m8a.xlarge`,
  `desiredCapacity: 0` — scaffolded only) + `akamas` (existing, unchanged) +
  `llm-serving` (existing A10G group, kept here as part of the full snapshot but not
  used by this study) + `llm-serving-l4` (new, `g6.12xlarge`, `desiredCapacity: 0` —
  scaffolded only). See the file's own comments for the AMI-selection caveat
  (`amiFamily: AmazonLinux2023` with no explicit `ami:` override — unverified for
  `g6`, confirmed to need a pin for `g7e` in `2-larger-model-g7e`).
- **`eks/storageclass.yaml`** — the default `gp3` StorageClass (Retain reclaim policy).
- **`eks/provision.sh`** — creates the cluster if it doesn't exist yet (all node
  groups), or, if it already exists, creates ONLY `llm-serving-l4` and `system-m8a` if
  they're missing — every other node group is left completely untouched. Then applies
  StorageClasses, the NVIDIA device plugin, namespaces, and this study's PVCs. Prints
  remaining manual steps at the end.
- **`k8s-bootstrap/00-namespaces.yaml`** — the three namespaces this study uses
  (`llm-serving`, `llm-benchmark`, `monitoring`) — identical to prior studies, applied
  idempotently (`kubectl apply` no-ops if they already exist from an earlier study).
- **`k8s-bootstrap/01-storage-classes.yaml`** — the second StorageClass,
  `gp3-ephemeral` (Delete reclaim policy, for the re-downloadable model cache).

## Prerequisites (local tooling, not provisioned by this folder)

`eksctl`, `kubectl`, `aws` CLI (with credentials for an account that can create/modify
EKS node groups), and `helm` (for the monitoring stack, already installed on this
cluster from `0-explorative`'s provisioning).

## Usage

```bash
cd studies/12-gpt-oss-20b-steady-throughput/infra/eks
./provision.sh                          # default region us-east-2
./provision.sh --region us-west-2       # different region (also edit cluster.yaml)
./provision.sh --profile my-aws-profile # named AWS CLI profile
```

After provisioning, verify the GPU actually came up correctly before trusting anything
downstream (this is the first study on this instance family in this repo):

```bash
kubectl describe node -l node-role=llm-serving-l4 | grep -A5 Allocatable
# Should show: nvidia.com/gpu: 4
```

If it doesn't, see `eks/cluster.yaml`'s own comment on the AMI-selection gap
`2-larger-model-g7e` hit for a different new instance family (`g7e`) — the fix there
was pinning `ami:` explicitly to the EKS-optimized AL2023 NVIDIA variant.

## Teardown

```bash
# Stop GPU billing without affecting other studies' node groups.
eksctl delete nodegroup --cluster vllm-bench --region us-east-2 --name llm-serving-l4 --approve

# Full cluster teardown — CAUTION: this removes every study's node groups, since they
# all share this same cluster (0-explorative/1-goodput-realistic-load/3-comparison-a10/
# 5-pack-changes' llm-serving, 2-larger-model-g7e's llm-serving-g7e, this study's
# llm-serving-l4). Confirm no other study still needs it before running this.
eksctl delete cluster --name vllm-bench --region us-east-2
```

## What this does NOT cover

- The Akamas platform itself (assumed already installed/reachable).
- The `toolbox` host this study's Akamas workflow SSHes into to run `kubectl`/`helm`
  commands — needs its own `kubectl` configured against this cluster and this repo
  checked out at the path the workflow references (see `1-goodput-realistic-load`'s own
  `infra/README.md` for the precedent this follows).
- Monitoring stack *installation* — already done on this cluster from
  `0-explorative`'s provisioning. This study needs its OWN DCGM Exporter release
  (`dcgm-exporter-l4`, see `k8s/monitoring/dcgm-exporter-values.yaml`), same pattern as
  `2-larger-model-g7e`'s `dcgm-exporter-g7e` — a single shared release can only target
  one node-role at a time.
- **Scaling `system` to 0** — done manually (`eksctl scale nodegroup`), not by
  anything in this repo, once you've confirmed the migration above is stable.
