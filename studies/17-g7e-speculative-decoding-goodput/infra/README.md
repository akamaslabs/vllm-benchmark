# infra/ — this study's cluster, from zero

This study is atomic: everything needed to go from an empty AWS account to a cluster
ready for the `akamas create`/`akamas start study` commands in this study's own
`README.md` lives here — nothing is assumed to already exist on a shared cluster.
Deliberately duplicated across studies rather than centralized (see the repo root
`README.md` on why studies are self-contained).

**This study deliberately targets the same cluster as `0-explorative`/
`1-goodput-realistic-load`** (same name `vllm-bench`, same region `us-east-2`) — an
explicit user decision (2026-08-20), not an oversight. Unlike those two studies (which
share the *same* A10G `llm-serving` node group because they use identical hardware),
this study's hardware is different, so it adds its **own, separate managed node
group** — `llm-serving-g7e` (`g7e.4xlarge`, 1x NVIDIA RTX PRO 6000 Blackwell Server
Edition 96GB) — rather than reusing `llm-serving`. `provision.sh` detects whether the
cluster and this specific node group already exist and only creates what's missing;
the existing A10G node group is never touched by this study's provisioning.

**Hardware swapped 2026-08-21**: originally `p5.4xlarge` (1x H100 80GB) — no capacity
available in `us-east-2` at provisioning time. Now `g7e.4xlarge` (RTX PRO 6000
Blackwell, GDDR7, ~1.6TB/s bandwidth vs H100's ~3.35TB/s HBM3) — a different GPU class,
not just a smaller/bigger H100, see `cluster.yaml`'s comment on this node group and the
study's own `README.md` for what this changes (decode-throughput bandwidth ceiling,
SM120 kernel-maturity risk on `attention_backend`/`kv_cache_dtype`). The node group
name `llm-serving-g7e` is kept for continuity with existing Akamas resource names and
k8s manifests — it no longer describes the actual GPU.

## Layout

- **`eks/cluster.yaml`** — the full `eksctl` `ClusterConfig` for the `vllm-bench`
  cluster, including the pre-existing `system`/`akamas`/`llm-serving` node groups (kept
  here so a from-scratch run of this script on an empty account still produces a
  complete cluster) plus this study's own `llm-serving-g7e` node group
  (`g7e.4xlarge`, RTX PRO 6000 Blackwell — see the hardware-swap note above).
- **`eks/storageclass.yaml`** — the default `gp3` StorageClass (Retain reclaim policy).
- **`eks/provision.sh`** — creates the cluster if it doesn't exist yet (all node
  groups), or, if it already exists, creates only the missing `llm-serving-g7e` node
  group via `eksctl create nodegroup --include`. Then applies StorageClasses, the
  NVIDIA device plugin, namespaces, and (once populated) this study's PVCs. Prints
  remaining manual steps at the end.
- **`k8s-bootstrap/00-namespaces.yaml`** — the three namespaces this study uses
  (`llm-serving`, `llm-benchmark`, `monitoring`) — identical to prior studies, applied
  idempotently (`kubectl apply` no-ops if they already exist from an earlier study).
- **`k8s-bootstrap/01-storage-classes.yaml`** — the second StorageClass,
  `gp3-ephemeral` (Delete reclaim policy, for the re-downloadable model cache).

## Prerequisites (local tooling, not provisioned by this folder)

`eksctl`, `kubectl`, `aws` CLI (with credentials for an account that can create/modify
EKS node groups), and `helm` (for the monitoring stack, already installed on this
cluster from `0-explorative`'s provisioning if reusing `vllm-bench`).

## Usage

```bash
cd studies/17-g7e-speculative-decoding-goodput/infra/eks
./provision.sh                          # default region us-east-2
./provision.sh --region us-west-2       # different region (also edit cluster.yaml)
./provision.sh --profile my-aws-profile # named AWS CLI profile
```

## Teardown

```bash
# Stop this GPU node's billing, keep the rest of the cluster (including the A10G node
# used by 0-explorative/1-goodput-realistic-load) running:
eksctl delete nodegroup --cluster vllm-bench --region us-east-2 --name llm-serving-g7e --approve

# Full cluster teardown — CAUTION: this also removes 0-explorative's and
# 1-goodput-realistic-load's node groups, since they share this same cluster. Confirm
# no other study still needs it before running this.
eksctl delete cluster --name vllm-bench --region us-east-2
```

## What this does NOT cover

- The Akamas platform itself (assumed already installed/reachable).
- The `toolbox` host this study's Akamas workflow SSHes into to run `kubectl`/`helm`
  commands — needs its own `kubectl` configured against this cluster and this repo
  checked out at the path the workflow references (see `1-goodput-realistic-load`'s own
  `infra/README.md` for the precedent this follows).
- Monitoring stack *installation* — already done on this cluster from
  `0-explorative`'s provisioning if reusing `vllm-bench`; this study's own
  `ServiceMonitor` (once `k8s/monitoring/` is populated, see the main `README.md`'s
  "Prerequisites still open") still needs to be applied so Prometheus scrapes this
  study's workload specifically.
