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
- **`eks/gpu-capacity-fallback.sh`** — capacity workarounds against the *live* cluster,
  added 2026-09-22 after a full day of `InsufficientInstanceCapacity` on `g7e.4xlarge`
  (see "GPU capacity" below). Deliberately kept out of `cluster.yaml`, which stays the
  from-scratch description of the cluster the study *wants*.
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

## GPU capacity — the alternatives, and why L4 is not one (2026-09-22)

`g7e.4xlarge` hit `InsufficientInstanceCapacity` 56+ consecutive times in `us-east-2`
across a single day, and a second attempt on `g7e.8xlarge` rolled back with the same
error. Every GPU instance type the region offers was then evaluated against the one
requirement that matters here: **the study serves a dense `Qwen/Qwen3-32B-FP8`
(~31 GiB of weights) plus a `Qwen/Qwen3-0.6B` drafter on a SINGLE GPU**, because the
whole point of the dense-model swap was to remove parallelism as a confound.

| Instance | GPU | VRAM/GPU | USD/h | 32B-FP8 on 1 GPU | Study changes |
|---|---|---|---|---|---|
| `g7e.4xlarge` | 1x RTX PRO 6000 Blackwell | 96 GB | 4.00 | yes, wide margin | none — the intended target |
| `g7e.8xlarge` | 1x RTX PRO 6000 Blackwell | 96 GB | 5.27 | yes, wide margin | none |
| `g7e.2xlarge` | 1x RTX PRO 6000 Blackwell | 96 GB | 3.36 | GPU yes, node no | pod requests `cpu: 8`/`memory: 32Gi`, above an 8-vCPU node's allocatable |
| `g6e.4xlarge` | 1x L40S (Ada) | 48 GB | 3.00 | yes, but tight | material, see `gpu-capacity-fallback.sh` |
| `g6.12xlarge` | 4x L4 (Ada) | 24 GB | 4.60 | no — needs TP4 | reintroduces the parallelism confound |
| `g6.4xlarge` | 1x L4 (Ada) | 24 GB | 1.32 | no — 31 GiB > 22.35 GiB | would need a much smaller model |
| `g5.12xlarge` | 4x A10G (Ampere) | 24 GB | 5.67 | no | Ampere has no FP8 tensor cores at all |

Three conclusions worth keeping:

1. **An L4 cannot host this model.** A single L4 exposes 22.35 GiB of VRAM against ~31 GiB
   of weights. The pre-existing `llm-serving-l4` node group (4x L4) fits the model only
   across tensor parallelism 4, and at 4.60 USD/h it is *more expensive* than the g7e node
   it would replace. It is not a cheaper fallback; it is a different, costlier study.
2. **`g7e.2xlarge` was never tried and still should not be**, even though it carries the
   same 96 GB GPU: 8 vCPU cannot schedule the vLLM pod as written, and a CPU-starved
   frontend at concurrency 2048 would cap goodput — contaminating the metric being tuned.
3. **The only true single-GPU fallback is the L40S**, and it costs KV cache: ~4-8 GiB of
   headroom versus ~54 GiB on the 96 GB node, which moves the saturation knee down to
   roughly 64-128 concurrent requests and turns the sweep's upper levels into preemption
   tests rather than speculative-decoding tests.

### What a direct capacity probe actually found (2026-09-22, 15:28 CEST)

The table above ranks the alternatives by *fit*. A probe then ranked them by what AWS
would actually hand over. `gpu-capacity-fallback.sh probe` asks by creating a capacity
reservation per type/zone and cancelling it in the same second: the API refuses outright
with `InsufficientInstanceCapacity` when the pool is empty, so the answer is immediate
and costs a second of billing instead of an ASG's four-minute retry cycle.

| Type | GPU | VRAM/GPU | Capacity in us-east-2 |
|---|---|---|---|
| `g6.4xlarge` | 1x L4 | 24 GB | 2a, 2b, 2c |
| `g5.2xlarge` | 1x A10G | 24 GB | 2a, 2b |
| `g5.12xlarge` | 4x A10G | 24 GB | 2b only |
| `g7e.*`, `g6e.*`, `g6.12xlarge` | >=48 GB | | none, in any zone |

**Everything with more than 24 GB per GPU was empty.** The `llm-serving-g7e-2a` node group
from the previous section reached `CREATE_FAILED` after 34 launch attempts, and AWS's
health message then flipped to recommending `us-east-2b` — the zone it had spent all day
rejecting. That hint names whichever zone you did not ask for; it carries no information
and should not be acted on again.

This inverts the decision. The constraint is no longer "which GPU suits a 32B dense
model" but "which model fits the only GPU obtainable". A single L4 holds 22.35 GiB, so
Qwen3-32B-FP8 is out and the candidates become `Qwen/Qwen3-8B-FP8` or
`Qwen/Qwen3-14B-FP8` — both dense, both FP8-capable on Ada/SM89, and both sharing the
151936-token Qwen3 vocabulary the `Qwen/Qwen3-0.6B` drafter requires, so the drafter is
unchanged. Verify the KV budget against the chosen checkpoint before committing: the
8B leaves noticeably more cache headroom than the 14B, and the study's `max_num_seqs`
domain was sized for 96 GB.

The scientific cost is smaller than it looks. The study asks when speculative decoding
helps, which is a question about regime rather than about absolute tokens per second. An
L4's ~300 GB/s of memory bandwidth sits far below the RTX PRO 6000's, pushing decode
deeper into the bandwidth-bound regime where speculation has the most to give — arguably
a sharper instrument for this particular question, at 1.32 USD/h instead of 4.00.

`eks/gpu-capacity-fallback.sh` implements the live-cluster actions this produced:

```bash
./gpu-capacity-fallback.sh status      # node groups + GPU instances actually running
./gpu-capacity-fallback.sh probe       # which types/zones have capacity RIGHT NOW
./gpu-capacity-fallback.sh l4-single   # 1x L4 (g6.4xlarge), desiredSize 1 — the only
                                       #   family with capacity; needs a model swap
./gpu-capacity-fallback.sh g7e-2a      # g7e pinned to us-east-2a, desiredSize 1
./gpu-capacity-fallback.sh l40s        # L40S node group, desiredSize 0 (opt-in, not free)
```

`g7e-2a` is kept for the day g7e capacity returns, but as of this writing it fails; the
probe is the cheapest way to find out before creating anything.

`g7e-2a` exists because AWS's own health message on the stuck node group says capacity is
available in `us-east-2a` while every recorded failure is in `us-east-2b`. The original
node group spans both zones, so its ASG *may* retry in 2a but keeps landing in 2b; the new
one has only the 2a subnet, so every retry is forced into the zone AWS points at. It
reuses the existing launch template verbatim, whose `nodeadm` userdata already writes
`node-role=llm-serving-g7e` and the `nvidia.com/gpu` taint — which is what makes it a
drop-in with zero changes to the study's manifests.

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
