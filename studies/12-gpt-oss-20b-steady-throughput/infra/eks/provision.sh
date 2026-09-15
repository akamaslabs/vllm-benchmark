#!/usr/bin/env bash
set -euo pipefail

# Provision the vllm-bench EKS cluster for this study (12-gpt-oss-20b-steady-throughput).
#
# Node layout (all defined in cluster.yaml, but NOT all created by this script — see
# below): system (m6i.xlarge), system-m8a (m8a.xlarge — scaffolded candidate
# replacement for `system`, NOT migrated to yet), akamas (r6i.xlarge), llm-serving
# (g5.2xlarge, 1x A10G, belongs to 0-explorative/1-goodput-realistic-load/
# 3-comparison-a10/5-pack-changes), and this study's own llm-serving-l4 (g6.12xlarge,
# 4x NVIDIA L4 24GB, tainted).
#
# Per explicit instruction (2026-09-07): "crea i nodegroup nuovi poi spostiamo dopo" —
# create the new node groups now, migrate onto them later as a separate, deliberate
# step. This script therefore:
#   - creates the whole cluster (all node groups in cluster.yaml) if `vllm-bench`
#     doesn't exist at all yet (e.g. this study is ever run standalone on a fresh
#     account, per this repo's atomic-per-study convention);
#   - otherwise, leaves the existing system/akamas/llm-serving node groups
#     COMPLETELY untouched (in particular: does NOT touch or migrate the live
#     `system` node group's workloads) and creates ONLY the two node groups that
#     don't exist yet: `llm-serving-l4` and `system-m8a`.
#
# What this script does NOT do, by design: migrate any workload from `system` onto
# `system-m8a`, or delete the old `system` node group. That's a separate, later step
# (draining `system`, confirming `system-m8a` is healthy, only then decommissioning
# the old one) — do it deliberately, not as a side effect of provisioning.
#
# Usage:
#   ./provision.sh
#   ./provision.sh --region us-west-2
#   ./provision.sh --profile my-aws-profile

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLUSTER_CONFIG="$SCRIPT_DIR/cluster.yaml"
STORAGE_CLASS="$SCRIPT_DIR/storageclass.yaml"
BOOTSTRAP_DIR="$STUDY_ROOT/infra/k8s-bootstrap"
K8S_DIR="$STUDY_ROOT/k8s"

CLUSTER_NAME="vllm-bench"
L4_NODEGROUP="llm-serving-l4"
SYSTEM_M8A_NODEGROUP="system-m8a"
AWS_REGION="us-east-2"
AWS_PROFILE=""

# --- Argument parsing ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --region)   AWS_REGION="$2"; shift 2 ;;
    --profile)  AWS_PROFILE="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--region <region>] [--profile <profile>]"
      exit 0
      ;;
    *) echo "Unknown argument: $1. Run $0 --help for usage."; exit 1 ;;
  esac
done

# --- Prerequisites ---
for cmd in eksctl kubectl aws helm; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' not found in PATH"; exit 1; }
done

PROFILE_ARG=""
if [[ -n "$AWS_PROFILE" ]]; then
  PROFILE_ARG="--profile $AWS_PROFILE"
  CALLER=$(aws sts get-caller-identity $PROFILE_ARG --query 'Arn' --output text)
  echo "AWS profile : $AWS_PROFILE"
  echo "Identity    : $CALLER"
fi

echo ""
echo "=== vllm-bench EKS Cluster (studies/12-gpt-oss-20b-steady-throughput) ==="
echo "Cluster    : $CLUSTER_NAME"
echo "Node groups: $L4_NODEGROUP (g6.12xlarge, 4x NVIDIA L4 24GB)"
echo "             $SYSTEM_M8A_NODEGROUP (m8a.xlarge — scaffolded, NOT migrated to yet)"
echo "Region     : $AWS_REGION"
echo ""

# --- 1. Cluster + new node groups ---
echo "[1/6] Cluster + node groups..."
if eksctl get cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" $PROFILE_ARG >/dev/null 2>&1; then
  echo "  Cluster '$CLUSTER_NAME' already exists."
  for NG in "$L4_NODEGROUP" "$SYSTEM_M8A_NODEGROUP"; do
    if eksctl get nodegroup --cluster "$CLUSTER_NAME" --region "$AWS_REGION" --name "$NG" $PROFILE_ARG >/dev/null 2>&1; then
      echo "  Node group '$NG' already exists — skipping creation."
    else
      echo "  Node group '$NG' missing — creating it (existing node groups untouched)..."
      eksctl create nodegroup --config-file="$CLUSTER_CONFIG" --include="$NG" $PROFILE_ARG
    fi
  done
else
  echo "  Cluster '$CLUSTER_NAME' does not exist — creating cluster with all node groups..."
  eksctl create cluster -f "$CLUSTER_CONFIG" $PROFILE_ARG
fi

# --- 2. Update kubeconfig ---
echo ""
echo "[2/6] Updating kubeconfig..."
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION" $PROFILE_ARG
echo "  Context: $(kubectl config current-context)"

# --- 3. StorageClasses ---
# Idempotent even if an earlier study already applied these — `kubectl apply` no-ops
# on an unchanged resource.
echo ""
echo "[3/6] Applying StorageClasses (gp3 default + gp3-ephemeral)..."
kubectl apply -f "$STORAGE_CLASS"
kubectl apply -f "$BOOTSTRAP_DIR/01-storage-classes.yaml"

# --- 4. NVIDIA device plugin ---
# Already running on this cluster from 0-explorative's provisioning, but re-applying
# is idempotent and ensures it's present even on a from-scratch run of this script.
# Its DaemonSet tolerates the nvidia.com/gpu:NoSchedule taint, so it schedules on the
# new llm-serving-l4 node the same way it already does on llm-serving/llm-serving-g7e.
echo ""
echo "[4/6] Ensuring NVIDIA device plugin is installed..."
kubectl apply -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
echo "  Waiting for DaemonSet rollout (up to 3 min)..."
kubectl rollout status daemonset/nvidia-device-plugin-daemonset \
  --namespace kube-system \
  --timeout=180s

# --- 5. Namespaces ---
echo ""
echo "[5/6] Applying Kubernetes namespaces (llm-serving, llm-benchmark, monitoring)..."
kubectl apply -f "$BOOTSTRAP_DIR/00-namespaces.yaml"

# --- 6. PVCs (one-time, persist across the whole study) ---
echo ""
echo "[6/6] Applying this study's PVCs..."
kubectl apply -f "$K8S_DIR/00-pvc.yaml"
kubectl apply -f "$K8S_DIR/01-pvc-model-cache.yaml"

# --- Summary ---
echo ""
echo "=== Done ==="
echo ""
kubectl get nodes -L node-role
echo ""
echo "Verify the GPU is visible to Kubernetes:"
echo "  kubectl describe node -l node-role=llm-serving-l4 | grep -A5 Allocatable"
echo "  # Should show: nvidia.com/gpu: 4"
echo ""
echo "If eksctl silently picked the non-GPU AMI for llm-serving-l4 (see cluster.yaml's"
echo "comment on this — confirmed to happen for g7e, unverified for g6), nvidia.com/gpu"
echo "won't show up at all — check with:"
echo "  kubectl exec -it \$(kubectl get pod -n kube-system -l name=nvidia-device-plugin-ds -o name | head -1) -n kube-system -- nvidia-smi"
echo ""
echo "Next steps (still manual, not run by this script):"
echo ""
echo "  1. Install this study's OWN DCGM Exporter release (dcgm-exporter-l4) — see"
echo "     k8s/monitoring/dcgm-exporter-values.yaml for the exact helm command; a"
echo "     ConfigMap from dcgm_counters.csv is a prerequisite:"
echo "       kubectl create configmap dcgm-custom-metrics \\"
echo "         --from-file=metrics=$K8S_DIR/monitoring/dcgm_counters.csv -n monitoring"
echo ""
echo "  2. Confirm the vllm-model-cache-gptoss PVC lands in the same AZ as wherever"
echo "     llm-serving-l4 actually schedules (the recurring AZ-mismatch pattern this"
echo "     repo has hit repeatedly) — recreate the PVC if it's pinned to a different"
echo "     AZ, same as every prior study."
echo ""
echo "  3. Deploy vLLM manually to sanity-check the stack before creating the Akamas"
echo "     study — this is the FIRST study on NVIDIA L4/Ada Lovelace in this repo, so"
echo "     don't assume the image/model/flags just work:"
echo "       kubectl apply -f $K8S_DIR/02-service.yaml"
echo "       kubectl apply -f $K8S_DIR/01-deployment.yaml"
echo ""
echo "  4. Create and start the Akamas study — see this study's own README.md"
echo "     ('How to run') for the exact akamas create/start commands."
echo ""
echo "  5. (Separate, later step — NOT part of this scaffold) migrate the 'system'"
echo "     node group's workloads onto system-m8a and decommission the old m6i.xlarge"
echo "     group. Do this deliberately, step by step, with the same care used for the"
echo "     AZ-pinning fix on the 'system' node group in an earlier study (verify"
echo "     historical Prometheus/Grafana data survives before deleting anything)."
echo ""
echo "Stop GPU node billing (keep the rest of the cluster running):"
echo "  eksctl delete nodegroup --cluster $CLUSTER_NAME --region $AWS_REGION --name $L4_NODEGROUP --approve $PROFILE_ARG"
echo ""
echo "Full teardown (affects every study sharing this cluster — confirm no other"
echo "study needs it first):"
echo "  eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION $PROFILE_ARG"
