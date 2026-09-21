#!/usr/bin/env bash
set -euo pipefail

# Provision the vllm-bench EKS cluster for this study (17-g7e-speculative-decoding-goodput).
#
# Node layout (all defined in cluster.yaml, but NOT all created by this script — see
# below): system (m6i.xlarge), akamas (r6i.xlarge), llm-serving (g5.2xlarge, 1x A10G,
# belongs to 0-explorative/1-goodput-realistic-load), and this study's own
# llm-serving-g7e (g7e.4xlarge, 1x RTX PRO 6000 Blackwell Server Edition 96GB,
# tainted — swapped 2026-08-21 from p5.4xlarge/H100 due to no capacity in this region;
# see cluster.yaml's comment on that node group for what this changes).
#
# Unlike 1-goodput-realistic-load's provision.sh (which reuses an *identical* node
# group set and can skip provisioning entirely if the cluster exists), this study
# reuses the SAME cluster but needs a node group that doesn't exist yet on it. So this
# script:
#   - creates the whole cluster (all node groups in cluster.yaml) if `vllm-bench`
#     doesn't exist at all yet (e.g. this study is ever run standalone on a fresh
#     account, per this repo's atomic-per-study convention);
#   - otherwise, leaves the existing system/akamas/llm-serving node groups untouched
#     and creates ONLY the missing llm-serving-g7e node group via
#     `eksctl create nodegroup --include`.
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
G7E_NODEGROUP="llm-serving-g7e"
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
echo "=== vllm-bench EKS Cluster (studies/17-g7e-speculative-decoding-goodput) ==="
echo "Cluster   : $CLUSTER_NAME"
echo "Node group: $G7E_NODEGROUP (g7e.4xlarge, 1x RTX PRO 6000 Blackwell 96GB)"
echo "Region    : $AWS_REGION"
echo ""

# --- 1. Cluster + GPU node group ---
echo "[1/6] Cluster + node groups..."
if eksctl get cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" $PROFILE_ARG >/dev/null 2>&1; then
  echo "  Cluster '$CLUSTER_NAME' already exists."
  if eksctl get nodegroup --cluster "$CLUSTER_NAME" --region "$AWS_REGION" --name "$G7E_NODEGROUP" $PROFILE_ARG >/dev/null 2>&1; then
    echo "  Node group '$G7E_NODEGROUP' already exists — skipping creation."
  else
    echo "  Node group '$G7E_NODEGROUP' missing — creating it (existing node groups untouched)..."
    eksctl create nodegroup --config-file="$CLUSTER_CONFIG" --include="$G7E_NODEGROUP" $PROFILE_ARG
  fi
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
# Idempotent even if 0-explorative/1-goodput-realistic-load already applied these —
# `kubectl apply` no-ops on an unchanged resource.
echo ""
echo "[3/6] Applying StorageClasses (gp3 default + gp3-ephemeral)..."
kubectl apply -f "$STORAGE_CLASS"
kubectl apply -f "$BOOTSTRAP_DIR/01-storage-classes.yaml"

# --- 4. NVIDIA device plugin ---
# Already running on this cluster from 0-explorative's provisioning, but re-applying
# is idempotent and ensures it's present even on a from-scratch run of this script.
# Its DaemonSet tolerates the nvidia.com/gpu:NoSchedule taint, so it schedules on the
# new llm-serving-g7e node the same way it already does on llm-serving.
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
echo "  TODO: this study's k8s/ manifests (PVCs included) are not yet populated —"
echo "  see this study's own README.md, 'Prerequisites still open,' item 5."
# kubectl apply -f "$K8S_DIR/00-pvc.yaml"
# kubectl apply -f "$K8S_DIR/01-pvc-model-cache.yaml"

# --- Summary ---
echo ""
echo "=== Done ==="
echo ""
kubectl get nodes -L node-role
echo ""
echo "Verify the GPU is visible to Kubernetes:"
echo "  kubectl describe node -l node-role=llm-serving-g7e | grep -A5 Allocatable"
echo "  # Should show: nvidia.com/gpu: 1"
echo ""
echo "Next steps (still manual, not run by this script):"
echo ""
echo "  1. Confirm this study's open prerequisites (README.md): vLLM/pack version for"
echo "     Qwen2.5-7B-Instruct on this hardware, and recalibrating the AIPerf"
echo "     concurrency sweep before trusting 1-goodput-realistic-load's"
echo "     carried-forward numbers."
echo ""
echo "  2. Populate $K8S_DIR/ with this study's actual manifests (vLLM Deployment/"
echo "     Service, AIPerf job, PVCs, monitoring) — mirror"
echo "     studies/1-goodput-realistic-load/k8s/, adapted for this model/image."
echo ""
echo "  3. Monitoring stack (DCGM Exporter, Prometheus/Grafana) is already installed"
echo "     on this cluster from 0-explorative's provisioning — confirm its"
echo "     ServiceMonitor also covers pods on the llm-serving-g7e node before"
echo "     assuming metrics are being scraped."
echo ""
echo "  4. Deploy vLLM manually to sanity-check the stack before creating the Akamas"
echo "     study, same as prior studies. If serving a gated model, create the"
echo "     HuggingFace token secret first:"
echo "       kubectl create secret generic hf-token \\"
echo "         --from-literal=token=<YOUR_HF_TOKEN> \\"
echo "         --namespace llm-serving"
echo ""
echo "  5. Create and start the Akamas study — see this study's own README.md"
echo "     ('How to run') for the exact akamas create/start commands."
echo ""
echo "Stop GPU node billing (keep the rest of the cluster running, including the A10G node):"
echo "  eksctl delete nodegroup --cluster $CLUSTER_NAME --region $AWS_REGION --name $G7E_NODEGROUP --approve $PROFILE_ARG"
echo ""
echo "Full teardown (affects 0-explorative/1-goodput-realistic-load too — confirm no"
echo "other study needs this cluster first):"
echo "  eksctl delete cluster --name $CLUSTER_NAME --region $AWS_REGION $PROFILE_ARG"
