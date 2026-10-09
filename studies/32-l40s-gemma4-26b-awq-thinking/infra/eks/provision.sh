#!/bin/bash
# Provision everything 32-l40s-gemma4-26b-awq-thinking needs on the vllm-bench cluster, up to
# `akamas create`. Idempotent: on the live cluster every step is a no-op or an in-place
# reconcile. Two preconditions it checks rather than creates: kube-prometheus-stack in
# `monitoring` (step 5), and the GPU node group not paused at desiredSize 0 (step 3: use
# gpu-nodegroup.sh --up first).
#
# Usage: ./provision.sh [--profile <aws-profile>]   (the lab account needs --profile lab)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLUSTER_NAME=vllm-bench
AWS_REGION=us-east-2
NG=llm-serving-l40s-1xl

while [[ $# -gt 0 ]]; do
  case $1 in
    --profile) export AWS_PROFILE="$2"; shift 2 ;;
    --help|-h) echo "Usage: $0 [--profile <aws-profile>]"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
for cmd in eksctl kubectl aws helm; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' not found in PATH"; exit 1; }
done
echo "Identity: $(aws sts get-caller-identity --query Arn --output text)"

echo "[1/5] Cluster + system node groups (eksctl)"
if eksctl get cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "  cluster exists"
else
  eksctl create cluster -f "$SCRIPT_DIR/cluster.yaml"
  # The cluster-wide NVIDIA device plugin (v0.18.0, the version the cluster runs): it
  # tolerates nvidia.com/gpu, the only taint of this study's node.
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.18.0/deployments/static/nvidia-device-plugin.yml
fi
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"

echo "[2/5] StorageClasses and namespaces"
kubectl apply -f "$SCRIPT_DIR/storageclass.yaml"
kubectl apply -f "$STUDY_ROOT/infra/k8s-bootstrap/01-storage-classes.yaml"
kubectl apply -f "$STUDY_ROOT/infra/k8s-bootstrap/00-namespaces.yaml"

echo "[3/5] GPU node group $NG (aws eks, not eksctl — see cluster.yaml)"
"$SCRIPT_DIR/gpu-nodegroup.sh"
DESIRED=$(aws eks describe-nodegroup --cluster-name "$CLUSTER_NAME" --region "$AWS_REGION" \
  --nodegroup-name $NG --query 'nodegroup.scalingConfig.desiredSize' --output text)
if [ "$DESIRED" = 0 ]; then
  echo "  node group is at 0 (paused): run ./gpu-nodegroup.sh --up --always-on, then re-run" >&2; exit 1
fi
kubectl wait --for=condition=Ready node -l node-role=$NG --timeout=900s

echo "[4/5] Study PVCs, Services"
kubectl apply -f "$STUDY_ROOT/k8s/00-pvc.yaml" -f "$STUDY_ROOT/k8s/06-hf-cache-pvc.yaml" \
  -f "$STUDY_ROOT/k8s/02-service.yaml"

echo "[5/5] ServiceMonitor"
kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1 || {
  echo "  kube-prometheus-stack is not installed (no ServiceMonitor CRD): install it in namespace monitoring, then re-run" >&2; exit 1; }
kubectl apply -f "$STUDY_ROOT/k8s/monitoring/servicemonitor.yaml"

cat <<'NOTE'

Done. Remaining manual steps (study README, "Morning runbook"):
  - Add this node group to the SHARED dcgm-exporter release (one more node role in its
    nodeAffinity; the other studies' filters are unaffected):
      helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring \
        --version 4.8.3 --reuse-values=false -f k8s/monitoring/dcgm-exporter-values.yaml
  - Startup probe (probe/), then the smoke study, then the study.
NOTE
kubectl get nodes -L node-role
