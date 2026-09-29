#!/bin/bash
# Provision everything 25-g7-4500-gpu-sharing-goodput needs on the vllm-bench cluster, up
# to `akamas create`. Idempotent: on the live cluster every step is a no-op or an in-place
# reconcile. Two preconditions it checks rather than creates: kube-prometheus-stack in
# `monitoring` (step 7), and the GPU node group not paused at desiredSize 0 (step 4).
#
# Usage: ./provision.sh [--profile <aws-profile>]   (the lab account needs --profile lab)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLUSTER_NAME=vllm-bench
AWS_REGION=us-east-2

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

echo "[1/7] Cluster + system node groups (eksctl)"
if eksctl get cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "  cluster exists"
else
  eksctl create cluster -f "$SCRIPT_DIR/cluster.yaml"
fi
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"

echo "[2/7] StorageClasses"
kubectl apply -f "$SCRIPT_DIR/storageclass.yaml"
kubectl apply -f "$STUDY_ROOT/infra/k8s-bootstrap/01-storage-classes.yaml"

echo "[3/7] Cluster-wide NVIDIA device plugin (every GPU node except this study's)"
# Same version the cluster runs (v0.18.0). It never reaches llm-serving-g7-4500: that
# node's akamas.io/gpu-sharing taint is not in its tolerations.
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.18.0/deployments/static/nvidia-device-plugin.yml

echo "[4/7] GPU node group llm-serving-g7-4500 (aws eks, not eksctl — see cluster.yaml)"
"$SCRIPT_DIR/gpu-nodegroup.sh"
DESIRED=$(aws eks describe-nodegroup --cluster-name "$CLUSTER_NAME" --region "$AWS_REGION" \
  --nodegroup-name llm-serving-g7-4500 --query 'nodegroup.scalingConfig.desiredSize' --output text)
if [ "$DESIRED" = 0 ]; then
  echo "  node group is scaled to 0 (paused): scale it to 1 first, then re-run" >&2; exit 1
fi
kubectl wait --for=condition=Ready node -l node-role=llm-serving-g7-4500 --timeout=900s

echo "[5/7] Namespaces"
kubectl apply -f "$STUDY_ROOT/infra/k8s-bootstrap/00-namespaces.yaml"

echo "[6/7] GPU sharing layer (study device plugin + gpu-admin)"
"$STUDY_ROOT/infra/gpu-sharing/install.sh"

echo "[7/7] Study PVCs, Services, ServiceMonitor"
# The ServiceMonitor needs kube-prometheus-stack's CRD; this script does not install the
# monitoring stack (k8s/monitoring/values-kube-prometheus.yaml).
kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1 || {
  echo "  kube-prometheus-stack is not installed (no ServiceMonitor CRD): install it in namespace monitoring, then re-run" >&2; exit 1; }
kubectl apply -f "$STUDY_ROOT/k8s/00-pvc.yaml" -f "$STUDY_ROOT/k8s/06-hf-cache-pvc.yaml" \
  -f "$STUDY_ROOT/k8s/02-service.yaml" -f "$STUDY_ROOT/k8s/monitoring/servicemonitor.yaml"

cat <<'NOTE'

Done. Remaining manual steps (see the study README, "Before starting"):
  - kube-prometheus-stack must already run in `monitoring` (k8s/monitoring/values-kube-prometheus.yaml).
  - Re-point the SHARED dcgm-exporter release at this node group — only when no other
    running study needs GPU telemetry from its current node:
      helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring \
        --version 4.8.3 --reuse-values=false -f k8s/monitoring/dcgm-exporter-values.yaml
  - For runs past 17:00 UTC: ./gpu-nodegroup.sh --always-on
  - Install the GPU optimization pack >= 1.3.0 (GPU.sharing_mode) before `akamas create`.
NOTE
kubectl get nodes -L node-role
