#!/bin/bash
# Installs (or reconciles) the GPU sharing layer on llm-serving-g7-4500. Idempotent.
# The same resources were created by hand in phase 0 (2026-09-29) under these exact
# names, so on the live cluster this is an upgrade in place, not a second copy.
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
NS=gpu-sharing
NODE=$(kubectl get nodes -l node-role=llm-serving-g7-4500 -o jsonpath='{.items[0].metadata.name}')
[ -n "$NODE" ] || { echo "no llm-serving-g7-4500 node — run ../eks/gpu-nodegroup.sh first" >&2; exit 1; }

kubectl get namespace $NS >/dev/null 2>&1 || kubectl create namespace $NS

# 1. The cluster-wide plugin must not run on this node. The node's akamas.io/gpu-sharing
#    taint stops it from being scheduled there again, but a pod that started before the
#    taint was applied stays until deleted (NoSchedule does not evict). One-time.
kubectl get node "$NODE" -o jsonpath='{.spec.taints[*].key}' | grep -q akamas.io/gpu-sharing \
  || { echo "node $NODE lacks the akamas.io/gpu-sharing taint — run ../eks/gpu-nodegroup.sh" >&2; exit 1; }
# `|| true`: on every re-run there is no such pod, grep matches nothing and exits 1,
# which under pipefail would abort the script before the helm upgrade (audit 2026-09-29).
{ kubectl -n kube-system get pods --field-selector spec.nodeName="$NODE" -o name \
  | grep nvidia-device-plugin || true; } | xargs -r kubectl -n kube-system delete

# 2. This study's device plugin.
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin >/dev/null 2>&1 || true
helm repo update nvdp >/dev/null
helm upgrade --install nvdp-g7 nvdp/nvidia-device-plugin --version 0.18.0 \
  --namespace $NS -f "$DIR/nvdp-values.yaml"
kubectl -n $NS rollout status ds/nvdp-g7-nvidia-device-plugin --timeout=180s

# 3. Host nvidia-smi access for apply_config.sh.
kubectl apply -f "$DIR/gpu-admin.yaml"
kubectl -n $NS rollout status ds/gpu-admin --timeout=120s

echo "node $NODE: config=$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.nvidia\.com/device-plugin\.config}') nvidia.com/gpu=$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
