#!/bin/bash
# Study layer of 36-l40s-gemma4-26b-awq-replay-users-v2 on the vllm-bench cluster (namespace llm-l40s,
# studies 30-34's): PVCs, Services, ServiceMonitor. Idempotent (kubectl apply).
# Usage: KUBE_CONTEXT=lab-vllm-bench bash infra/provision.sh   (default on the toolbox)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY=$(dirname "$HERE")
KUBE_CONTEXT=${KUBE_CONTEXT:-default}
kubectl() { command kubectl --context "$KUBE_CONTEXT" "$@"; }
kubectl get ns llm-l40s >/dev/null || kubectl create namespace llm-l40s
kubectl apply -f "$STUDY/k8s/00-pvc.yaml" -f "$STUDY/k8s/06-hf-cache-pvc.yaml" -f "$STUDY/k8s/02-service.yaml"
kubectl apply -f "$STUDY/k8s/monitoring/servicemonitor.yaml"
kubectl get nodes -l node-role=llm-serving-l40s-1xl
