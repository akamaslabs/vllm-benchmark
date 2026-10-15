#!/bin/bash
# Study layer of 38-l40s-gemma4-26b-awq-replay-users-v3 on the existing vllm-bench-gatling cluster:
# namespace, PVCs, Services, ServiceMonitor. Idempotent (kubectl apply). The cluster, its system
# node, the monitoring stack (kube-prometheus-stack + dcgm-exporter, Prometheus behind an NLB) and
# the L40S node groups belong to akamas-gatling-llms-optimization (infra/eks/cluster.yaml); this
# study does not change them (README.md here).
# Usage: KUBE_CONTEXT=lab-vllm-bench-gatling bash infra/provision.sh   (gatling on the toolbox)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY=$(dirname "$HERE")
KUBE_CONTEXT=${KUBE_CONTEXT:-gatling}
kubectl() { command kubectl --context "$KUBE_CONTEXT" "$@"; }

kubectl get nodes -l node-role=system >/dev/null || { echo "error: cluster unreachable with context $KUBE_CONTEXT" >&2; exit 1; }
kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null || { echo "error: no ServiceMonitor CRD (monitoring stack missing)" >&2; exit 1; }
kubectl apply -f "$HERE/k8s-bootstrap/00-namespaces.yaml"
kubectl apply -f "$STUDY/k8s/00-pvc.yaml" -f "$STUDY/k8s/06-hf-cache-pvc.yaml" -f "$STUDY/k8s/02-service.yaml"
kubectl apply -f "$STUDY/k8s/monitoring/servicemonitor.yaml"
echo "done: namespace llm-replay, PVCs, Services, ServiceMonitor vllm-replay"
echo "GPU node: scale llm-serving-l40s (us-east-2c) or llm-serving-l40s-ab (us-east-2a/2b) to 1 (README.md)"
