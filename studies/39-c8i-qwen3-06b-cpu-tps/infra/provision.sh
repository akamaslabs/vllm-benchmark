#!/bin/bash
# Provision everything 39-c8i-qwen3-06b-cpu-tps needs on the EXISTING vllm-bench cluster, up
# to `akamas create`. It never creates the cluster: the cluster, its system-2b / akamas node
# groups, the gp3 / gp3-ephemeral StorageClasses and kube-prometheus-stack are shared and
# checked, not created (studies/30-l40s-gemma4-26b-tps/infra/eks/ owns them).
# Idempotent: every step is a no-op or an in-place reconcile on the live cluster.
#
# Preconditions it checks:
#   - the CPU node group is not paused at desiredSize 0 (run eks/cpu-nodegroup.sh --up first)
#   - the ServiceMonitor CRD exists (kube-prometheus-stack in `monitoring`)
#
# Usage: ./provision.sh [--profile <aws-profile>]   (the lab account needs --profile lab)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLUSTER_NAME=vllm-bench
AWS_REGION=us-east-2
NG=llm-serving-c8i
NODE_ROLE=llm-serving-cpu
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

while [[ $# -gt 0 ]]; do
  case $1 in
    --profile) export AWS_PROFILE="$2"; shift 2 ;;
    --help|-h) echo "Usage: $0 [--profile <aws-profile>]"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
for cmd in kubectl aws; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' not found in PATH"; exit 1; }
done
echo "Identity: $(aws sts get-caller-identity --query Arn --output text)"

echo "[1/5] Existing cluster $CLUSTER_NAME"
aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query cluster.status --output text >/dev/null || {
  echo "  cluster $CLUSTER_NAME not found in $AWS_REGION (this study does not create it)" >&2; exit 1; }
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"
for sc in gp3 gp3-ephemeral; do
  kubectl get storageclass "$sc" >/dev/null 2>&1 || {
    echo "  StorageClass $sc missing: apply studies/30-l40s-gemma4-26b-tps/infra/eks/storageclass.yaml and k8s-bootstrap/01-storage-classes.yaml" >&2; exit 1; }
done

echo "[2/5] Namespace llm-cpu"
kubectl apply -f "$SCRIPT_DIR/k8s-bootstrap/00-namespaces.yaml"

echo "[3/5] CPU node group $NG"
"$SCRIPT_DIR/eks/cpu-nodegroup.sh"
DESIRED=$(aws eks describe-nodegroup --cluster-name "$CLUSTER_NAME" --region "$AWS_REGION" \
  --nodegroup-name $NG --query 'nodegroup.scalingConfig.desiredSize' --output text)
if [ "$DESIRED" = 0 ]; then
  echo "  node group is at 0 (paused): run ./eks/cpu-nodegroup.sh --up --always-on, then re-run" >&2; exit 1
fi
kubectl wait --for=condition=Ready node -l node-role=$NODE_ROLE --timeout=900s

echo "[4/5] Study PVCs, Services"
# An explicit list, never a glob of k8s/: the *_template.yaml files there are rendered by
# the workflow scripts, not applied as they are.
REQUIRED=("$STUDY_ROOT/k8s/00-pvc.yaml" "$STUDY_ROOT/k8s/02-service.yaml")
OPTIONAL=("$STUDY_ROOT/k8s/06-hf-cache-pvc.yaml")
for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] || { echo "  required manifest missing: ${f#"$STUDY_ROOT"/}" >&2; exit 1; }
  kubectl apply -f "$f"
done
for f in "${OPTIONAL[@]}"; do
  if [ -f "$f" ]; then kubectl apply -f "$f"; else echo "  (no ${f#"$STUDY_ROOT"/}, skipped)"; fi
done

echo "[5/5] ServiceMonitor"
kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1 || {
  echo "  kube-prometheus-stack is not installed (no ServiceMonitor CRD): install it in namespace monitoring, then re-run" >&2; exit 1; }
SM="$STUDY_ROOT/k8s/monitoring/servicemonitor.yaml"
[ -f "$SM" ] || { echo "  required manifest missing: k8s/monitoring/servicemonitor.yaml" >&2; exit 1; }
kubectl apply -f "$SM"

cat <<'NOTE'

Done. Next (study README, runbook):
  - Startup probe (probe/): checks the exclusive cpuset, lscpu / NUMA, the KV cache size
    vLLM reports and its OMP binding lines.
  - Smoke study (calibrates RT_RATE / RT_RAMP_S), then the study.
NOTE
kubectl get nodes -L node-role
