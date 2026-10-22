#!/bin/bash
# Provision what 41-l4-dsv2-lite-context-parallel needs on the EXISTING vllm-bench cluster, up
# to `akamas create`. It never creates the cluster or the node group: the cluster, its
# system-2b / akamas node groups, the GPU node group llm-serving-l4, the gp3 / gp3-ephemeral
# StorageClasses, namespace llm-serving, kube-prometheus-stack, the NVIDIA device plugin and
# dcgm-exporter are shared and checked, not created. Kubernetes objects of this study are
# created only if MISSING (`kubectl get` first): the model cache PVC already exists (smoke run)
# and is never modified.
#
# Usage: ./provision.sh [--profile <aws-profile>] [--scale 0|1]
#   --scale 1  scales node group llm-serving-l4 to 1, waits for the node and checks its AZ
#              (must be us-east-2a, where the model PVC lives); --scale 0 scales it to 0 and stops.
# Idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLUSTER_NAME=vllm-bench
AWS_REGION=us-east-2
NG=llm-serving-l4
NODE_ROLE=llm-serving-l4
NS=llm-serving
MODEL_AZ=us-east-2a
SCALE=""
: "${AWS_PROFILE:=lab}"; export AWS_PROFILE

while [[ $# -gt 0 ]]; do
  case $1 in
    --profile) export AWS_PROFILE="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --help|-h) echo "Usage: $0 [--profile <aws-profile>] [--scale 0|1]"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
[[ -z "$SCALE" || "$SCALE" =~ ^[01]$ ]] || { echo "--scale takes 0 or 1" >&2; exit 1; }
for cmd in kubectl aws; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' not found in PATH"; exit 1; }
done
echo "Identity: $(aws sts get-caller-identity --query Arn --output text)"

echo "[1/6] Existing cluster $CLUSTER_NAME"
aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query cluster.status --output text >/dev/null || {
  echo "  cluster $CLUSTER_NAME not found in $AWS_REGION (this study does not create it)" >&2; exit 1; }
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"
for sc in gp3 gp3-ephemeral; do
  kubectl get storageclass "$sc" >/dev/null 2>&1 || {
    echo "  StorageClass $sc missing: apply studies/30-l40s-gemma4-26b-tps/infra/eks/storageclass.yaml and k8s-bootstrap/01-storage-classes.yaml" >&2; exit 1; }
done
kubectl get namespace "$NS" >/dev/null 2>&1 || { echo "  namespace $NS missing (shared with other studies; create it by hand)" >&2; exit 1; }

echo "[2/6] GPU node group $NG"
NG_JSON=$(aws eks describe-nodegroup --cluster-name "$CLUSTER_NAME" --region "$AWS_REGION" --nodegroup-name "$NG" \
  --query 'nodegroup.[resources.autoScalingGroups[0].name, scalingConfig.desiredSize, scalingConfig.maxSize]' --output text) || {
  echo "  node group $NG not found: it is shared (eksctl, studies 7-24); this study does not create it" >&2; exit 1; }
read -r ASG DESIRED MAX <<<"$NG_JSON"
echo "  desiredSize $DESIRED (max $MAX), ASG $ASG"
ALWAYS_ON=$(aws autoscaling describe-tags --region "$AWS_REGION" \
  --filters "Name=auto-scaling-group,Values=$ASG" "Name=key,Values=AlwaysOn" --query 'Tags[0].Value' --output text)
[ "$ALWAYS_ON" = true ] || echo "  WARNING: ASG $ASG has no AlwaysOn=true tag: the 17:00 UTC Lambda will stop the node (infra/README.md)"
if [ -n "$SCALE" ] && [ "$SCALE" != "$DESIRED" ]; then
  echo "  scaling $NG to $SCALE"
  aws eks update-nodegroup-config --cluster-name "$CLUSTER_NAME" --region "$AWS_REGION" --nodegroup-name "$NG" \
    --scaling-config "minSize=0,maxSize=1,desiredSize=$SCALE" >/dev/null
  DESIRED=$SCALE
fi
if [ "$DESIRED" = 0 ]; then
  echo "  node group is at 0 (paused): re-run with --scale 1 to bring the node up" >&2
  [ "$SCALE" = 0 ] && exit 0 || exit 1
fi
echo "  waiting for a Ready node with node-role=$NODE_ROLE"
for _ in $(seq 1 90); do
  kubectl get nodes -l node-role=$NODE_ROLE -o name 2>/dev/null | grep -q . && break; sleep 10
done
kubectl wait --for=condition=Ready node -l node-role=$NODE_ROLE --timeout=900s
NODE=$(kubectl get nodes -l node-role=$NODE_ROLE -o jsonpath='{.items[0].metadata.name}')
ZONE=$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
echo "  node $NODE in $ZONE, allocatable GPUs $(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
if [ "$ZONE" != "$MODEL_AZ" ]; then
  cat >&2 <<EOF
  ERROR: the node is in $ZONE, the model PVC vllm-model-cache-dsv2lite is an EBS volume in $MODEL_AZ.
  Replace the instance until it lands in $MODEL_AZ (infra/README.md, "Wrong AZ"):
    aws autoscaling terminate-instance-in-auto-scaling-group --region $AWS_REGION \\
      --instance-id <id> --no-should-decrement-desired-capacity
EOF
  exit 1
fi

echo "[3/6] Model cache PVC (must already exist, bound in $MODEL_AZ)"
if kubectl -n "$NS" get pvc vllm-model-cache-dsv2lite >/dev/null 2>&1; then
  PV=$(kubectl -n "$NS" get pvc vllm-model-cache-dsv2lite -o jsonpath='{.spec.volumeName}')
  echo "  exists, volume $PV, zone $(kubectl get pv "$PV" -o jsonpath='{.spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].values[0]}' 2>/dev/null)"
else
  echo "  missing: step 4 creates it EMPTY (the first vLLM start downloads 31.4 GB; it binds in the node's AZ)"
fi

echo "[4/6] Study PVCs, Services (created only if missing)"
create_missing() {  # $1 manifest: kubectl create each object that does not exist yet
  local f=$1 out
  out=$(kubectl create -f "$f" 2>&1) && { echo "$out"; return 0; }
  # Already existing objects are left untouched; anything else is a real error.
  # kubectl create goes on past an existing object, so the output can mix "created" lines
  # with AlreadyExists errors: only other errors are fatal.
  if echo "$out" | grep -i 'error' | grep -v 'AlreadyExists' | grep -q .; then echo "$out" >&2; return 1; fi
  echo "$out" | sed 's/^/  /'
}
# An explicit list, never a glob of k8s/: the *_template.yaml files there are rendered by
# the workflow scripts, not applied as they are.
for f in "$STUDY_ROOT/k8s/00-pvc.yaml" "$STUDY_ROOT/k8s/06-hf-cache-pvc.yaml" "$STUDY_ROOT/k8s/02-service.yaml"; do
  [ -f "$f" ] || { echo "  required manifest missing: ${f#"$STUDY_ROOT"/}" >&2; exit 1; }
  create_missing "$f"
done

echo "[5/6] ServiceMonitor"
kubectl get crd servicemonitors.monitoring.coreos.com >/dev/null 2>&1 || {
  echo "  kube-prometheus-stack is not installed (no ServiceMonitor CRD)" >&2; exit 1; }
create_missing "$STUDY_ROOT/k8s/monitoring/servicemonitor.yaml"

echo "[6/6] GPU stack on the node"
kubectl -n monitoring get pods -o wide 2>/dev/null | grep dcgm-exporter | grep "$NODE" >/dev/null \
  && echo "  dcgm-exporter running on $NODE" || echo "  WARNING: no dcgm-exporter pod on $NODE: the gpu0..gpu3 metrics will be empty"
OTHER=$(kubectl get pods -A --field-selector "spec.nodeName=$NODE,status.phase=Running" \
  -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}:{.spec.containers[*].resources.requests.nvidia\.com/gpu}{"\n"}{end}' \
  | grep -E ':[^:]*[1-9]' | grep -v "^$NS/vllm-dsv2-" || true)
[ -z "$OTHER" ] || echo "  WARNING: pods holding GPUs on $NODE (apply_config.sh will refuse to start): $OTHER"

cat <<'NOTE'

Done. Next (study README, "Before starting"):
  - Calibration run (Smoke study): TP4 baseline + TP4+DCP4, places RT_USERS / RT_RAMP_S and
    checks prefix caching with DCP.
  - Then the main study. Scale to 0 afterwards: ./provision.sh --scale 0
NOTE
kubectl get nodes -L node-role,topology.kubernetes.io/zone
