#!/bin/bash
# Phase 0 spike: start N probe pods, each with its own claim from TEMPLATE, print what each
# got, then (with --keep absent) delete them and time how long the claims take to go.
# Usage: ./probe-pods.sh <template> <n> [--keep]
#   e.g. ./probe-pods.sh mig-1g24gb 4 ; ./probe-pods.sh mig-2g48gb 2 ; ./probe-pods.sh gpu-full 1
set -euo pipefail
T=$1; N=$2; KEEP=${3:-}
NS=dra-mig
t0=$(date +%s)
for i in $(seq 0 $((N-1))); do
kubectl apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata: {namespace: $NS, name: probe-$T-$i, labels: {app: dra-probe}}
spec:
  restartPolicy: Never
  nodeSelector: {node-role: llm-serving-g7e-2xl}
  tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
  containers:
  - name: probe
    image: ubuntu:22.04
    command: ["bash","-c","nvidia-smi -L; nvidia-smi --query-gpu=memory.total --format=csv,noheader; sleep 3600"]
    resources: {claims: [{name: gpu}]}
  resourceClaims: [{name: gpu, resourceClaimTemplateName: $T}]
YAML
done
kubectl -n $NS wait --for=condition=Ready pod -l app=dra-probe --timeout=600s
echo "ready after $(( $(date +%s) - t0 )) s"
for i in $(seq 0 $((N-1))); do echo "== probe-$T-$i"; kubectl -n $NS logs probe-$T-$i; done
kubectl -n $NS get resourceclaims -o custom-columns=NAME:.metadata.name,DEV:.status.allocation.devices.results[*].device,STATE:.status.reservedFor[*].name
if [ "$KEEP" != --keep ]; then
  t1=$(date +%s)
  kubectl -n $NS delete pod -l app=dra-probe --wait=true
  while kubectl -n $NS get resourceclaims -o name 2>/dev/null | grep -q .; do sleep 2; done
  echo "pods + claims gone after $(( $(date +%s) - t1 )) s"
fi
