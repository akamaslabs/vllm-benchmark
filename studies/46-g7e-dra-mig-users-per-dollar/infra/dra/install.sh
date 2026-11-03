#!/bin/bash
# Installs the NVIDIA DRA driver for study 46 and keeps the classic device plugin off the
# DRA nodes. A DRA driver and a device plugin must never manage the same GPU.
#
# 1. kube-system/nvidia-device-plugin-daemonset (static manifest v0.18.0) excludes node
#    groups through a NotIn list on eks.amazonaws.com/nodegroup: add this study's GPU node
#    groups (llm-serving-g7e-2xl used in Phase 0, llm-serving-g7e-4xl for the study).
#    Only adds to the list, so no other node group changes.
#    (nvdp-g7 in gpu-sharing is pinned to node-role=llm-serving-g7-4500: nothing to do.)
# 2. helm upgrade --install dra-driver-nvidia-gpu 0.5.1 with ./values.yaml.
#
# Usage: ./install.sh             (idempotent)
#        ./install.sh --uninstall (removes the Helm release; leaves the exclusions in place)
set -euo pipefail
cd "$(dirname "$0")"
NGS="llm-serving-g7e-2xl llm-serving-g7e-4xl"
NS=dra-driver-nvidia-gpu
CHART=oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu
VERSION=0.5.1

if [ "${1:-}" = --uninstall ]; then
  helm uninstall dra-driver-nvidia-gpu -n $NS
  exit 0
fi

vals=$(kubectl -n kube-system get ds nvidia-device-plugin-daemonset -o json | python3 -c '
import json,sys
d=json.load(sys.stdin)
for t in d["spec"]["template"]["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]:
  for e in t["matchExpressions"]:
    if e["key"]=="eks.amazonaws.com/nodegroup" and e["operator"]=="NotIn": print(" ".join(e["values"]))')
for NG in $NGS; do
  if [[ " $vals " == *" $NG "* ]]; then
    echo "device plugin already excludes $NG"
  else
    kubectl -n kube-system patch ds nvidia-device-plugin-daemonset --type=json -p "[{\"op\":\"add\",\"path\":\"/spec/template/spec/affinity/nodeAffinity/requiredDuringSchedulingIgnoredDuringExecution/nodeSelectorTerms/0/matchExpressions/0/values/-\",\"value\":\"$NG\"}]"
    echo "device plugin now excludes $NG"
  fi
done

helm upgrade --install dra-driver-nvidia-gpu $CHART --version $VERSION \
  --namespace $NS --create-namespace -f values.yaml
kubectl -n $NS get pods -o wide
