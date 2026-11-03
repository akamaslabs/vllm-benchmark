#!/bin/bash
# Usage: ./vllm-up.sh <claim-template> <replicas> [gmu=0.90] [max_num_seqs=256]
# CPU / memory per replica = what is free on the GPU node (allocatable minus a 400m /
# 6 GiB margin for DaemonSets), split evenly. NODE_ROLE picks the node group.
set -euo pipefail
cd "$(dirname "$0")"
CLAIM=$1; N=$2; GMU=${3:-0.90}; SEQS=${4:-256}
NODE_ROLE=${NODE_ROLE:-llm-serving-g7e-4xl}
read -r ACPU AMEM < <(kubectl get node -l node-role=$NODE_ROLE -o jsonpath='{.items[0].status.allocatable.cpu} {.items[0].status.allocatable.memory}{"\n"}')
ACPU=${ACPU%m}; AMEM_GI=$(( ${AMEM%Ki} / 1048576 ))
CPU=$(( (ACPU - 400) / N )); MEM=$(( (AMEM_GI - 6) / N ))
kubectl -n dra-mig delete statefulset vllm --ignore-not-found --wait=true
kubectl -n dra-mig wait --for=delete pod -l app=vllm --timeout=300s 2>/dev/null || true
sed -e "s/@CLAIM@/$CLAIM/" -e "s/@REPLICAS@/$N/" -e "s/@GMU@/$GMU/" -e "s/@MAX_NUM_SEQS@/$SEQS/" \
    -e "s/@CPU@/$CPU/" -e "s/@MEM@/$MEM/" -e "s/@NODE_ROLE@/$NODE_ROLE/" 10-vllm-statefulset.yaml.templ | kubectl apply -f -
t0=$(date +%s)
kubectl -n dra-mig rollout status statefulset/vllm --timeout=40m
echo "all $N replicas Ready after $(( $(date +%s) - t0 )) s (cpu ${CPU}m, mem ${MEM}Gi each)"
for i in $(seq 0 $((N-1))); do
  echo "== vllm-$i"; kubectl -n dra-mig logs vllm-$i | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency' | sed 's/^.*\] //'
done
