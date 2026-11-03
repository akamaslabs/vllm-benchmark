#!/bin/bash
# Usage: ./vllm-up.sh <claim-template> <replicas> [gmu=0.90] [max_num_seqs=256]
# CPU / memory per replica = what is free on the 8-vCPU / 64 GiB node, split evenly
# (7600m and 52 GiB in total, measured 2026-11-03 with only DaemonSets on the node).
set -euo pipefail
cd "$(dirname "$0")"
CLAIM=$1; N=$2; GMU=${3:-0.90}; SEQS=${4:-256}
CPU=$((7600 / N)); MEM=$((52 / N))
kubectl -n dra-mig delete statefulset vllm --ignore-not-found --wait=true
kubectl -n dra-mig wait --for=delete pod -l app=vllm --timeout=300s 2>/dev/null || true
sed -e "s/@CLAIM@/$CLAIM/" -e "s/@REPLICAS@/$N/" -e "s/@GMU@/$GMU/" -e "s/@MAX_NUM_SEQS@/$SEQS/" \
    -e "s/@CPU@/$CPU/" -e "s/@MEM@/$MEM/" 10-vllm-statefulset.yaml.templ | kubectl apply -f -
t0=$(date +%s)
kubectl -n dra-mig rollout status statefulset/vllm --timeout=40m
echo "all $N replicas Ready after $(( $(date +%s) - t0 )) s (cpu ${CPU}m, mem ${MEM}Gi each)"
for i in $(seq 0 $((N-1))); do
  echo "== vllm-$i"; kubectl -n dra-mig logs vllm-$i | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency' | sed 's/^.*\] //'
done
