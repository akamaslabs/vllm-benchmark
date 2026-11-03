#!/bin/bash
# Phase 0 point 6: load every vLLM replica at once and read CPU throttling of the vLLM
# containers (cAdvisor via Prometheus) plus the AIPerf pods' CPU.
# Usage: ./cpu-check.sh <replicas> [concurrency=64] [duration_s=240]
set -euo pipefail
cd "$(dirname "$0")"
N=$1; CONC=${2:-64}; DUR=${3:-240}
kubectl -n dra-mig delete job -l app=aiperf --ignore-not-found --wait=true
for i in $(seq 0 $((N-1))); do
  sed -e "s/@REPLICA@/$i/g" -e "s/@CONC@/$CONC/" -e "s/@DUR@/$DUR/" 20-aiperf-job.yaml.templ | kubectl apply -f -
done
kubectl -n dra-mig wait --for=condition=complete job -l app=aiperf --timeout=$((DUR + 900))s
P="kubectl -n monitoring exec prometheus-kube-prometheus-stack-prometheus-0 -c prometheus -- promtool query instant http://localhost:9090"
W="$((DUR / 60 + 1))m"
echo "== vLLM throttled fraction of CFS periods (last $W)"
$P "sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{namespace=\"dra-mig\",container=\"vllm\"}[$W])) / sum by (pod) (rate(container_cpu_cfs_periods_total{namespace=\"dra-mig\",container=\"vllm\"}[$W]))"
echo "== vLLM CPU cores used"
$P "sum by (pod) (rate(container_cpu_usage_seconds_total{namespace=\"dra-mig\",container=\"vllm\"}[$W]))"
echo "== AIPerf CPU cores used"
$P "sum by (pod) (rate(container_cpu_usage_seconds_total{namespace=\"dra-mig\",container=\"aiperf\"}[$W]))"
for i in $(seq 0 $((N-1))); do
  echo "== aiperf-r$i"; kubectl -n dra-mig logs job/aiperf-r$i | grep -E 'Time to First Token|Inter Token Latency|Request Throughput|Output Token Throughput|Goodput' | head -8
done
