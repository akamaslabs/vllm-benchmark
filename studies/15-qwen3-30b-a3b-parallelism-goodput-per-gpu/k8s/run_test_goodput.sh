#!/bin/bash
# Load-test step for 15-qwen3-30b-a3b-parallelism-goodput-per-gpu (runs on the toolbox host via the
# Akamas Executor). Re-applies the AIPerf Job, waits for it, dumps its logs, then FAILS
# THE TRIAL IF vLLM RESTARTED DURING THE TEST — the guard study 9 lacked (its "best"
# experiment 64 had 8 restarts in CrashLoopBackOff after minute 43 and still scored VALID,
# because the stability window landed before the crash and nothing checked restarts; see
# ~/akamas/2026-09-14-study9-exp64-collasso-throughput.md).
set -e
BENCH_FILE=/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/k8s/05-job.yaml

# Restart count of the vLLM container BEFORE the test (should be 0 after a fresh rollout).
RESTARTS_BEFORE=$(kubectl get pod -n llm-serving -l app=vllm \
  -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="vllm")].restartCount}' 2>/dev/null || echo 0)
RESTARTS_BEFORE=${RESTARTS_BEFORE:-0}
echo "vLLM restartCount before the test: $RESTARTS_BEFORE"

kubectl delete -f "$BENCH_FILE" --ignore-not-found ; kubectl apply -f "$BENCH_FILE"

# --timeout=5700s (95m): up to 15 min one-time dataset prep on a cold cache + 12 x 300 s
# levels (60 min) + ~20 min buffer (pip install, per-level dataset generation).
set +e
kubectl wait --for=condition=complete job/aiperf-benchmark -n llm-benchmark --timeout=5700s
WAIT_EXIT=$?
set -e

echo "--- wait-for-vllm init container logs ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c wait-for-vllm --tail=-1 || true
echo "--- aiperf container logs (full) ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c aiperf --tail=-1 || true

# --- Restart guard ---
RESTARTS_AFTER=$(kubectl get pod -n llm-serving -l app=vllm \
  -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="vllm")].restartCount}' 2>/dev/null || echo 0)
RESTARTS_AFTER=${RESTARTS_AFTER:-0}
echo "vLLM restartCount after the test: $RESTARTS_AFTER"
if [ "$RESTARTS_AFTER" -gt "$RESTARTS_BEFORE" ]; then
  echo "--- vLLM RESTARTED $((RESTARTS_AFTER - RESTARTS_BEFORE)) time(s) during the test: previous container logs (the traceback study 9 never preserved) ---"
  kubectl logs deployment/vllm -n llm-serving -c vllm --previous --tail=400 || true
  echo "--- current container logs (tail) ---"
  kubectl logs deployment/vllm -n llm-serving -c vllm --tail=100 || true
  echo "--- pod status ---"
  kubectl get pod -n llm-serving -l app=vllm -o wide || true
  kubectl describe pod -n llm-serving -l app=vllm | grep -A12 'Last State' || true
  echo "FAILING THE TRIAL: a configuration that crashes under load is not a valid result."
  exit 1
fi

exit $WAIT_EXIT
