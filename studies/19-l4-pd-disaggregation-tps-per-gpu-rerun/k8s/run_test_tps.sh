#!/bin/bash
# Load-test step for study 18 (runs on the toolbox host via the Akamas Executor). Re-applies
# the AIPerf Job, waits for it, dumps its logs, dumps the router's and every instance's view
# of the KV transfer, then FAILS THE TRIAL IF ANY CONTAINER OF THE SERVING POD RESTARTED
# DURING THE TEST. This is the same guard as studies 13-16: study 9's "best" experiment
# crash-looped after minute 43 and still scored VALID.
set -e
K8S=/work/vllm-benchmark/studies/19-l4-pd-disaggregation-tps-per-gpu-rerun/k8s
BENCH_FILE=$K8S/05-job.yaml
NS=llm-serving

restarts() {  # total restarts of engine + router in the current vllm-pd pod
  kubectl get pod -n "$NS" -l app=vllm-pd \
    -o jsonpath='{range .items[0].status.containerStatuses[*]}{.restartCount}{"\n"}{end}' 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}
RESTARTS_BEFORE=$(restarts)
echo "serving pod restarts before the test: $RESTARTS_BEFORE"

kubectl delete -f "$BENCH_FILE" --ignore-not-found ; kubectl apply -f "$BENCH_FILE"

# --timeout=4800s (80 min): 6 x 600 s levels (60 min) + pip install + synthetic dataset
# generation + per-level warm-up. The workflow task's own timeout (95 min) must stay
# ABOVE this, or Akamas kills the task before these logs are dumped.
set +e
kubectl wait --for=condition=complete job/aiperf-benchmark -n llm-benchmark --timeout=4800s
WAIT_EXIT=$?
set -e

echo "--- wait-for-router init container logs ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c wait-for-router --tail=-1 || true
echo "--- aiperf container logs (full) ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c aiperf --tail=-1 || true

echo "--- router counters at the end of the test (requests, failures, delivered tokens) ---"
kubectl exec deployment/vllm-pd -n "$NS" -c router -- python3 -c \
  "import urllib.request;[print(l) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith(('vllm:request_success_total','pd_router_request_failures_total','vllm:prompt_tokens_total','vllm:generation_tokens_total'))]" 2>&1 || true
echo "--- vLLM periodic KV-transfer / throughput log lines (last 60) ---"
kubectl logs deployment/vllm-pd -n "$NS" -c engine --tail=5000 2>/dev/null \
  | grep -iE 'KV Transfer metrics|nixl|Avg prompt throughput|Running:|preempt|expired' | tail -60 || true

# --- Restart guard ---
RESTARTS_AFTER=$(restarts)
echo "serving pod restarts after the test: $RESTARTS_AFTER"
if [ "$RESTARTS_AFTER" -gt "$RESTARTS_BEFORE" ]; then
  echo "--- SERVING POD RESTARTED $((RESTARTS_AFTER - RESTARTS_BEFORE)) time(s) during the test: previous engine logs ---"
  kubectl logs deployment/vllm-pd -n "$NS" -c engine --previous --tail=400 || true
  echo "--- previous router logs ---"
  kubectl logs deployment/vllm-pd -n "$NS" -c router --previous --tail=100 || true
  echo "--- pod status ---"
  kubectl get pod -n "$NS" -l app=vllm-pd -o wide || true
  kubectl describe pod -n "$NS" -l app=vllm-pd | grep -A12 'Last State' || true
  echo "FAILING THE TRIAL: a configuration that crashes under load is not a valid result."
  exit 1
fi

exit $WAIT_EXIT
