#!/bin/bash
# Load-test step for study 18 (runs on the toolbox host via the Akamas Executor). Re-applies
# the AIPerf Job, waits for it, dumps its logs, dumps the router's and every instance's view
# of the KV transfer, then FAILS THE TRIAL IF ANY CONTAINER OF THE SERVING POD RESTARTED
# DURING THE TEST. This is the same guard as studies 13-16: study 9's "best" experiment
# crash-looped after minute 43 and still scored VALID.
#
# 2026-09-29: the wait fails fast. It used to be `kubectl wait --for=condition=complete`,
# which never returns early on a job that FAILED. So a trial already dead waited the full
# 80 minutes (study 23 exp 12: prefill CUDA OOM at 07:33, engine crash loop, AIPerf OOMKilled
# at 07:36, step failed by timeout at ~08:28). The loop below ends the test as soon as one of
# these happens, then dumps every log, deletes the job and fails the trial:
#   - the AIPerf job failed;
#   - a serving-pod container restarted (engine or router);
#   - the serving pod was replaced (its restart counters would start again from 0);
#   - no request completed at the router for 15 min after the first one (a stall, e.g. study
#     22's push + cuda-buffer trials, where the KV transfers took 10-150 s), or none at all
#     within 25 min of the start (pip install + dataset generation take ~5-8 min);
#   - the 80-min deadline passed.
# Poll interval and stall thresholds can be overridden by env (RT_*) for testing only.
set -e
K8S=/work/vllm-benchmark/studies/20-l4-pd-disaggregation-tuned/k8s
BENCH_FILE=$K8S/05-job.yaml
NS=llm-serving

restarts() {  # total restarts of engine + router in the current vllm-pd pod
  kubectl get pod -n "$NS" -l app=vllm-pd \
    -o jsonpath='{range .items[0].status.containerStatuses[*]}{.restartCount}{"\n"}{end}' 2>/dev/null \
    | awk '{s+=$1} END {print s+0}'
}
serving_pod() {
  kubectl get pod -n "$NS" -l app=vllm-pd -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}
router_successes() {  # requests the router completed so far (empty if unreachable)
  kubectl exec deployment/vllm-pd -n "$NS" -c router -- python3 -c \
    "import urllib.request;print(int(sum(float(l.split()[-1]) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith('vllm:request_success_total'))))" 2>/dev/null
}
fail_job_cleanup() {  # stop AIPerf hammering a dead or stalled server; logs are dumped already
  kubectl delete job aiperf-benchmark -n llm-benchmark --ignore-not-found --wait=false || true
}
RESTARTS_BEFORE=$(restarts)
POD_BEFORE=$(serving_pod)
echo "serving pod before the test: ${POD_BEFORE:-none}, restarts: $RESTARTS_BEFORE"

kubectl delete -f "$BENCH_FILE" --ignore-not-found ; kubectl apply -f "$BENCH_FILE"

# Deadline 4800 s (80 min): 6 x 600 s levels (60 min) + pip install + synthetic dataset
# generation + per-level warm-up. The workflow task's own timeout (95 min) must stay
# ABOVE this, or Akamas kills the task before these logs are dumped.
POLL_S=${RT_POLL_S:-15}
DEADLINE_S=${RT_DEADLINE_S:-4800}
STALL_S=${RT_STALL_S:-900}          # no new completion for this long, after the first one
FIRST_OK_S=${RT_FIRST_OK_S:-1500}   # no completion at all this long after the start
PROGRESS_EVERY_S=${RT_PROGRESS_EVERY_S:-60}
set +e
FAIL_REASON=""
T0=$SECONDS; LAST_OK=""; LAST_PROGRESS=$SECONDS; NEXT_PROGRESS=$SECONDS
while true; do
  SUCC=$(kubectl get job aiperf-benchmark -n llm-benchmark -o jsonpath='{.status.succeeded}' 2>/dev/null)
  FAILED=$(kubectl get job aiperf-benchmark -n llm-benchmark -o jsonpath='{.status.failed}' 2>/dev/null)
  if [ "${SUCC:-0}" -ge 1 ]; then break; fi
  if [ "${FAILED:-0}" -ge 1 ]; then FAIL_REASON="the AIPerf job failed"; break; fi
  POD_NOW=$(serving_pod)
  if [ -n "$POD_BEFORE" ] && [ "$POD_NOW" != "$POD_BEFORE" ]; then
    FAIL_REASON="the serving pod was replaced during the test ($POD_BEFORE -> ${POD_NOW:-none})"; break
  fi
  RESTARTS_NOW=$(restarts)
  if [ "$RESTARTS_NOW" -gt "$RESTARTS_BEFORE" ]; then
    FAIL_REASON="the serving pod restarted during the test ($RESTARTS_BEFORE -> $RESTARTS_NOW restarts)"; break
  fi
  if [ "$SECONDS" -ge "$NEXT_PROGRESS" ]; then
    NEXT_PROGRESS=$((SECONDS + PROGRESS_EVERY_S))
    OK=$(router_successes)
    if [[ "$OK" =~ ^[0-9]+$ ]]; then   # skip the check when the router cannot be read
      if [ "$OK" != "$LAST_OK" ]; then LAST_OK=$OK; LAST_PROGRESS=$SECONDS; fi
      if [ "$OK" -gt 0 ] && [ $((SECONDS - LAST_PROGRESS)) -ge "$STALL_S" ]; then
        FAIL_REASON="stalled: no request completed at the router for $((SECONDS - LAST_PROGRESS)) s (at $OK completed)"; break
      fi
      if [ "$OK" -eq 0 ] && [ $((SECONDS - T0)) -ge "$FIRST_OK_S" ]; then
        FAIL_REASON="stalled: no request completed at the router $((SECONDS - T0)) s after the start"; break
      fi
    fi
  fi
  if [ $((SECONDS - T0)) -ge "$DEADLINE_S" ]; then FAIL_REASON="test timeout (${DEADLINE_S} s)"; break; fi
  sleep "$POLL_S"
done
set -e
if [ -n "$FAIL_REASON" ]; then
  echo "error: $FAIL_REASON, after $((SECONDS - T0)) s. Dumping logs, then failing the trial."
else
  echo "AIPerf job completed after $((SECONDS - T0)) s."
fi

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
  fail_job_cleanup
  exit 1
fi

if [ -n "$FAIL_REASON" ]; then
  echo "--- serving pod status and recent events ---"
  kubectl get pod -n "$NS" -l app=vllm-pd -o wide || true
  kubectl get events -n "$NS" --sort-by=.lastTimestamp 2>/dev/null | tail -15 || true
  kubectl get pod -n llm-benchmark -l job-name=aiperf-benchmark \
    -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.containerStatuses[*].state}{"\n"}{end}' 2>/dev/null || true
  echo "FAILING THE TRIAL: $FAIL_REASON."
  fail_job_cleanup
  exit 1
fi
exit 0
