#!/bin/bash
# RunTest step for 25-g7-4500-gpu-sharing-goodput: re-create the AIPerf Job and watch it.
#
# Guards ported from study 24's run_test_tps.sh (audit 2026-09-29), adapted to one or two
# vLLM replicas. A plain `kubectl wait --for=condition=complete` had two holes: it never
# returns early on a Failed Job (every failed trial burned the full 90 min), and a replica
# that crashed and restarted mid-ramp still let the experiment score VALID — study 9's
# failure, where the stability window landed before the crash. Now the trial fails as
# soon as the Job fails, a vLLM pod is replaced or restarts, or no request completes for
# too long.
set -e
K8S=${K8S:-/work/vllm-benchmark/studies/25-g7-4500-gpu-sharing-goodput/k8s}   # overridable for manual tests
BENCH_FILE=$K8S/05-job.yaml
NS=gpu-sharing
JOB=aiperf-gpu-sharing

pods_state() {  # "name:uid:restarts" for every vLLM replica, sorted
  kubectl get pods -n "$NS" -l app=vllm \
    -o jsonpath='{range .items[*]}{.metadata.name}:{.metadata.uid}:{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort
}
successes() {  # requests completed so far, summed over the replicas (empty if unreadable)
  local total=0 n p
  for p in $(kubectl get pods -n "$NS" -l app=vllm -o name 2>/dev/null); do
    n=$(kubectl exec -n "$NS" "$p" -c vllm -- python3 -c \
      "import urllib.request;print(int(sum(float(l.split()[-1]) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith('vllm:request_success_total'))))" 2>/dev/null) || return 0
    [[ "$n" =~ ^[0-9]+$ ]] || return 0
    total=$((total + n))
  done
  echo "$total"
}

STATE_BEFORE=$(pods_state)
echo "vLLM replicas before the test:"; echo "${STATE_BEFORE:-none}" | sed 's/^/  /'
[ -n "$STATE_BEFORE" ] || { echo "error: no vLLM replica is running — Apply config did not leave a server up" >&2; exit 2; }

kubectl delete -f "$BENCH_FILE" --ignore-not-found ; kubectl apply -f "$BENCH_FILE"

# Budget (DEADLINE_S 5400 = 90 min): ~2 min pip install, up to ~10 min one-time ShareGPT
# prep on the first experiment, 60 s warm-up, 12 x 300 s ramp = 60 min, per-level
# overhead. It MUST stay below the Akamas RunTest task timeout (105m in the workflow),
# or the log dump below never runs.
POLL_S=15
DEADLINE_S=5400
STALL_S=900          # no new completion for 15 min once requests have started completing
FIRST_OK_S=1500      # nothing completed 25 min after the start (covers pip + prep + warm-up)
PROGRESS_EVERY_S=60
set +e
FAIL_REASON=""
T0=$SECONDS; LAST_OK=""; LAST_PROGRESS=$SECONDS; NEXT_PROGRESS=$SECONDS
while true; do
  SUCC=$(kubectl get job "$JOB" -n "$NS" -o jsonpath='{.status.succeeded}' 2>/dev/null)
  FAILED=$(kubectl get job "$JOB" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null)
  if [ "${SUCC:-0}" -ge 1 ]; then break; fi
  if [ "${FAILED:-0}" -ge 1 ]; then FAIL_REASON="the AIPerf job failed"; break; fi
  STATE_NOW=$(pods_state)
  if [ "$STATE_NOW" != "$STATE_BEFORE" ]; then
    FAIL_REASON="a vLLM replica restarted or was replaced during the test (before: $(echo $STATE_BEFORE) / now: $(echo ${STATE_NOW:-none}))"; break
  fi
  if [ "$SECONDS" -ge "$NEXT_PROGRESS" ]; then
    NEXT_PROGRESS=$((SECONDS + PROGRESS_EVERY_S))
    OK=$(successes)
    if [[ "$OK" =~ ^[0-9]+$ ]]; then   # skip the check when a replica cannot be read
      if [ "$OK" != "$LAST_OK" ]; then LAST_OK=$OK; LAST_PROGRESS=$SECONDS; fi
      if [ "$OK" -gt 0 ] && [ $((SECONDS - LAST_PROGRESS)) -ge "$STALL_S" ]; then
        FAIL_REASON="stalled: no request completed for $((SECONDS - LAST_PROGRESS)) s (at $OK completed)"; break
      fi
      if [ "$OK" -eq 0 ] && [ $((SECONDS - T0)) -ge "$FIRST_OK_S" ]; then
        FAIL_REASON="stalled: no request completed $((SECONDS - T0)) s after the start"; break
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
# Full job logs (init + main container), unconditionally: a failed run is as debuggable
# from the Akamas task output as a passing one.
echo "--- job/$JOB: full logs ---"
kubectl logs "job/$JOB" -n "$NS" --all-containers --tail=-1 || true
if [ -n "$FAIL_REASON" ]; then
  echo "--- vLLM replicas at failure ---"
  kubectl get pods -n "$NS" -l app=vllm -o wide || true
  for p in $(kubectl get pods -n "$NS" -l app=vllm -o name 2>/dev/null); do
    echo "--- $p: last 200 lines (current and previous container) ---"
    kubectl logs -n "$NS" "$p" --tail=200 || true
    kubectl logs -n "$NS" "$p" --tail=200 --previous 2>/dev/null || true
  done
  # Stop AIPerf hammering a dead or stalled server.
  kubectl delete job "$JOB" -n "$NS" --ignore-not-found --wait=false || true
  exit 1
fi
