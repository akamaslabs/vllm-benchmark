#!/bin/bash
# RunTest step for 41-l4-dsv2-lite-context-parallel (toolbox, Akamas Executor task).
#
# One AIPerf Job against vllm-dsv2-0: 60 s warm-up, then a closed-loop users ramp, 1 -> RT_USERS
# concurrent sessions over RT_RAMP_S s (render_job.sh), of synthetic multi-turn long-context
# sessions with 8k-token shared prefixes (lib_workload.sh).
# TODO(calibrate): RT_USERS=96 / RT_RAMP_S=3600 are placeholders until the calibration run
# (README "Before starting": one TP4 baseline + one TP4+DCP4 run) places the knees; the
# workflow command sets both explicitly.
# The trial FAILS (exit 1) if the Job fails, the vLLM pod restarts or is replaced,
# vllm-dsv2-0 completes nothing for RT_STALL_S, or the deadline passes (study 27/28/30/39
# guards). The watchdog ENDS the test with SUCCESS (exit 0) once vllm-dsv2-0's TTFT p95
# (150 s) > RT_WD_TTFT_MS or TPOT p95 (150 s) > RT_WD_TPOT_MS for RT_WD_HOLD_S: in a closed
# loop more users past the knee only queue longer, every later window is invalid anyway, and
# Akamas scores the best valid window before it. Defaults = 2 x the study's SLO (TTFT p95 <=
# 10000 ms, TPOT p95 <= 50 ms), so the run goes a little past the breach and the scored
# window before it is complete. TPOT is vllm:request_time_per_output_token_seconds, the
# histogram behind the constraint metric vllm.time_per_output_token_p95 (present in vLLM
# 0.32.0, v1/metrics/loggers.py); it is observed when a request completes, so it lags the
# load by about one request. Armed RT_WD_ARM_DELAY_S after the measured run starts.
set -euo pipefail
K8S=${K8S:-/work/vllm-benchmark/studies/41-l4-dsv2-lite-context-parallel/k8s}
NS=llm-serving
APP=vllm-dsv2
POD=vllm-dsv2-0
MODEL=dsv2-lite
USERS=${RT_USERS:-96}
RAMP_S=${RT_RAMP_S:-3600}
# pip (~2 min) + synthetic dataset generation (4000 long sessions: to be measured by the
# calibration run, budgeted ~15 min) + 60 s warm-up + the ramp + grace. The workflow's
# RunTest timeout must stay above it, or Akamas kills the task before the log dump.
DEADLINE_S=${RT_DEADLINE_S:-$((RAMP_S + 2400))}
WD_TTFT_MS=${RT_WD_TTFT_MS:-20000}; WD_TPOT_MS=${RT_WD_TPOT_MS:-100}
WD_HOLD_S=${RT_WD_HOLD_S:-120}; WD_ARM_DELAY_S=${RT_WD_ARM_DELAY_S:-150}
POLL_S=${RT_POLL_S:-15}; STALL_S=${RT_STALL_S:-900}; FIRST_OK_S=${RT_FIRST_OK_S:-2100}
PROGRESS_EVERY_S=${RT_PROGRESS_EVERY_S:-60}
PROM=${RT_PROM:-http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090}
TMP=$(mktemp -d /tmp/run-test-41-XXXXXX)
JOB=aiperf-dsv2
# shellcheck source=lib_watchdog.sh
source "$K8S/lib_watchdog.sh"

pod_state() {  # "name:uid:restarts" of the vLLM pod
  kubectl get pods -n "$NS" -l app=$APP \
    -o jsonpath='{range .items[*]}{.metadata.name}:{.metadata.uid}:{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort
}
successes() {  # requests vllm-dsv2-0 completed so far, summed over its DP engines (empty if unreadable)
  kubectl exec -n "$NS" $POD -c vllm -- python3 -c \
    "import urllib.request;print(int(sum(float(l.split()[-1]) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith('vllm:request_success_total'))))" 2>/dev/null || true
}
p95_ms() {  # $1 vLLM histogram name: vllm-dsv2-0's p95 over 150 s in ms (empty if no data)
  python3 - "$PROM" "$1" "$MODEL" "$POD" <<'EOF' 2>/dev/null || true
import json, sys, urllib.parse, urllib.request
prom, h, model, pod = sys.argv[1:5]
q = ('histogram_quantile(0.95, sum by(le)(rate(vllm:%s_bucket{model_name="%s",pod="%s"}[150s])))*1000'
     % (h, model, pod))
r = json.load(urllib.request.urlopen(prom + '/api/v1/query?' + urllib.parse.urlencode({'query': q}), timeout=10))
v = r['data']['result']
if v and v[0]['value'][1] not in ('NaN', '+Inf'):
    print(int(float(v[0]['value'][1])))
EOF
}

READY=$(kubectl -n "$NS" get sts $APP -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)
if [ "${READY:-0}" != 1 ]; then
  echo "error: $POD is not ready (readyReplicas=${READY:-0}): Apply config did not leave it up" >&2
  exit 2
fi
STATE_BEFORE=$(pod_state)
echo "vLLM pod before the test: $STATE_BEFORE"
echo "users ramp 1 -> $USERS over $RAMP_S s; watchdog: TTFT p95 > $WD_TTFT_MS ms or TPOT p95 > $WD_TPOT_MS ms for $WD_HOLD_S s, armed $WD_ARM_DELAY_S s after the measured run starts; deadline $DEADLINE_S s"

# A Job left over from a stopped study would keep loading vLLM (study 29, exp 16).
kubectl -n "$NS" delete job -l app=$JOB --ignore-not-found --wait=true
bash "$K8S/render_job.sh" "$USERS" "$RAMP_S" "$TMP/job.yaml"
kubectl apply -f "$TMP/job.yaml"

set +e
FAIL_REASON=""; WATCHDOG=""
T0=$SECONDS; LAST_OK=""; LAST_PROGRESS=$SECONDS; NEXT_PROGRESS=$SECONDS; MARKER_AT=""; OVER_SINCE=""
while true; do
  S=$(kubectl get job "$JOB" -n "$NS" -o jsonpath='{.status.succeeded}' 2>/dev/null)
  F=$(kubectl get job "$JOB" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null)
  [ "${F:-0}" -ge 1 ] && { FAIL_REASON="the AIPerf job failed"; break; }
  [ "${S:-0}" -ge 1 ] && break
  STATE_NOW=$(pod_state)
  if [ "$STATE_NOW" != "$STATE_BEFORE" ]; then
    FAIL_REASON="the vLLM pod restarted or was replaced during the test (before: $STATE_BEFORE / now: ${STATE_NOW:-none})"; break
  fi
  if [ "$SECONDS" -ge "$NEXT_PROGRESS" ]; then
    NEXT_PROGRESS=$((SECONDS + PROGRESS_EVERY_S))
    OK=$(successes)
    if [[ "$OK" =~ ^[0-9]+$ ]]; then
      if [ "$OK" != "$LAST_OK" ]; then LAST_OK=$OK; LAST_PROGRESS=$SECONDS; fi
      if [ "$OK" -gt 0 ] && [ $((SECONDS - LAST_PROGRESS)) -ge "$STALL_S" ]; then
        FAIL_REASON="stalled: $POD completed no request for $((SECONDS - LAST_PROGRESS)) s (at $OK)"; break
      fi
      if [ "$OK" -eq 0 ] && [ $((SECONDS - T0)) -ge "$FIRST_OK_S" ]; then
        FAIL_REASON="stalled: $POD completed no request $((SECONDS - T0)) s after the start"; break
      fi
    fi
  fi
  # --- Watchdog ---
  if [ -z "$MARKER_AT" ] && wd_armed "$(kubectl logs "job/$JOB" -n "$NS" -c aiperf --tail=400 2>/dev/null)"; then
    MARKER_AT=$SECONDS
    echo "$(date -u +%T) measured run started; watchdog armed in $WD_ARM_DELAY_S s"
  fi
  if wd_ready "$SECONDS" "$MARKER_AT"; then
    TT=$(p95_ms time_to_first_token_seconds); TP=$(p95_ms request_time_per_output_token_seconds)
    if wd_over "$TT" "$TP"; then NOW_OVER=0; else NOW_OVER=1; fi
    NEW_SINCE=$(wd_next_since "$SECONDS" "$OVER_SINCE" "$NOW_OVER")
    if [ -z "$OVER_SINCE" ] && [ -n "$NEW_SINCE" ]; then echo "$(date -u +%T) watchdog: over (TTFT p95 ${TT:-n/a} ms, TPOT p95 ${TP:-n/a} ms)"; fi
    if [ -n "$OVER_SINCE" ] && [ -z "$NEW_SINCE" ]; then echo "$(date -u +%T) watchdog: back under (TTFT p95 ${TT:-n/a} ms, TPOT p95 ${TP:-n/a} ms)"; fi
    OVER_SINCE=$NEW_SINCE
    if wd_fired "$SECONDS" "$OVER_SINCE"; then
      WATCHDOG="TTFT p95 ${TT:-n/a} ms, TPOT p95 ${TP:-n/a} ms for $((SECONDS - OVER_SINCE)) s"; break
    fi
  fi
  if [ $((SECONDS - T0)) -ge "$DEADLINE_S" ]; then FAIL_REASON="test timeout (${DEADLINE_S} s)"; break; fi
  sleep "$POLL_S"
done
set -e

if [ -n "$FAIL_REASON" ]; then
  echo "error: $FAIL_REASON, after $((SECONDS - T0)) s. Dumping logs, then failing the trial."
elif [ -n "$WATCHDOG" ]; then
  echo "Watchdog ended the test after $((SECONDS - T0)) s: $WATCHDOG. The test is complete (Akamas scores the best valid window before it)."
else
  echo "AIPerf job completed after $((SECONDS - T0)) s (the ramp reached $USERS users without breaking 2 x the SLO: raise RT_USERS)."
fi
echo "--- job/$JOB: full logs ---"
kubectl logs "job/$JOB" -n "$NS" --all-containers --tail=-1 || true
if [ -n "$FAIL_REASON" ]; then
  echo "--- vLLM pod at failure ---"
  kubectl get pods -n "$NS" -l app=$APP -o wide || true
  kubectl logs -n "$NS" $POD --tail=200 || true
  kubectl logs -n "$NS" $POD --tail=200 --previous 2>/dev/null || true
  kubectl -n "$NS" delete job -l app=$JOB --ignore-not-found --wait=false || true
  rm -rf "$TMP"; exit 1
fi
if [ -n "$WATCHDOG" ]; then  # logs dumped: now stop the load
  kubectl -n "$NS" delete job -l app=$JOB --ignore-not-found --wait=false || true
fi
rm -rf "$TMP"
exit 0
