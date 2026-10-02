#!/bin/bash
# RunTest step for 28-g7-4500-mig-min-cost-fixed-load (toolbox, Akamas Executor task).
#
# One AIPerf Job per vLLM replica, each on its own pod through the headless Service:
# vllm-0 is the tenant under test, vllm-1 (mig_profile 1g.16gb only) the busy neighbour
# with the same traffic. Modes (render_job.sh):
#   RT_MODE=fixed (the study)        60 s warm-up, then 780 s at RT_RATE (default 3.3 req/s)
#   RT_MODE=ramp  (calibration)      60 s warm-up, then 0 -> RT_RATE (default 12 req/s) over
#                                    RT_RAMP_S (default 2400 s)
# The trial FAILS (exit 1) if a Job fails, a vLLM pod restarts or is replaced, vllm-0
# completes nothing for RT_STALL_S, or the deadline passes (study 26/27 guards). The
# watchdog ENDS the test with SUCCESS (exit 0) once vllm-0's TTFT p95 (150 s) >
# RT_WD_TTFT_MS or ITL p95 (150 s) > RT_WD_ITL_MS for RT_WD_HOLD_S: the goal is then INVALID
# by its own constraints. It is armed RT_WD_ARM_DELAY_S after the measured run starts, so
# the warm-up is out of its 150 s view.
set -euo pipefail
K8S=${K8S:-/work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load/k8s}
NS=gpu-sharing
MODEL=qwen3-8b-mig
MODE=${RT_MODE:-fixed}
case "$MODE" in
  # Deadlines: pip (~2 min) + one-time ShareGPT prep (~10 min, first trial only) + 60 s
  # warm-up + the run + grace. The workflow's RunTest timeout must stay above them (45m /
  # 65m), or Akamas kills the task before the log dump.
  fixed) RATE=${RT_RATE:-3.3}; DEADLINE_S=${RT_DEADLINE_S:-2100} ;;
  ramp)  RATE=${RT_RATE:-12};  DEADLINE_S=${RT_DEADLINE_S:-3300} ;;
  *) echo "error: RT_MODE '$MODE' is not fixed or ramp" >&2; exit 2 ;;
esac
RAMP_S=${RT_RAMP_S:-2400}
WD_TTFT_MS=${RT_WD_TTFT_MS:-3000}; WD_ITL_MS=${RT_WD_ITL_MS:-600}
WD_HOLD_S=${RT_WD_HOLD_S:-120}; WD_ARM_DELAY_S=${RT_WD_ARM_DELAY_S:-150}
POLL_S=${RT_POLL_S:-15}; STALL_S=${RT_STALL_S:-900}; FIRST_OK_S=${RT_FIRST_OK_S:-1500}
PROGRESS_EVERY_S=${RT_PROGRESS_EVERY_S:-60}
PROM=${RT_PROM:-http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090}
TMP=$(mktemp -d /tmp/run-test-28-XXXXXX)
# shellcheck source=lib_watchdog.sh
source "$K8S/lib_watchdog.sh"

pods_state() {  # "name:uid:restarts" for every vLLM replica, sorted
  kubectl get pods -n "$NS" -l app=vllm \
    -o jsonpath='{range .items[*]}{.metadata.name}:{.metadata.uid}:{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort
}
r0_successes() {  # requests vllm-0 completed so far (empty if unreadable)
  kubectl exec -n "$NS" vllm-0 -c vllm -- python3 -c \
    "import urllib.request;print(int(sum(float(l.split()[-1]) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith('vllm:request_success_total'))))" 2>/dev/null || true
}
p95_ms() {  # $1 vLLM histogram name: vllm-0's p95 over 150 s in ms (empty if no data)
  python3 - "$PROM" "$1" "$MODEL" <<'EOF' 2>/dev/null || true
import json, sys, urllib.parse, urllib.request
prom, h, model = sys.argv[1:4]
q = ('histogram_quantile(0.95, sum by(le)(rate(vllm:%s_bucket{model_name="%s",pod="vllm-0"}[150s])))*1000'
     % (h, model))
r = json.load(urllib.request.urlopen(prom + '/api/v1/query?' + urllib.parse.urlencode({'query': q}), timeout=10))
v = r['data']['result']
if v and v[0]['value'][1] not in ('NaN', '+Inf'):
    print(int(float(v[0]['value'][1])))
EOF
}

# One Job per replica the StatefulSet ASKS for (1 with none, 2 with 1g.16gb), and every one of
# them must be Ready: a neighbour that died after Apply config (e.g. OOMKilled by its warm-up)
# would otherwise leave the tenant measured with an idle neighbour, ~13 % faster (study 25).
REPLICAS=$(kubectl -n "$NS" get sts vllm -o jsonpath='{.spec.replicas}' 2>/dev/null || true)
READY=$(kubectl -n "$NS" get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)
if ! [[ "${REPLICAS:-}" =~ ^[12]$ ]] || [ "${READY:-0}" != "$REPLICAS" ]; then
  echo "error: ${READY:-0} of ${REPLICAS:-0} vLLM replica(s) ready (expected 1 or 2, all ready): Apply config did not leave the layout up" >&2
  exit 2
fi
STATE_BEFORE=$(pods_state)
echo "vLLM replicas before the test ($REPLICAS):"; echo "$STATE_BEFORE" | sed 's/^/  /'
echo "mode=$MODE rate=$RATE ramp_s=$RAMP_S; watchdog: TTFT p95 > $WD_TTFT_MS ms or ITL p95 > $WD_ITL_MS ms for $WD_HOLD_S s, armed $WD_ARM_DELAY_S s after the measured run starts"

kubectl -n "$NS" delete job -l app=aiperf-mig --ignore-not-found --wait=true
JOBS=""
for i in $(seq 0 $((REPLICAS - 1))); do
  bash "$K8S/render_job.sh" "$i" "$MODE" "$RATE" "$RAMP_S" "$TMP/job-r$i.yaml"
  kubectl apply -f "$TMP/job-r$i.yaml"
  JOBS="$JOBS aiperf-mig-r$i"
done

set +e
FAIL_REASON=""; WATCHDOG=""
T0=$SECONDS; LAST_OK=""; LAST_PROGRESS=$SECONDS; NEXT_PROGRESS=$SECONDS; MARKER_AT=""; OVER_SINCE=""
while true; do
  DONE=0
  for j in $JOBS; do
    S=$(kubectl get job "$j" -n "$NS" -o jsonpath='{.status.succeeded}' 2>/dev/null)
    F=$(kubectl get job "$j" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null)
    [ "${F:-0}" -ge 1 ] && FAIL_REASON="the AIPerf job $j failed"
    [ "${S:-0}" -ge 1 ] && DONE=$((DONE + 1))
  done
  [ -n "$FAIL_REASON" ] && break
  [ "$DONE" -eq "$REPLICAS" ] && break
  STATE_NOW=$(pods_state)
  if [ "$STATE_NOW" != "$STATE_BEFORE" ]; then
    FAIL_REASON="a vLLM replica restarted or was replaced during the test (before: $(echo "$STATE_BEFORE" | tr '\n' ' ')/ now: $(echo "${STATE_NOW:-none}" | tr '\n' ' '))"; break
  fi
  if [ "$SECONDS" -ge "$NEXT_PROGRESS" ]; then
    NEXT_PROGRESS=$((SECONDS + PROGRESS_EVERY_S))
    OK=$(r0_successes)
    if [[ "$OK" =~ ^[0-9]+$ ]]; then
      if [ "$OK" != "$LAST_OK" ]; then LAST_OK=$OK; LAST_PROGRESS=$SECONDS; fi
      if [ "$OK" -gt 0 ] && [ $((SECONDS - LAST_PROGRESS)) -ge "$STALL_S" ]; then
        FAIL_REASON="stalled: vllm-0 completed no request for $((SECONDS - LAST_PROGRESS)) s (at $OK)"; break
      fi
      if [ "$OK" -eq 0 ] && [ $((SECONDS - T0)) -ge "$FIRST_OK_S" ]; then
        FAIL_REASON="stalled: vllm-0 completed no request $((SECONDS - T0)) s after the start"; break
      fi
    fi
  fi
  # --- Watchdog on vllm-0 ---
  if [ -z "$MARKER_AT" ] && wd_armed "$(kubectl logs job/aiperf-mig-r0 -n "$NS" -c aiperf --tail=400 2>/dev/null)"; then
    MARKER_AT=$SECONDS
    echo "$(date -u +%T) measured run started; watchdog armed in $WD_ARM_DELAY_S s"
  fi
  if wd_ready "$SECONDS" "$MARKER_AT"; then
    TT=$(p95_ms time_to_first_token_seconds); IT=$(p95_ms inter_token_latency_seconds)
    if wd_over "$TT" "$IT"; then NOW_OVER=0; else NOW_OVER=1; fi
    NEW_SINCE=$(wd_next_since "$SECONDS" "$OVER_SINCE" "$NOW_OVER")
    if [ -z "$OVER_SINCE" ] && [ -n "$NEW_SINCE" ]; then echo "$(date -u +%T) watchdog: over (TTFT p95 ${TT:-n/a} ms, ITL p95 ${IT:-n/a} ms)"; fi
    if [ -n "$OVER_SINCE" ] && [ -z "$NEW_SINCE" ]; then echo "$(date -u +%T) watchdog: back under (TTFT p95 ${TT:-n/a} ms, ITL p95 ${IT:-n/a} ms)"; fi
    OVER_SINCE=$NEW_SINCE
    if wd_fired "$SECONDS" "$OVER_SINCE"; then
      WATCHDOG="TTFT p95 ${TT:-n/a} ms, ITL p95 ${IT:-n/a} ms for $((SECONDS - OVER_SINCE)) s"; break
    fi
  fi
  if [ $((SECONDS - T0)) -ge "$DEADLINE_S" ]; then FAIL_REASON="test timeout (${DEADLINE_S} s)"; break; fi
  sleep "$POLL_S"
done
set -e

if [ -n "$FAIL_REASON" ]; then
  echo "error: $FAIL_REASON, after $((SECONDS - T0)) s. Dumping logs, then failing the trial."
elif [ -n "$WATCHDOG" ]; then
  echo "Watchdog ended the test after $((SECONDS - T0)) s: $WATCHDOG. The test is complete (the goal will be INVALID by its constraints)."
else
  echo "AIPerf job(s) completed after $((SECONDS - T0)) s."
fi
for j in $JOBS; do
  echo "--- job/$j: full logs ---"
  kubectl logs "job/$j" -n "$NS" --all-containers --tail=-1 || true
done
if [ -n "$FAIL_REASON" ]; then
  echo "--- vLLM replicas at failure ---"
  kubectl get pods -n "$NS" -l app=vllm -o wide || true
  for p in $(kubectl get pods -n "$NS" -l app=vllm -o name 2>/dev/null); do
    echo "--- $p: last 200 lines (current and previous container) ---"
    kubectl logs -n "$NS" "$p" --tail=200 || true
    kubectl logs -n "$NS" "$p" --tail=200 --previous 2>/dev/null || true
  done
  kubectl -n "$NS" delete job -l app=aiperf-mig --ignore-not-found --wait=false || true
  rm -rf "$TMP"; exit 1
fi
if [ -n "$WATCHDOG" ]; then  # logs dumped: now stop the load
  kubectl -n "$NS" delete job -l app=aiperf-mig --ignore-not-found --wait=false || true
fi
rm -rf "$TMP"
exit 0
