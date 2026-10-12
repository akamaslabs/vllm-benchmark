#!/bin/bash
# RunTest step for 34-l40s-gemma4-26b-awq-customer-replay (toolbox, Akamas Executor task).
#
# One AIPerf Job against vllm-0: the customer's replay (05-job_template.yaml), users ramping from
# 1 to RT_USERS over RT_RAMP_S s (closed loop: each user waits for its reply, then its think
# time). Their dataset template is not in git: RT_TEMPLATE on the toolbox (../load/README.md);
# it goes, gzipped, into the ConfigMap aiperf-replay with render_dataset.py and their
# run_benchmark.sh, which the Job mounts: it renders the dataset and runs their script.
# Defaults RT_USERS=100 (the users of their full dataset), RT_RAMP_S=6000, as the workflow
# command. PROVISIONAL (study README "Load"): study 32's knees (thinking on, synthetic load:
# compose ~0.59 req/s, best ~1.3) over a user's request rate with thinking on (~20 s per reply
# plus ~13 s of mean think time: ~0.03 req/s) put the compose's knee near ~20 users and the best
# near ~45: inside a ramp to 100.
# The trial FAILS (exit 1) if the Job fails, the vLLM pod restarts or is replaced, vllm-0
# completes nothing for RT_STALL_S, or the deadline passes (study 27/28 guards). The
# watchdog ENDS the test with SUCCESS (exit 0) once vllm-0's TTFT p95 (150 s) > RT_WD_TTFT_MS
# or ITL p95 (150 s) > RT_WD_ITL_MS for RT_WD_HOLD_S: past the capacity of an open loop the
# queue only grows, so every later window would be invalid anyway, and Akamas scores the
# best valid window before it. Armed RT_WD_ARM_DELAY_S after the measured run starts, so
# Apply config's warm-up is out of its 150 s view.
# RT_STOP_AIPERF=0 (the workflow): the Job is left running, on failure too, for the workflow's
# Stop AIPerf task (stop_aiperf.sh: graceful stop, records saved on the toolbox); 1 (default: the
# script run by hand): deleted here, as before.
set -euo pipefail
K8S=${K8S:-/work/vllm-benchmark/studies/34-l40s-gemma4-26b-awq-customer-replay/k8s}
NS=llm-l40s
MODEL=aiservices_gemma
USERS=${RT_USERS:-100}
RAMP_S=${RT_RAMP_S:-6000}
TEMPLATE=${RT_TEMPLATE:-/work/customer-data/34-l40s-gemma4-26b-awq-customer-replay/dataset_u000.template.jsonl}
# pip (~2 min) + the dataset render and AIPerf's load of it (~1-3 min) + the ramp + grace. The workflow's RunTest timeout must stay above it, or Akamas kills the task
# before the log dump.
DEADLINE_S=${RT_DEADLINE_S:-$((RAMP_S + 1500))}
WD_TTFT_MS=${RT_WD_TTFT_MS:-3000}; WD_ITL_MS=${RT_WD_ITL_MS:-600}
WD_HOLD_S=${RT_WD_HOLD_S:-120}; WD_ARM_DELAY_S=${RT_WD_ARM_DELAY_S:-150}
POLL_S=${RT_POLL_S:-15}; STALL_S=${RT_STALL_S:-900}; FIRST_OK_S=${RT_FIRST_OK_S:-1500}
PROGRESS_EVERY_S=${RT_PROGRESS_EVERY_S:-60}
STOP_AIPERF=${RT_STOP_AIPERF:-1}
PROM=${RT_PROM:-http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090}
TMP=$(mktemp -d /tmp/run-test-34-XXXXXX)
JOB=aiperf-l40s
# shellcheck source=lib_watchdog.sh
source "$K8S/lib_watchdog.sh"

stop_job() {
  if [ "$STOP_AIPERF" = 0 ]; then
    echo "job/$JOB left running for the Stop AIPerf task (RT_STOP_AIPERF=0)"
  else
    kubectl -n "$NS" delete job -l app=aiperf-l40s --ignore-not-found --wait=false || true
  fi
}

pod_state() {  # "name:uid:restarts" of the vLLM pod
  kubectl get pods -n "$NS" -l app=vllm \
    -o jsonpath='{range .items[*]}{.metadata.name}:{.metadata.uid}:{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort
}
successes() {  # requests vllm-0 completed so far (empty if unreadable)
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

READY=$(kubectl -n "$NS" get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)
if [ "${READY:-0}" != 1 ]; then
  echo "error: vllm-0 is not ready (readyReplicas=${READY:-0}): Apply config did not leave it up" >&2
  exit 2
fi
STATE_BEFORE=$(pod_state)
echo "vLLM pod before the test: $STATE_BEFORE"
echo "users 1 -> $USERS over $RAMP_S s; watchdog: TTFT p95 > $WD_TTFT_MS ms or ITL p95 > $WD_ITL_MS ms for $WD_HOLD_S s, armed $WD_ARM_DELAY_S s after the measured run starts; deadline $DEADLINE_S s"

# A Job left over from a stopped study would keep loading vLLM (study 29, exp 16).
kubectl -n "$NS" delete job -l app=aiperf-l40s --ignore-not-found --wait=true
[ -s "$TEMPLATE" ] || { echo "error: dataset template $TEMPLATE missing on the toolbox (../load/README.md)" >&2; exit 2; }
gzip -c "$TEMPLATE" > "$TMP/template.jsonl.gz"
kubectl -n "$NS" create configmap aiperf-replay --from-file=template.jsonl.gz="$TMP/template.jsonl.gz" \
  --from-file=render_dataset.py="$K8S/render_dataset.py" --from-file=run_benchmark.sh="$K8S/run_benchmark.sh" \
  --dry-run=client -o yaml | kubectl apply -f -
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
        FAIL_REASON="stalled: vllm-0 completed no request for $((SECONDS - LAST_PROGRESS)) s (at $OK)"; break
      fi
      if [ "$OK" -eq 0 ] && [ $((SECONDS - T0)) -ge "$FIRST_OK_S" ]; then
        FAIL_REASON="stalled: vllm-0 completed no request $((SECONDS - T0)) s after the start"; break
      fi
    fi
  fi
  # --- Watchdog ---
  if [ -z "$MARKER_AT" ] && wd_armed "$(kubectl logs "job/$JOB" -n "$NS" -c aiperf --tail=400 2>/dev/null)"; then
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
  echo "Watchdog ended the test after $((SECONDS - T0)) s: $WATCHDOG. The test is complete (Akamas scores the best valid window before it)."
else
  echo "AIPerf job completed after $((SECONDS - T0)) s (the ramp reached $USERS users without breaking the SLA: raise RT_USERS)."
fi
echo "--- job/$JOB: full logs ---"
kubectl logs "job/$JOB" -n "$NS" --all-containers --tail=-1 || true
if [ -n "$FAIL_REASON" ]; then
  echo "--- vLLM pod at failure ---"
  kubectl get pods -n "$NS" -l app=vllm -o wide || true
  kubectl logs -n "$NS" vllm-0 --tail=200 || true
  kubectl logs -n "$NS" vllm-0 --tail=200 --previous 2>/dev/null || true
  stop_job
  rm -rf "$TMP"; exit 1
fi
if [ -n "$WATCHDOG" ]; then  # logs dumped: now stop the load
  stop_job
fi
rm -rf "$TMP"
exit 0
