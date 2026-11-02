#!/bin/bash
# Study 45 idle guard, run on the toolbox with setsid nohup (survives laptop sleep and an
# expired Akamas CLI login; uses only kubectl and the toolbox's IAM policy):
#   setsid nohup bash /work/study45/45-gpu-choice-tps-per-dollar/k8s/idle_guard.sh \
#     > /work/study45/idle_guard.log 2>&1 < /dev/null &
# Every 10 min: if no serving pod (app=vllm-pd-mn) and no AIPerf job exist for 30+ min while
# some study-45 node group still has desired > 0, run scale_down.sh. Between experiments the
# workflow leaves nothing running for more than a few minutes, so 30 min of idleness means the
# study stopped (finished, ERROR, killed workflow) without its Scale down task.
# Stops by itself after GUARD_HOURS (default 96).
K8S=${STUDY45_K8S:-/work/study45/45-gpu-choice-tps-per-dollar/k8s}
# shellcheck disable=SC1091
source "$K8S/nodes.sh"
END=$(( $(date +%s) + ${GUARD_HOURS:-96} * 3600 ))
IDLE_SINCE=""
while [ "$(date +%s)" -lt "$END" ]; do
  busy=$(kubectl get pod -n llm-serving -l app=vllm-pd-mn --no-headers 2>/dev/null | wc -l)
  job=$(kubectl get job aiperf-pd-mn -n llm-benchmark --no-headers 2>/dev/null | wc -l)
  up=""
  for g in $ALL_GROUPS; do d=$(desired_of "$g"); [ -n "$d" ] && [ "$d" != "0" ] && up="$up $g=$d"; done
  if [ "$busy" -eq 0 ] && [ "$job" -eq 0 ] && [ -n "$up" ]; then
    [ -z "$IDLE_SINCE" ] && IDLE_SINCE=$(date +%s) && echo "$(date -u +%FT%T) idle with nodes up:$up"
    if [ $(( $(date +%s) - IDLE_SINCE )) -ge 1800 ]; then
      echo "$(date -u +%FT%T) idle for 30 min with nodes up:$up -> scale_down.sh"
      bash "$K8S/scale_down.sh"
      IDLE_SINCE=""
    fi
  else
    IDLE_SINCE=""
  fi
  sleep 600
done
echo "$(date -u +%FT%T) guard ended after ${GUARD_HOURS:-96} h"
