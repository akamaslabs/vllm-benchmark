#!/bin/bash
# Hold task of study 46: keep every replica at the level run_ramp.sh found
# (k8s/level.current) for HOLD_SECONDS. The study windows on this task (trim), so the goal
# (requests in flight) and the SLA constraints are read only here, in steady state.
set -euo pipefail
source "$(dirname "$0")/lib_load.sh"

HOLD_SECONDS=${HOLD_SECONDS:-240}
L=$(cat "$K8S/level.current" 2>/dev/null || true)
[ -n "$L" ] || die "no $K8S/level.current: the Ramp task did not complete"

R0=$(restarts)
t "holding $REPLICAS replica(s) at concurrency $L each ($((L * REPLICAS)) users) for ${HOLD_SECONDS} s"
OUT=$(run_level "$L" "$HOLD_SECONDS" "hold")
echo "$OUT"
[ "$(restarts)" = "$R0" ] || die "a vLLM replica restarted or was replaced during the hold: $R0 -> $(restarts)" 5
echo "$OUT" | grep -q '^R[0-9]* .* ok=0$' && t "WARNING: the SLA did not hold client-side during the hold (Akamas constraints decide)"
# Full AIPerf output of the hold, every replica, so the Akamas task log is enough to debug.
for i in $(seq 0 $((REPLICAS - 1))); do
  echo "--- aiperf-client-$i: hold log ---"
  kubectl -n $NS exec aiperf-client-$i -- cat /work/out/hold.log 2>/dev/null || true
done
t "done"
