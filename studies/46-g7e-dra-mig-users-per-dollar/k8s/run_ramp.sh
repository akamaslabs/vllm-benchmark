#!/bin/bash
# Ramp task of study 46: find the highest per-replica closed-loop concurrency at which
# EVERY replica meets the SLA (client-side, per replica: TTFT p95 <= 1500 ms, ITL p95 <=
# 300 ms, >= 95 % completed). All replicas step together, so a slice is always measured
# with busy neighbours. Stops at the first level where any replica fails. Writes the
# chosen level to k8s/level.current for run_hold.sh (the lowest level if even that one
# fails: the Hold task then runs there and Akamas' SLA constraints mark the experiment).
#
# Score (design): users within SLA = N_replicas x min over replicas of the highest passing
# level. Because all replicas step together and the ramp stops at the first failure, the
# last passing level IS that minimum; the Hold task holds it and Akamas reads
# num_requests_running + num_requests_waiting = N x level.
set -euo pipefail
source "$(dirname "$0")/lib_load.sh"

LEVELS=${LEVELS:-"8 16 24 32 48 64 96 128 160 192 256 320 384 512 640 768 1024"}
LEVEL_SECONDS=${LEVEL_SECONDS:-120}

R0=$(restarts)
t "$REPLICAS replica(s); levels per replica: $LEVELS; ${LEVEL_SECONDS} s each (+20 s warm-up)"
BEST=""
FIRST=""
for L in $LEVELS; do
  FIRST=${FIRST:-$L}
  OUT=$(run_level "$L" "$LEVEL_SECONDS" "ramp-$L")
  echo "$OUT"
  [ "$(restarts)" = "$R0" ] || die "a vLLM replica restarted or was replaced during level $L: $R0 -> $(restarts)" 5
  if echo "$OUT" | grep -q '^R[0-9]* .* ok=0$'; then
    t "level $L fails on at least one replica: stop"
    break
  fi
  BEST=$L
  t "level $L passes on all $REPLICAS replica(s)"
done
if [ -z "$BEST" ]; then
  echo "${FIRST}" > "$K8S/level.current"
  t "RESULT: no level passes; Hold will run at $FIRST and is expected to violate the SLA"
else
  echo "$BEST" > "$K8S/level.current"
  [ "$BEST" = "${LEVELS##* }" ] && t "WARNING: the top level $BEST passed: the score is capped by the ramp, not by the SLA"
  t "RESULT: highest passing level $BEST per replica -> $((BEST * REPLICAS)) users within SLA on the GPU"
fi
