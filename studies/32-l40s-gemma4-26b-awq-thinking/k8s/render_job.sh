#!/bin/bash
# Renders the AIPerf Job (32-l40s-gemma4-26b-awq-thinking).
# Usage: render_job.sh <rate req/s> <ramp_s> <output>        study 27's open-loop linear rate ramp
#        render_job.sh --closed <concurrency> <count> <output>  the length run (smoke/smoke_manual.sh):
#                                                             closed loop, <count> requests
# Exit codes: 0 rendered; 2 invalid input (nothing written).
# ENTRIES, the synthetic conversation pool, is sized above the run's conversation count so no
# conversation repeats (a repeat would hit the prefix cache on its whole first turn): requests
# over 2 (the profile averages ~3 turns; 2 leaves room for short conversations), at least 100.
# For the ramp the requests are R x D / 2 if it ran to the end (the watchdog stops it earlier).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
TEMPLATE=$(cd "$(dirname "$0")" && pwd)/05-job_template.yaml
if [ "${1:-}" = --closed ]; then
  [ $# -eq 4 ] || die "usage: render_job.sh --closed <concurrency> <count> <output>"
  CONC=$2 COUNT=$3 OUT=$4
  [[ "$CONC" =~ ^[0-9]+$ ]] && [ "$CONC" -gt 0 ] || die "concurrency '$CONC' is not a positive integer"
  [[ "$COUNT" =~ ^[0-9]+$ ]] && [ "$COUNT" -ge "$CONC" ] || die "count '$COUNT' is not an integer >= concurrency"
  LOAD_ARGS="--concurrency $CONC --request-count $COUNT --random-seed 30"
  ENTRIES=$(awk -v n="$COUNT" 'BEGIN { e = int(n / 2) + 1; print (e < 100 ? 100 : e) }')
else
  [ $# -eq 3 ] || die "usage: render_job.sh <rate> <ramp_s> <output>"
  RATE=$1 RAMP_S=$2 OUT=$3
  [[ "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "rate '$RATE' is not a number"
  [[ "$RAMP_S" =~ ^[0-9]+$ ]] && [ "$RAMP_S" -gt 0 ] || die "ramp_s '$RAMP_S' is not a positive integer"
  # One fixed seed: the same arrival sequence in every trial (study 30's seed).
  LOAD_ARGS="--request-rate $RATE --request-rate-ramp-duration $RAMP_S --arrival-pattern gamma --arrival-smoothness 4 --random-seed 30 --benchmark-duration $RAMP_S --benchmark-grace-period 60"
  ENTRIES=$(awk -v r="$RATE" -v d="$RAMP_S" 'BEGIN { e = int(r * d / 2 / 2) + 1; print (e < 100 ? 100 : e) }')
fi
sed -e "s|@LOAD_ARGS@|$LOAD_ARGS|g" -e "s|@ENTRIES@|$ENTRIES|g" "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
