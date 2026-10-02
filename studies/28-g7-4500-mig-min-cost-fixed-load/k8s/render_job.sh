#!/bin/bash
# Renders the AIPerf Job for one vLLM replica (28-g7-4500-mig-min-cost-fixed-load).
# Usage: render_job.sh <replica 0|1> <mode fixed|ramp> <rate req/s> <ramp_s> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 5 ] || die "usage: render_job.sh <replica> <mode> <rate> <ramp_s> <output>"
REPLICA=$1 MODE=$2 RATE=$3 RAMP_S=$4 OUT=$5
TEMPLATE=$(cd "$(dirname "$0")" && pwd)/05-job_template.yaml
[[ "$REPLICA" =~ ^[01]$ ]] || die "replica '$REPLICA' is not 0 or 1"
[[ "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "rate '$RATE' is not a number"
[[ "$RAMP_S" =~ ^[0-9]+$ ]] || die "ramp_s '$RAMP_S' is not an integer"
SEED=$((28 + REPLICA))   # one sequence per slice, the same in every trial
ARRIVALS="--arrival-pattern gamma --arrival-smoothness 4 --random-seed $SEED"
case "$MODE" in
  fixed) LOAD_ARGS="--request-rate $RATE $ARRIVALS --benchmark-duration 780 --benchmark-grace-period 30" ;;
  ramp)  [ "$RAMP_S" -gt 0 ] || die "ramp mode needs ramp_s > 0"
         LOAD_ARGS="--request-rate $RATE --request-rate-ramp-duration $RAMP_S $ARRIVALS --benchmark-duration $RAMP_S --benchmark-grace-period 60" ;;
  *) die "mode '$MODE' is not fixed or ramp" ;;
esac
if [ "$REPLICA" = 0 ]; then CACHE_ROLE="make"; else CACHE_ROLE="wait"; fi
sed -e "s|@REPLICA@|$REPLICA|g" -e "s|@LOAD_ARGS@|$LOAD_ARGS|g" -e "s|@CACHE_ROLE@|$CACHE_ROLE|g" \
  "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
