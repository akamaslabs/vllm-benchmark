#!/bin/bash
# Renders the AIPerf Job (30-l40s-gemma4-26b-tps): study 27's open-loop linear rate ramp.
# Usage: render_job.sh <rate req/s> <ramp_s> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_job.sh <rate> <ramp_s> <output>"
RATE=$1 RAMP_S=$2 OUT=$3
TEMPLATE=$(cd "$(dirname "$0")" && pwd)/05-job_template.yaml
[[ "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "rate '$RATE' is not a number"
[[ "$RAMP_S" =~ ^[0-9]+$ ]] && [ "$RAMP_S" -gt 0 ] || die "ramp_s '$RAMP_S' is not a positive integer"
# One fixed seed: the same arrival sequence in every trial.
LOAD_ARGS="--request-rate $RATE --request-rate-ramp-duration $RAMP_S --arrival-pattern gamma --arrival-smoothness 4 --random-seed 30 --benchmark-duration $RAMP_S --benchmark-grace-period 60"
sed -e "s|@LOAD_ARGS@|$LOAD_ARGS|g" "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
