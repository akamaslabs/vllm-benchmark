#!/bin/bash
# Renders the AIPerf Job (39-c8i-qwen3-06b-cpu-tps): study 27's open-loop linear rate ramp
# over the workload of lib_workload.sh.
# Usage: render_job.sh <rate req/s> <ramp_s> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_job.sh <rate> <ramp_s> <output>"
RATE=$1 RAMP_S=$2 OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
TEMPLATE=$HERE/05-job_template.yaml
# shellcheck source=lib_workload.sh
source "$HERE/lib_workload.sh"
[[ "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "rate '$RATE' is not a number"
[[ "$RATE" =~ [1-9] ]] || die "rate '$RATE' is zero"
{ [[ "$RAMP_S" =~ ^[0-9]+$ ]] && [ "$RAMP_S" -gt 0 ]; } || die "ramp_s '$RAMP_S' is not a positive integer"
# One fixed seed (WORKLOAD_SEED, in the template): the same arrival sequence in every trial.
LOAD_ARGS="--request-rate $RATE --request-rate-ramp-duration $RAMP_S --arrival-pattern gamma --arrival-smoothness 4 --benchmark-duration $RAMP_S --benchmark-grace-period 60"
sed -e "s|@WORKLOAD_ARGS@|$WORKLOAD_ARGS|g" -e "s|@LOAD_ARGS@|$LOAD_ARGS|g" \
    -e "s|@MEASURED_ENTRIES@|$MEASURED_ENTRIES|g" -e "s|@WORKLOAD_SEED@|$WORKLOAD_SEED|g" \
    -e "s|@WARMUP_ENTRIES@|$WARMUP_ENTRIES|g" -e "s|@WARMUP_SEED@|$WARMUP_SEED|g" \
    "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
