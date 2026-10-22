#!/bin/bash
# Renders the AIPerf Job (41-l4-dsv2-lite-context-parallel): a closed-loop users ramp, 1 -> N
# concurrent sessions over RAMP s (as study 38's run_benchmark.sh), over the workload of
# lib_workload.sh.
# Usage: render_job.sh <users> <ramp_s> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_job.sh <users> <ramp_s> <output>"
USERS=$1 RAMP_S=$2 OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
TEMPLATE=$HERE/05-job_template.yaml
# shellcheck source=lib_workload.sh
source "$HERE/lib_workload.sh"
{ [[ "$USERS" =~ ^[0-9]+$ ]] && [ "$USERS" -ge 2 ]; } || die "users '$USERS' is not an integer >= 2"
{ [[ "$RAMP_S" =~ ^[0-9]+$ ]] && [ "$RAMP_S" -gt 0 ]; } || die "ramp_s '$RAMP_S' is not a positive integer"
# One fixed seed (WORKLOAD_SEED, in the template): the same sessions in every trial. The run
# lasts the ramp; the watchdog (run_test.sh) normally ends it earlier, past the SLO.
LOAD_ARGS="--concurrency $USERS --concurrency-ramp-duration $RAMP_S --benchmark-duration $RAMP_S --benchmark-grace-period 60"
sed -e "s|@WORKLOAD_ARGS@|$WORKLOAD_ARGS|g" -e "s|@LOAD_ARGS@|$LOAD_ARGS|g" \
    -e "s|@MEASURED_ENTRIES@|$MEASURED_ENTRIES|g" -e "s|@WORKLOAD_SEED@|$WORKLOAD_SEED|g" \
    -e "s|@WARMUP_ENTRIES@|$WARMUP_ENTRIES|g" -e "s|@WARMUP_SEED@|$WARMUP_SEED|g" \
    "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
