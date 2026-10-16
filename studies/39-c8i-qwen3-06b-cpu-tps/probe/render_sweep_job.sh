#!/bin/bash
# Renders the probe's AIPerf rate sweep (sweep_job_template.yaml) with the study's workload
# (../k8s/lib_workload.sh).
# Usage: render_sweep_job.sh "<rate> <rate> ..." <step_s> <output>
# Env: KP_STOP_TTFT_MS / KP_STOP_TPOT_MS (default 2x the SLO: 6000 / 200) end the sweep early.
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 3 ] || die "usage: render_sweep_job.sh \"<rates>\" <step_s> <output>"
RATES=$1 STEP_S=$2 OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../k8s/lib_workload.sh
source "$HERE/../k8s/lib_workload.sh"
STOP_TTFT_MS=${KP_STOP_TTFT_MS:-6000} STOP_TPOT_MS=${KP_STOP_TPOT_MS:-200}
[ -n "$RATES" ] || die "no rates"
for r in $RATES; do
  { [[ "$r" =~ ^[0-9]+(\.[0-9]+)?$ ]] && [[ "$r" =~ [1-9] ]]; } || die "rate '$r' is not a positive number"
done
{ [[ "$STEP_S" =~ ^[0-9]+$ ]] && [ "$STEP_S" -ge 60 ]; } || die "step_s '$STEP_S' is not an integer >= 60"
for v in STOP_TTFT_MS STOP_TPOT_MS; do [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"; done
sed -e "s|@WORKLOAD_ARGS@|$WORKLOAD_ARGS|g" -e "s|@RATES@|$RATES|g" -e "s|@STEP_S@|$STEP_S|g" \
    -e "s|@MEASURED_ENTRIES@|$MEASURED_ENTRIES|g" -e "s|@WORKLOAD_SEED@|$WORKLOAD_SEED|g" \
    -e "s|@WARMUP_ENTRIES@|$WARMUP_ENTRIES|g" -e "s|@WARMUP_SEED@|$WARMUP_SEED|g" \
    -e "s|@STOP_TTFT_MS@|$STOP_TTFT_MS|g" -e "s|@STOP_TPOT_MS@|$STOP_TPOT_MS|g" \
    "$HERE/sweep_job_template.yaml" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
