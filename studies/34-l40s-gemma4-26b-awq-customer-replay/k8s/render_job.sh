#!/bin/bash
# Renders the AIPerf Job (34-l40s-gemma4-26b-awq-customer-replay): the customer's replay.
# Usage: render_job.sh <users> <ramp_s> <output>
# Writes <users> and <ramp_s> into the benchmark.env the Job gives the customer's run_benchmark.sh:
# USERS, RAMP and DURATION (their --concurrency, --concurrency-ramp-duration, --benchmark-duration):
# users ramp from 1 to <users> over <ramp_s> s and the run lasts <ramp_s> s. USERS_RENDERED, the users
# render_dataset.py writes, is above <users> (+20 %, at least +10): AIPerf starts the next trace
# session when one ends, and a session must not repeat (it would hit the prefix cache).
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
TEMPLATE=$(cd "$(dirname "$0")" && pwd)/05-job_template.yaml
[ $# -eq 3 ] || die "usage: render_job.sh <users> <ramp_s> <output>"
USERS=$1 RAMP_S=$2 OUT=$3
[[ "$USERS" =~ ^[0-9]+$ ]] && [ "$USERS" -gt 0 ] || die "users '$USERS' is not a positive integer"
[[ "$RAMP_S" =~ ^[0-9]+$ ]] && [ "$RAMP_S" -gt 0 ] || die "ramp_s '$RAMP_S' is not a positive integer"
EXTRA=$((USERS / 5)); [ "$EXTRA" -ge 10 ] || EXTRA=10
RENDERED=$((USERS + EXTRA))
sed -e "s|@USERS_RENDERED@|$RENDERED|g" -e "s|@USERS@|$USERS|g" -e "s|@RAMP_S@|$RAMP_S|g" "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
