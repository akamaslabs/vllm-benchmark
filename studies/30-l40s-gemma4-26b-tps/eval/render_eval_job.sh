#!/bin/bash
# Renders the lm-eval Job of study 30's accuracy check (eval_job_template.yaml).
# Usage: render_eval_job.sh <run> <protocols> <limit|""> <output>
#   protocols: space-separated subset of "greedy card"; limit: empty (full tasks) or N.
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 4 ] || die "usage: render_eval_job.sh <run> <protocols> <limit> <output>"
RUN=$1 PROTOCOLS=$2 LIMIT=$3 OUT=$4
TEMPLATE=$(cd "$(dirname "$0")" && pwd)/eval_job_template.yaml
[[ "$RUN" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "run '$RUN': lowercase letters, digits and hyphens only"
[ -n "$PROTOCOLS" ] || die "no protocol"
for p in $PROTOCOLS; do
  case $p in greedy|card) ;; *) die "unknown protocol '$p'" ;; esac
done
[ -z "$LIMIT" ] || [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "limit '$LIMIT' is not a positive integer"
sed -e "s|@RUN@|$RUN|g" -e "s|@PROTOCOLS@|$PROTOCOLS|g" -e "s|@LIMIT@|$LIMIT|g" "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
