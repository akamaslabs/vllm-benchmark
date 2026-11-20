#!/bin/bash
# AIPerf to CSV step of study 54 (toolbox, Akamas Executor task after Stop AIPerf; critical: without it
# the trial has no goal or constraint values).
# Converts the run's two exports (RUNS_DIR/<run>/c0.jsonl.gz, c1.jsonl.gz) with aiperf_to_csv.py into
# CSV_DIR/<run>.csv (read by the CSV File telemetry instance, cut at the first sustained SLO breach)
# and CSV_FULL_DIR/<run>.csv (every bucket), with the SLO and knee rule of workload.env and the pools
# of LOAD_DIR; prints the knee and the offline estimate of the score (score_offline.py).
# Usage: bash aiperf_csv.sh [RUN]   (default: RUNS_DIR/current). Exit 0; 2 an export is missing or has
# no record; 3 a session was replayed twice (the pool ran out: the trial must not be scored).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OVR="RUNS_DIR LOAD_DIR CSV_DIR CSV_FULL_DIR"
for v in $OVR; do eval "OV_$v=\${$v-}"; done
# shellcheck disable=SC1091
source "$HERE/workload.env"
for v in $OVR; do eval "[ -z \"\$OV_$v\" ] || $v=\$OV_$v"; done
RUN=${1:-$(cat "$RUNS_DIR/current" 2>/dev/null)}
D=$RUNS_DIR/$RUN
for i in 0 1; do
  [ -s "$D/c$i.jsonl.gz" ] || { echo "error: $D/c$i.jsonl.gz missing: Stop AIPerf did not save client $i's records" >&2; exit 2; }
  gzip -dc "$D/c$i.jsonl.gz" > "$D/c$i.jsonl" || { echo "error: $D/c$i.jsonl.gz is not a valid gzip (truncated?)" >&2; exit 2; }
done
mkdir -p "$CSV_DIR" "$CSV_FULL_DIR"
conv() { if [ -n "${CONVERTER:-}" ]; then "$CONVERTER" "$@"; else python3 -I "$HERE/aiperf_to_csv.py" "$@"; fi; }
score() { if [ -n "${SCORER:-}" ]; then "$SCORER" "$@"; else python3 -I "$HERE/score_offline.py" "$@"; fi; }
conv --pair "$D/c0.jsonl" "$LOAD_DIR/pool_0.jsonl.gz" --pair "$D/c1.jsonl" "$LOAD_DIR/pool_1.jsonl.gz" \
  --out "$CSV_DIR/$RUN.csv" --full-out "$CSV_FULL_DIR/$RUN.csv" \
  --slo-ttft-ms "$SLO_TTFT_MS" --slo-itl-ms "$SLO_ITL_MS" --min-window-reqs "$MIN_WINDOW_REQS" --breach-buckets "$BREACH_BUCKETS"
RC=$?
rm -f "$D/c0.jsonl" "$D/c1.jsonl"
[ "$RC" = 0 ] || exit "$RC"
echo "offline estimate of the Akamas score: $(score "$CSV_DIR/$RUN.csv" --slo-ttft-ms "$SLO_TTFT_MS" --slo-itl-ms "$SLO_ITL_MS")"
echo "uncut, for reference: $(score "$CSV_FULL_DIR/$RUN.csv" --slo-ttft-ms "$SLO_TTFT_MS" --slo-itl-ms "$SLO_ITL_MS")"
