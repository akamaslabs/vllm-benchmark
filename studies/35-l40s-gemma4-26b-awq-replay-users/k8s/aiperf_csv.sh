#!/bin/bash
# AIPerf to CSV step for 35-l40s-gemma4-26b-awq-replay-users (toolbox, Akamas Executor task after
# Stop AIPerf). Converts the measured run's per-request records, saved by stop_aiperf.sh as
# SA_DEST/<epoch>.jsonl, into the CSV that the study's CSV File telemetry instance reads
# (../akamas/telemetry/aiperf-csv.yaml): AC_CSV_DIR/<epoch>.csv, one row per 30 s, component
# `aiperf`, one column per AIPerf pack metric (aiperf_to_csv.py, a byte-for-byte copy of the AIPerf
# optimization pack 1.0.0's tools/aiperf_to_csv.py).
#
# Which records: the newest <epoch>.jsonl, and only if it was written in the last AC_MAX_AGE_S s
# (default 1800): when Stop AIPerf could not save this trial's records, an older file would hand
# Akamas another trial's run. Its rows would fall outside this trial's window anyway, but failing
# here says why the trial has no client-side metrics.
# Exit 1 on no fresh records or a failed conversion: the goal and the constraint read the CSV, so a
# trial without it cannot be scored (the task is critical).
set -euo pipefail
K8S=$(cd "$(dirname "$0")" && pwd)
SRC=${SA_DEST:-/work/aiperf-exports/35-l40s-gemma4-26b-awq-replay-users}
DEST=${AC_CSV_DIR:-/work/aiperf-csv/35-l40s-gemma4-26b-awq-replay-users}
MAX_AGE_S=${AC_MAX_AGE_S:-1800}

NEWEST=$(ls -t "$SRC"/*.jsonl 2>/dev/null | head -1 || true)
[ -n "$NEWEST" ] || { echo "error: no records in $SRC (Stop AIPerf saved none)" >&2; exit 1; }
AGE=$(( $(date +%s) - $(stat -c %Y "$NEWEST") ))
[ "$AGE" -le "$MAX_AGE_S" ] || { echo "error: newest records $NEWEST are $AGE s old (> $MAX_AGE_S s): not this trial's" >&2; exit 1; }
mkdir -p "$DEST"
BASE=$(basename "$NEWEST" .jsonl)
python3 "$K8S/aiperf_to_csv.py" "$NEWEST" "$DEST/$BASE.csv.tmp" --component aiperf
mv "$DEST/$BASE.csv.tmp" "$DEST/$BASE.csv"
ROWS=$(($(wc -l < "$DEST/$BASE.csv") - 1))
echo "$NEWEST -> $DEST/$BASE.csv: $ROWS rows of 30 s ($(head -2 "$DEST/$BASE.csv" | tail -1 | cut -d, -f1) to $(tail -1 "$DEST/$BASE.csv" | cut -d, -f1) UTC)"
