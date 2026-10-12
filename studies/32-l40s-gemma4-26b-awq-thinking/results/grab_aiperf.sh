#!/bin/bash
# Save each AIPerf run's per-request records (profile_export.jsonl) while the study runs.
#
# Usage: AWS_PROFILE=lab ./grab_aiperf.sh [SECONDS] [DEST_DIR]   (default 12 h, ./aiperf_runs)
#
# The Job writes them on the results PVC under /benchmarks/study32/aiperf-<epoch>/, in batches
# of 100 records, and the next experiment's Job deletes that folder (rm -rf at its start): the
# file is only readable from the running aiperf-l40s pod, so this loop copies it every 2 minutes
# and keeps the largest copy. The first run's inputs.json (the synthetic dataset, ~64 MB) is
# copied once. Analyse a copy, even a partial one, with analyze_requests.py.
END=$(( $(date +%s) + ${1:-43200} ))
DST=${2:-$(dirname "$0")/aiperf_runs}
mkdir -p "$DST"
while [ "$(date +%s)" -lt "$END" ]; do
  P=$(kubectl -n llm-l40s get pod -l app=aiperf-l40s --field-selector=status.phase=Running -o name 2>/dev/null | head -1)
  if [ -n "$P" ]; then
    for D in $(kubectl -n llm-l40s exec "$P" -c aiperf -- sh -c 'ls -d /benchmarks/study32/aiperf-* 2>/dev/null' 2>/dev/null); do
      N=$(basename "$D")
      kubectl -n llm-l40s exec "$P" -c aiperf -- cat "$D/profile_export.jsonl" > "$DST/$N.jsonl.tmp" 2>/dev/null
      NEW=$(wc -c < "$DST/$N.jsonl.tmp")
      OLD=$( [ -f "$DST/$N.jsonl" ] && wc -c < "$DST/$N.jsonl" || echo 0)
      if [ "$NEW" -ge "$OLD" ]; then mv "$DST/$N.jsonl.tmp" "$DST/$N.jsonl"; else rm -f "$DST/$N.jsonl.tmp"; fi
      ls "$DST"/*.inputs.json >/dev/null 2>&1 \
        || kubectl -n llm-l40s exec "$P" -c aiperf -- cat "$D/inputs.json" > "$DST/$N.inputs.json" 2>/dev/null
    done
  fi
  sleep 120
done
