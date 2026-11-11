#!/bin/bash
# fetch_all.sh: brings every exported trace and the ramp's logs and AIPerf records to this folder.
# traces/ is git-ignored (hundreds of MB), and so are the per-request jsonl files in results/aiperf/.
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p traces results/aiperf
for f in $(kubectl -n nsight-probe exec nsight -- sh -c 'ls /work/traces/ramp/*.zip'); do
  [ -f "traces/$(basename "$f")" ] || ./fetch.sh "$f" "traces/$(basename "$f")" || echo "FAILED $f"
done
for f in metrics.log capture.log post.log steps-summary.txt; do
  ./fetch.sh /work/traces/ramp/$f results/$f || echo "FAILED $f"
done
for c in c1 c2 c3 c4 c5 c6 c7; do
  for f in profile_export.jsonl profile_export_aiperf.json; do
    POD=aiperf ./fetch.sh /work/out/ramp/$c/run/$f results/aiperf/ramp-$c-$f || echo "FAILED $c $f"
  done
done
for f in profile_export.jsonl profile_export_aiperf.json; do
  POD=aiperf ./fetch.sh /work/out/run/$f results/aiperf/steady-64users-$f || echo "FAILED steady $f"
done
POD=aiperf ./fetch.sh /work/out/ramp.log results/ramp.log || echo "FAILED ramp.log"
echo "fetch_all done"
