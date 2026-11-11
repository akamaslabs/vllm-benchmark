#!/bin/bash
# post.sh <HH:MM:SS UTC start>: runs in the nsight pod after the ramp's last capture.
# Exports every step's whole 10 s capture to a Perfetto zip named after its load, keeps the 1.5 s
# window next to it, and appends a summary of each full trace (steps.py) to steps-summary.txt.
set -u
T0=$(date -u -d "today $1" +%s); while [ "$(date +%s)" -lt "$T0" ]; do sleep 1; done
BIN=/work/sdk/build/bin; R=/work/traces/ramp; U=(0 64 128 192 256 320 384 448)
cd $BIN; : > $R/steps-summary.txt
for i in 1 2 3 4 5 6 7; do
  d=$R/step$i; tsf=$(ls -d $d/nsight.*); n=ramp-step$i-${U[$i]}users
  /usr/bin/time -f "export %es, peak %M KB" ./nvtrace_perfetto_trace_exporter --tsf-database-dir "$tsf" > $d/export-full.log 2>&1
  mv "$tsf.perfetto-trace.zip" $R/$n-10s.perfetto-trace.zip
  cp $d/window.perfetto-trace.zip $R/$n-1.5s.perfetto-trace.zip
  mkdir -p /work/tmp/$n && unzip -o -q $R/$n-10s.perfetto-trace.zip -d /work/tmp/$n
  python3 /work/steps.py "$n=$(ls /work/tmp/$n/*.perfetto-trace)" >> $R/steps-summary.txt 2>&1
  rm -rf /work/tmp/$n
  echo "$(date -u +%H:%M:%S) $n: $(du -sm $R/$n-10s.perfetto-trace.zip | cut -f1) MB, $(tail -1 $d/export-full.log)"
done
tsf=$(ls -d /work/traces/long30/nsight.*)
/usr/bin/time -f "export %es, peak %M KB" ./nvtrace_perfetto_trace_exporter --tsf-database-dir "$tsf" > /work/traces/long30/export-full.log 2>&1
mv "$tsf.perfetto-trace.zip" $R/steady-64users-30s.perfetto-trace.zip
echo "$(date -u +%H:%M:%S) steady-64users-30s: $(du -sm $R/steady-64users-30s.perfetto-trace.zip | cut -f1) MB, $(tail -1 /work/traces/long30/export-full.log)"
echo "$(date -u +%H:%M:%S) post done"
