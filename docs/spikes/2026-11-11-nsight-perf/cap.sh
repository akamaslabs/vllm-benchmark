#!/bin/bash
# cap.sh <name> <duration-ms> <window-start-ms> <window-end-ms>: capture, export in full and one window.
set -u; name=$1; dur=$2; ws=$3; we=$4
cd /work/sdk/build/bin; out=/work/traces/$name; mkdir -p $out
t0=$(date +%s%3N); date -u +"start %H:%M:%S"
./nvtrace_streaming_consumer --output-dir $out --duration-ms $dur 2>&1 | tail -1
d=$(ls -d $out/nsight.*); echo "tsf $(du -sm $d | cut -f1) MB"
if [ "${FULL:-1}" = 1 ]; then
  /usr/bin/time -f "full export %es, peak %M KB" ./nvtrace_perfetto_trace_exporter --tsf-database-dir $d 2>&1 | tail -2
  mv $d.perfetto-trace.zip $out/full.perfetto-trace.zip
fi
/usr/bin/time -f "window export %es, peak %M KB" ./nvtrace_perfetto_trace_exporter --tsf-database-dir $d --start-time-utc-ms $((t0+ws)) --end-time-utc-ms $((t0+we)) 2>&1 | tail -2
mv $d.perfetto-trace.zip $out/window.perfetto-trace.zip
ls -la $out/*.zip; date -u +"end %H:%M:%S"
