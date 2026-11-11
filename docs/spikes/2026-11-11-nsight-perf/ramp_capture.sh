#!/bin/bash
# ramp_capture.sh <HH:MM:SS UTC start> [step-seconds] [steps] [offset-seconds] [capture-ms]
# Runs in the nsight pod next to ramp.sh. In every step it captures one Nsight trace, starting
# offset seconds into the step, and exports a 1.5 s window of it. A sampler writes the vLLM
# counters every 15 s to /work/traces/ramp/metrics.log for the whole ramp.
set -u
T0=$(date -u -d "today $1" +%s); STEP=${2:-240}; N=${3:-7}; OFF=${4:-200}; DUR=${5:-10000}
BIN=/work/sdk/build/bin; OUT=/work/traces/ramp; mkdir -p $OUT
( end=$(( T0 + N * STEP + 60 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    { echo "# $(date -u +%H:%M:%S) $(date +%s)"
      curl -s --max-time 5 http://vllm.nsight-probe.svc:8000/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting\{|kv_cache_usage_perc|generation_tokens_total|prompt_tokens_total|request_success_total|time_to_first_token_seconds_(sum|count|bucket)|inter_token_latency_seconds_(sum|count))'
      nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader | sed 's/^/gpu: /'; } >> $OUT/metrics.log
    sleep 15
  done ) &
cd $BIN
for i in $(seq 1 "$N"); do
  at=$(( T0 + (i - 1) * STEP + OFF )); while [ "$(date +%s)" -lt "$at" ]; do sleep 1; done
  d=$OUT/step$i; mkdir -p $d; t0=$(date +%s%3N)
  ./nvtrace_streaming_consumer --output-dir $d --duration-ms "$DUR" > $d/consumer.log 2>&1
  tsf=$(ls -d $d/nsight.*)
  ./nvtrace_perfetto_trace_exporter --tsf-database-dir "$tsf" --start-time-utc-ms $((t0 + 4000)) --end-time-utc-ms $((t0 + 5500)) > $d/export-window.log 2>&1
  mv "$tsf.perfetto-trace.zip" $d/window.perfetto-trace.zip
  echo "$(date -u +%H:%M:%S) step $i: tsf $(du -sm "$tsf" | cut -f1) MB, window $(du -sm $d/window.perfetto-trace.zip | cut -f1) MB"
done
wait; echo "$(date -u +%H:%M:%S) captures done"
