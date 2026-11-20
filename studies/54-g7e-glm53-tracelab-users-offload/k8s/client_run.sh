#!/bin/sh
# Study 54, in an aiperf-client pod (copied to /tmp/client_run.sh by run_test.sh): one TraceLab replay
# against the vLLM Service with the shared system prompt /tmp/sys8k.txt, detached by the caller
# (setsid nohup) so kubectl exec returns. Writes /tmp/a13/runs/RUN/{aiperf.log,cgroup.log,exit} and
# AIPerf's artifacts (profile_export.jsonl) under /tmp/a13/runs/RUN/aiperf/. cgroup.log samples the
# pod's CPU (usage, throttling) and memory every 15 s; exit holds "<rc> <seconds>".
# The flags are the calibration's (2026-11-20, design "Load"): users ramp 1 -> CONC over RAMP_S s, a
# lane replays sessions one after the other, developer pauses capped at 1200 s, outputs as recorded.
# Usage: client_run.sh POOL RUN CONC RAMP_S DUR_S URL MODEL
POOL=$1 RUN=$2 CONC=$3 RAMP=$4 DUR=$5 URL=$6 MODEL=$7
W=/tmp/a13/runs/$RUN; mkdir -p "$W" /tmp/a13/mmap /tmp/a13/cache
(
  while [ ! -f "$W/exit" ]; do
    echo "$(date +%s) $(tr '\n' ' ' < /sys/fs/cgroup/cpu.stat) mem $(cat /sys/fs/cgroup/memory.current)" >> "$W/cgroup.log"
    sleep 15
  done
) &
t0=$(date +%s)
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 \
AIPERF_DATASET_MMAP_PREFAULT=false AIPERF_DATASET_MMAP_BASE_PATH=/tmp/a13/mmap AIPERF_DATASET_MMAP_CACHE_DIR=/tmp/a13/cache \
aiperf profile --model "$MODEL" --tokenizer /work/tokenizer \
  --url "$URL" --endpoint-type chat --streaming --custom-dataset-type tracelab \
  --input-file "$POOL" --system-prompt-file /tmp/sys8k.txt --no-fixed-schedule \
  --dataset-sampling-strategy shuffle --random-seed 1 \
  --concurrency "$CONC" --concurrency-ramp-duration "$RAMP" --inter-turn-delay-cap-seconds 1200 \
  --extra-inputs ignore_eos:true \
  --benchmark-duration "$DUR" --benchmark-grace-period 30 --ui simple --output-artifact-dir "$W/aiperf" > "$W/aiperf.log" 2>&1
rc=$?
echo "$rc $(( $(date +%s) - t0 ))" > "$W/exit"
