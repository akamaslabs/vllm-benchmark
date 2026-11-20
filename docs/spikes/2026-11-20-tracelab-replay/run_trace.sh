#!/bin/bash
# Spike: AIPerf 0.13.0 replays a prepared TraceLab sample in session-concurrency mode with a ramp
# against the stub. Usage: [AIPERF=/path/to/aiperf] run_trace.sh OUT_DIR INPUT [CONC RAMP_S DUR_S DELAY_CAP_S PORT]
# AIPERF defaults to `aiperf` on PATH (must be 0.13.0: 0.11.0 has no tracelab loader).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
AIPERF=${AIPERF:-aiperf}
OUT=$1 IN=$2 CONC=${3:-30} RAMP=${4:-60} DUR=${5:-150} CAP=${6:-20} PORT=${7:-18090}
mkdir -p "$OUT"
LOG="$OUT/stub.jsonl"; : > "$LOG"
python3 -I "$HERE/stub_trace.py" "$PORT" "$LOG" & STUB=$!
# Sum of the RSS of every aiperf process, every 2 s.
( while sleep 2; do echo "$(date +%s) $(ps -axo rss=,command= | awk 'tolower($0) ~ /aiperf/ && !/awk/ && !/stub_trace/ {s+=$1} END {printf "%d", s/1024}')"; done ) > "$OUT/rss.log" 2>/dev/null & RSS=$!
trap 'kill $STUB $RSS 2>/dev/null || true' EXIT
sleep 1
HF_HUB_OFFLINE=1 "$AIPERF" profile --model stub --tokenizer Qwen/Qwen3-0.6B \
  --url "http://127.0.0.1:$PORT" --endpoint-type chat --streaming \
  --custom-dataset-type tracelab --input-file "$IN" --no-fixed-schedule \
  --dataset-sampling-strategy shuffle --random-seed 1 \
  --concurrency "$CONC" --concurrency-ramp-duration "$RAMP" --inter-turn-delay-cap-seconds "$CAP" \
  --benchmark-duration "$DUR" --benchmark-grace-period 5 \
  --ui simple --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1 || echo "aiperf exit $?"
tail -5 "$OUT/aiperf.log"
echo "peak RSS of aiperf processes: $(sort -k2 -n "$OUT/rss.log" | tail -1)"
