#!/bin/bash
# Spike: does AIPerf 0.11.0's --concurrency-ramp-duration open sessions gradually and keep each one
# open through its think time (--conversation-turn-delay-mean) until its last turn? Scaled down:
# seconds instead of minutes, tiny prompts, a local stub server. Usage:
#   run_ramp.sh OUT_DIR [CONCURRENCY RAMP_S DELAY_MS DURATION_S SEED PORT]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1 CONC=${2:-30} RAMP=${3:-90} DELAY_MS=${4:-20000} DUR=${5:-150} SEED=${6:-1} PORT=${7:-18080}
mkdir -p "$OUT"
LOG="$OUT/stub.jsonl"; : > "$LOG"
python3 -I "$HERE/stub_server.py" "$PORT" "$LOG" & STUB=$!
trap 'kill $STUB 2>/dev/null || true' EXIT
sleep 1
HF_HUB_OFFLINE=1 aiperf profile --model stub --tokenizer Qwen/Qwen3-0.6B --url "http://127.0.0.1:$PORT" \
  --endpoint-type chat --streaming \
  --shared-system-prompt-length 100 --user-context-prompt-length 200 --num-dataset-entries "$CONC" \
  --isl 50 --isl-stddev 0 --osl 10 --osl-stddev 0 \
  --conversation-turn-mean 200 --conversation-turn-stddev 0 --conversation-num "$CONC" \
  --conversation-turn-delay-mean "$DELAY_MS" --conversation-turn-delay-stddev 0 \
  --concurrency "$CONC" --concurrency-ramp-duration "$RAMP" \
  --benchmark-duration "$DUR" --benchmark-grace-period 5 --random-seed "$SEED" \
  --ui simple --output-artifact-dir "$OUT/aiperf" > "$OUT/aiperf.log" 2>&1 || echo "aiperf exit $?"
tail -5 "$OUT/aiperf.log"
python3 -I "$HERE/analyze.py" "$LOG" 10
