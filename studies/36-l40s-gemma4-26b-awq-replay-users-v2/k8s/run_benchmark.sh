#!/usr/bin/env bash
# Replay benchmark with AIPerf. Settings are in benchmark.env (copy benchmark.env.example).
# Any setting can be overridden on the command line:   ./run_benchmark.sh USERS=10 NAME=u10
set -euo pipefail
cd "$(dirname "$0")"
source ./benchmark.env
for kv in "$@"; do export "$kv"; done

OUT="data/results/run_$(date +%Y%m%d%H%M)_$NAME"
mkdir -p "$OUT"
echo "endpoint=$ENDPOINT model=$MODEL dataset=$DATASET"
echo "users=$USERS duration=${DURATION}s ramp=${RAMP}s cap=${CAP}s grace=${GRACE}s -> $OUT"

../llm-coding-benchmark/.venv/bin/aiperf profile \
  --url "$ENDPOINT" \
  --endpoint-type responses \
  --model "$MODEL" \
  ${TOKENIZER:+--tokenizer "$TOKENIZER"} \
  ${API_KEY:+--api-key "$API_KEY"} \
  ${HEADER:+--header "$HEADER"} \
  --input-file "$DATASET" \
  --custom-dataset-type mooncake_trace \
  --concurrency "$USERS" \
  --concurrency-ramp-duration "$RAMP" \
  --benchmark-duration "$DURATION" \
  --benchmark-grace-period "$GRACE" \
  --inter-turn-delay-cap-seconds "$CAP" \
  --random-seed 42 \
  --connection-reuse-strategy sticky-user-sessions \
  --use-server-token-count \
  --streaming \
  --artifact-dir "$OUT" \
  --export-level records \
  --ui simple \
  2>&1 | tee "$OUT/run.log"
