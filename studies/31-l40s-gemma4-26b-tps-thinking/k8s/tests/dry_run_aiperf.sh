#!/bin/bash
# Local dry run of the study's AIPerf load against mock_openai.py: AIPerf 0.11.0 must load the
# Gemma 4 tokenizer and accept the ramp arguments exactly as render_job.sh writes them (a
# shorter ramp), and the ramp must send requests from the first minutes (study 27's
# AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10 fix).
# Usage: bash k8s/tests/dry_run_aiperf.sh [ramp_s] [rate]   (network: pip and the tokenizer)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
D=${1:-120}; R=${2:-20}
W=$(mktemp -d /tmp/aiperf-dry-XXXXXX)
python3 -m venv "$W/venv"; "$W/venv/bin/pip" install --quiet aiperf==0.11.0
rm -f /tmp/mock_openai.count
python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
A="$W/venv/bin/aiperf"; export AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10
COMMON=(--model gemma4-26b-l40s-think --tokenizer RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic --url http://127.0.0.1:18000 --endpoint-type chat --streaming --ui simple)
# An inputs.json in AIPerf's own format (the Job uses the ShareGPT one, same format).
"$A" profile "${COMMON[@]}" --synthetic-input-tokens-mean 100 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 8 --num-dataset-entries 50 --concurrency 1 --request-count 1 \
  --output-artifact-dir "$W/prep" > "$W/prep.log" 2>&1 || { tail -30 "$W/prep.log"; exit 1; }
rm -f /tmp/mock_openai.count
bash "$K8S/render_job.sh" "$R" "$D" "$W/job.yaml"
LOAD=$(grep -m1 'MEASURED RUN START: ' "$W/job.yaml" | sed -e 's/.*MEASURED RUN START: //' -e 's/"$//')
echo "load args: $LOAD"
# shellcheck disable=SC2086  # LOAD is a list of flags
"$A" profile "${COMMON[@]}" --input-file "$W/prep/inputs.json" --custom-dataset-type inputs_json $LOAD \
  --output-artifact-dir "$W/run" > "$W/run.log" 2>&1 || { tail -30 "$W/run.log"; exit 1; }
grep -h -i 'ramp\|interval generator' "$W/run/logs/aiperf.log" | sed 's/.* - INFO - //' | head -5
python3 - "$D" "$R" <<'PY'
import sys
d, r = float(sys.argv[1]), float(sys.argv[2])
t = [float(x) for x in open('/tmp/mock_openai.count')]
t0 = t[0]
first_half = sum(1 for x in t if x - t0 < d / 2); second = len(t) - first_half
print('requests %d in %.0f s (linear ramp to %.0f req/s expects ~%.0f); first half %d, second half %d (expect ~1:3)'
      % (len(t), d, r, r * d / 2, first_half, second))
PY
