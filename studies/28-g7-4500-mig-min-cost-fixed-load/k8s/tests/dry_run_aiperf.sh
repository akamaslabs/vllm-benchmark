#!/bin/bash
# Local dry run of the study's AIPerf load (fixed mode) against mock_openai.py: AIPerf 0.11.0
# must accept --request-rate + gamma + seed + grace period with an inputs_json dataset, and
# send at R from the first second (no ramp, so no dead time).
# Usage: bash k8s/tests/dry_run_aiperf.sh [duration_s]   (network: pip and the tokenizer)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
D=${1:-60}
W=$(mktemp -d /tmp/aiperf-dry-XXXXXX)
python3 -m venv "$W/venv"; "$W/venv/bin/pip" install --quiet aiperf==0.11.0
rm -f /tmp/mock_openai.count
python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
A="$W/venv/bin/aiperf"; export AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10
COMMON=(--model qwen3-8b-mig --tokenizer Qwen/Qwen3-8B-FP8 --url http://127.0.0.1:18000 --endpoint-type chat --streaming --ui simple)
# An inputs.json in AIPerf's own format (the Job uses the ShareGPT one, same format).
"$A" profile "${COMMON[@]}" --synthetic-input-tokens-mean 100 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 8 --num-dataset-entries 50 --concurrency 1 --request-count 1 \
  --output-artifact-dir "$W/prep" > "$W/prep.log" 2>&1
rm -f /tmp/mock_openai.count
# The load arguments exactly as render_job.sh writes them, with a shorter duration.
bash "$K8S/render_job.sh" 0 fixed 3.3 0 "$W/job.yaml"
LOAD=$(grep -m1 'MEASURED RUN START: ' "$W/job.yaml" | sed -e 's/.*MEASURED RUN START: //' -e 's/"$//' -e "s/--benchmark-duration 780/--benchmark-duration $D/")
echo "load args: $LOAD"
# shellcheck disable=SC2086  # LOAD is a list of flags
"$A" profile "${COMMON[@]}" --input-file "$W/prep/inputs.json" --custom-dataset-type inputs_json $LOAD \
  --output-artifact-dir "$W/run" > "$W/run.log" 2>&1
grep -h 'Creating interval generator' "$W/run/logs/aiperf.log" | sed 's/.* - INFO - //'
python3 - "$D" <<'PY'
import statistics as st, sys
d = float(sys.argv[1])
t = [float(x) for x in open('/tmp/mock_openai.count')]
iv = [b - a for a, b in zip(t, t[1:])]
print('requests %d in %.0f s = %.2f req/s; interval cv %.2f (gamma k=4: 0.50)' % (len(t), d, len(t) / d, st.pstdev(iv) / st.mean(iv)))
PY
