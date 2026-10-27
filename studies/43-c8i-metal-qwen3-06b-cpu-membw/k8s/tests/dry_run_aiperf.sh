#!/bin/bash
# Local dry run of the study's AIPerf load against mock_openai.py: AIPerf 0.11.0 must load the
# Qwen3-0.6B tokenizer and accept the workload (lib_workload.sh) and ramp arguments exactly as
# render_job.sh writes them (a shorter ramp), and the ramp must send requests from the first
# minutes (study 27's AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10 fix).
# Usage: bash k8s/tests/dry_run_aiperf.sh [ramp_s] [rate]   (network: pip and the tokenizer)
# DRY_VENV=<dir> reuses an existing venv with aiperf==0.11.0.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
D=${1:-120}; R=${2:-10}
W=$(mktemp -d /tmp/aiperf-dry-XXXXXX)
VENV=${DRY_VENV:-$W/venv}
if [ ! -x "$VENV/bin/aiperf" ]; then python3 -m venv "$VENV"; "$VENV/bin/pip" install --quiet aiperf==0.11.0; fi
rm -f /tmp/mock_openai.count
python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
A="$VENV/bin/aiperf"; export AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10
# shellcheck source=../lib_workload.sh
source "$K8S/lib_workload.sh"
COMMON=(--model qwen3-06b --tokenizer Qwen/Qwen3-0.6B --url http://127.0.0.1:18000 --ui simple)
bash "$K8S/render_job.sh" "$R" "$D" "$W/job.yaml"
LOAD=$(grep -m1 'MEASURED RUN START: ' "$W/job.yaml" | sed -e 's/.*MEASURED RUN START: //' -e 's/"$//')
echo "load args: $LOAD"
# The warm-up as the Job runs it (few sessions, concurrency 4), shortened.
# shellcheck disable=SC2086  # WORKLOAD_ARGS / LOAD are lists of flags
"$A" profile "${COMMON[@]}" $WORKLOAD_ARGS --num-dataset-entries "$WARMUP_ENTRIES" --random-seed "$WARMUP_SEED" \
  --concurrency 4 --benchmark-duration 10 --output-artifact-dir "$W/warmup" > "$W/warmup.log" 2>&1 \
  || { tail -30 "$W/warmup.log"; exit 1; }
echo "warm-up ok"
rm -f /tmp/mock_openai.count
# The measured run (fewer sessions than the Job, to keep the dataset generation short).
# shellcheck disable=SC2086
"$A" profile "${COMMON[@]}" $WORKLOAD_ARGS --num-dataset-entries "${DRY_ENTRIES:-1000}" --random-seed "$WORKLOAD_SEED" \
  $LOAD --goodput "time_to_first_token:3000 inter_token_latency:100" \
  --output-artifact-dir "$W/run" > "$W/run.log" 2>&1 || { tail -30 "$W/run.log"; exit 1; }
grep -h -i 'ramp\|interval generator\|prefix\|conversation' "$W/run/logs/aiperf.log" | sed 's/.* - INFO - //' | head -8 || true
python3 - "$D" "$R" "$W/run/profile_export_aiperf.json" <<'PY'
import json, sys
d, r = float(sys.argv[1]), float(sys.argv[2])
t = [float(x) for x in open('/tmp/mock_openai.count')]
t0 = t[0]
first_half = sum(1 for x in t if x - t0 < d / 2); second = len(t) - first_half
print('requests %d in %.0f s (linear ramp to %.0f req/s expects ~%.0f); first half %d, second half %d (expect ~1:3)'
      % (len(t), d, r, r * d / 2, first_half, second))
j = json.load(open(sys.argv[3]))
print('AIPerf JSON: ttft p95 %s ms, itl p95 %s ms, req/s %s, total tok/s %s' % (
    (j.get('time_to_first_token') or {}).get('p95'), (j.get('inter_token_latency') or {}).get('p95'),
    (j.get('request_throughput') or {}).get('avg'), (j.get('total_token_throughput') or {}).get('avg')))
PY
