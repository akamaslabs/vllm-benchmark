#!/bin/bash
# Local dry run of the study's AIPerf load against mock_openai.py: AIPerf 0.11.0 must load the
# DeepSeek-V2-Lite-Chat tokenizer and accept the workload (lib_workload.sh) and the closed-loop
# users ramp exactly as render_job.sh writes them (a shorter ramp, fewer users and sessions),
# and the number of sessions in flight must grow over the ramp.
# Usage: bash k8s/tests/dry_run_aiperf.sh [ramp_s] [users]   (network: pip and the tokenizer)
# DRY_VENV=<dir> reuses an existing venv with aiperf==0.11.0; DRY_ENTRIES (default 200) sets
# the measured sessions (the Job uses 4000: time its generation with DRY_ENTRIES=4000).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
D=${1:-120}; U=${2:-16}
W=$(mktemp -d /tmp/aiperf-dry-XXXXXX)
VENV=${DRY_VENV:-$W/venv}
if [ ! -x "$VENV/bin/aiperf" ]; then python3 -m venv "$VENV"; "$VENV/bin/pip" install --quiet aiperf==0.11.0; fi
rm -f /tmp/mock_openai.count
python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
A="$VENV/bin/aiperf"
# shellcheck source=../lib_workload.sh
source "$K8S/lib_workload.sh"
COMMON=(--model dsv2-lite --tokenizer deepseek-ai/DeepSeek-V2-Lite-Chat --url http://127.0.0.1:18000 --ui simple)
bash "$K8S/render_job.sh" "$U" "$D" "$W/job.yaml"
LOAD=$(grep -m1 'MEASURED RUN START: ' "$W/job.yaml" | sed -e 's/.*MEASURED RUN START: //' -e 's/"$//')
echo "load args: $LOAD"
# The warm-up as the Job runs it (few sessions, concurrency 4), shortened.
# shellcheck disable=SC2086  # WORKLOAD_ARGS / LOAD are lists of flags
"$A" profile "${COMMON[@]}" $WORKLOAD_ARGS --num-dataset-entries "$WARMUP_ENTRIES" --random-seed "$WARMUP_SEED" \
  --concurrency 4 --benchmark-duration 10 --output-artifact-dir "$W/warmup" > "$W/warmup.log" 2>&1 \
  || { tail -30 "$W/warmup.log"; exit 1; }
echo "warm-up ok"
rm -f /tmp/mock_openai.count
T0=$(date +%s)
# shellcheck disable=SC2086
"$A" profile "${COMMON[@]}" $WORKLOAD_ARGS --num-dataset-entries "${DRY_ENTRIES:-200}" --random-seed "$WORKLOAD_SEED" \
  $LOAD --goodput "time_to_first_token:10000 inter_token_latency:50" \
  --output-artifact-dir "$W/run" > "$W/run.log" 2>&1 || { tail -30 "$W/run.log"; exit 1; }
echo "measured run took $(( $(date +%s) - T0 )) s (ramp $D s + dataset generation + grace)"
grep -h -i 'ramp\|concurren\|prefix\|conversation' "$W/run/logs/aiperf.log" | sed 's/.* - INFO - //' | head -8 || true
python3 - "$D" "$U" "$W/run/profile_export_aiperf.json" <<'PY'
import json, sys
d, u = float(sys.argv[1]), int(sys.argv[2])
t = sorted(float(x) for x in open('/tmp/mock_openai.count'))
t0 = t[0]
first_half = sum(1 for x in t if x - t0 < d / 2); second = len(t) - first_half
print('requests %d in %.0f s, users ramp 1 -> %d; first half %d, second half %d (a linear users ramp expects ~1:3, think time included)'
      % (len(t), d, u, first_half, second))
j = json.load(open(sys.argv[3]))
print('AIPerf JSON: ttft p95 %s ms, isl avg %s, req/s %s, total tok/s %s' % (
    (j.get('time_to_first_token') or {}).get('p95'), (j.get('input_sequence_length') or {}).get('avg'),
    (j.get('request_throughput') or {}).get('avg'), (j.get('total_token_throughput') or {}).get('avg')))
PY
