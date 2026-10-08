#!/bin/bash
# Local end-to-end dry run of study 31's AIPerf Job script (thinking mode): the rendered Job's
# own shell (render_job.sh --closed), under dash as in python:3.12-slim, with real AIPerf 0.11.0
# against mock_openai.py streaming `reasoning` deltas, then content. Checks:
#   1. the copy without max_tokens is built, and no request reaches the server with max_tokens /
#      max_completion_tokens (the cache it starts from has them, as the ShareGPT one);
#   2. the warm-up passes when every request lasts more than a minute (thinking), and so does
#      the measured run (a 60 s duration-based warm-up failed here: no request completed);
#   3. profile_export.jsonl is written by default and the LENGTHS lines count reasoning tokens.
# Usage: bash k8s/tests/dry_run_thinking.sh   (network: pip and the Gemma 4 tokenizer; ~6 min)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
MODEL=gemma4-26b-l40s-think
W=$(mktemp -d /tmp/aiperf-think-XXXXXX)
python3 -m venv "$W/venv"; "$W/venv/bin/pip" install --quiet aiperf==0.11.0
# Each request: 6 reasoning + 6 content chunks, 6 s apart = 72 s, longer than the 60 s warm-up.
rm -f /tmp/mock_openai.count
MOCK_REASONING_CHUNKS=6 MOCK_CHUNKS=6 MOCK_CHUNK_S=6 MOCK_BODIES="$W/bodies.jsonl" \
  python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
# A cache in AIPerf's inputs.json format with max_completion_tokens set (the Job's ShareGPT cache
# has the same format; its prep run is not repeated here: it downloads ShareGPT).
"$W/venv/bin/aiperf" profile --model $MODEL --tokenizer RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic \
  --url http://127.0.0.1:18000 --endpoint-type chat --streaming --ui simple \
  --synthetic-input-tokens-mean 100 --synthetic-input-tokens-stddev 0 --output-tokens-mean 8 \
  --num-dataset-entries 50 --concurrency 1 --request-count 1 \
  --output-artifact-dir "$W/prep" > "$W/prep.log" 2>&1 || { tail -30 "$W/prep.log"; exit 1; }
grep -qE '"max_(completion_)?tokens"' "$W/prep/inputs.json" && ok "the starting cache carries max_tokens" \
  || ko "the starting cache carries max_tokens"
mkdir -p "$W/bench/sharegpt-cache"; cp "$W/prep/inputs.json" "$W/bench/sharegpt-cache/inputs-$MODEL.json"
: > "$W/bodies.jsonl"
bash "$K8S/render_job.sh" --closed 4 8 "$W/job.yaml"
yq '.spec.template.spec.containers[0].args[0]' "$W/job.yaml" \
  | sed -e "s|/tmp/warmup|@W@/warmup|g" -e "s|/tmp/sharegpt-prep|@W@/sharegpt-prep|g" \
        -e "s|/benchmarks|@W@/bench|g" -e "s|http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000|http://127.0.0.1:18000|" \
  | sed -e "s|@W@|$W|g" > "$W/job.sh"
if PATH="$W/venv/bin:$PATH" /bin/dash "$W/job.sh" > "$W/job.log" 2>&1; then ok "job script exits 0 (warm-up and run)"
else ko "job script exits 0 (warm-up and run)"; tail -40 "$W/job.log"; fi
grep -E '^nomax:|LENGTHS' "$W/job.log" || true
[ -f "$W/bench/sharegpt-cache/inputs-$MODEL-nomax.json" ] && ok "nomax copy built" || ko "nomax copy built"
N=$(wc -l < "$W/bodies.jsonl" | tr -d ' ')
CAPPED=$(grep -cE '"max_(completion_)?tokens"' "$W/bodies.jsonl" || true)
{ [ "$N" -gt 0 ] && [ "$CAPPED" = 0 ]; } && ok "$N requests sent, none with max_tokens" || ko "requests without max_tokens ($CAPPED of $N capped)"
grep -qE 'LENGTHS reasoning_token_count +n= *[1-9]' "$W/job.log" && ok "LENGTHS counts reasoning tokens" || ko "LENGTHS counts reasoning tokens"
grep -qE 'LENGTHS time_to_first_output_token +n= *[1-9]' "$W/job.log" && ok "LENGTHS has the time to the first answer token" \
  || ko "LENGTHS has the time to the first answer token"
grep -qE 'LENGTHS output >= 15300 tokens .*: 0 of [1-9]' "$W/job.log" && ok "LENGTHS max_model_len check" || ko "LENGTHS max_model_len check"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
