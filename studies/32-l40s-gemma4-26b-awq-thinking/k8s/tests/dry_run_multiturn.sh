#!/bin/bash
# Local end-to-end dry run of study 32's AIPerf Job script (synthetic multi-turn, thinking): the
# rendered Job's own shell, under dash as in python:3.12-slim, with real AIPerf 0.11.0 against
# mock_openai.py streaming `reasoning` deltas, then content. Two runs of the script:
#   A. closed loop (render_job.sh --closed 4 24): the warm-up and the run exit 0; no request
#      carries max_tokens / max_completion_tokens; every measured request opens with the same
#      system prompt (the warm-up has its own seed, hence its own); later turns carry the history
#      (earlier user messages and the mock's replies), so the prompt grows turn after turn, and a
#      reply in the history holds its reasoning too (AIPerf 0.11.0 keeps reasoning-only chunks:
#      the profile is calibrated for it, 05-job_template.yaml); the LENGTHS lines report
#      input_sequence_length and the max_model_len check.
#   B. the open-loop ramp exactly as render_job.sh writes it (shorter: R req/s over D s): AIPerf
#      accepts it with conversations, and the requests follow the ramp (first half : second
#      half ~ 1:3, study 27's AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10 fix).
# Usage: bash k8s/tests/dry_run_multiturn.sh [ramp_s] [rate]   (network: pip and the tokenizer;
# ~8 min with the defaults 180 s, 4 req/s)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
D=${1:-180}; R=${2:-4}
W=$(mktemp -d /tmp/aiperf-multiturn-XXXXXX)
python3 -m venv "$W/venv"; "$W/venv/bin/pip" install --quiet aiperf==0.11.0
rm -f /tmp/mock_openai.count
MOCK_REASONING_CHUNKS=3 MOCK_CHUNKS=20 MOCK_CHUNK_S=0.05 MOCK_BODIES="$W/bodies.jsonl" \
  python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
job_script() {  # rendered Job -> local shell script ($1 job.yaml, $2 out.sh)
  yq '.spec.template.spec.containers[0].args[0]' "$1" \
    | sed -e "s|/tmp/warmup|@W@/warmup|g" -e "s|/benchmarks|@W@/bench|g" \
          -e "s|http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000|http://127.0.0.1:18000|" \
    | sed -e "s|@W@|$W|g" > "$2"
}

# --- A. closed loop ------------------------------------------------------------------------
: > "$W/bodies.jsonl"
bash "$K8S/render_job.sh" --closed 4 24 "$W/a.yaml"; job_script "$W/a.yaml" "$W/a.sh"
if PATH="$W/venv/bin:$PATH" AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10 /bin/dash "$W/a.sh" > "$W/a.log" 2>&1; then
  ok "A: job script exits 0 (warm-up and run)"
else ko "A: job script exits 0 (warm-up and run)"; tail -40 "$W/a.log"; fi
grep -E 'LENGTHS' "$W/a.log" || true
python3 - "$W/bodies.jsonl" > "$W/a.check" <<'PY'
import json, sys
bodies = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
measured = bodies[8:]   # the first 8 are the warm-up (concurrency 4, 8 requests)
capped = sum(1 for b in bodies if "max_tokens" in b or "max_completion_tokens" in b)
systems = {json.dumps(b["messages"][0]) for b in measured if b["messages"][0]["role"] == "system"}
no_system = sum(1 for b in bodies if b["messages"][0]["role"] != "system")
lens = [len(b["messages"]) for b in bodies]
assistant = [m for b in bodies for m in b["messages"] if m["role"] == "assistant"]
with_reasoning = sum(1 for m in assistant if "think" in json.dumps(m))
chars = [len(json.dumps(b["messages"])) for b in bodies]
print(" bodies", len(bodies), "capped", capped, "distinct_system", len(systems), "no_system", no_system,
      "max_messages", max(lens), "assistant_msgs", len(assistant), "with_reasoning", with_reasoning,
      "chars_min", min(chars), "chars_max", max(chars))
PY
cat "$W/a.check"
v() { sed -n "s/.* $1 \([0-9]*\).*/\1/p" "$W/a.check"; }
[ "$(v bodies)" -gt 24 ] && [ "$(v capped)" = 0 ] && ok "A: $(v bodies) requests, none with max_tokens" || ko "A: requests without max_tokens"
{ [ "$(v distinct_system)" = 1 ] && [ "$(v no_system)" = 0 ]; } && ok "A: one system prompt shared by every measured request" \
  || ko "A: one system prompt shared by every measured request"
{ [ "$(v max_messages)" -ge 4 ] && [ "$(v assistant_msgs)" -gt 0 ]; } && ok "A: later turns carry the history ($(v max_messages) messages)" \
  || ko "A: later turns carry the history"
[ "$(v with_reasoning)" = "$(v assistant_msgs)" ] && ok "A: every reply in the history holds its reasoning (as calibrated)" \
  || ko "A: replies in the history: $(v with_reasoning) of $(v assistant_msgs) with reasoning (the calibration assumes all)"
grep -qE 'LENGTHS input_sequence_length +n= *[1-9]' "$W/a.log" && ok "A: LENGTHS reports the input length" || ko "A: LENGTHS input length"
grep -qE 'LENGTHS reasoning_token_count +n= *[1-9]' "$W/a.log" && ok "A: LENGTHS counts reasoning tokens" || ko "A: LENGTHS reasoning tokens"
grep -qE 'LENGTHS input \+ output >= 95000 tokens .*: 0 of [1-9]' "$W/a.log" && ok "A: LENGTHS max_model_len check" || ko "A: LENGTHS max_model_len check"

# --- B. the ramp ---------------------------------------------------------------------------
: > "$W/bodies.jsonl"; rm -f /tmp/mock_openai.count
bash "$K8S/render_job.sh" "$R" "$D" "$W/b.yaml"; job_script "$W/b.yaml" "$W/b.sh"
if PATH="$W/venv/bin:$PATH" AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10 /bin/dash "$W/b.sh" > "$W/b.log" 2>&1; then
  ok "B: ramp run exits 0"
else ko "B: ramp run exits 0"; tail -40 "$W/b.log"; fi
python3 - "$D" "$R" <<'PY' | tee "$W/b.check"
import sys
d, r = float(sys.argv[1]), float(sys.argv[2])
t = sorted(float(x) for x in open('/tmp/mock_openai.count'))[8:]   # the first 8 are the warm-up
t0 = t[0]
t = [x for x in t if x - t0 < d]
first = sum(1 for x in t if x - t0 < d / 2); second = len(t) - first
print('requests %d in %.0f s (linear ramp to %.0f req/s expects ~%.0f); first half %d, second half %d, ratio %.2f (expect ~3)'
      % (len(t), d, r, r * d / 2, first, second, second / max(first, 1)))
PY
RATIO=$(sed -n 's/.*ratio \([0-9.]*\).*/\1/p' "$W/b.check")
awk -v x="${RATIO:-0}" 'BEGIN { exit !(x >= 2 && x <= 4.5) }' && ok "B: the requests follow the ramp (ratio $RATIO)" \
  || ko "B: the requests follow the ramp (ratio ${RATIO:-none})"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
