#!/bin/bash
# Tests for ../render_job.sh. Run: bash k8s/tests/test_render_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }
bash "$R" 4 6000 "$TMP/j.yaml" >/dev/null 2>&1 || ko "renders"
if [ -f "$TMP/j.yaml" ]; then
  [ "$(yq '.kind' "$TMP/j.yaml")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/j.yaml")" = aiperf-l40s ] && ok "name" || ko "name"
  has "$TMP/j.yaml" "--request-rate 4 --request-rate-ramp-duration 6000 --arrival-pattern gamma --arrival-smoothness 4 --random-seed 30 --benchmark-duration 6000" \
    && ok "ramp load args" || ko "ramp load args"
  has "$TMP/j.yaml" "http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000" && ok "targets vllm-0" || ko "targets vllm-0"
  has "$TMP/j.yaml" "MEASURED RUN START" && ok "marker" || ko "marker"
  [ "$(yq '.spec.template.spec.containers[0].env[] | select(.name=="AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL") | .value' "$TMP/j.yaml")" = 10 ] \
    && ok "ramp update interval 10" || ko "ramp update interval 10"
  grep -q '@[A-Z_]*@' "$TMP/j.yaml" && ko "no token left" || ok "no token left"
  # Multi-turn synthetic load: the measured run carries the profile and a pool of
  # 4 x 6000 / 2 / 2 + 1 = 6001 conversations; the warm-up is single-turn.
  has "$TMP/j.yaml" 'aiperf profile $COMMON $PROMPT $TURNS --num-dataset-entries 6001' && ok "run: profile and pool 6001" || ko "run: profile and pool 6001"
  has "$TMP/j.yaml" 'PROMPT="--shared-system-prompt-length 1000 --synthetic-input-tokens-mean 300 --synthetic-input-tokens-stddev 150"' \
    && ok "prompt profile" || ko "prompt profile"
  has "$TMP/j.yaml" 'TURNS="--conversation-turn-mean 3 --conversation-turn-stddev 1 --conversation-turn-delay-mean 15000 --conversation-turn-delay-stddev 5000"' \
    && ok "turn profile" || ko "turn profile"
  has "$TMP/j.yaml" 'aiperf profile $COMMON $PROMPT --conversation-turn-mean 1' && ok "warm-up single-turn" || ko "warm-up single-turn"
  has "$TMP/j.yaml" '--tokenizer $TOKENIZER --tokenizer-revision $TOKENIZER_REV' && ok "tokenizer with revision" || ko "tokenizer with revision"
  has "$TMP/j.yaml" 'TOKENIZER=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit' && ok "served checkpoint's tokenizer" || ko "served checkpoint's tokenizer"
  # No output cap (thinking runs to its end), no session stop, no file replay.
  for f in --output-tokens-mean --conversation-num --num-conversations --input-file; do
    grep -v '^ *#' "$TMP/j.yaml" | grep -qF -- "$f" && ko "no $f (outside comments)" || ok "no $f (outside comments)"
  done
fi
bash "$R" --closed 32 320 "$TMP/c.yaml" >/dev/null 2>&1 || ko "renders closed"
if [ -f "$TMP/c.yaml" ]; then
  [ "$(yq '.kind' "$TMP/c.yaml")" = Job ] && ok "closed: valid YAML" || ko "closed: valid YAML"
  has "$TMP/c.yaml" "MEASURED RUN START: --concurrency 32 --request-count 320 --random-seed 30" \
    && ok "closed load args" || ko "closed load args"
  grep -F "MEASURED RUN START" "$TMP/c.yaml" | grep -qF -- "--request-rate" && ko "closed: no rate args" || ok "closed: no rate args"
  has "$TMP/c.yaml" '--num-dataset-entries 161' && ok "closed: pool 320 / 2 + 1" || ko "closed: pool 320 / 2 + 1"
fi
bash "$R" --closed 4 8 "$TMP/m.yaml" >/dev/null 2>&1 && has "$TMP/m.yaml" '$TURNS --num-dataset-entries 100' \
  && ok "pool at least 100" || ko "pool at least 100"
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "rate not a number" abc 4500
reject "ramp 0" 30 0
reject "ramp not an integer" 30 45.5
reject "closed: count below concurrency" --closed 32 16
reject "closed: concurrency 0" --closed 0 10
reject "closed: missing count" --closed 32
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
