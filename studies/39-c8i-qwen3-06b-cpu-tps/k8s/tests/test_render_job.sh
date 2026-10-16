#!/bin/bash
# Tests for ../render_job.sh. Run: bash k8s/tests/test_render_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }
bash "$R" 12 3000 "$TMP/j.yaml" >/dev/null 2>&1 || ko "renders"
if [ -f "$TMP/j.yaml" ]; then
  [ "$(yq '.kind' "$TMP/j.yaml")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/j.yaml")" = aiperf-cpu ] && ok "name" || ko "name"
  [ "$(yq '.metadata.namespace' "$TMP/j.yaml")" = llm-cpu ] && ok "namespace" || ko "namespace"
  [ "$(yq '.spec.template.spec.nodeSelector."node-role"' "$TMP/j.yaml")" = system-m8a ] && ok "on the system node" || ko "on the system node"
  has "$TMP/j.yaml" "--request-rate 12 --request-rate-ramp-duration 3000 --arrival-pattern gamma --arrival-smoothness 4 --benchmark-duration 3000" \
    && ok "ramp load args" || ko "ramp load args"
  for f in "--prefix-prompt-pool-size 16" "--prefix-prompt-length 1536" "--conversation-turn-mean 4" \
           "--synthetic-input-tokens-mean 384" "--output-tokens-mean 192" "--extra-inputs ignore_eos:true" \
           "--num-dataset-entries 8000 --random-seed 39" "--num-dataset-entries 100 --random-seed 7" "aiperf==0.11.0"; do
    has "$TMP/j.yaml" "$f" && ok "has $f" || ko "has $f"
  done
  has "$TMP/j.yaml" "--conversation-num" && ko "no --conversation-num (a session stop condition)" || ok "no --conversation-num (a session stop condition)"
  has "$TMP/j.yaml" "--shared-system-prompt-length" && ko "no shared system prompt (exclusive with the pool)" || ok "no shared system prompt (exclusive with the pool)"
  has "$TMP/j.yaml" "http://vllm-0.vllm-headless.llm-cpu.svc.cluster.local:8000" && ok "targets vllm-0" || ko "targets vllm-0"
  has "$TMP/j.yaml" "MEASURED RUN START" && ok "marker" || ko "marker"
  has "$TMP/j.yaml" "Qwen/Qwen3-0.6B" && ok "tokenizer" || ko "tokenizer"
  grep -q 'hf-token\|HF_TOKEN' "$TMP/j.yaml" && ko "no HF token" || ok "no HF token"
  [ "$(yq '.spec.template.spec.containers[0].env[] | select(.name=="AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL") | .value' "$TMP/j.yaml")" = 10 ] \
    && ok "ramp update interval 10" || ko "ramp update interval 10"
  grep -q '@[A-Z_]*@' "$TMP/j.yaml" && ko "no token left" || ok "no token left"
fi
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "rate not a number" abc 3000
reject "rate 0" 0 3000
reject "ramp 0" 12 0
reject "ramp not an integer" 12 45.5
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
