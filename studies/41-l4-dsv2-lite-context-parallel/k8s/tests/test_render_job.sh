#!/bin/bash
# Tests for ../render_job.sh. Run: bash k8s/tests/test_render_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }
bash "$R" 96 3600 "$TMP/j.yaml" >/dev/null 2>&1 || ko "renders"
if [ -f "$TMP/j.yaml" ]; then
  [ "$(yq '.kind' "$TMP/j.yaml")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/j.yaml")" = aiperf-dsv2 ] && ok "name" || ko "name"
  [ "$(yq '.metadata.namespace' "$TMP/j.yaml")" = llm-serving ] && ok "namespace" || ko "namespace"
  [ "$(yq '.spec.template.spec.nodeSelector."node-role"' "$TMP/j.yaml")" = system-m8a ] && ok "on the system node" || ko "on the system node"
  has "$TMP/j.yaml" "--concurrency 96 --concurrency-ramp-duration 3600 --benchmark-duration 3600 --benchmark-grace-period 60" \
    && ok "closed-loop users ramp" || ko "closed-loop users ramp"
  has "$TMP/j.yaml" "--request-rate" && ko "no request rate (closed loop)" || ok "no request rate (closed loop)"
  for f in "--prefix-prompt-pool-size 32" "--prefix-prompt-length 8192" "--synthetic-input-tokens-mean 2048" \
           "--synthetic-input-tokens-stddev 512" "--output-tokens-mean 256" "--output-tokens-stddev 64" \
           "--conversation-turn-mean 5" "--conversation-turn-stddev 1" "--conversation-turn-delay-mean 10000" \
           "--conversation-turn-delay-stddev 3000" "--extra-inputs ignore_eos:true" "--tokenizer-trust-remote-code" \
           "--num-dataset-entries 4000 --random-seed 40" "--num-dataset-entries 50 --random-seed 7" \
           "time_to_first_token:10000 inter_token_latency:50" "aiperf==0.11.0"; do
    has "$TMP/j.yaml" "$f" && ok "has $f" || ko "has $f"
  done
  has "$TMP/j.yaml" "--conversation-num" && ko "no --conversation-num (a session stop condition)" || ok "no --conversation-num (a session stop condition)"
  has "$TMP/j.yaml" "--shared-system-prompt-length" && ko "no shared system prompt (exclusive with the pool)" || ok "no shared system prompt (exclusive with the pool)"
  has "$TMP/j.yaml" "http://vllm-dsv2-0.vllm-dsv2-headless.llm-serving.svc.cluster.local:8000" && ok "targets vllm-dsv2-0" || ko "targets vllm-dsv2-0"
  has "$TMP/j.yaml" "MEASURED RUN START" && ok "marker" || ko "marker"
  has "$TMP/j.yaml" "deepseek-ai/DeepSeek-V2-Lite-Chat" && ok "tokenizer" || ko "tokenizer"
  has "$TMP/j.yaml" "MODEL_NAME=dsv2-lite" && ok "served model name" || ko "served model name"
  grep -q 'hf-token\|HF_TOKEN' "$TMP/j.yaml" && ko "no HF token" || ok "no HF token"
  [ "$(yq '.spec.template.spec.volumes[] | select(.name=="results") | .persistentVolumeClaim.claimName' "$TMP/j.yaml")" = aiperf-results-dsv2 ] && ok "results PVC" || ko "results PVC"
  [ "$(yq '.spec.template.spec.volumes[] | select(.name=="hf-cache") | .persistentVolumeClaim.claimName' "$TMP/j.yaml")" = aiperf-hf-cache-dsv2 ] && ok "HF cache PVC" || ko "HF cache PVC"
  grep -q '@[A-Z_]*@' "$TMP/j.yaml" && ko "no token left" || ok "no token left"
fi
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "users not a number" abc 3600
reject "users 1" 1 3600
reject "users fractional" 9.5 3600
reject "ramp 0" 96 0
reject "ramp not an integer" 96 45.5
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
