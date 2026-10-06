#!/bin/bash
# Tests for ../render_job.sh. Run: bash k8s/tests/test_render_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }
bash "$R" 30 4500 "$TMP/j.yaml" >/dev/null 2>&1 || ko "renders"
if [ -f "$TMP/j.yaml" ]; then
  [ "$(yq '.kind' "$TMP/j.yaml")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/j.yaml")" = aiperf-l40s ] && ok "name" || ko "name"
  has "$TMP/j.yaml" "--request-rate 30 --request-rate-ramp-duration 4500 --arrival-pattern gamma --arrival-smoothness 4 --random-seed 30 --benchmark-duration 4500" \
    && ok "ramp load args" || ko "ramp load args"
  has "$TMP/j.yaml" "http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000" && ok "targets vllm-0" || ko "targets vllm-0"
  has "$TMP/j.yaml" "MEASURED RUN START" && ok "marker" || ko "marker"
  [ "$(yq '.spec.template.spec.containers[0].env[] | select(.name=="AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL") | .value' "$TMP/j.yaml")" = 10 ] \
    && ok "ramp update interval 10" || ko "ramp update interval 10"
  grep -q '@[A-Z_]*@' "$TMP/j.yaml" && ko "no token left" || ok "no token left"
fi
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "rate not a number" abc 4500
reject "ramp 0" 30 0
reject "ramp not an integer" 30 45.5
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
