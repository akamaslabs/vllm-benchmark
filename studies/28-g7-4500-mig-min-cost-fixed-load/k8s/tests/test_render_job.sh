#!/bin/bash
# Tests for ../render_job.sh. Run: bash k8s/tests/test_render_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }

bash "$R" 0 fixed 3.3 0 "$TMP/r0.yaml" >/dev/null 2>&1 || ko "fixed r0 renders"
if [ -f "$TMP/r0.yaml" ]; then
  [ "$(yq '.kind' "$TMP/r0.yaml")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/r0.yaml")" = aiperf-mig-r0 ] && ok "name r0" || ko "name r0"
  [ "$(yq '.metadata.labels.app' "$TMP/r0.yaml")" = aiperf-mig ] && ok "label" || ko "label"
  has "$TMP/r0.yaml" "--request-rate 3.3 --arrival-pattern gamma --arrival-smoothness 4 --random-seed 28 --benchmark-duration 780" \
    && ok "fixed load args" || ko "fixed load args"
  has "$TMP/r0.yaml" "ramp-duration" && ko "no ramp in fixed mode" || ok "no ramp in fixed mode"
  has "$TMP/r0.yaml" "http://vllm-0.vllm-headless.gpu-sharing.svc.cluster.local:8000" && ok "r0 targets vllm-0" || ko "r0 targets vllm-0"
  has "$TMP/r0.yaml" '[ "make" = make ]' && ok "r0 makes the cache" || ko "r0 makes the cache"
  has "$TMP/r0.yaml" "MEASURED RUN START" && ok "marker" || ko "marker"
  grep -q '@[A-Z_]*@' "$TMP/r0.yaml" && ko "no token left" || ok "no token left"
fi
bash "$R" 1 fixed 3.3 0 "$TMP/r1.yaml" >/dev/null 2>&1 || ko "fixed r1 renders"
if [ -f "$TMP/r1.yaml" ]; then
  [ "$(yq '.metadata.name' "$TMP/r1.yaml")" = aiperf-mig-r1 ] && ok "name r1" || ko "name r1"
  has "$TMP/r1.yaml" "--random-seed 29" && ok "r1 seed 29" || ko "r1 seed 29"
  has "$TMP/r1.yaml" "http://vllm-1.vllm-headless.gpu-sharing.svc.cluster.local:8000" && ok "r1 targets vllm-1" || ko "r1 targets vllm-1"
  has "$TMP/r1.yaml" '[ "wait" = make ]' && ok "r1 waits for the cache" || ko "r1 waits for the cache"
fi
bash "$R" 0 ramp 12 2400 "$TMP/ramp.yaml" >/dev/null 2>&1 || ko "ramp renders"
[ -f "$TMP/ramp.yaml" ] && has "$TMP/ramp.yaml" "--request-rate 12 --request-rate-ramp-duration 2400 --arrival-pattern gamma --arrival-smoothness 4 --random-seed 28 --benchmark-duration 2400" \
  && ok "ramp load args" || ko "ramp load args"
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "replica 2" 2 fixed 3.3 0
reject "unknown mode" 0 burst 3.3 0
reject "rate not a number" 0 fixed abc 0
reject "ramp without duration" 0 ramp 12 0
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
