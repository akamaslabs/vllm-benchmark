#!/bin/bash
# Tests for ../render_job.sh. Run: bash k8s/tests/test_render_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }
code() { grep -v '^ *#' "$1"; }  # the rendered Job without comment lines
# The customer's script, byte-identical to what they sent (2026-10-12): never edit it, wrap it.
SUM=$( (sha256sum "$K8S/run_benchmark.sh" 2>/dev/null || shasum -a 256 "$K8S/run_benchmark.sh") | awk '{print $1}')
[ "$SUM" = a9d6b5cd9a725061e03c37e79639a9e4271ad3c9e96c143b5874f62ab8299532 ] \
  && ok "run_benchmark.sh unchanged (sha256)" || ko "run_benchmark.sh unchanged (sha256 $SUM)"
for f in "--endpoint-type responses" "--custom-dataset-type mooncake_trace" '--concurrency "$USERS"' \
         '--concurrency-ramp-duration "$RAMP"' '--benchmark-duration "$DURATION"' '--inter-turn-delay-cap-seconds "$CAP"' \
         "--connection-reuse-strategy sticky-user-sessions" "--use-server-token-count" "--export-level records" \
         '../llm-coding-benchmark/.venv/bin/aiperf profile' 'source ./benchmark.env'; do
  grep -qF -- "$f" "$K8S/run_benchmark.sh" && ok "their script: $f" || ko "their script: $f"
done
bash "$R" 100 6000 "$TMP/j.yaml" >/dev/null 2>&1 || ko "renders"
if [ -f "$TMP/j.yaml" ]; then
  [ "$(yq '.kind' "$TMP/j.yaml")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/j.yaml")" = aiperf-replay ] && ok "name" || ko "name"
  grep -q '@[A-Z_]*@' "$TMP/j.yaml" && ko "no token left" || ok "no token left"
  S=$(yq '.spec.template.spec.containers[0].args[0]' "$TMP/j.yaml")
  # benchmark.env, what their script sources: our values.
  for kv in ENDPOINT=http://vllm-0.vllm-headless.llm-replay.svc.cluster.local:8000 MODEL=aiservices_gemma \
            TOKENIZER=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit DATASET=/benchmarks/study35/dataset.jsonl \
            USERS=100 RAMP=6000 DURATION=6000 CAP=60 GRACE=60; do
    echo "$S" | grep -qx -- "$kv" && ok "benchmark.env $kv" || ko "benchmark.env $kv"
  done
  echo "$S" | grep -qF 'cp "$REPLAY/run_benchmark.sh" "$B/replay-benchmark/run_benchmark.sh"' && ok "their script copied as is" || ko "their script copied as is"
  echo "$S" | grep -qF 'ln -sfn /opt/aiperf-venv "$B/llm-coding-benchmark/.venv"' && ok "the venv where their script looks" || ko "the venv where their script looks"
  echo "$S" | grep -qF 'pip install --quiet aiperf==0.11.0' && ok "AIPerf 0.11.0" || ko "AIPerf 0.11.0"
  echo "$S" | grep -qF 'bash "$B/replay-benchmark/run_benchmark.sh" NAME=$(date +%s)' && ok "their script run, NAME = epoch" || ko "their script run"
  echo "$S" | grep -n 'MEASURED RUN START\|run_benchmark.sh" NAME' | awk -F: 'NR==1{a=$1} NR==2{b=$1} END{exit !(a<b)}' \
    && ok "marker before their script" || ko "marker before their script"
  echo "$S" | grep -qF '"$REPLAY/template.jsonl.gz" "$C/corpus_it.json.gz" 120 "$B/dataset.jsonl" --seed 42 --mix "$REPLAY/mix.json"' && ok "renders 120 users" || ko "renders 120 users"
  [ "$(yq '.spec.template.spec.volumes[] | select(.name=="replay") | .configMap.name' "$TMP/j.yaml")" = aiperf-replay ] \
    && ok "ConfigMap volume aiperf-replay" || ko "ConfigMap volume aiperf-replay"
  # No AIPerf command of our own: only theirs.
  echo "$S" | grep -q 'aiperf profile' && ko "no aiperf profile of our own" || ok "no aiperf profile of our own"
fi
bash "$R" 20 600 "$TMP/s.yaml" >/dev/null 2>&1 && has "$TMP/s.yaml" 'corpus_it.json.gz" 30 "$B' \
  && ok "small runs render at least 10 more users" || ko "small runs render at least 10 more users"
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "users not an integer" abc 4500
reject "users 0" 0 4500
reject "ramp 0" 30 0
reject "ramp not an integer" 30 45.5
rm -f "$TMP/x.yaml"; bash "$R" 30 "$TMP/x.yaml" >/dev/null 2>&1; rc=$?
[ $rc = 2 ] && ok "reject: missing argument" || ko "reject: missing argument (rc=$rc)"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
