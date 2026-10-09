#!/bin/bash
# Tests for ../render_eval_job.sh. Run: bash eval/tests/test_render_eval_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); E=$(dirname "$HERE"); R=$E/render_eval_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }
bash "$R" baseline-a "greedy card" "" "$TMP/j.yaml" >/dev/null 2>&1 || ko "renders"
if [ -f "$TMP/j.yaml" ]; then
  J=$TMP/j.yaml
  [ "$(yq '.kind' "$J")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$J")" = lmeval-l40s ] && ok "name" || ko "name"
  [ "$(yq '.spec.template.metadata.labels.app' "$J")" = lmeval-l40s ] && ok "label" || ko "label"
  [ "$(yq '.spec.template.spec.nodeSelector["node-role"]' "$J")" = system-m8a ] && ok "system-m8a" || ko "system-m8a"
  [ "$(yq '.spec.template.spec.containers[0].name' "$J")" = lmeval ] && ok "container lmeval" || ko "container lmeval"
  has "$J" 'RUN=baseline-a' && ok "run" || ko "run"
  has "$J" 'PROTOCOLS="greedy card"' && ok "protocols" || ko "protocols"
  has "$J" 'LIMIT=""' && ok "no limit" || ko "no limit"
  has "$J" '"lm_eval[api,ifeval]==0.4.13"' && ok "lm-eval pinned" || ko "lm-eval pinned"
  has "$J" 'lm_eval run --model local-chat-completions' && ok "chat completions" || ko "chat completions"
  has "$J" 'model=gemma4-26b-l40s,base_url=$URL/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200,max_length=4096' \
    && ok "model_args" || ko "model_args"
  has "$J" 'greedy) GEN="do_sample=false,temperature=0,max_gen_toks=2048"' && ok "greedy 2048 tokens" || ko "greedy 2048 tokens"
  has "$J" 'card) GEN="do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=2048,seed=1234"' \
    && ok "card 2048 tokens" || ko "card 2048 tokens"
  has "$J" 'for TASK in gsm8k_platinum_cot_llama ifeval' && ok "tasks" || ko "tasks"
  has "$J" '--num_fewshot 0 --apply_chat_template' && ok "0-shot, chat template" || ko "0-shot, chat template"
  has "$J" '--seed 1234 --log_samples' && ok "seed, samples" || ko "seed, samples"
  has "$J" 'http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000' && ok "targets vllm-0" || ko "targets vllm-0"
  has "$J" 'touch "$OUT/DONE"' && ok "DONE marker" || ko "DONE marker"
  grep -q '@[A-Z_]*@' "$J" && ko "no token left" || ok "no token left"
fi
bash "$R" best greedy 20 "$TMP/l.yaml" >/dev/null 2>&1 && has "$TMP/l.yaml" 'LIMIT="20"' && ok "limit 20" || ko "limit 20"
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "uppercase run" Best greedy ""
reject "underscore run" best_1 greedy ""
reject "unknown protocol" best nucleus ""
reject "no protocol" best "" ""
reject "limit 0" best greedy 0
reject "limit not a number" best greedy abc
rm -f "$TMP/x.yaml"; bash "$R" best greedy >/dev/null 2>&1; [ $? = 2 ] && ok "reject: 2 args" || ko "reject: 2 args"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
