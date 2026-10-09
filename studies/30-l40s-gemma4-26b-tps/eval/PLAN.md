# Study 30 accuracy check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the lm-eval accuracy check of `eval/README.md`: baseline vs study
30's best configuration (plus ablations and a repeated baseline), with a paired verdict per
task.

**Architecture:** `run_eval.sh` (workstation or toolbox, kubectl only) loops over runs: it
writes a run's `params.env` (a base file in `configs/` plus overrides), restarts vLLM through
the study's own `../k8s/apply_config.sh`, renders and applies an lm-eval Job
(`render_eval_job.sh` + `eval_job_template.yaml`, on `system-m8a`), waits for its `DONE`
marker, copies `/benchmarks/eval/<run>/` back to `results/<run>/lmeval/`. `compare.py`
(stdlib only) reads the per-item samples and writes `results/comparison.{md,json}`.

**Tech Stack:** bash, kubectl, yq v4 (tests only), Python 3.12 stdlib, lm-evaluation-harness
0.4.13 in `python:3.12-slim`, vLLM 0.29.0 (already deployed by the study).

**Spec:** `studies/30-l40s-gemma4-26b-tps/eval/README.md`

## Global Constraints

- lm-eval **0.4.13**, `pip install "lm_eval[api,ifeval]==0.4.13"`, subcommand `lm_eval run`.
- Model type `local-chat-completions`; model_args exactly
  `model=gemma4-26b-l40s,base_url=http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200,max_length=4096`.
- Tasks `gsm8k_platinum_cot_llama` and `ifeval`, `--num_fewshot 0 --apply_chat_template --seed 1234 --log_samples`.
- Protocols: greedy `do_sample=false,temperature=0,max_gen_toks=2048`; card
  `do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=2048,seed=1234`.
- vLLM's fixed flags are untouched (`--max-model-len=4096`, thinking off): every run goes
  through `../k8s/apply_config.sh`.
- Margins 1.0 point (GSM8K Platinum, strict-match) and 2.0 points (IFEval prompt-level
  strict); paired bootstrap 10,000 resamples; anchor 95.37 +- 1.4 and 89.34 +- 3.1;
  truncation warning above 1 %.
- Never run while study 30 has an experiment in flight (an `aiperf-l40s` Job exists).
- Repo files in English. No commits by the agent: the user commits `vllm-benchmark`
  (stated preference), so the "commit" steps below are "leave the files for the user".

## Review Focus

- **The 256-token default:** if `max_gen_toks=2048` is lost from a protocol, every chain of
  thought is cut and both runs look equally (and wrongly) bad. Task 1's render test asserts
  it for both protocols; Task 4 surfaces truncations from vLLM's counters.
- **kubectl failing (logged out, cluster unreachable) read as "no aiperf Job":** the guard
  must fail closed. Task 2 tests a failing `get job` stops before any apply.
- **Stale or duplicated samples** (a rerun into an existing directory): the Job and the
  runner both wipe the run directory; `compare.py` refuses zero or two samples files. Tests
  in Tasks 2 and 4.
- **An ablation running with the wrong parameters** (override not applied, or run when best
  does not use the lever): Task 2 checks the merged `params.env` and the skip rule.
- **A Job that never finishes** (pod Pending on a full `system-m8a`, lm-eval hung): the
  runner times out with exit 5 instead of waiting forever. Task 2 tests it.

---

### Task 1: Baseline config, Job template, renderer

**Files:**
- Create: `eval/configs/baseline.params.env`
- Create: `eval/eval_job_template.yaml`
- Create: `eval/render_eval_job.sh`
- Test: `eval/tests/test_render_eval_job.sh`

**Interfaces:**
- Produces: `render_eval_job.sh <run> <protocols> <limit|""> <output>` (exit 0 rendered;
  2 invalid input, nothing written). Run names `^[a-z0-9][a-z0-9-]*$`; protocols a
  space-separated subset of `greedy card`; limit empty or a positive integer.
- Produces: Job `lmeval-l40s` (label `app=lmeval-l40s`, container `lmeval`) writing
  `/benchmarks/eval/<run>/<protocol>/<task>/{finished_before.json,finished_after.json,lm_eval.log,gemma4-26b-l40s/{results_*.json,samples_<task>_*.jsonl}}`
  and `/benchmarks/eval/<run>/DONE`.

- [ ] **Step 1: Write the failing test** — `eval/tests/test_render_eval_job.sh`:

```bash
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash studies/30-l40s-gemma4-26b-tps/eval/tests/test_render_eval_job.sh`
Expected: `FAIL renders` and the rejects failing (no renderer yet), non-zero exit.

- [ ] **Step 3: Write `eval/configs/baseline.params.env`**

```bash
# Study 30's baseline step (akamas/30-L40S-Gemma4-TPS.yaml, experiment 1): vLLM 0.29.0
# defaults, every parameter written out. Sourced by ../../k8s/apply_config.sh.
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_DTYPE=auto
PERFORMANCE_MODE=balanced
OPTIMIZATION_LEVEL=2
ENFORCE_EAGER=false
SCHEDULING_POLICY=fcfs
ASYNC_SCHEDULING=true
MAX_CUDAGRAPH_CAPTURE_SIZE=512
BLOCK_SIZE=16
LINEAR_BACKEND=auto
SPEC_METHOD=none
SPEC_TOKENS=0
```

- [ ] **Step 4: Write `eval/eval_job_template.yaml`**

```yaml
# lm-eval Job for study 30's accuracy check (eval/README.md). Rendered by
# render_eval_job.sh (tokens RUN, PROTOCOLS, LIMIT between at-signs).
#
# lm-evaluation-harness 0.4.13, local-chat-completions against the deployed vllm-0 (every
# tuned flag in effect): GSM8K Platinum and IFEval, 0-shot, chat template, under each
# protocol (greedy: the comparisons; card: RedHat's sampling, the anchor). Around each task
# it snapshots vLLM's request_success_total by finished_reason ("length" = truncated).
# Writes /benchmarks/eval/<run>/, touches DONE, then sleeps so run_eval.sh can copy it.
apiVersion: batch/v1
kind: Job
metadata:
  name: lmeval-l40s
  namespace: llm-l40s
  labels:
    app: lmeval-l40s
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 7200
  template:
    metadata:
      labels:
        app: lmeval-l40s
    spec:
      restartPolicy: Never
      nodeSelector:
        node-role: system-m8a
      initContainers:
        - name: wait-for-vllm
          image: alpine:3
          command:
            - sh
            - -c
            - |
              until wget -qO- http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000/health >/dev/null 2>&1; do
                echo "$(date -u +%T) waiting for vllm-0..."
                sleep 10
              done
              echo "vllm-0 is ready."
      containers:
        - name: lmeval
          image: python:3.12-slim
          command: ["bash", "-c"]
          args:
            - |
              set -euo pipefail
              RUN=@RUN@
              PROTOCOLS="@PROTOCOLS@"
              LIMIT="@LIMIT@"
              URL=http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000
              OUT=/benchmarks/eval/$RUN
              rm -rf "$OUT"; mkdir -p "$OUT"
              pip install --quiet --root-user-action=ignore "lm_eval[api,ifeval]==0.4.13"
              cat > /tmp/finished.py <<'EOF'
              import json, re, sys, urllib.request
              text = urllib.request.urlopen(sys.argv[1] + "/metrics").read().decode()
              counts = {}
              for m in re.finditer(r'^vllm:request_success_total\{[^}]*finished_reason="(\w+)"[^}]*\} (\S+)$', text, re.M):
                  counts[m.group(1)] = counts.get(m.group(1), 0) + float(m.group(2))
              print(json.dumps(counts))
              EOF
              MODEL_ARGS="model=gemma4-26b-l40s,base_url=$URL/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200,max_length=4096"
              LIMIT_ARGS=()
              [ -z "$LIMIT" ] || LIMIT_ARGS=(--limit "$LIMIT")
              for P in $PROTOCOLS; do
                case $P in
                  greedy) GEN="do_sample=false,temperature=0,max_gen_toks=2048" ;;
                  card) GEN="do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=2048,seed=1234" ;;
                  *) echo "unknown protocol $P" >&2; exit 2 ;;
                esac
                for TASK in gsm8k_platinum_cot_llama ifeval; do
                  D=$OUT/$P/$TASK; mkdir -p "$D"
                  echo "$(date -u +%T) $RUN: $P $TASK"
                  python3 /tmp/finished.py "$URL" > "$D/finished_before.json"
                  lm_eval run --model local-chat-completions --model_args "$MODEL_ARGS" \
                    --tasks "$TASK" --num_fewshot 0 --apply_chat_template \
                    --gen_kwargs "$GEN" --seed 1234 --log_samples --output_path "$D" \
                    "${LIMIT_ARGS[@]}" 2>&1 | tee "$D/lm_eval.log"
                  python3 /tmp/finished.py "$URL" > "$D/finished_after.json"
                done
              done
              touch "$OUT/DONE"
              echo "$(date -u +%T) $RUN: DONE, waiting for the copy"
              sleep 1800
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
                  optional: true
            - name: HF_HOME
              value: /hf-cache
            - name: NLTK_DATA
              value: /hf-cache/nltk
          resources:
            requests:
              cpu: 1000m
              memory: 2Gi
          volumeMounts:
            - name: results
              mountPath: /benchmarks
            - name: hf-cache
              mountPath: /hf-cache
      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: aiperf-results
        - name: hf-cache
          persistentVolumeClaim:
            claimName: hf-cache
```

- [ ] **Step 5: Write `eval/render_eval_job.sh`**

```bash
#!/bin/bash
# Renders the lm-eval Job of study 30's accuracy check (eval_job_template.yaml).
# Usage: render_eval_job.sh <run> <protocols> <limit|""> <output>
#   protocols: space-separated subset of "greedy card"; limit: empty (full tasks) or N.
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 4 ] || die "usage: render_eval_job.sh <run> <protocols> <limit> <output>"
RUN=$1 PROTOCOLS=$2 LIMIT=$3 OUT=$4
TEMPLATE=$(cd "$(dirname "$0")" && pwd)/eval_job_template.yaml
[[ "$RUN" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "run '$RUN': lowercase letters, digits and hyphens only"
[ -n "$PROTOCOLS" ] || die "no protocol"
for p in $PROTOCOLS; do
  case $p in greedy|card) ;; *) die "unknown protocol '$p'" ;; esac
done
[ -z "$LIMIT" ] || [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "limit '$LIMIT' is not a positive integer"
sed -e "s|@RUN@|$RUN|g" -e "s|@PROTOCOLS@|$PROTOCOLS|g" -e "s|@LIMIT@|$LIMIT|g" "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `bash studies/30-l40s-gemma4-26b-tps/eval/tests/test_render_eval_job.sh`
Expected: every line `ok`, `0 failure(s)`, exit 0.

- [ ] **Step 7: Leave the files for the user to commit.**

### Task 2: The runner

**Files:**
- Create: `eval/run_eval.sh`
- Create: `eval/tests/stub/kubectl`, `eval/tests/stub/apply_config.sh`
- Test: `eval/tests/test_run_eval.sh`

**Interfaces:**
- Consumes: `render_eval_job.sh <run> <protocols> <limit> <output>` and the Job layout of
  Task 1; `../k8s/apply_config.sh` reading `STUDY_DIR`, `PARAMS`, `RENDERED` (exit != 0 on
  failure).
- Produces: `run_eval.sh [--limit N] [run ...]` with runs `baseline-a best best-kv-auto
  best-no-mtp baseline-b`; exit 0 done, 2 bad input / missing config, 3 study experiment in
  flight, 4 apply_config failed, 5 Job failed or timed out. Writes
  `results/<run>/{params.env,sts.yaml,apply_config.log,job.yaml,job.log,lmeval/<protocol>/<task>/...}`
  with `samples_*.jsonl` gzipped. Env overrides: `EVAL_OUT`, `APPLY`, `STUDY_DIR`,
  `WAIT_S` (default 5400), `POLL_S` (default 20).

- [ ] **Step 1: Write the stubs**

`eval/tests/stub/kubectl`:

```bash
#!/bin/bash
# Stub kubectl for test_run_eval.sh: logs every call to $STUB_LOG and answers run_eval.sh's
# calls. STUB_AIPERF=1: an aiperf-l40s Job exists; STUB_GET_FAILS=1: `get job` fails (logged
# out); STUB_FAILED: the lmeval Job's .status.failed; STUB_NOT_DONE=1: no DONE marker ever.
echo "kubectl $*" >> "$STUB_LOG"
args="$*"
case "$args" in
  *"get job -l app=aiperf-l40s"*)
    [ "${STUB_GET_FAILS:-0}" = 1 ] && exit 1
    [ "${STUB_AIPERF:-0}" = 1 ] && echo "job.batch/aiperf-l40s" ;;
  *"get job lmeval-l40s"*failed*) echo "${STUB_FAILED:-}" ;;
  *"get pods -l app=lmeval-l40s"*) echo "lmeval-l40s-x1" ;;
  *" exec "*"test -f"*) [ "${STUB_NOT_DONE:-0}" = 1 ] && exit 1 ;;
  *"apply -f "*)
    f=${args##*-f }
    echo "applied-job $(grep -m1 'RUN=' "$f" | sed 's/.*RUN=//') $(grep -m1 'PROTOCOLS=' "$f" | sed 's/.*PROTOCOLS=//') $(grep -m1 'LIMIT=' "$f" | sed 's/.*LIMIT=//')" >> "$STUB_LOG" ;;
  *" cp "*)
    dst=${args##* }; mkdir -p "$dst/greedy/ifeval/gemma4-26b-l40s"
    echo '{}' > "$dst/greedy/ifeval/gemma4-26b-l40s/samples_ifeval_x.jsonl" ;;
  *" logs "*) echo "lm-eval log" ;;
  *) : ;;
esac
exit 0
```

`eval/tests/stub/apply_config.sh`:

```bash
#!/bin/bash
# Stub apply_config.sh for test_run_eval.sh: logs the run (the params.env's directory) and
# its parameters to $STUB_LOG; exits $STUB_APPLY_RC (default 0).
echo "apply $(basename "$(dirname "$PARAMS")") $(grep -v '^#' "$PARAMS" | tr '\n' ' ')" >> "$STUB_LOG"
exit "${STUB_APPLY_RC:-0}"
```

Run: `chmod +x studies/30-l40s-gemma4-26b-tps/eval/tests/stub/kubectl studies/30-l40s-gemma4-26b-tps/eval/tests/stub/apply_config.sh`

- [ ] **Step 2: Write the failing test** — `eval/tests/test_run_eval.sh`:

```bash
#!/bin/bash
# Tests for ../run_eval.sh with a kubectl stub and a stub apply_config.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); E=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
# A private copy of eval/ per case, with a best.params.env using the given KV dtype and
# speculative method (MAX_NUM_SEQS=448 marks it as best's).
setup() {  # kv_dtype spec_method | "none-file" for no best.params.env
  rm -rf "$TMP/e" "$TMP/out" "$TMP/log"; mkdir -p "$TMP/e/configs"; : > "$TMP/log"
  cp "$E/run_eval.sh" "$E/render_eval_job.sh" "$E/eval_job_template.yaml" "$TMP/e/"
  cp "$E/configs/baseline.params.env" "$TMP/e/configs/"
  [ "$1" = none-file ] && return
  local k=0; [ "$2" = none ] || k=3
  sed -e "s/^KV_CACHE_DTYPE=.*/KV_CACHE_DTYPE=$1/" -e "s/^SPEC_METHOD=.*/SPEC_METHOD=$2/" \
      -e "s/^SPEC_TOKENS=.*/SPEC_TOKENS=$k/" -e "s/^MAX_NUM_SEQS=.*/MAX_NUM_SEQS=448/" \
      "$E/configs/baseline.params.env" > "$TMP/e/configs/best.params.env"
}
run() {
  PATH="$HERE/stub:$PATH" STUB_LOG=$TMP/log APPLY=$HERE/stub/apply_config.sh EVAL_OUT=$TMP/out \
    POLL_S=0 bash "$TMP/e/run_eval.sh" "$@" > "$TMP/stdout" 2>&1
}
order() { grep '^apply ' "$TMP/log" | awk '{print $2}' | tr '\n' ' '; }
val() { sed -n "s/^$1=//p" "$2"; }

setup fp8 mtp; run; rc=$?
[ $rc = 0 ] && ok "fp8+mtp: exit 0" || ko "fp8+mtp: exit 0 (rc=$rc, $(tail -1 "$TMP/stdout"))"
[ "$(order)" = "baseline-a best best-kv-auto best-no-mtp baseline-b " ] && ok "fp8+mtp: all five runs in order" || ko "fp8+mtp: order '$(order)'"
P=$TMP/out/best-kv-auto/params.env
{ [ "$(val KV_CACHE_DTYPE "$P")" = auto ] && [ "$(val SPEC_METHOD "$P")" = mtp ] && [ "$(val MAX_NUM_SEQS "$P")" = 448 ]; } \
  && ok "kv-auto ablation = best + KV auto" || ko "kv-auto ablation = best + KV auto"
P=$TMP/out/best-no-mtp/params.env
{ [ "$(val SPEC_METHOD "$P")" = none ] && [ "$(val SPEC_TOKENS "$P")" = 0 ] && [ "$(val KV_CACHE_DTYPE "$P")" = fp8 ] && [ "$(val MAX_NUM_SEQS "$P")" = 448 ]; } \
  && ok "no-mtp ablation = best - MTP" || ko "no-mtp ablation = best - MTP"
grep -qF 'applied-job baseline-a "greedy card" ""' "$TMP/log" && ok "baseline-a: greedy + card" || ko "baseline-a: greedy + card"
grep -qF 'applied-job best "greedy" ""' "$TMP/log" && ok "best: greedy only" || ko "best: greedy only"
[ -f "$TMP/out/baseline-a/lmeval/greedy/ifeval/gemma4-26b-l40s/samples_ifeval_x.jsonl.gz" ] && ok "samples gzipped" || ko "samples gzipped"
[ -f "$TMP/out/baseline-a/job.log" ] && ok "job log saved" || ko "job log saved"
grep -q 'scale sts vllm --replicas=0' "$TMP/log" && ok "vLLM to 0 at the end" || ko "vLLM to 0 at the end"
[ "$(grep -c 'delete job lmeval-l40s' "$TMP/log")" = 5 ] && ok "each Job deleted" || ko "each Job deleted"

setup auto none; run; rc=$?
{ [ $rc = 0 ] && [ "$(order)" = "baseline-a best baseline-b " ]; } && ok "auto+none: ablations skipped" || ko "auto+none: order '$(order)' rc=$rc"

setup fp8 none; run
[ "$(order)" = "baseline-a best best-kv-auto baseline-b " ] && ok "fp8 only: kv ablation only" || ko "fp8 only: order '$(order)'"

setup fp8 mtp; STUB_AIPERF=1 run; rc=$?
{ [ $rc = 3 ] && ! grep -q '^apply ' "$TMP/log"; } && ok "refuses under a live experiment" || ko "refuses under a live experiment (rc=$rc)"

setup fp8 mtp; STUB_GET_FAILS=1 run; rc=$?
{ [ $rc != 0 ] && ! grep -q '^apply ' "$TMP/log"; } && ok "kubectl failing: fails closed" || ko "kubectl failing: fails closed (rc=$rc)"

setup fp8 mtp; run --limit 20 baseline-a; rc=$?
{ [ $rc = 0 ] && [ "$(order)" = "baseline-a " ] && grep -qF 'applied-job baseline-a "greedy card" "20"' "$TMP/log"; } \
  && ok "--limit 20 baseline-a" || ko "--limit 20 baseline-a (rc=$rc, order '$(order)')"

setup fp8 mtp; run nope; rc=$?
{ [ $rc = 2 ] && [ ! -s "$TMP/log" ]; } && ok "unknown run: exit 2, cluster untouched" || ko "unknown run (rc=$rc)"

setup fp8 mtp; run --limit 0 baseline-a; rc=$?
{ [ $rc = 2 ] && [ ! -s "$TMP/log" ]; } && ok "limit 0 rejected" || ko "limit 0 rejected (rc=$rc)"

setup none-file; run; rc=$?
{ [ $rc = 2 ] && [ ! -s "$TMP/log" ]; } && ok "no best.params.env: exit 2, cluster untouched" || ko "no best.params.env (rc=$rc)"

setup none-file; run baseline-a baseline-b; rc=$?
{ [ $rc = 0 ] && [ "$(order)" = "baseline-a baseline-b " ]; } && ok "baselines only without best" || ko "baselines only without best (rc=$rc)"

setup fp8 mtp; STUB_APPLY_RC=4 run; rc=$?
[ $rc = 4 ] && ok "apply_config failure: exit 4" || ko "apply_config failure: exit 4 (rc=$rc)"

setup fp8 mtp; STUB_FAILED=1 run; rc=$?
{ [ $rc = 5 ] && [ -f "$TMP/out/baseline-a/job.log" ]; } && ok "Job failed: exit 5, log kept" || ko "Job failed (rc=$rc)"

setup fp8 mtp; STUB_NOT_DONE=1 WAIT_S=0 run; rc=$?
[ $rc = 5 ] && ok "no DONE: timeout exit 5" || ko "no DONE: timeout exit 5 (rc=$rc)"

echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 3: Run it to verify it fails**

Run: `bash studies/30-l40s-gemma4-26b-tps/eval/tests/test_run_eval.sh`
Expected: FAIL lines (no `run_eval.sh` yet), non-zero exit.

- [ ] **Step 4: Write `eval/run_eval.sh`**

```bash
#!/bin/bash
# Accuracy check for study 30 (eval/README.md). Per run: params.env = a base file in configs/
# plus the run's overrides, ../k8s/apply_config.sh (vLLM restarted with it), the lm-eval Job
# (render_eval_job.sh), then /benchmarks/eval/<run>/ copied to results/<run>/lmeval/.
# Ends with vLLM at 0 replicas. Never while a study 30 experiment is in flight.
#
# Usage: run_eval.sh [--limit N] [run ...]   (no run: the whole plan, in order)
# Exit codes: 0 done; 2 bad input or missing config; 3 a study experiment is running;
# 4 apply_config failed; 5 the lm-eval Job failed or timed out.
set -euo pipefail
EVAL_DIR=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=${STUDY_DIR:-$(dirname "$EVAL_DIR")}
NS=llm-l40s
OUT=${EVAL_OUT:-$EVAL_DIR/results}
APPLY=${APPLY:-$STUDY_DIR/k8s/apply_config.sh}
WAIT_S=${WAIT_S:-5400}
POLL_S=${POLL_S:-20}
t() { echo "$(date -u +%T) run_eval: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# run | base params.env in configs/ | overrides | protocols
PLAN=(
  "baseline-a|baseline.params.env||greedy card"
  "best|best.params.env||greedy"
  "best-kv-auto|best.params.env|KV_CACHE_DTYPE=auto|greedy"
  "best-no-mtp|best.params.env|SPEC_METHOD=none SPEC_TOKENS=0|greedy"
  "baseline-b|baseline.params.env||greedy"
)

LIMIT=""
if [ "${1:-}" = --limit ]; then
  [ $# -ge 2 ] || die "--limit needs a value"
  LIMIT=$2; shift 2
fi
[ -z "$LIMIT" ] || [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || die "limit '$LIMIT' is not a positive integer"

val() { sed -n "s/^$1=//p" "$2" | tail -1; }   # KEY file -> value

# An ablation only when best uses the lever it removes.
needed() {
  case $1 in
    best-kv-auto) [ "$(val KV_CACHE_DTYPE "$EVAL_DIR/configs/best.params.env")" != auto ] ;;
    best-no-mtp)  [ "$(val SPEC_METHOD "$EVAL_DIR/configs/best.params.env")" != none ] ;;
    *) return 0 ;;
  esac
}

# --- 0. Select and validate every run before touching the cluster ---------------------
for a in "$@"; do
  printf '%s\n' "${PLAN[@]}" | grep -q "^$a|" || die "unknown run '$a'"
done
SELECTED=()
for line in "${PLAN[@]}"; do
  name=${line%%|*}
  if [ $# -gt 0 ]; then
    want=0; for a in "$@"; do [ "$a" = "$name" ] && want=1; done
    [ $want = 1 ] || continue
  fi
  SELECTED+=("$line")
done
for line in "${SELECTED[@]}"; do
  IFS='|' read -r name base _ _ <<<"$line"
  [ -f "$EVAL_DIR/configs/$base" ] || die "$name: configs/$base missing"
done

# --- 1. Never under a live study experiment (fails closed if kubectl fails) ----------
live=$(kubectl -n $NS get job -l app=aiperf-l40s -o name)
[ -z "$live" ] || die "an aiperf-l40s Job exists: a study 30 experiment is running" 3

# --- 2. One vLLM start and one lm-eval Job per run ------------------------------------
for line in "${SELECTED[@]}"; do
  IFS='|' read -r name base overrides protocols <<<"$line"
  if ! needed "$name"; then t "$name: skipped (best does not use that lever)"; continue; fi
  D=$OUT/$name; rm -rf "$D"; mkdir -p "$D"
  cp "$EVAL_DIR/configs/$base" "$D/params.env"
  for kv in $overrides; do
    k=${kv%%=*}
    grep -q "^$k=" "$D/params.env" || die "$name: $k not in configs/$base"
    sed -i.bak "s|^$k=.*|$kv|" "$D/params.env" && rm -f "$D/params.env.bak"
  done
  t "$name: apply_config ($(grep -v '^#' "$D/params.env" | tr '\n' ' '))"
  STUDY_DIR=$STUDY_DIR PARAMS=$D/params.env RENDERED=$D/sts.yaml bash "$APPLY" > "$D/apply_config.log" 2>&1 \
    || die "$name: apply_config failed, see $D/apply_config.log" 4
  bash "$EVAL_DIR/render_eval_job.sh" "$name" "$protocols" "$LIMIT" "$D/job.yaml"
  kubectl -n $NS delete job -l app=lmeval-l40s --ignore-not-found --wait=true >/dev/null
  kubectl apply -f "$D/job.yaml" >/dev/null
  t "$name: lm-eval Job started ($protocols${LIMIT:+, limit $LIMIT})"
  waited=0; pod=""
  while :; do
    failed=$(kubectl -n $NS get job lmeval-l40s -o jsonpath='{.status.failed}')
    if [ -n "$failed" ] && [ "$failed" != 0 ]; then
      kubectl -n $NS logs job/lmeval-l40s -c lmeval > "$D/job.log" 2>&1 || true
      die "$name: the lm-eval Job failed, see $D/job.log" 5
    fi
    pod=$(kubectl -n $NS get pods -l app=lmeval-l40s -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [ -n "$pod" ] && kubectl -n $NS exec "$pod" -c lmeval -- test -f "/benchmarks/eval/$name/DONE" >/dev/null 2>&1; then
      break
    fi
    [ "$waited" -lt "$WAIT_S" ] || die "$name: no DONE after ${WAIT_S}s (Job left in place)" 5
    sleep "$POLL_S"; waited=$((waited + POLL_S))
  done
  kubectl -n $NS logs "$pod" -c lmeval > "$D/job.log"
  kubectl -n $NS cp -c lmeval "$pod:/benchmarks/eval/$name" "$D/lmeval" >/dev/null
  find "$D/lmeval" -name 'samples_*.jsonl' -exec gzip -f {} +
  kubectl -n $NS delete job lmeval-l40s --wait=false >/dev/null
  t "$name: done -> $D"
done

kubectl -n $NS scale sts vllm --replicas=0 >/dev/null
t "vLLM scaled to 0"
```

Run: `chmod +x studies/30-l40s-gemma4-26b-tps/eval/run_eval.sh studies/30-l40s-gemma4-26b-tps/eval/render_eval_job.sh`

- [ ] **Step 5: Run the test to verify it passes**

Run: `bash studies/30-l40s-gemma4-26b-tps/eval/tests/test_run_eval.sh`
Expected: every line `ok`, `0 failure(s)`, exit 0. Also re-run Task 1's test (shared template).

- [ ] **Step 6: Leave the files for the user to commit.**

### Task 3: Cluster smoke (real vLLM, 20 items per task)

Needs: study 30 stopped with no `aiperf-l40s` Job (checked 2026-10-08: deleted), the L40S
node up, `AWS_PROFILE=lab`. Restarts vLLM with the baseline (~5-6 min) and evaluates 20
items per task under both protocols (~5 min with the pip install).

- [ ] **Step 1: Run the smoke**

Run (from `studies/30-l40s-gemma4-26b-tps`):
`AWS_PROFILE=lab EVAL_OUT=/tmp/eval30-smoke bash eval/run_eval.sh --limit 20 baseline-a`
Expected: `baseline-a: done -> /tmp/eval30-smoke/baseline-a`, then `vLLM scaled to 0`, exit 0.

- [ ] **Step 2: Check the layout and the samples format**

Run: `find /tmp/eval30-smoke/baseline-a/lmeval -type f | sort` and
`gzip -dc /tmp/eval30-smoke/baseline-a/lmeval/greedy/gsm8k_platinum_cot_llama/*/samples_*.jsonl.gz | head -2 | python3 -I -c "import json,sys; [print(sorted(json.loads(l))) for l in sys.stdin]"`
Expected: per protocol and task `finished_before.json`, `finished_after.json`, `lm_eval.log`,
one `results_*.json`, one `samples_*.jsonl.gz`; sample keys include `doc_id`, `filter`,
`exact_match` (GSM8K: two lines per doc, filters `strict-match` and `flexible-extract`;
IFEval: `prompt_level_strict_acc`, `inst_level_strict_acc`). If the keys differ, adapt
Task 4's fixtures and `compare.py` to what is printed here before writing them.

- [ ] **Step 3: Check the counters**

Run: `cat /tmp/eval30-smoke/baseline-a/lmeval/greedy/*/finished_*.json`
Expected: JSON objects with `stop` / `length` / `abort` / `error` / `repetition` counts;
after - before sums to 20 requests per task (more only if lm-eval retried). The 400 check
needs vLLM up, so it runs in Task 5 Step 3, not here (no extra restart).

- [ ] **Step 4: Note the smoke in `eval/README.md`** ("Smoke" paragraph under "Tests":
date, the 20-item scores, the sample keys, the truncation counts). Leave for the user.

### Task 4: compare.py

**Files:**
- Create: `eval/compare.py`
- Test: `eval/tests/test_compare.sh`

**Interfaces:**
- Consumes: `results/<run>/lmeval/<protocol>/<task>/{finished_before.json,finished_after.json,<model dir>/results_*.json,<model dir>/samples_<task>_*.jsonl[.gz]}` (Tasks 1-3).
- Produces: `python3 -I compare.py <results_dir>` -> `<results_dir>/comparison.md` and
  `comparison.json` = `{"anchor": {task: {accuracy, reference, tolerance, within,
  truncations}}, "accuracy": {task: {run: {metric: pct}}}, "truncations": {task: {run:
  {requests, length, share}}}, "pairs": [{task, a, b, metric, margin, n, acc_a, acc_b, delta,
  ci: [lo, hi], lost: [doc_id], gained: [doc_id], mcnemar_p, verdict}], "warnings": [str]}`.
  Exit 2 on inconsistent inputs.

- [ ] **Step 1: Write the failing test** — `eval/tests/test_compare.sh`:

```bash
#!/bin/bash
# compare.py on synthetic lm-eval 0.4.13 outputs with known deltas and verdicts.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); E=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
# mk root run protocol task n wrong_ranges [version] [truncated]
#   wrong_ranges: "a:b,c:d" = docs a..b-1 and c..d-1 answered wrong ("" = none)
mk() {
  python3 -I - "$@" <<'EOF'
import json, os, sys
root, run, protocol, task, n, wrong = sys.argv[1:7]
version = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
trunc = int(sys.argv[8]) if len(sys.argv) > 8 else 0
n = int(n)
bad = set()
for r in filter(None, wrong.split(",")):
    a, b = map(int, r.split(":")); bad.update(range(a, b))
td = os.path.join(root, run, "lmeval", protocol, task)
d = os.path.join(td, "gemma4-26b-l40s")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, f"samples_{task}_2026-10-08T00-00-00.jsonl"), "w") as f:
    for i in range(n):
        ok = i not in bad
        if task == "ifeval":
            f.write(json.dumps({"doc_id": i, "filter": "none", "prompt_level_strict_acc": ok,
                                "inst_level_strict_acc": [ok, True]}) + "\n")
        else:
            for flt in ("strict-match", "flexible-extract"):
                f.write(json.dumps({"doc_id": i, "filter": flt, "exact_match": 1.0 if ok else 0.0}) + "\n")
with open(os.path.join(d, "results_2026-10-08T00-00-00.json"), "w") as f:
    json.dump({"versions": {task: version}}, f)
with open(os.path.join(td, "finished_before.json"), "w") as f:
    json.dump({"stop": 100.0, "length": 5.0, "abort": 0.0}, f)
with open(os.path.join(td, "finished_after.json"), "w") as f:
    json.dump({"stop": 100.0 + n - trunc, "length": 5.0 + trunc, "abort": 0.0}, f)
EOF
}
G=gsm8k_platinum_cot_llama
q() { python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$1/comparison.json" "$2"; }
pair() { echo "[p for p in r['pairs'] if p['task']=='$1' and p['b']=='$2'][0]"; }
cmp() { python3 -I "$E/compare.py" "$1" > "$1/stdout" 2>&1; }

# S1: same accuracy, 5 lost + 5 gained on GSM8K; identical IFEval -> no degradation.
S=$TMP/s1
mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:45,950:955
mk $S baseline-a greedy ifeval 500 0:50; mk $S best greedy ifeval 500 0:50
cmp $S && ok "s1: exit 0" || ko "s1: exit 0 ($(tail -2 $S/stdout))"
[ "$(q $S "$(pair $G best)['verdict']")" = "no degradation" ] && ok "s1: gsm8k no degradation" || ko "s1: gsm8k verdict"
[ "$(q $S "($(pair $G best)['delta'], len($(pair $G best)['lost']), len($(pair $G best)['gained']))")" = "(0.0, 5, 5)" ] \
  && ok "s1: delta 0, 5 lost, 5 gained" || ko "s1: delta/flips"
[ "$(q $S "$(pair ifeval best)['verdict']")" = "no degradation" ] && ok "s1: ifeval no degradation" || ko "s1: ifeval verdict"
[ "$(q $S "round(r['accuracy']['$G']['best']['exact_match'], 2)")" = 95.0 ] && ok "s1: accuracy 95.0" || ko "s1: accuracy"
[ "$(q $S "round(r['accuracy']['ifeval']['best']['inst_level_strict_acc,none'], 2)")" = 95.0 ] && ok "s1: inst-level secondary" || ko "s1: inst-level secondary"
grep -q 'no degradation' "$S/comparison.md" && ok "s1: markdown written" || ko "s1: markdown written"

# S2: 60 lost of 1000 -> degradation, McNemar tiny.
S=$TMP/s2
mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50,900:960
cmp $S
[ "$(q $S "$(pair $G best)['verdict']")" = degradation ] && ok "s2: degradation" || ko "s2: verdict"
[ "$(q $S "round($(pair $G best)['delta'], 2)")" = -6.0 ] && ok "s2: delta -6.0" || ko "s2: delta"
[ "$(q $S "$(pair $G best)['mcnemar_p'] < 1e-6")" = True ] && ok "s2: McNemar p tiny" || ko "s2: McNemar"

# S3: 5 lost of 2000 -> measurable drop within the margin.
S=$TMP/s3
mk $S baseline-a greedy $G 2000 0:100; mk $S best greedy $G 2000 0:100,1990:1995
cmp $S
[ "$(q $S "$(pair $G best)['verdict']")" = "measurable drop within the margin" ] && ok "s3: drop within margin" || ko "s3: verdict $(q $S "$(pair $G best)['ci']")"

# S4: IFEval, 100 prompts, 4 lost 2 gained -> inconclusive.
S=$TMP/s4
mk $S baseline-a greedy ifeval 100 0:10; mk $S best greedy ifeval 100 0:8,90:94
cmp $S
[ "$(q $S "$(pair ifeval best)['verdict']")" = inconclusive ] && ok "s4: inconclusive" || ko "s4: verdict $(q $S "$(pair ifeval best)['ci']")"

# S5: noise floor and ablations paired against baseline-a; anchor; truncation warning.
S=$TMP/s5
mk $S baseline-a greedy $G 1000 0:50; mk $S baseline-b greedy $G 1000 0:49,999:1000
mk $S best-kv-auto greedy $G 1000 0:50 1.0 20
mk $S baseline-a card $G 1000 0:47; mk $S baseline-a card ifeval 500 0:150
cmp $S
[ "$(q $S "$(pair $G baseline-b)['a']")" = baseline-a ] && ok "s5: noise floor pair" || ko "s5: noise floor pair"
[ "$(q $S "$(pair $G best-kv-auto)['a']")" = baseline-a ] && ok "s5: ablation pair" || ko "s5: ablation pair"
[ "$(q $S "r['anchor']['$G']['within']")" = True ] && ok "s5: gsm8k anchor within" || ko "s5: gsm8k anchor"
[ "$(q $S "r['anchor']['ifeval']['within']")" = False ] && ok "s5: ifeval anchor outside" || ko "s5: ifeval anchor"
[ "$(q $S "r['truncations']['$G']['best-kv-auto']['length']")" = 20.0 ] && ok "s5: 20 truncations counted" || ko "s5: truncations"
[ "$(q $S "any('best-kv-auto' in w and 'truncated' in w for w in r['warnings'])")" = True ] && ok "s5: truncation warning" || ko "s5: truncation warning"
[ "$(q $S "any('anchor' in w and 'ifeval' in w for w in r['warnings'])")" = True ] && ok "s5: anchor warning" || ko "s5: anchor warning"

# Refusals.
S=$TMP/r1; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 999 0:50
cmp $S; [ $? = 2 ] && grep -q 'doc_id' "$S/stdout" && ok "refuse: different doc_id sets" || ko "refuse: different doc_id sets"
S=$TMP/r2; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50 2.0
cmp $S; [ $? = 2 ] && grep -q 'version' "$S/stdout" && ok "refuse: different task versions" || ko "refuse: different task versions"
S=$TMP/r3; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50
cp $S/best/lmeval/greedy/$G/gemma4-26b-l40s/samples_${G}_2026-10-08T00-00-00.jsonl $S/best/lmeval/greedy/$G/gemma4-26b-l40s/samples_${G}_2026-10-09T00-00-00.jsonl
cmp $S; [ $? = 2 ] && grep -q 'found 2' "$S/stdout" && ok "refuse: two samples files" || ko "refuse: two samples files"
S=$TMP/r4; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50
gzip $S/best/lmeval/greedy/$G/gemma4-26b-l40s/samples_*.jsonl
cmp $S && [ "$(q $S "$(pair $G best)['n']")" = 1000 ] && ok "reads gzipped samples" || ko "reads gzipped samples"

echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash studies/30-l40s-gemma4-26b-tps/eval/tests/test_compare.sh`
Expected: FAIL lines (no `compare.py` yet), non-zero exit.

- [ ] **Step 3: Write `eval/compare.py`**

```python
#!/usr/bin/env python3
"""Paired accuracy comparison for study 30's accuracy check (eval/README.md, "Decision rule").

Usage: compare.py <results_dir>

Reads <results_dir>/<run>/lmeval/<protocol>/<task>/: lm-eval 0.4.13's results_*.json and
samples_<task>_*.jsonl[.gz] (--log_samples, one level down), and the vLLM finished-reason
snapshots finished_{before,after}.json. Writes comparison.md and comparison.json into
<results_dir>. Exit 2 on inconsistent inputs: different doc_id sets, different task
versions, a missing or duplicated file. Standard library only.
"""
import glob
import gzip
import json
import math
import os
import random
import sys

# task -> (filter, primary metric, margin in points, (filter, secondary metric))
TASKS = {
    "gsm8k_platinum_cot_llama": ("strict-match", "exact_match", 1.0, ("flexible-extract", "exact_match")),
    "ifeval": ("none", "prompt_level_strict_acc", 2.0, ("none", "inst_level_strict_acc")),
}
# RedHat's FP8 column (no thinking) and the tolerance of one run against a 3-seed mean.
ANCHOR = {"gsm8k_platinum_cot_llama": (95.37, 1.4), "ifeval": (89.34, 3.1)}
# (B, A): every comparison is B - A against the first baseline.
PAIRS = [("best", "baseline-a"), ("best-kv-auto", "baseline-a"),
         ("best-no-mtp", "baseline-a"), ("baseline-b", "baseline-a")]
BOOTSTRAP = 10000
SEED = 30
TRUNCATION_WARN = 1.0  # percent of a task's requests ending on "length"


class InputError(Exception):
    pass


def task_dir(root, run, protocol, task):
    return os.path.join(root, run, "lmeval", protocol, task)


def one_file(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if len(files) != 1:
        raise InputError(f"expected one file for {pattern}, found {len(files)}")
    return files[0]


def load_samples(root, run, protocol, task):
    """-> {(doc_id, filter): record}"""
    path = one_file(os.path.join(task_dir(root, run, protocol, task), "**", f"samples_{task}_*.jsonl*"))
    opener = gzip.open if path.endswith(".gz") else open
    out = {}
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["doc_id"], r["filter"])] = r
    return out


def task_version(root, run, protocol, task):
    path = one_file(os.path.join(task_dir(root, run, protocol, task), "**", "results_*.json"))
    with open(path) as f:
        return json.load(f)["versions"][task]


def scores(samples, flt, metric):
    """-> {doc_id: value}; an instruction-level value stays a list."""
    return {d: r[metric] for (d, f), r in samples.items() if f == flt}


def accuracy(values):
    flat = [float(x) for v in values for x in (v if isinstance(v, list) else [v])]
    return 100.0 * sum(flat) / len(flat)


def mcnemar_p(lost, gained):
    """Exact two-sided McNemar p-value on the discordant pairs."""
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b, seed=SEED, resamples=BOOTSTRAP):
    """a, b: {doc_id: 0/1}. Delta = b - a in points, 95 % CI by paired bootstrap."""
    if set(a) != set(b):
        raise InputError(f"different doc_id sets ({len(a)} vs {len(b)} docs)")
    ids = sorted(a)
    d = [float(b[i]) - float(a[i]) for i in ids]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=n)) / n for _ in range(resamples))
    lo, hi = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    lost = [i for i in ids if a[i] and not b[i]]
    gained = [i for i in ids if b[i] and not a[i]]
    return {"n": n, "acc_a": accuracy(a.values()), "acc_b": accuracy(b.values()),
            "delta": 100.0 * sum(d) / n, "ci": [100.0 * lo, 100.0 * hi],
            "lost": lost, "gained": gained, "mcnemar_p": mcnemar_p(len(lost), len(gained))}


def verdict(lo, hi, margin):
    if lo > -margin:
        return "no degradation" if hi >= 0 else "measurable drop within the margin"
    return "degradation" if hi < 0 else "inconclusive"


def truncations(root, run, protocol, task):
    d = task_dir(root, run, protocol, task)
    with open(os.path.join(d, "finished_before.json")) as f:
        before = json.load(f)
    with open(os.path.join(d, "finished_after.json")) as f:
        after = json.load(f)
    diff = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    total = sum(diff.values())
    length = diff.get("length", 0)
    return {"requests": total, "length": length, "share": 100.0 * length / total if total else 0.0}


def analyse(root):
    runs = sorted(r for r in os.listdir(root) if os.path.isdir(os.path.join(root, r, "lmeval")))
    report = {"anchor": {}, "accuracy": {}, "truncations": {}, "pairs": [], "warnings": []}
    for task, (flt, metric, margin, (flt2, metric2)) in TASKS.items():
        greedy, versions = {}, {}
        for run in runs:
            if not os.path.isdir(task_dir(root, run, "greedy", task)):
                continue
            s = load_samples(root, run, "greedy", task)
            greedy[run] = s
            versions[run] = task_version(root, run, "greedy", task)
            report["accuracy"].setdefault(task, {})[run] = {
                metric: accuracy(scores(s, flt, metric).values()),
                f"{metric2},{flt2}": accuracy(scores(s, flt2, metric2).values())}
            tr = truncations(root, run, "greedy", task)
            report["truncations"].setdefault(task, {})[run] = tr
            if tr["share"] > TRUNCATION_WARN:
                report["warnings"].append(
                    f"{run} {task}: {tr['share']:.1f} % of requests truncated (> {TRUNCATION_WARN} %)")
        if len(set(versions.values())) > 1:
            raise InputError(f"{task}: different task versions {versions}")
        if os.path.isdir(task_dir(root, "baseline-a", "card", task)):
            acc = accuracy(scores(load_samples(root, "baseline-a", "card", task), flt, metric).values())
            ref, tol = ANCHOR[task]
            within = abs(acc - ref) <= tol
            report["anchor"][task] = {"accuracy": acc, "reference": ref, "tolerance": tol, "within": within,
                                      "truncations": truncations(root, "baseline-a", "card", task)}
            if not within:
                report["warnings"].append(
                    f"anchor {task}: {acc:.2f} vs card {ref} +- {tol}: find out why before reading any delta")
        for b, a in PAIRS:
            if a in greedy and b in greedy:
                p = paired(scores(greedy[a], flt, metric), scores(greedy[b], flt, metric))
                p.update(task=task, a=a, b=b, metric=metric, margin=margin,
                         verdict=verdict(p["ci"][0], p["ci"][1], margin))
                report["pairs"].append(p)
    return report


def markdown(r):
    out = ["# Study 30 accuracy check: comparison", ""]
    if r["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    out += ["## Anchor (baseline-a, card protocol, vs RedHat's FP8 column)", "",
            "| Task | Accuracy | Card | Tolerance | Within | Truncated |", "|---|---|---|---|---|---|"]
    for task, x in r["anchor"].items():
        out.append(f"| {task} | {x['accuracy']:.2f} | {x['reference']} | +-{x['tolerance']} | "
                   f"{'yes' if x['within'] else 'NO'} | {x['truncations']['length']:.0f} |")
    out += ["", "## Accuracy (greedy)", "", "| Task | Run | Primary | Secondary | Truncated |", "|---|---|---|---|---|"]
    for task, runs in r["accuracy"].items():
        for run, m in runs.items():
            (k1, v1), (k2, v2) = list(m.items())
            tr = r["truncations"][task][run]
            out.append(f"| {task} | {run} | {k1} {v1:.2f} | {k2} {v2:.2f} | "
                       f"{tr['length']:.0f} / {tr['requests']:.0f} |")
    out += ["", "## Paired comparisons (greedy, primary metric, B - A)", "",
            "| Task | B vs A | n | A | B | Delta | 95 % CI | Lost | Gained | McNemar p | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in r["pairs"]:
        out.append(f"| {p['task']} | {p['b']} vs {p['a']} | {p['n']} | {p['acc_a']:.2f} | {p['acc_b']:.2f} | "
                   f"{p['delta']:+.2f} | [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] | {len(p['lost'])} | "
                   f"{len(p['gained'])} | {p['mcnemar_p']:.3g} | {p['verdict']} |")
    out += ["", "## Flips (doc_id)", ""]
    for p in r["pairs"]:
        out.append(f"- {p['task']}, {p['b']} vs {p['a']}: lost {p['lost'][:30]}, gained {p['gained'][:30]}")
    return "\n".join(out) + "\n"


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = argv[1]
    try:
        report = analyse(root)
    except (InputError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    with open(os.path.join(root, "comparison.json"), "w") as f:
        json.dump(report, f, indent=1)
    md = markdown(report)
    with open(os.path.join(root, "comparison.md"), "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bash studies/30-l40s-gemma4-26b-tps/eval/tests/test_compare.sh`
Expected: every line `ok`, `0 failure(s)`, exit 0.

- [ ] **Step 5: Run it on the smoke output** (real file format)

Run: `cp -R /tmp/eval30-smoke/baseline-a /tmp/eval30-smoke/baseline-b && rm -rf /tmp/eval30-smoke/baseline-b/lmeval/card && python3 -I studies/30-l40s-gemma4-26b-tps/eval/compare.py /tmp/eval30-smoke`
Expected: exit 0; anchor rows for both tasks (20 items, `within` may be anything at n=20);
a `baseline-b vs baseline-a` row per task with delta 0 and 0 flips (same files).

- [ ] **Step 6: Leave the files for the user to commit.**

### Task 5: Best configuration and the full run

- [ ] **Step 1: Get the best experiment's parameters.** Akamas 4.1 CLI (needs a login
  in the toolbox, `akamas login`; expired on 2026-10-08):
  `kubectl -n akamas-41 exec deploy/toolbox -c toolbox -- akamas list --no-pagination -o json experiment 30-L40S-Gemma4-TPS --workspace default`
  -> the VALID experiment with the highest score; or the user gives its number from the UI.
  Then `akamas describe experiment ...` / the export's `study.json` for its 13 values.

- [ ] **Step 2: Write `eval/configs/best.params.env`** in the format of
  `configs/baseline.params.env` (same 14 lines, `ENFORCE_EAGER=false`), with a header
  comment naming the experiment, its score and the date. Check: `diff
  eval/configs/baseline.params.env eval/configs/best.params.env` shows only the tuned values.

- [ ] **Step 3: Full run** (~70-80 min; workstation awake):
  `mkdir -p /tmp/eval30 && AWS_PROFILE=lab caffeinate -i nohup bash eval/run_eval.sh > /tmp/eval30/run.log 2>&1 &`
  from `studies/30-l40s-gemma4-26b-tps`. Past 17:00 UTC the GPU node must carry
  `AlwaysOn=true` (it does, per the study README) and so must the `system-m8a` node.
  While `baseline-a` evaluates (vLLM up), check the 400 once:

```bash
kubectl -n llm-l40s exec vllm-0 -- python3 -c '
import json, urllib.request as u, urllib.error as e
body = {"model": "gemma4-26b-l40s", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4096}
req = u.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
try:
    u.urlopen(req); print("accepted: vLLM did NOT reject max_tokens = max_model_len")
except e.HTTPError as x:
    print(x.code, x.read()[:300])
'
```

  Expected: `400 b'{"error": ... maximum context length is 4096 tokens ...'`. If it is
  accepted instead, vLLM clamps silently: note it in `eval/README.md` (truncations are
  still counted by `finished_reason`, so the comparison holds).
  Expected at the end: `vLLM scaled to 0`, exit 0, `results/<run>/` for each run that was
  not skipped.

### Task 6: Compare and record

- [ ] **Step 1:** `python3 -I eval/compare.py eval/results` -> `eval/results/comparison.md`.
  Expected: no warnings (else stop: anchor outside or truncations > 1 %, see the spec).
- [ ] **Step 2:** Fill `eval/README.md` "Results": anchor table, the paired table, the
  verdict per task in one sentence each, which lever explains a drop (if any), and the flip
  counts of best against baseline-b's. Set **Status** to DONE with the date.
- [ ] **Step 3:** Add one line to study 30's `README.md` "Running notes" pointing at
  `eval/README.md`; the study-recap skill carries it into the study's Results later.
- [ ] **Step 4:** Leave everything for the user to commit (results: `comparison.*`,
  `results_*.json`, `samples_*.jsonl.gz`, logs; a few MB).
