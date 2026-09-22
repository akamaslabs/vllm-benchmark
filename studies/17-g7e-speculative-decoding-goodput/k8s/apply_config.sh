#!/bin/bash
# Apply-config step for 17-g7e-speculative-decoding-goodput (runs on the toolbox host
# via the Akamas workflow's Executor task).
#
# `set -e` ADDED 2026-09-22 and it is not cosmetic. Without it, a failed `kubectl apply`
# on line ~63 did not stop the script: execution fell through to `kubectl rollout status`,
# which then reported on the deployment ALREADY RUNNING — the previous trial's
# configuration — and exited 0 if that one was healthy. Akamas would have benchmarked the
# previous experiment's config while attributing the result to the current one. A silent,
# results-corrupting failure, not a loud one. Study 16's version has had `set -e` since it
# was written; this study inherited study 2's copy, which never did.
#
# Safe against the rest of the script: every `grep -q` sits in an `if`/`elif` condition
# (exempt from set -e), the rollout wait is explicitly bracketed by `set +e`/`set -e`, and
# both `kubectl logs` calls end in `|| true`.
set -e

DEPLOY_FILE=/work/vllm-benchmark/studies/17-g7e-speculative-decoding-goodput/k8s/01-deployment.yaml

# --- Step 1: boolean CLI flags (same fix as prior studies) ---
# vLLM's boolean flags (enforce-eager, disable-cascade-attn, async-scheduling,
# enable-expert-parallel, disable-custom-all-reduce) use argparse.BooleanOptionalAction,
# which rejects an explicit "--flag=value" form — only bare --flag / --no-flag is
# accepted. The Akamas vLLM pack declares these as categorical "true"/"false" string
# parameters, so FileConfigurator renders "--flag=true"/"--flag=false" into the
# deployment args; rewrite those into the accepted form here, right before applying.
for flag in enforce-eager disable-cascade-attn async-scheduling enable-expert-parallel disable-custom-all-reduce; do
  sed -i "s/--${flag}=true/--${flag}/" "$DEPLOY_FILE"
  sed -i "s/--${flag}=false/--no-${flag}/" "$DEPLOY_FILE"
done

# --- Step 1b: speculative decoding off means the flags must not exist at all ---
# NEW in this study. vLLM 0.29.0 treats speculative decoding as "off" only when none of
# --speculative-config/--spec-method/--spec-model/--spec-tokens is present on the
# command line (create_speculative_config in vllm/engine/arg_utils.py). Passing a
# sentinel instead of omitting the flags is a hard startup failure, in two different
# ways: argparse rejects "none" as a --spec-method choice, and Pydantic rejects
# num_speculative_tokens=0 (the field --spec-tokens maps to declares gt=0). So when
# Akamas renders the pack's documented off-sentinel (spec_method="none", which its own
# parameterConstraints pair with spec_tokens=0), BOTH lines are deleted here rather
# than lowered. This is the generalization of Step 1's boolean rewrite that the vLLM
# pack's README asks the consuming study to implement.
if grep -qE '^[[:space:]]*-[[:space:]]*"--spec-method=none"[[:space:]]*$' "$DEPLOY_FILE"; then
  echo "apply_config: spec_method=none — removing --spec-method/--spec-tokens/--spec-model (vLLM requires absence, not a sentinel)"
  sed -i -E '/^[[:space:]]*-[[:space:]]*"--spec-method=/d; /^[[:space:]]*-[[:space:]]*"--spec-tokens=/d; /^[[:space:]]*-[[:space:]]*"--spec-model=/d' "$DEPLOY_FILE"
elif grep -qE '^[[:space:]]*-[[:space:]]*"--spec-method=draft_model"[[:space:]]*$' "$DEPLOY_FILE"; then
  # draft_model is the ONLY method that takes an explicit --spec-model. The template
  # always renders the drafter line; here it simply survives. Fail loudly if it is
  # missing, because vLLM would otherwise die with "num_speculative_tokens was provided
  # but without speculative model." after a full image pull and weight load.
  if ! grep -qE '^[[:space:]]*-[[:space:]]*"--spec-model=[^"]+"[[:space:]]*$' "$DEPLOY_FILE"; then
    echo "error: spec_method=draft_model but no --spec-model line survived rendering — vLLM cannot start without a drafter reference" >&2
    exit 2
  fi
  echo "apply_config: spec_method=draft_model — keeping --spec-model $(grep -oE '\-\-spec-model=[^\"]+' "$DEPLOY_FILE")"
else
  # ngram / ngram_gpu / suffix / mtp: each self-assigns an internal model placeholder,
  # and an explicit --spec-model fights it. Drop only that line, keep method and tokens.
  echo "apply_config: drafter-free speculative method — removing --spec-model"
  sed -i -E '/^[[:space:]]*-[[:space:]]*"--spec-model=/d' "$DEPLOY_FILE"
fi

# Defensive, whatever branch ran above: a real method left with the 0 sentinel would
# crash vLLM at config construction. The study's parameterConstraints forbid that
# combination, so reaching here means the study YAML and this script disagree — fail
# loudly rather than spend a 45-minute trial discovering it in a crash loop.
if grep -qE '^[[:space:]]*-[[:space:]]*"--spec-tokens=0"[[:space:]]*$' "$DEPLOY_FILE"; then
  echo "error: spec_tokens=0 rendered together with a non-none spec_method — the study's sentinel parameterConstraints are missing or wrong" >&2
  exit 2
fi

# --- Step 2: strip any vLLM parameter flag left with no rendered value ---
# Only matters if this study's baseline step ever excludes some parameters from
# rendering (as 1-goodput-realistic-load's did) — a no-op on trials where every
# ${vLLM.*} token gets a real, non-empty computed value. Kept as a generic safety net
# regardless of which baseline-rendering design this study's akamas/study.yaml ends up
# using (not yet built — see README "Prerequisites still open" #4).
sed -i -E '/\$\{vLLM\./d; /^[[:space:]]*-[[:space:]]*"--[A-Za-z0-9_-]+="[[:space:]]*$/d' "$DEPLOY_FILE"

if ! kubectl apply -f "$DEPLOY_FILE" -n llm-serving; then
  echo "error: kubectl apply failed — the cluster still runs the PREVIOUS trial's configuration. Failing the task rather than benchmarking the wrong config." >&2
  exit 3
fi

# Don't let a failed rollout exit immediately — print vLLM's own container logs first,
# so they land in this task's stdout and show up in the Akamas UI (experiment/trial
# view) without needing separate kubectl access.
set +e
# 1740s (29 min) — just under the Deployment's own 1800s progressDeadlineSeconds, so
# the rollout's verdict comes from Kubernetes rather than from this wait expiring first.
# Raised with the model swap: 29.03 GiB of FP8 weights, cold, do not load in 25 minutes.
kubectl rollout status deployment/vllm -n llm-serving --timeout=1740s
ROLLOUT_EXIT=$?
set -e

echo "--- vLLM container logs (current pod) ---"
kubectl logs deployment/vllm -n llm-serving --tail=200 || true
echo "--- vLLM container logs (previous pod, if it crashed and restarted) ---"
kubectl logs deployment/vllm -n llm-serving --tail=200 --previous 2>/dev/null || true

exit $ROLLOUT_EXIT
