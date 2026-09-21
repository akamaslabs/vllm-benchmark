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
  echo "apply_config: spec_method=none — removing --spec-method/--spec-tokens (vLLM requires absence, not a sentinel)"
  sed -i -E '/^[[:space:]]*-[[:space:]]*"--spec-method=/d; /^[[:space:]]*-[[:space:]]*"--spec-tokens=/d' "$DEPLOY_FILE"
else
  # Defensive: a real method with the 0 sentinel would crash vLLM at config
  # construction. The study's parameterConstraints forbid this combination, so
  # reaching here means the study YAML and this script disagree — fail loudly rather
  # than spend a 40-minute trial discovering it in a crash loop.
  if grep -qE '^[[:space:]]*-[[:space:]]*"--spec-tokens=0"[[:space:]]*$' "$DEPLOY_FILE"; then
    echo "error: spec_tokens=0 rendered together with a non-none spec_method — the study's sentinel parameterConstraints are missing or wrong" >&2
    exit 2
  fi
fi

# --- Step 2: strip any vLLM parameter flag left with no rendered value ---
# Only matters if this study's baseline step ever excludes some parameters from
# rendering (as 1-goodput-realistic-load's did) — a no-op on trials where every
# ${vLLM.*} token gets a real, non-empty computed value. Kept as a generic safety net
# regardless of which baseline-rendering design this study's akamas/study.yaml ends up
# using (not yet built — see README "Prerequisites still open" #4).
sed -i -E '/\$\{vLLM\./d; /^[[:space:]]*-[[:space:]]*"--[A-Za-z0-9_-]+="[[:space:]]*$/d' "$DEPLOY_FILE"

kubectl apply -f "$DEPLOY_FILE" -n llm-serving

# Don't let a failed rollout exit immediately — print vLLM's own container logs first,
# so they land in this task's stdout and show up in the Akamas UI (experiment/trial
# view) without needing separate kubectl access.
set +e
kubectl rollout status deployment/vllm -n llm-serving --timeout=1500s
ROLLOUT_EXIT=$?
set -e

echo "--- vLLM container logs (current pod) ---"
kubectl logs deployment/vllm -n llm-serving --tail=200 || true
echo "--- vLLM container logs (previous pod, if it crashed and restarted) ---"
kubectl logs deployment/vllm -n llm-serving --tail=200 --previous 2>/dev/null || true

exit $ROLLOUT_EXIT
