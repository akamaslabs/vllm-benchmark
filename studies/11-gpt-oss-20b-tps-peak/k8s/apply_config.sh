#!/bin/bash
# Apply-config step for 11-gpt-oss-20b-tps-peak (runs on the toolbox host via the
# Akamas Executor, after FileConfigurator rendered 01-deployment_template.yaml).
set -e
DEPLOY_FILE=/work/vllm-benchmark/studies/11-gpt-oss-20b-tps-peak/k8s/01-deployment.yaml

# --- Step 1: boolean CLI flags ---
# vLLM's boolean flags use argparse.BooleanOptionalAction and reject "--flag=value"; the
# Akamas vLLM pack declares them as categorical "true"/"false", so FileConfigurator renders
# "--flag=true"/"--flag=false". Rewrite into the accepted bare form. The list must contain
# EVERY boolean parameter this study renders (study 9's list did not cover the 1.9.0
# booleans — keep this in sync with 01-deployment_template.yaml).
for flag in enforce-eager async-scheduling enable-expert-parallel disable-cascade-attn disable-custom-all-reduce enable-chunked-prefill enable-prefix-caching scheduler-reserve-full-isl; do
  sed -i "s/--${flag}=true/--${flag}/" "$DEPLOY_FILE"
  sed -i "s/--${flag}=false/--no-${flag}/" "$DEPLOY_FILE"
done

# --- Step 2: strip any vLLM parameter flag left with no rendered value ---
# The baseline step leaves every tuned parameter but gpu_memory_utilization unrendered
# (doNotRenderParameters), so vLLM's own defaults apply; the optional 1.9.0 tokens are
# stripped the same way until that pack version is installed.
sed -i -E '/\$\{vLLM\./d; /^[[:space:]]*-[[:space:]]*"--[A-Za-z0-9_-]+="[[:space:]]*$/d' "$DEPLOY_FILE"

echo "--- rendered vLLM args ---"
sed -n '/args:/,/env:/p' "$DEPLOY_FILE" | grep -E '^\s*- "' || true

kubectl apply -f "$DEPLOY_FILE" -n llm-serving

# Don't exit immediately on a failed rollout: print vLLM's own logs first so they land in
# this task's stdout (visible in the Akamas experiment view without kubectl access).
set +e
kubectl rollout status deployment/vllm -n llm-serving --timeout=1500s
ROLLOUT_EXIT=$?
set -e

echo "--- vLLM container logs (current pod, full) ---"
kubectl logs deployment/vllm -n llm-serving -c vllm --tail=-1 || true
echo "--- kernel/backend selection lines (expected on L4: TRITON_ATTN, Marlin MXFP4 MoE) ---"
kubectl logs deployment/vllm -n llm-serving -c vllm --tail=-1 2>/dev/null | grep -iE 'attention backend|mxfp4|marlin|quantization|Using .* backend|expert parallel|Loading weights took|KV cache|Maximum concurrency' || true
echo "--- vLLM container logs (previous pod, if it crashed and restarted) ---"
kubectl logs deployment/vllm -n llm-serving -c vllm --tail=300 --previous 2>/dev/null || true

exit $ROLLOUT_EXIT
