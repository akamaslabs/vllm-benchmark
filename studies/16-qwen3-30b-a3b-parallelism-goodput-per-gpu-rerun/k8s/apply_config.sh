#!/bin/bash
# Apply-config step for 15-qwen3-30b-a3b-parallelism-goodput-per-gpu (runs on the toolbox host via the
# Akamas Executor, after FileConfigurator rendered 01-deployment_template.yaml).
set -e
DEPLOY_FILE=/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu/k8s/01-deployment.yaml

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

# Epoch seconds just before the apply: only pods created after this instant belong to
# this rollout (used by the crash-loop guard below).
APPLY_TS=$(date -u +%s)
kubectl apply -f "$DEPLOY_FILE" -n llm-serving

# Don't exit immediately on a failed rollout: print vLLM's own logs first so they land in
# this task's stdout (visible in the Akamas experiment view without kubectl access).
#
# Fail fast on a crash loop (added for study 13, 2026-09-15). Study 12's experiment 12
# died deterministically ~70 s after every start (sampler-warmup CUDA OOM, see the study
# manifest's sampler-warmup parameterConstraint) yet the trial burned the whole progress
# deadline in CrashLoopBackOff before failing. On this study the deadline is 30 minutes
# (31 GB of weights to download cold), so the guard matters more, not less. A healthy vLLM start never restarts, so once
# the vllm container of a pod created by THIS rollout has restarted twice there is nothing
# left to wait for: stop polling, dump the logs below and fail the trial now. Pods are
# filtered by creation time so the old pod of a previous crashed trial, still terminating
# under the Recreate strategy, cannot trip the guard with its own restart count.
set +e
ROLLOUT_EXIT=1
DEADLINE=$((SECONDS + 2100))
while true; do
  OUT=$(kubectl rollout status deployment/vllm -n llm-serving --timeout=30s 2>&1)
  RC=$?
  echo "$OUT" | tail -1
  if [ "$RC" -eq 0 ]; then ROLLOUT_EXIT=0; break; fi
  if echo "$OUT" | grep -q 'exceeded its progress deadline'; then ROLLOUT_EXIT=1; break; fi
  RESTARTS=$(kubectl get pod -n llm-serving -l app=vllm \
    -o jsonpath='{range .items[*]}{.metadata.creationTimestamp}{" "}{.status.containerStatuses[?(@.name=="vllm")].restartCount}{"\n"}{end}' 2>/dev/null \
    | while read -r ts rc; do
        [ -n "$ts" ] && [ "$(date -u -d "$ts" +%s 2>/dev/null || echo 0)" -ge "$APPLY_TS" ] && echo "${rc:-0}"
      done | sort -n | tail -1)
  if [ "${RESTARTS:-0}" -ge 2 ]; then
    echo "error: vLLM container restarted ${RESTARTS} times during this rollout (crash loop) — failing fast instead of waiting for the progress deadline"
    ROLLOUT_EXIT=2; break
  fi
  if [ "$SECONDS" -ge "$DEADLINE" ]; then echo "error: rollout did not complete within 1500s"; ROLLOUT_EXIT=1; break; fi
done
set -e

echo "--- vLLM container logs (current pod, full) ---"
kubectl logs deployment/vllm -n llm-serving -c vllm --tail=-1 || true
echo "--- kernel/backend selection lines (expected on L4: TRITON_ATTN, FP8 Marlin w8a16 for the block-quantized MoE) ---"
kubectl logs deployment/vllm -n llm-serving -c vllm --tail=-1 2>/dev/null | grep -iE 'attention backend|marlin|fp8|quantization|Using .* backend|expert parallel|Loading weights took|KV cache|Maximum concurrency' || true
echo "--- vLLM container logs (previous pod, if it crashed and restarted) ---"
kubectl logs deployment/vllm -n llm-serving -c vllm --tail=300 --previous 2>/dev/null || true

exit $ROLLOUT_EXIT
