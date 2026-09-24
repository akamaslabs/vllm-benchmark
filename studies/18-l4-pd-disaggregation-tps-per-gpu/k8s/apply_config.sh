#!/bin/bash
# Apply-config step for study 18 (runs on the toolbox host via the Akamas Executor, after
# FileConfigurator rendered 01-deployment_template.yaml into 01-deployment.yaml).
set -e
K8S=/work/vllm-benchmark/studies/18-l4-pd-disaggregation-tps-per-gpu/k8s
DEPLOY_FILE=$K8S/01-deployment.yaml
NS=llm-serving

# --- Step 1: drop any flag left with no rendered value ---
# One flag per line in the pd-config ConfigMap, so an unrendered ${component.param} token
# means "use vLLM's (or the launcher's) default": delete that line.
sed -i -E '/\$\{(vllm_prefill|vllm_decode|pd_topology)\./d' "$DEPLOY_FILE"

# --- Step 2: launcher + router scripts, regenerated from the repo on every trial ---
kubectl create configmap pd-scripts -n "$NS" \
  --from-file=launcher.sh="$K8S/launcher.sh" --from-file=pd_router.py="$K8S/pd_router.py" \
  --dry-run=client -o yaml | kubectl apply -f -

# --- Step 3: force a fresh pod per trial ---
# A ConfigMap change alone does not restart a Deployment. The pod-template annotation
# pd-config-sha changes on every trial: it hashes the rendered config, both scripts and
# the time. The time is in on purpose, so two trials with identical settings still both
# start cold.
SHA=$( { awk '/^---$/{exit} {print}' "$DEPLOY_FILE"; cat "$K8S/launcher.sh" "$K8S/pd_router.py"; date -u +%s; } | sha256sum | cut -c1-16)
sed -i "s/__PD_CONFIG_SHA__/$SHA/" "$DEPLOY_FILE"

echo "--- rendered pd-config (topology + per-role flags) ---"
awk '/^---$/{exit} {print}' "$DEPLOY_FILE" | sed -n '/^data:/,$p'

APPLY_TS=$(date -u +%s)
kubectl apply -f "$DEPLOY_FILE"

# --- Step 4: wait for the rollout, failing fast on a crash loop ---
# The router's startup probe is 200 only when EVERY instance of the topology is healthy,
# so a completed rollout means the whole topology is up. The launcher exits as soon as
# any instance dies, so a broken configuration shows up as restarts of `engine`. Two
# restarts of a pod created by THIS rollout end the wait immediately (same guard as
# studies 13-16). Pods older than the apply are ignored.
set +e
ROLLOUT_EXIT=1
DEADLINE=$((SECONDS + 1500))
while true; do
  OUT=$(kubectl rollout status deployment/vllm-pd -n "$NS" --timeout=30s 2>&1)
  RC=$?
  echo "$OUT" | tail -1
  if [ "$RC" -eq 0 ]; then ROLLOUT_EXIT=0; break; fi
  if echo "$OUT" | grep -q 'exceeded its progress deadline'; then ROLLOUT_EXIT=1; break; fi
  RESTARTS=$(kubectl get pod -n "$NS" -l app=vllm-pd \
    -o jsonpath='{range .items[*]}{.metadata.creationTimestamp}{" "}{.status.containerStatuses[?(@.name=="engine")].restartCount}{"\n"}{end}' 2>/dev/null \
    | while read -r ts rc; do
        [ -n "$ts" ] && [ "$(date -u -d "$ts" +%s 2>/dev/null || echo 0)" -ge "$APPLY_TS" ] && echo "${rc:-0}"
      done | sort -n | tail -1)
  if [ "${RESTARTS:-0}" -ge 2 ]; then
    echo "error: engine container restarted ${RESTARTS} times during this rollout (crash loop) — failing fast"
    ROLLOUT_EXIT=2; break
  fi
  if [ "$SECONDS" -ge "$DEADLINE" ]; then echo "error: rollout did not complete within 1500s"; ROLLOUT_EXIT=1; break; fi
done
set -e

echo "--- engine container logs (all instances, prefixed by role; full) ---"
kubectl logs deployment/vllm-pd -n "$NS" -c engine --tail=-1 || true
echo "--- router container logs ---"
kubectl logs deployment/vllm-pd -n "$NS" -c router --tail=-1 || true
echo "--- key lines: topology, GPU P2P, backend, KV capacity, NIXL/UCX transport ---"
kubectl logs deployment/vllm-pd -n "$NS" -c engine --tail=-1 2>/dev/null \
  | grep -iE 'launcher:|GPU[0-9]|P2P|attention backend|Using .* backend|fp8|KV cache|Maximum concurrency|nixl|ucx|cuda_ipc|kv_transfer|Loading weights took' \
  | grep -v -iE 'GET /(health|metrics)' | head -400 || true
echo "--- router /health ---"
kubectl exec deployment/vllm-pd -n "$NS" -c router -- python3 -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).read().decode())" 2>&1 || true
echo "--- engine container logs (previous run, if it crashed and restarted) ---"
kubectl logs deployment/vllm-pd -n "$NS" -c engine --tail=300 --previous 2>/dev/null || true

exit $ROLLOUT_EXIT
