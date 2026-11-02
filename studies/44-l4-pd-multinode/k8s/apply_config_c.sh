#!/bin/bash
# Study 44 tappa C apply-config step (runs on the toolbox via the Akamas Executor, after the
# FileConfigurator rendered 40-c-config_template.yaml into 40-c-config.yaml). Study 27's
# apply_config.sh, adapted to one pod per GPU on two nodes:
#   1. drop flag lines left unrendered or rendered empty (doNotRenderParameters);
#   2. ConfigMaps: the rendered config, the scripts (serve.sh, pd_router.py);
#   3. read the topology (PD_PREFILL_INSTANCES) and fill 41-c-serving_template.yaml:
#        1 -> L40S prefill, A10G decode           (1P1D)
#        0 -> both aggregated, router "least"     (0P2D)
#   4. apply, wait for all three Deployments, fail fast on a crash loop;
#   5. dump every pod's full logs (convention: the trial's logs live in Akamas).
set -e
K8S=${STUDY44_K8S:-/work/study44/44-l4-pd-multinode/k8s}
CONFIG_FILE=$K8S/40-c-config.yaml
NS=llm-serving
SVC=llm-serving.svc.cluster.local

# --- Step 1: unrendered / empty flags ---
sed -i -E '/\$\{(vllm_l40s|vllm_a10g|pd_topology)\./d' "$CONFIG_FILE"
sed -i -E '/^[[:space:]]+--[A-Za-z0-9-]+=[[:space:]]*$/d' "$CONFIG_FILE"

# --- Step 2: ConfigMaps ---
kubectl apply -f "$CONFIG_FILE"
kubectl create configmap pd-mn-scripts -n "$NS" \
  --from-file=serve.sh="$K8S/serve.sh" --from-file=pd_router.py="$K8S/pd_router.py" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$K8S/02-service.yaml" -f "$K8S/monitoring/servicemonitor.yaml"
# Tappa A/B role Services would select nothing here, but keep them gone: see the template.
kubectl delete service vllm-pd-mn-prefill vllm-pd-mn-decode -n "$NS" --ignore-not-found

# --- Step 3: topology ---
P=$(sed -n -E 's/^[[:space:]]*PD_PREFILL_INSTANCES=([0-9]+).*/\1/p' "$CONFIG_FILE" | head -1)
case "${P:-}" in
  1) L40S_ROLE=prefill; A10G_ROLE=decode; TOPO=1P1D
     PREFILL_URLS="http://vllm-pd-mn-l40s.$SVC:8000"; DECODE_URLS="http://vllm-pd-mn-a10g.$SVC:8000" ;;
  0) L40S_ROLE=aggregated; A10G_ROLE=aggregated; TOPO=0P2D
     PREFILL_URLS=""; DECODE_URLS="http://vllm-pd-mn-l40s.$SVC:8000,http://vllm-pd-mn-a10g.$SVC:8000" ;;
  *) echo "error: PD_PREFILL_INSTANCES='${P:-}' (expected 0 or 1)"; exit 3 ;;
esac
# A fresh pod per trial: the hash includes the time, so two identical trials both start cold.
SHA=$( { cat "$CONFIG_FILE" "$K8S/serve.sh" "$K8S/pd_router.py"; date -u +%s; } | sha256sum | cut -c1-16)
echo "--- topology $TOPO: L40S $L40S_ROLE, A10G $A10G_ROLE; rendered config ---"
sed -n '/^data:/,$p' "$CONFIG_FILE"

SERVING=$(mktemp --suffix=.yaml)
sed -e "s|__L40S_ROLE__|$L40S_ROLE|g" -e "s|__A10G_ROLE__|$A10G_ROLE|g" -e "s|__TOPOLOGY__|$TOPO|g" \
    -e "s|__PREFILL_URLS__|$PREFILL_URLS|g" -e "s|__DECODE_URLS__|$DECODE_URLS|g" -e "s|__CONFIG_SHA__|$SHA|g" \
    "$K8S/41-c-serving_template.yaml" > "$SERVING"
# Tappa A/B Deployments (other names) must not hold the GPUs.
kubectl delete deployment vllm-pd-mn-prefill vllm-pd-mn-decode -n "$NS" --ignore-not-found
APPLY_TS=$(date -u +%s)
kubectl apply -f "$SERVING"
rm -f "$SERVING"

# --- Step 4: wait, failing fast on a crash loop (2 restarts of a pod from this rollout) ---
set +e
ROLLOUT_EXIT=0
DEADLINE=$((SECONDS + 1500))
for d in vllm-pd-mn-l40s vllm-pd-mn-a10g vllm-pd-mn-router; do
  while true; do
    OUT=$(kubectl rollout status deployment/$d -n "$NS" --timeout=30s 2>&1); RC=$?
    echo "$OUT" | tail -1
    [ "$RC" -eq 0 ] && break
    if echo "$OUT" | grep -q 'exceeded its progress deadline'; then ROLLOUT_EXIT=1; break 2; fi
    RESTARTS=$(kubectl get pod -n "$NS" -l app=vllm-pd-mn \
      -o jsonpath='{range .items[*]}{.metadata.creationTimestamp}{" "}{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null \
      | while read -r ts rc; do
          [ -n "$ts" ] && [ "$(date -u -d "$ts" +%s 2>/dev/null || echo 0)" -ge "$APPLY_TS" ] && echo "${rc:-0}"
        done | sort -n | tail -1)
    if [ "${RESTARTS:-0}" -ge 2 ]; then
      echo "error: a container restarted ${RESTARTS} times during this rollout (crash loop) — failing fast"
      ROLLOUT_EXIT=2; break 2
    fi
    if [ "$SECONDS" -ge "$DEADLINE" ]; then echo "error: rollout did not complete within 1500s"; ROLLOUT_EXIT=1; break 2; fi
  done
done
set -e

# --- Step 5: logs ---
for d in vllm-pd-mn-l40s vllm-pd-mn-a10g; do
  echo "--- $d engine logs (full) ---"
  kubectl logs deployment/$d -n "$NS" --tail=-1 || true
  echo "--- $d previous run, if it restarted ---"
  kubectl logs deployment/$d -n "$NS" --tail=300 --previous 2>/dev/null || true
done
echo "--- router logs ---"
kubectl logs deployment/vllm-pd-mn-router -n "$NS" --tail=-1 || true
echo "--- key lines: role, GPU, KV capacity, kernels, NIXL ---"
for d in vllm-pd-mn-l40s vllm-pd-mn-a10g; do
  kubectl logs deployment/$d -n "$NS" --tail=-1 2>/dev/null \
    | grep -iE 'serve:|GPU 0:|KV cache size|Maximum concurrency|Using .*Kernel|attention backend|NIXL compatibility|Loading weights took' \
    | grep -v -iE 'tuned config' | sed "s/^/[$d] /" | head -60 || true
done
echo "--- router /health ---"
kubectl exec deployment/vllm-pd-mn-router -n "$NS" -- python3 -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).read().decode())" 2>&1 || true
kubectl get pod -n "$NS" -l app=vllm-pd-mn -o wide || true

exit $ROLLOUT_EXIT
