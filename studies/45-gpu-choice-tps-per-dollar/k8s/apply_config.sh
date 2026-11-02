#!/bin/bash
# Study 45 apply-config step (toolbox, Akamas Executor, after the FileConfigurator rendered
# 50-config_template.yaml into 50-config.yaml). Study 44 tappa C's apply_config_c.sh plus the
# node choice:
#   1. drop flag lines left unrendered / rendered empty;
#   2. topology and instances from the rendered config; reject invalid combinations;
#   3. scale the chosen node groups up (2 nodes when node_a and node_b are the same instance)
#      and every other study-45 group to 0; wait for the nodes (fail after 15 min: no capacity);
#   4. ConfigMaps, then the serving layout from 51-serving_template.yaml (node_b's pod only
#      when the topology uses node_b), with each node's price for the router;
#   5. wait for the Deployments, failing fast on a crash loop; dump every log.
set -e
K8S=${STUDY45_K8S:-/work/study45/45-gpu-choice-tps-per-dollar/k8s}
CONFIG_FILE=$K8S/50-config.yaml
NS=llm-serving
SVC=llm-serving.svc.cluster.local
# shellcheck disable=SC1091
source "$K8S/nodes.sh"

# --- Step 1 ---
sed -i -E '/\$\{(vllm_a|vllm_b|pd_topology|node_a|node_b)\./d' "$CONFIG_FILE"
sed -i -E '/^[[:space:]]+--[A-Za-z0-9-]+=[[:space:]]*$/d' "$CONFIG_FILE"

# --- Step 2 ---
val() { sed -n -E "s/^[[:space:]]*$1=([^[:space:]]+).*/\1/p" "$CONFIG_FILE" | head -1; }
P=$(val PD_PREFILL_INSTANCES); D=$(val PD_DECODE_INSTANCES)
A_INST="$(val NODE_A_TYPE).$(val NODE_A_SIZE)"; B_INST="$(val NODE_B_TYPE).$(val NODE_B_SIZE)"
case "$P$D" in
  01) TOPO=P0D1; USE_B=0; A_ROLE=aggregated; B_ROLE=none
      PREFILL_URLS=""; DECODE_URLS="http://vllm-pd-mn-a.$SVC:8000" ;;
  02) TOPO=P0D2; USE_B=1; A_ROLE=aggregated; B_ROLE=aggregated
      PREFILL_URLS=""; DECODE_URLS="http://vllm-pd-mn-a.$SVC:8000,http://vllm-pd-mn-b.$SVC:8000" ;;
  11) TOPO=P1D1; USE_B=1; A_ROLE=prefill; B_ROLE=decode
      PREFILL_URLS="http://vllm-pd-mn-a.$SVC:8000"; DECODE_URLS="http://vllm-pd-mn-b.$SVC:8000" ;;
  *) echo "error: topology P=$P D=$D is not one of P0D1, P0D2, P1D1"; exit 3 ;;
esac
A_GROUP=$(group_of "$A_INST") || { echo "error: node_a instance '$A_INST' is not in the study's list"; exit 3; }
A_PRICE=$(price_of "$A_INST")
if [ "$USE_B" = 1 ]; then
  B_GROUP=$(group_of "$B_INST") || { echo "error: node_b instance '$B_INST' is not in the study's list"; exit 3; }
  B_PRICE=$(price_of "$B_INST")
else
  B_GROUP=""; B_INST=none; B_PRICE=0
fi
echo "--- experiment: topology $TOPO, node_a $A_INST (\$$A_PRICE/h, $A_ROLE), node_b $B_INST (\$$B_PRICE/h, $B_ROLE) ---"
sed -n '/^data:/,$p' "$CONFIG_FILE"

# --- Step 3: nodes ---
kubectl delete deployment -n "$NS" -l app=vllm-pd-mn --ignore-not-found --wait=true
for g in $ALL_GROUPS; do
  want=0
  [ "$g" = "$A_GROUP" ] && want=$((want + 1))
  [ -n "$B_GROUP" ] && [ "$g" = "$B_GROUP" ] && want=$((want + 1))
  [ "$(desired_of "$g")" = "$want" ] || scale_group "$g" "$want"
done
wait_nodes() {  # group count
  local g=$1 n=$2 deadline=$((SECONDS + 900)) ready
  while true; do
    ready=$(kubectl get nodes -l node-role="$g" --no-headers 2>/dev/null | grep -c ' Ready' || true)
    [ "${ready:-0}" -ge "$n" ] && { echo "$g: $ready node(s) Ready"; return 0; }
    [ "$SECONDS" -ge "$deadline" ] && { echo "error: $g has $ready/$n Ready nodes after 15 min (no capacity?)"; return 1; }
    sleep 20
  done
}
if [ -n "$B_GROUP" ] && [ "$B_GROUP" = "$A_GROUP" ]; then
  wait_nodes "$A_GROUP" 2 || exit 4
else
  wait_nodes "$A_GROUP" 1 || exit 4
  if [ -n "$B_GROUP" ]; then wait_nodes "$B_GROUP" 1 || exit 4; fi
fi
kubectl get nodes -l 'node-role in (llm-serving-l4-pd-xl,llm-serving-g6-2xl,llm-serving-g5-xl,llm-serving-a10g-pd,llm-serving-g6e-xl,llm-serving-l40s-pd)' \
  -L node-role,node.kubernetes.io/instance-type,topology.kubernetes.io/zone

# --- Step 4: ConfigMaps and serving layout ---
kubectl apply -f "$CONFIG_FILE"
kubectl create configmap pd-mn-scripts -n "$NS" \
  --from-file=serve.sh="$K8S/serve.sh" --from-file=pd_router.py="$K8S/pd_router.py" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$K8S/02-service.yaml" -f "$K8S/monitoring/servicemonitor.yaml"
SHA=$( { cat "$CONFIG_FILE" "$K8S/serve.sh" "$K8S/pd_router.py"; date -u +%s; } | sha256sum | cut -c1-16)
read -r A_CPU A_MEMR A_MEML A_SHM <<< "$(resources_of "$A_INST")"
if [ "$USE_B" = 1 ]; then read -r B_CPU B_MEMR B_MEML B_SHM <<< "$(resources_of "$B_INST")"; ROUTER_GROUP=$B_GROUP
else B_CPU=1; B_MEMR=1Gi; B_MEML=1Gi; B_SHM=1Gi; ROUTER_GROUP=$A_GROUP; fi
SERVING=$(mktemp --suffix=.yaml)
sed -e "s|__A_ROLE__|$A_ROLE|g" -e "s|__B_ROLE__|$B_ROLE|g" -e "s|__TOPOLOGY__|$TOPO|g" \
    -e "s|__A_NODE_ROLE__|$A_GROUP|g" -e "s|__B_NODE_ROLE__|${B_GROUP:-none}|g" -e "s|__ROUTER_NODE_ROLE__|$ROUTER_GROUP|g" \
    -e "s|__A_CPU_REQ__|$A_CPU|g" -e "s|__A_MEM_REQ__|$A_MEMR|g" -e "s|__A_MEM_LIM__|$A_MEML|g" -e "s|__A_SHM__|$A_SHM|g" \
    -e "s|__B_CPU_REQ__|$B_CPU|g" -e "s|__B_MEM_REQ__|$B_MEMR|g" -e "s|__B_MEM_LIM__|$B_MEML|g" -e "s|__B_SHM__|$B_SHM|g" \
    -e "s|__A_INSTANCE__|$A_INST|g" -e "s|__B_INSTANCE__|$B_INST|g" -e "s|__A_PRICE__|$A_PRICE|g" -e "s|__B_PRICE__|$B_PRICE|g" \
    -e "s|__PREFILL_URLS__|$PREFILL_URLS|g" -e "s|__DECODE_URLS__|$DECODE_URLS|g" -e "s|__CONFIG_SHA__|$SHA|g" \
    "$K8S/51-serving_template.yaml" > "$SERVING"
if [ "$USE_B" = 0 ]; then  # drop document 2 (vllm-pd-mn-b) of the multi-document file
  python3 - "$SERVING" <<'EOF'
import sys
p = sys.argv[1]
docs = open(p).read().split("\n---\n")
open(p, "w").write("\n---\n".join(d for d in docs if "name: vllm-pd-mn-b\n  namespace: llm-serving\n  labels: {app: vllm-pd-mn, gpu: b}\nspec:\n  replicas" not in d))
EOF
fi
APPLY_TS=$(date -u +%s)
kubectl apply -f "$SERVING"
rm -f "$SERVING"

# --- Step 5: wait, fail fast on a crash loop ---
DEPLOYS="vllm-pd-mn-a vllm-pd-mn-router"; [ "$USE_B" = 1 ] && DEPLOYS="vllm-pd-mn-a vllm-pd-mn-b vllm-pd-mn-router"
set +e
ROLLOUT_EXIT=0
DEADLINE=$((SECONDS + 1500))
for d in $DEPLOYS; do
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

for d in vllm-pd-mn-a vllm-pd-mn-b; do
  kubectl get deployment $d -n "$NS" >/dev/null 2>&1 || continue
  echo "--- $d logs (full) ---"
  kubectl logs deployment/$d -n "$NS" --tail=-1 || true
  echo "--- $d previous run, if it restarted ---"
  kubectl logs deployment/$d -n "$NS" --tail=300 --previous 2>/dev/null || true
  kubectl describe pod -n "$NS" -l app=vllm-pd-mn,gpu=${d##*-} | grep -A6 'Last State' || true
done
echo "--- router logs ---"
kubectl logs deployment/vllm-pd-mn-router -n "$NS" --tail=-1 || true
echo "--- key lines ---"
for d in vllm-pd-mn-a vllm-pd-mn-b; do
  kubectl logs deployment/$d -n "$NS" --tail=-1 2>/dev/null \
    | grep -iE 'serve:|GPU 0:|KV cache size|Maximum concurrency|attention backend|NIXL compatibility|Loading weights took' \
    | sed "s/^/[$d] /" | head -40 || true
done
echo "--- router /health and node prices ---"
kubectl exec deployment/vllm-pd-mn-router -n "$NS" -- python3 -c \
  "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).read().decode());[print(l) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith('pd_node_price')]" 2>&1 || true
kubectl get pod -n "$NS" -l app=vllm-pd-mn -o wide || true

exit $ROLLOUT_EXIT
