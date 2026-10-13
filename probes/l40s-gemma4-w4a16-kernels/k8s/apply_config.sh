#!/bin/bash
# Starts one vLLM combination on the L40S for the W4A16 kernel probe (outside Akamas).
# Ported from study 2's (akamas-gatling-llms-optimization, gemma4-kernels/k8s/apply_config.sh):
# validate -> free the GPU -> start one replica -> warm-up -> health check -> startup summary.
# Env: PARAMS (params.env), RENDERED (output YAML), TUNED_DIR, KUBE_CONTEXT, ROLLOUT_S.
# Exit codes: 0 ready; 2 invalid parameters; 3 cluster problem; 4 vLLM did not start.
set -euo pipefail

STUDY_DIR=$(cd "$(dirname "$0")/.." && pwd)
NS=llm-serving
STS=vllm-w4a16
NODE_LABEL=gpu=l40s
MODEL=gemma4-w4a16
# Explicit cluster, never the current context.
KUBE_CONTEXT=${KUBE_CONTEXT:-lab-vllm-bench-gatling}
kubectl() { command kubectl --context "$KUBE_CONTEXT" "$@"; }
TUNED_DIR=${TUNED_DIR:-$STUDY_DIR/kernel-bench/tuned-configs}
PARAMS=${PARAMS:?PARAMS=<params.env> is required}
TEMPLATE=$STUDY_DIR/k8s/01-statefulset_template.yaml
RENDERED=${RENDERED:-$(mktemp --suffix=.yaml)}
ROLLOUT_S=${ROLLOUT_S:-2400}
# shellcheck source=lib_health.sh
source "$STUDY_DIR/k8s/lib_health.sh"

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

bash "$STUDY_DIR/k8s/render_statefulset.sh" "$PARAMS" "$TEMPLATE" "$RENDERED" || die "invalid parameters in $PARAMS"
t "parameters: $(grep -v '^#' "$PARAMS" | grep . | tr '\n' ' ')"
t "vLLM flags: $(grep -E '^ +- "--' "$RENDERED" | sed -e 's/^ *- "//' -e 's/"$//' | tr '\n' ' ')"

NODE=$(kubectl get nodes -l $NODE_LABEL -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[ -n "$NODE" ] || die "no node with $NODE_LABEL (node group llm-serving-l40s scaled to 0?)" 3
if grep -q 'VLLM_TUNED_CONFIG_FOLDER' "$PARAMS"; then
  ls "$TUNED_DIR"/*.json >/dev/null 2>&1 || die "tuned configs requested but none in $TUNED_DIR"
fi

# --- 1. Free the GPU: ours and study 2's StatefulSet `vllm` (same node group) ---------
t "1/3 freeing the GPU"
kubectl -n $NS scale sts $STS --replicas=0 2>/dev/null || true
kubectl -n $NS delete pod -l app=$STS --ignore-not-found --wait=true --timeout=180s >/dev/null
OTHER=$(kubectl -n $NS get pods -l app=vllm -o name 2>/dev/null | wc -l | tr -d ' ')
[ "$OTHER" = 0 ] || die "study 2's vllm pod is running on this cluster: refusing to share the GPU" 3

# --- 2. Start the replica ----------------------------------------------------------------
t "2/3 starting vLLM on $NODE"
TUNED_FILE="$TUNED_DIR/E=128,N=704,device_name=NVIDIA_L40S,dtype=int4_w4a16.json"
if [ -f "$TUNED_FILE" ]; then
  TMP_CFG=$(mktemp -d)
  cp "$TUNED_FILE" "$TMP_CFG/moe-e128-n704-l40s-int4.json"
  kubectl -n $NS create configmap moe-tuned-configs-w4a16 --from-file="$TMP_CFG/moe-e128-n704-l40s-int4.json" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  rm -rf "$TMP_CFG"
fi
kubectl apply -f "$RENDERED" >/dev/null || die "kubectl apply failed" 3
kubectl -n $NS scale sts $STS --replicas=1 >/dev/null

set +e
ROLLOUT_EXIT=1
T0=$SECONDS
while [ $((SECONDS - T0)) -lt "$ROLLOUT_S" ]; do
  READY=$(kubectl -n $NS get sts $STS -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  if [ "${READY:-0}" = 1 ]; then ROLLOUT_EXIT=0; break; fi
  RESTARTS=$(kubectl -n $NS get pod $STS-0 -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  if [ "${RESTARTS:-0}" -ge 1 ]; then
    echo "error: $STS-0 restarted ${RESTARTS} time(s) during startup, failing now" >&2
    ROLLOUT_EXIT=4; break
  fi
  sleep 15
done
[ $ROLLOUT_EXIT = 1 ] && { echo "error: $STS-0 not Ready after $ROLLOUT_S s" >&2; ROLLOUT_EXIT=4; }
t "rollout exit=$ROLLOUT_EXIT after $((SECONDS - T0)) s"
set -e

# --- 3. Warm-up ---------------------------------------------------------------------------
if [ $ROLLOUT_EXIT -eq 0 ]; then
  t "3/3 warm-up: 8 requests"
  WARM_FAIL=0
  kubectl -n $NS exec $STS-0 -c vllm -- python3 -c '
import json, sys, time, urllib.request, concurrent.futures as cf
def one(i):
    body = {"model": sys.argv[1], "max_tokens": 64,
            "messages": [{"role": "user", "content": "Write a short story about the number %d." % i}]}
    s = time.perf_counter()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read()
    return time.perf_counter() - s
with cf.ThreadPoolExecutor(4) as ex:
    print("warm-up e2e (s):", " ".join("%.2f" % x for x in ex.map(one, range(8))))
' "$MODEL" || { echo "error: warm-up requests failed" >&2; WARM_FAIL=1; }
  READY=$(kubectl -n $NS get sts $STS -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  RESTARTS=$(kubectl -n $NS get pod $STS-0 -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  if ! REASON=$(replicas_healthy 1 "$READY" "${RESTARTS:-0}" "$WARM_FAIL"); then
    echo "error: $REASON" >&2
    ROLLOUT_EXIT=4
  fi
fi

echo "--- $STS-0: startup summary ---"
kubectl -n $NS logs $STS-0 --all-containers --tail=-1 2>/dev/null \
  | grep -E 'Selected .*Kernel|Using .*[Bb]ackend|[Bb]ackend|MoE config|configuration from|Marlin|Humming|humming|TRITON_ATTN|Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|Error|Traceback' | head -60 || true
echo "--- $STS-0: full logs ---"
kubectl -n $NS logs $STS-0 --all-containers --tail=-1 2>/dev/null || true
if [ $ROLLOUT_EXIT -ne 0 ]; then
  echo "--- $STS-0: previous container ---"
  kubectl -n $NS logs $STS-0 --all-containers --tail=-1 --previous 2>/dev/null || true
  kubectl -n $NS describe pod $STS-0 2>/dev/null | tail -30 || true
fi
exit $ROLLOUT_EXIT
