#!/bin/bash
# Apply-config step for 47-g7-kv-cache-offload-ttft. Runs on the akamas-41 toolbox via the Akamas
# workflow's Executor task, after FileConfigurator has rendered params.env.
#
# One experiment = one vLLM configuration on the single RTX PRO 4500 (node group
# llm-serving-g7-kv). The Deployment is replaced (Recreate) with the new flags; the AIPerf pod
# is kept. /dev/shm and the pod memory follow the CPU KV offload size (see 01-vllm_template.yaml).
#
# Fail-fast by design: a half-applied configuration would benchmark the previous experiment's
# setup under the current one's name. Exit codes: 2 bad parameters, 3 node missing / apply
# failed, 4 vLLM crash loop or not Ready, 6 AIPerf pod not Ready.
# RENDER_ONLY=1 prints the flags (one per line) and "shm=<Gi> mem=<Gi>", then exits: local tests
# (test_apply_config.sh).
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/study47/47-g7-kv-cache-offload-ttft}   # overridable for manual tests
NS=kv-offload
NODE_ROLE=llm-serving-g7-kv
K8S=$STUDY_DIR/k8s
PARAMS=$K8S/params.env
HERE=$(cd "$(dirname "$0")" && pwd)   # manifests live next to the script
OFFLOAD_MAX=28   # GiB: 28 + 20 (vLLM) + 6 (AIPerf) fits the node's 57 GiB allocatable

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# --- 0. Parameters -------------------------------------------------------------------
[ -f "$PARAMS" ] || die "no $PARAMS: the Write config task did not run"
grep -q '\${' "$PARAMS" && die "params.env still has unsubstituted tokens: $(grep '\${' "$PARAMS")"
VARS="KV_OFFLOADING_SIZE KV_CACHE_DTYPE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE"
for v in $VARS; do
  grep -q "^$v=" "$PARAMS" || die "$v has no line in params.env (empty = vLLM's default, absent = broken template)"
  declare "$v="
done
# shellcheck disable=SC1090
source "$PARAMS"
t "params.env as rendered: $(grep -E '^[A-Z_]+=' "$PARAMS" | tr '\n' ' ')"

# Akamas may render a real in scientific notation (study 35) or, for tiny values, without its
# exponent (study 46: 2.2e-15 rendered "2.19"); the domain check below refuses the latter.
if [[ "$GPU_MEMORY_UTILIZATION" =~ ^[0-9]+(\.[0-9]+)?[eE][-+]?[0-9]+$ ]]; then
  LC_ALL=C printf -v GPU_MEMORY_UTILIZATION '%.6f' "$GPU_MEMORY_UTILIZATION"
fi
# Integers may come as "12.0".
for v in KV_OFFLOADING_SIZE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE; do
  [[ "${!v}" =~ ^([0-9]+)\.0+$ ]] && declare "$v=${BASH_REMATCH[1]}"
done
int() { [ -z "${!1}" ] || [[ "${!1}" =~ ^[0-9]+$ ]] || die "$1='${!1}' is not an integer"; }
for v in KV_OFFLOADING_SIZE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS BLOCK_SIZE; do int $v; done
[ -z "$KV_OFFLOADING_SIZE" ] || [ "$KV_OFFLOADING_SIZE" -le $OFFLOAD_MAX ] \
  || die "kv_offloading_size $KV_OFFLOADING_SIZE GiB is above $OFFLOAD_MAX (node RAM)"
[ -z "$KV_CACHE_DTYPE" ] || [[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8)$ ]] || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto or fp8"
[ -z "$GPU_MEMORY_UTILIZATION" ] || awk -v g="$GPU_MEMORY_UTILIZATION" 'BEGIN { exit !(g ~ /^[0-9.]+$/ && g >= 0.5 && g <= 0.98) }' \
  || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not in [0.5, 0.98]"
[ -z "$BLOCK_SIZE" ] || [[ "$BLOCK_SIZE" =~ ^(16|32|64|128)$ ]] || die "block_size '$BLOCK_SIZE' is not 16, 32, 64 or 128"
if [ -n "$MAX_NUM_SEQS" ] && [ -n "$MAX_NUM_BATCHED_TOKENS" ] && [ "$MAX_NUM_BATCHED_TOKENS" -lt "$MAX_NUM_SEQS" ]; then
  die "max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS < max_num_seqs $MAX_NUM_SEQS (vLLM refuses it)"
fi

# Empty = not rendered = no flag (vLLM's default). Offload 0 = no offload flags.
ARGS=()
OFF=${KV_OFFLOADING_SIZE:-0}
if [ "$OFF" -gt 0 ]; then ARGS+=("--kv-offloading-size=$OFF" "--kv-offloading-backend=native"); fi
[ -z "$KV_CACHE_DTYPE" ] || ARGS+=("--kv-cache-dtype=$KV_CACHE_DTYPE")
[ -z "$GPU_MEMORY_UTILIZATION" ] || ARGS+=("--gpu-memory-utilization=$GPU_MEMORY_UTILIZATION")
[ -z "$MAX_NUM_SEQS" ] || ARGS+=("--max-num-seqs=$MAX_NUM_SEQS")
[ -z "$MAX_NUM_BATCHED_TOKENS" ] || ARGS+=("--max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS")
[ -z "$BLOCK_SIZE" ] || ARGS+=("--block-size=$BLOCK_SIZE")
if [ "$OFF" -gt 0 ]; then SHM=$((OFF + 1)); else SHM=2; fi
MEM=$((OFF + 20))
t "vLLM flags: ${ARGS[*]:-(none: vLLM defaults)}; shm ${SHM}Gi, memory ${MEM}Gi"
if [ "${RENDER_ONLY:-0}" = 1 ]; then
  [ ${#ARGS[@]} -eq 0 ] || printf '%s\n' "${ARGS[@]}"
  echo "shm=$SHM mem=$MEM"
  exit 0
fi

# --- 1. Node up ------------------------------------------------------------------------
GPU_NODE=$(kubectl get nodes -l node-role=$NODE_ROLE -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[ -n "$GPU_NODE" ] || die "no node with node-role=$NODE_ROLE: infra/eks/gpu-nodegroup.sh --up --always-on" 3

# --- 2. Render and apply ---------------------------------------------------------------
INDENT=$(grep -m1 '@TUNED_ARGS@' "$HERE/01-vllm_template.yaml" | sed 's/@TUNED_ARGS@.*//')
[ -n "$INDENT" ] || die "01-vllm_template.yaml has no @TUNED_ARGS@ line"
TMP=$(mktemp)
[ ${#ARGS[@]} -eq 0 ] || for a in "${ARGS[@]}"; do printf '%s- "%s"\n' "$INDENT" "$a"; done > "$TMP"
sed -e "s/@SHM_GI@/$SHM/g" -e "s/@MEM_GI@/$MEM/g" -e "s/@APPLIED_AT@/$(date -u +%Y%m%dT%H%M%SZ)/g" \
    -e "/@TUNED_ARGS@/r $TMP" -e '/@TUNED_ARGS@/d' \
    "$HERE/01-vllm_template.yaml" > "$K8S/01-vllm.yaml"
rm -f "$TMP"
grep -q '@[A-Z_]*@' "$K8S/01-vllm.yaml" && die "01-vllm.yaml still has a token"
kubectl apply -f "$HERE/monitoring/servicemonitor.yaml" >/dev/null
# An evicted pod (e.g. /dev/shm over its sizeLimit, node memory pressure) stays as Failed with the
# app=vllm label and would block the readiness check below forever (final review 2026-11-05).
kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Failed -o wide 2>/dev/null | grep -q . && {
  echo "--- Failed vLLM pods left from a previous experiment (deleted now) ---"
  kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Failed \
    -o custom-columns=POD:.metadata.name,REASON:.status.reason,MESSAGE:.status.message
  kubectl -n $NS delete pods -l app=vllm --field-selector=status.phase=Failed --wait=false >/dev/null
}
APPLY_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
kubectl apply -f "$K8S/01-vllm.yaml" || die "kubectl apply failed: refusing to benchmark a stale configuration" 3
kubectl apply -f "$HERE/02-aiperf.yaml" >/dev/null || die "kubectl apply of the AIPerf pod failed" 3

# --- 3. Wait for the new vLLM pod, fail fast on a crash loop; print the logs either way --
set +e
GEN=$(kubectl -n $NS get deploy vllm -o jsonpath='{.metadata.generation}')
EXIT=1; T0=$SECONDS
# 2400 s: cold node (10 GB image pull + 9 GB model download), ~8 min in the spike.
while [ $((SECONDS - T0)) -lt 2400 ]; do
  OBS=$(kubectl -n $NS get deploy vllm -o jsonpath='{.status.observedGeneration}')
  RDY=$(kubectl -n $NS get deploy vllm -o jsonpath='{.status.readyReplicas}')
  UPD=$(kubectl -n $NS get deploy vllm -o jsonpath='{.status.updatedReplicas}')
  if [ "$OBS" = "$GEN" ] && [ "${RDY:-0}" = 1 ] && [ "${UPD:-0}" = 1 ] \
     && [ "$(kubectl -n $NS get pods -l app=vllm --field-selector=status.phase!=Failed -o name | wc -l | tr -d ' ')" = 1 ]; then EXIT=0; break; fi
  # The new pod evicted during startup: no container restart, so the check below would miss it.
  if kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Failed -o jsonpath='{.items[*].metadata.creationTimestamp}' 2>/dev/null \
       | tr ' ' '\n' | awk -v t0="$APPLY_TS" '$0 >= t0 { f = 1 } END { exit !f }'; then
    echo "error: the new vLLM pod failed during startup (evicted?):" >&2
    kubectl -n $NS get pods -l app=vllm -o custom-columns=POD:.metadata.name,PHASE:.status.phase,REASON:.status.reason,MESSAGE:.status.message >&2
    EXIT=4; break
  fi
  R=$(kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{"\n"}{end}' | sort -n | tail -1)
  if [ "${R:-0}" -ge 2 ]; then echo "error: vLLM restarted $R times during startup: crash loop" >&2; EXIT=4; break; fi
  sleep 15
done
[ $EXIT = 1 ] && { echo "error: vLLM not Ready after 2400 s" >&2; EXIT=4; }
t "vLLM rollout exit=$EXIT after $((SECONDS - T0)) s"
POD=$(kubectl -n $NS get pods -l app=vllm --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
echo "--- $POD: startup summary ---"
kubectl -n $NS logs "$POD" --tail=-1 2>/dev/null | grep -E 'non-default args|Model loading took|Available KV cache memory|GPU KV cache size|offload|Error|Traceback' || true
echo "--- $POD: full log ---"
kubectl -n $NS logs "$POD" --tail=-1 2>/dev/null || true
if [ $EXIT -ne 0 ]; then
  echo "--- $POD: previous container ---"
  kubectl -n $NS logs "$POD" --tail=-1 --previous 2>/dev/null || true
  kubectl -n $NS describe pod "$POD" | sed -n '/^Events/,$p'
fi
set -e
[ $EXIT -eq 0 ] || exit $EXIT

# --- 4. AIPerf pod ready (pip install, ~1 min the first time) ---------------------------
kubectl -n $NS wait --for=condition=Ready pod/aiperf --timeout=600s \
  || { kubectl -n $NS logs aiperf --tail=50 || true; die "AIPerf pod not Ready after 600 s" 6; }
t "done: vLLM $POD and AIPerf ready on $GPU_NODE"
