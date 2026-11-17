#!/bin/bash
# Apply-config step for 52-g7e-glm53-coding-sessions-within-slo. Runs on the akamas-41 toolbox via
# the Akamas workflow's Executor task, after FileConfigurator has rendered params.env.
#
# One experiment = one vLLM configuration of GLM-5.3-Flash (FP8, TP4) on the g7e.24xlarge spot
# (node group llm-serving-g7e-24xl-spot). The vLLM container keeps the target deployment's shape:
# `vllm serve /mnt/models ... ${VLLM_ADDITIONAL_ARGS}`, where VLLM_ADDITIONAL_ARGS is the target's
# fixed arguments plus the tuned flags rendered here. The Deployment is replaced (Recreate) on every
# experiment; the AIPerf clients are kept. /dev/shm and the pod memory follow the CPU KV offload size.
#
# Fail-fast by design: a half-applied configuration would benchmark the previous experiment's setup
# under the current one's name. Exit codes: 2 bad parameters, 3 node missing / apply failed, 4 vLLM
# failed to start (the reason line from the log is printed), 6 AIPerf clients not Ready,
# 8 the running pod does not have the rendered configuration (study 49), 9 the node was lost during
# the start and none came back within the start budget (spot).
# RENDER_ONLY=1 prints the VLLM_ADDITIONAL_ARGS value on one line and "shm=<Gi> mem=<Gi>", then
# exits: local tests (test_apply_config.sh).
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/study52/52-g7e-glm53-coding-sessions-within-slo}   # overridable for manual tests
NS=glm53-coding
K8S=$STUDY_DIR/k8s
PARAMS=$K8S/params.env
HERE=$(cd "$(dirname "$0")" && pwd)   # manifests live next to the script
OFFLOAD_MAX=512   # GiB: the pack's maximum; the node has 1 TiB of RAM
MEM_BASE=96       # GiB of pod memory without offload (vLLM workers, loader buffers)
NODE_WAIT=1800    # s: a reclaimed spot node is replaced by its ASG
START_WAIT=4800   # s: first start on an empty node = 328 GB download + JIT warmup + weight load
MOE_ALLOWED="auto triton cutlass flashinfer_cutlass deep_gemm"   # narrowed by the smoke (k8s/smoke.sh)
# The target deployment's arguments that are not tuned, plus --language-model-only (text-only coding).
FIXED_ARGS="--served-model-name zai-org/GLM-5.3-Flash --kv-cache-dtype fp8 --max-model-len 262144 --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --tensor-parallel-size 4 --language-model-only"
t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }
# shellcheck disable=SC1091
source "$HERE/lib_apply.sh"

# --- 0. Parameters -------------------------------------------------------------------
[ -f "$PARAMS" ] || die "no $PARAMS: the Write config task did not run"
grep -q '\${' "$PARAMS" && die "params.env still has unsubstituted tokens: $(grep '\${' "$PARAMS")"
VARS="KV_OFFLOADING_SIZE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS SPEC_TOKENS DECODE_CONTEXT_PARALLEL_SIZE MOE_BACKEND MAX_CUDAGRAPH_CAPTURE_SIZE ASYNC_SCHEDULING PERFORMANCE_MODE OPTIMIZATION_LEVEL"
for v in $VARS; do
  grep -q "^$v=" "$PARAMS" || die "$v has no line in params.env (empty = vLLM's default, absent = broken template)"
  declare "$v="
done
# shellcheck disable=SC1090
source "$PARAMS"
t "params.env as rendered: $(grep -E '^[A-Z_]+=' "$PARAMS" | tr '\n' ' ')"
# Akamas may render a real in scientific notation (study 35) or, for tiny values, without its
# exponent (study 46); the domain check below refuses the latter.
if [[ "$GPU_MEMORY_UTILIZATION" =~ ^[0-9]+(\.[0-9]+)?[eE][-+]?[0-9]+$ ]]; then
  LC_ALL=C printf -v GPU_MEMORY_UTILIZATION '%.6f' "$GPU_MEMORY_UTILIZATION"
fi
INTS="KV_OFFLOADING_SIZE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS SPEC_TOKENS DECODE_CONTEXT_PARALLEL_SIZE MAX_CUDAGRAPH_CAPTURE_SIZE OPTIMIZATION_LEVEL"
for v in $INTS; do   # integers may come as "12.0"
  [[ "${!v}" =~ ^([0-9]+)\.0+$ ]] && declare "$v=${BASH_REMATCH[1]}"
  [ -z "${!v}" ] || [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"
done
[ -z "$KV_OFFLOADING_SIZE" ] || [ "$KV_OFFLOADING_SIZE" -le $OFFLOAD_MAX ] \
  || die "kv_offloading_size $KV_OFFLOADING_SIZE GiB is above $OFFLOAD_MAX"
[ -z "$GPU_MEMORY_UTILIZATION" ] || awk -v g="$GPU_MEMORY_UTILIZATION" 'BEGIN { exit !(g ~ /^[0-9.]+$/ && g >= 0.5 && g <= 1.0) }' \
  || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not in [0.5, 1.0]"
[ -z "$SPEC_TOKENS" ] || [ "$SPEC_TOKENS" -le 5 ] || die "spec_tokens $SPEC_TOKENS is above 5 (the design's domain)"
[ -z "$DECODE_CONTEXT_PARALLEL_SIZE" ] || [[ "$DECODE_CONTEXT_PARALLEL_SIZE" =~ ^(1|2|4)$ ]] \
  || die "decode_context_parallel_size '$DECODE_CONTEXT_PARALLEL_SIZE' is not 1, 2 or 4 (TP4)"
[ -z "$MOE_BACKEND" ] || [[ " $MOE_ALLOWED " == *" $MOE_BACKEND "* ]] || die "moe_backend '$MOE_BACKEND' is not one of: $MOE_ALLOWED"
[ -z "$ASYNC_SCHEDULING" ] || [[ "$ASYNC_SCHEDULING" =~ ^(true|false)$ ]] || die "async_scheduling '$ASYNC_SCHEDULING' is not true or false"
[ -z "$PERFORMANCE_MODE" ] || [[ "$PERFORMANCE_MODE" =~ ^(balanced|interactivity|throughput)$ ]] \
  || die "performance_mode '$PERFORMANCE_MODE' is not balanced, interactivity or throughput"
[ -z "$OPTIMIZATION_LEVEL" ] || [ "$OPTIMIZATION_LEVEL" -le 3 ] || die "optimization_level $OPTIMIZATION_LEVEL is above 3"
if [ -n "$MAX_NUM_SEQS" ] && [ -n "$MAX_NUM_BATCHED_TOKENS" ] && [ "$MAX_NUM_BATCHED_TOKENS" -lt "$MAX_NUM_SEQS" ]; then
  die "max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS < max_num_seqs $MAX_NUM_SEQS (vLLM refuses it)"
fi

# Empty = not rendered = no flag (vLLM's default). Offload 0 and MTP 0 = no flag. The target's
# style: "--flag value", space-separated, one line.
TUNED=""
add() { TUNED="$TUNED $*"; }
OFF=${KV_OFFLOADING_SIZE:-0}
[ "$OFF" -gt 0 ] && add --kv-offloading-size "$OFF" --kv-offloading-backend native
[ -z "$GPU_MEMORY_UTILIZATION" ] || add --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
[ -z "$MAX_NUM_SEQS" ] || add --max-num-seqs "$MAX_NUM_SEQS"
[ -z "$MAX_NUM_BATCHED_TOKENS" ] || add --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
if [ "${SPEC_TOKENS:-0}" -gt 0 ]; then
  add --speculative-config.method mtp --speculative-config.num_speculative_tokens "$SPEC_TOKENS"
fi
[ -z "$DECODE_CONTEXT_PARALLEL_SIZE" ] || add --decode-context-parallel-size "$DECODE_CONTEXT_PARALLEL_SIZE"
[ -z "$MOE_BACKEND" ] || [ "$MOE_BACKEND" = auto ] || add --moe-backend "$MOE_BACKEND"
[ -z "$MAX_CUDAGRAPH_CAPTURE_SIZE" ] || add --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
case "$ASYNC_SCHEDULING" in true) add --async-scheduling ;; false) add --no-async-scheduling ;; esac
[ -z "$PERFORMANCE_MODE" ] || add --performance-mode "$PERFORMANCE_MODE"
[ -z "$OPTIMIZATION_LEVEL" ] || add --optimization-level "$OPTIMIZATION_LEVEL"
ADDITIONAL_ARGS="$FIXED_ARGS$TUNED"
if [ "$OFF" -gt 0 ]; then SHM=$((OFF + 1)); else SHM=2; fi
MEM=$((OFF + MEM_BASE))
t "VLLM_ADDITIONAL_ARGS: $ADDITIONAL_ARGS; shm ${SHM}Gi, memory ${MEM}Gi"
if [ "${RENDER_ONLY:-0}" = 1 ]; then
  echo "$ADDITIONAL_ARGS"
  echo "shm=$SHM mem=$MEM"
  exit 0
fi

# --- 1. Node up (a reclaimed spot node is replaced by the ASG) --------------------------
wait_for_node $NODE_WAIT || die "no Ready node with node-role=$NODE_ROLE after ${NODE_WAIT}s: infra/eks/nodegroups.sh up" 3
GPU_NODE=$(kubectl get nodes -l node-role="$NODE_ROLE" -o jsonpath='{.items[0].metadata.name}')

# --- 2. Render and apply ---------------------------------------------------------------
sed -e "s|@ADDITIONAL_ARGS@|$ADDITIONAL_ARGS|g" -e "s/@SHM_GI@/$SHM/g" -e "s/@MEM_GI@/$MEM/g" \
    -e "s/@APPLIED_AT@/$(date -u +%Y%m%dT%H%M%SZ)/g" "$HERE/01-vllm_template.yaml" > "$K8S/01-vllm.yaml"
grep -q '@[A-Z_]*@' "$K8S/01-vllm.yaml" && die "01-vllm.yaml still has a token"
kubectl apply -f "$HERE/00-base.yaml" >/dev/null
kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Failed -o name 2>/dev/null | grep -q . && {
  echo "--- Failed vLLM pods left from a previous experiment (deleted now) ---"
  kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Failed \
    -o custom-columns=POD:.metadata.name,REASON:.status.reason,MESSAGE:.status.message
  kubectl -n $NS delete pods -l app=vllm --field-selector=status.phase=Failed --wait=false >/dev/null
}
APPLY_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
kubectl apply -f "$K8S/01-vllm.yaml" || die "kubectl apply failed: refusing to benchmark a stale configuration" 3
kubectl apply -f "$HERE/02-aiperf-clients.yaml" >/dev/null || die "kubectl apply of the AIPerf clients failed" 3

# --- 3. Wait for the new vLLM pod, fail fast on a crash loop; print the logs either way --
set +e
GEN=$(kubectl -n $NS get deploy vllm -o jsonpath='{.metadata.generation}')
EXIT=1; T0=$SECONDS
while [ $((SECONDS - T0)) -lt $START_WAIT ]; do
  OBS=$(kubectl -n $NS get deploy vllm -o jsonpath='{.status.observedGeneration}')
  RDY=$(kubectl -n $NS get deploy vllm -o jsonpath='{.status.readyReplicas}')
  UPD=$(kubectl -n $NS get deploy vllm -o jsonpath='{.status.updatedReplicas}')
  if [ "$OBS" = "$GEN" ] && [ "${RDY:-0}" = 1 ] && [ "${UPD:-0}" = 1 ] \
     && [ "$(kubectl -n $NS get pods -l app=vllm --field-selector=status.phase!=Failed -o name | wc -l | tr -d ' ')" = 1 ]; then EXIT=0; break; fi
  FAILED=$(kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Failed \
    -o jsonpath='{range .items[*]}{.metadata.creationTimestamp} {.metadata.name}{"\n"}{end}' 2>/dev/null \
    | awk -v t0="$APPLY_TS" '$1 >= t0 { print $2; exit }')
  if [ -n "$FAILED" ]; then
    kubectl -n $NS get pods -l app=vllm -o custom-columns=POD:.metadata.name,PHASE:.status.phase,REASON:.status.reason,MESSAGE:.status.message >&2
    # A spot reclaim during the start (up to 80 min) takes the pod down with its node: wait for the
    # ASG's replacement and let the Deployment start again, instead of scoring a configuration that
    # never had a node (final review I5). Exit 9 if the time runs out.
    if node_lost "$FAILED"; then
      echo "the vLLM pod $FAILED went down with its node (spot reclaim?): deleting it, waiting for a new node" >&2
      kubectl -n $NS delete pod "$FAILED" --wait=false >/dev/null 2>&1
      LEFT=$((START_WAIT - (SECONDS - T0)))
      [ $LEFT -gt 0 ] && wait_for_node $LEFT || { echo "error: no node came back within the start budget" >&2; EXIT=9; break; }
      continue
    fi
    echo "error: the new vLLM pod failed during startup (evicted?)" >&2
    EXIT=4; break
  fi
  # Only this apply's pod: the previous experiment's pod may still be listed (smoke 2026-11-17).
  read -r CP R <<< "$(newest_pod_since "$APPLY_TS")"
  # The first crash whose log names a start-up failure that repeats (Mamba blocks, no KV block, OOM)
  # ends the wait at once instead of after three starts (final review I4).
  if [ -n "$CP" ] && [ "${R:-0}" -ge 1 ]; then
    if WHY=$(crash_reason "$CP"); then echo "error: vLLM failed to start: $WHY" >&2; EXIT=4; break; fi
  fi
  if [ "${R:-0}" -ge 2 ]; then echo "error: vLLM restarted $R times during startup: crash loop" >&2; EXIT=4; break; fi
  sleep 15
done
[ $EXIT = 1 ] && { echo "error: vLLM not Ready after $START_WAIT s" >&2; EXIT=4; }
t "vLLM rollout exit=$EXIT after $((SECONDS - T0)) s"
read -r POD _ <<< "$(newest_pod_since "$APPLY_TS")"
[ -n "$POD" ] || POD=$(kubectl -n $NS get pods -l app=vllm --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
LOG=$(mktemp)
kubectl -n $NS logs "$POD" --tail=-1 > "$LOG" 2>/dev/null
[ $EXIT -eq 0 ] || kubectl -n $NS logs "$POD" --tail=-1 --previous >> "$LOG" 2>/dev/null
echo "--- $POD: startup summary ---"
grep -E 'non-default args|Model loading took|Using .* backend|Available KV cache memory|Memory after warmup|GPU KV cache size|Maximum concurrency|JIT kernel warmup finished|Error|Traceback' "$LOG" || true
if [ $EXIT -ne 0 ]; then
  echo "--- $POD: why it did not start ---"
  startup_failure_reason "$LOG" || true
  echo "--- $POD: log tail ---"
  tail -80 "$LOG"
  kubectl -n $NS describe pod "$POD" | sed -n '/^Events/,$p'
fi
rm -f "$LOG"
set -e
[ $EXIT -eq 0 ] || exit $EXIT

# --- 4. The pod runs what was rendered (study 49: a template bug ran 23 experiments at defaults) --
REAL=$(kubectl -n $NS get pod "$POD" -o jsonpath='{.spec.containers[0].env[?(@.name=="VLLM_ADDITIONAL_ARGS")].value}')
[ "$REAL" = "$ADDITIONAL_ARGS" ] || die "the pod's VLLM_ADDITIONAL_ARGS differs from the rendered one: '$REAL'" 8
CACHE=$(kubectl -n $NS exec "$POD" -- python3 -c '
import urllib.request
for l in urllib.request.urlopen("http://localhost:8000/metrics").read().decode().splitlines():
    if l.startswith("vllm:cache_config_info"): print(l)' 2>/dev/null | head -1)
echo "cache_config_info: $CACHE"
[ -n "$CACHE" ] || die "vllm:cache_config_info not found on the pod's /metrics" 8
if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
  gmu_matches "$CACHE" "$GPU_MEMORY_UTILIZATION" \
    || die "cache_config_info does not report gpu_memory_utilization $GPU_MEMORY_UTILIZATION" 8
fi
echo "$CACHE" | grep -Eq 'num_gpu_blocks="[1-9][0-9]*"' || die "cache_config_info reports no GPU KV block" 8

# --- 5. AIPerf clients ready (StatefulSet aiperf-client on the m8a-2xlarge node; pip install,
#        ~1 min the first time) ------------------------------------------------------------
for c in aiperf-client-0 aiperf-client-1; do
  kubectl -n $NS wait --for=condition=Ready "pod/$c" --timeout=600s \
    || { kubectl -n $NS logs "$c" --tail=50 || true; die "AIPerf client $c not Ready after 600 s" 6; }
done
t "done: vLLM $POD on $GPU_NODE and both AIPerf clients ready"
