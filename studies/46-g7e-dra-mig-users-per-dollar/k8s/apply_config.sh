#!/bin/bash
# Apply-config step for 46-g7e-dra-mig-users-per-dollar. Runs on the toolbox host via the
# Akamas workflow's Executor task, after FileConfigurator has rendered params.env.
#
# One experiment = one way of splitting the GPU (gpu0.mig_profile) + one vLLM configuration,
# the same on every replica. The GPU is always fully used, one replica per slice:
#   none     1 replica, ResourceClaimTemplate gpu-full   (whole GPU, no MIG)
#   2g.48gb  2 replicas, ResourceClaimTemplate mig-2g48gb
#   1g.24gb  4 replicas, ResourceClaimTemplate mig-1g24gb
# The DRA driver (dra-driver-nvidia-gpu 0.5.1, DynamicMIG) creates the MIG partition when a
# pod's claim is allocated and destroys it when the claim goes. No device plugin, no manual
# nvidia-smi mig (Phase 0, study README).
#
# Fail-fast by design: a half-applied configuration would benchmark the previous
# experiment's setup under the current one's name.
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/study46/46-g7e-dra-mig-users-per-dollar}   # overridable for manual tests
NS=dra-mig
NODE_ROLE=llm-serving-g7e-4xl
LOAD_NODE_ROLE=m8a-2xlarge
K8S=$STUDY_DIR/k8s
PARAMS=$K8S/params.env

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# --- 0. Parameters -------------------------------------------------------------------
grep -q '\${' "$PARAMS" && die "params.env still has unsubstituted tokens: $(grep '\${' "$PARAMS")"
MAX_NUM_BATCHED_TOKENS="" KV_CACHE_DTYPE="" PERFORMANCE_MODE="" OPTIMIZATION_LEVEL="" ASYNC_SCHEDULING=""
MAX_CUDAGRAPH_CAPTURE_SIZE="" BLOCK_SIZE="" WATERMARK="" SCHEDULER_RESERVE_FULL_ISL="" STREAM_INTERVAL=""
# shellcheck disable=SC1090
source "$PARAMS"
for v in MIG_PROFILE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS; do
  [ -n "${!v:-}" ] || die "$v is empty in params.env (always rendered: never in doNotRenderParameters)"
done
for v in MAX_NUM_BATCHED_TOKENS KV_CACHE_DTYPE PERFORMANCE_MODE OPTIMIZATION_LEVEL ASYNC_SCHEDULING \
         MAX_CUDAGRAPH_CAPTURE_SIZE BLOCK_SIZE WATERMARK SCHEDULER_RESERVE_FULL_ISL STREAM_INTERVAL; do
  grep -q "^$v=" "$PARAMS" || die "$v has no line in params.env (empty = vLLM's default, absent = broken template)"
done
case $MIG_PROFILE in
  none)    CLAIM=gpu-full;   REPLICAS=1 ;;
  2g.48gb) CLAIM=mig-2g48gb; REPLICAS=2 ;;
  1g.24gb) CLAIM=mig-1g24gb; REPLICAS=4 ;;
  *) die "mig_profile '$MIG_PROFILE' is not one of none / 2g.48gb / 1g.24gb" ;;
esac
# Akamas writes a tiny real in scientific notation (study 35: watermark 7.57e-17): make it decimal.
t "params.env as rendered: $(grep -E '^[A-Z_]+=' "$PARAMS" | tr '\n' ' ')"
for v in GPU_MEMORY_UTILIZATION WATERMARK; do
  if [[ "${!v}" =~ ^[0-9]+(\.[0-9]+)?[eE][-+]?[0-9]+$ ]]; then LC_ALL=C printf -v "$v" '%.6f' "${!v}"; fi
done
# Experiment 10 (2026-11-04): Akamas chose watermark 2.199e-15 but params.env got "2.19", the
# exponent dropped. The study domain is [0, 0.10], so a watermark above 0.10 can only be a tiny
# value that lost its exponent: use 0 (what the optimizer meant) instead of failing the trial.
if [[ "$WATERMARK" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v w="$WATERMARK" 'BEGIN { exit !(w > 0.10) }'; then
  t "WARNING: watermark '$WATERMARK' is outside the study domain [0, 0.10]: a tiny value rendered without its exponent, using 0"
  WATERMARK=0
fi
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not a fraction"
for v in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_CUDAGRAPH_CAPTURE_SIZE BLOCK_SIZE OPTIMIZATION_LEVEL STREAM_INTERVAL; do
  [ -z "${!v}" ] || [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"
done
# vLLM 0.29.0 config/scheduler.py raises ValueError when max_num_batched_tokens < max_num_seqs.
[ -z "$MAX_NUM_BATCHED_TOKENS" ] || [ "$MAX_NUM_BATCHED_TOKENS" -ge "$MAX_NUM_SEQS" ] || die "max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS < max_num_seqs $MAX_NUM_SEQS"
[ -z "$OPTIMIZATION_LEVEL" ] || [ "$OPTIMIZATION_LEVEL" -le 3 ] || die "optimization_level '$OPTIMIZATION_LEVEL' is not 0-3"
[ -z "$BLOCK_SIZE" ] || { [ "$BLOCK_SIZE" -ge 16 ] && [ $((BLOCK_SIZE % 16)) = 0 ]; } || die "block_size '$BLOCK_SIZE' is not a multiple of 16"
[ -z "$KV_CACHE_DTYPE" ] || [[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8)$ ]] || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto or fp8"
[ -z "$PERFORMANCE_MODE" ] || [[ "$PERFORMANCE_MODE" =~ ^(balanced|interactivity|throughput)$ ]] || die "performance_mode '$PERFORMANCE_MODE' is not balanced, interactivity or throughput"
for v in ASYNC_SCHEDULING SCHEDULER_RESERVE_FULL_ISL; do
  [ -z "${!v}" ] || [[ "${!v}" =~ ^(true|false)$ ]] || die "$v='${!v}' is not true or false"
done
[ -z "$WATERMARK" ] || [[ "$WATERMARK" =~ ^0(\.[0-9]+)?$ ]] || die "watermark '$WATERMARK' is not a fraction in [0, 1)"
[ -z "$STREAM_INTERVAL" ] || [ "$STREAM_INTERVAL" -ge 1 ] || die "stream_interval '$STREAM_INTERVAL' is below 1"

flag() { [ "$2" = true ] && echo "--$1" || echo "--no-$1"; }
ARGS=("--gpu-memory-utilization=$GPU_MEMORY_UTILIZATION" "--max-num-seqs=$MAX_NUM_SEQS")
# Empty = not rendered = no flag (vLLM's default).
[ -z "$MAX_NUM_BATCHED_TOKENS" ] || ARGS+=("--max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS")
[ -z "$KV_CACHE_DTYPE" ] || ARGS+=("--kv-cache-dtype=$KV_CACHE_DTYPE")
[ -z "$PERFORMANCE_MODE" ] || ARGS+=("--performance-mode=$PERFORMANCE_MODE")
[ -z "$OPTIMIZATION_LEVEL" ] || ARGS+=("--optimization-level=$OPTIMIZATION_LEVEL")
[ -z "$ASYNC_SCHEDULING" ] || ARGS+=("$(flag async-scheduling "$ASYNC_SCHEDULING")")
[ -z "$MAX_CUDAGRAPH_CAPTURE_SIZE" ] || ARGS+=("--max-cudagraph-capture-size=$MAX_CUDAGRAPH_CAPTURE_SIZE")
[ -z "$BLOCK_SIZE" ] || ARGS+=("--block-size=$BLOCK_SIZE")
[ -z "$WATERMARK" ] || ARGS+=("--watermark=$WATERMARK")
[ -z "$SCHEDULER_RESERVE_FULL_ISL" ] || ARGS+=("$(flag scheduler-reserve-full-isl "$SCHEDULER_RESERVE_FULL_ISL")")
[ -z "$STREAM_INTERVAL" ] || ARGS+=("--stream-interval=$STREAM_INTERVAL")
t "mig_profile=$MIG_PROFILE -> $REPLICAS replica(s), claim template $CLAIM; vLLM flags: ${ARGS[*]}"
echo "$REPLICAS" > "$K8S/replicas.current"
[ "${RENDER_ONLY:-0}" = 1 ] && { printf '%s\n' "${ARGS[@]}"; exit 0; }   # local test of the rendering

# --- 1. Nodes up ----------------------------------------------------------------------
# The study does not scale node groups: both nodes must be up before it starts (study README).
GPU_NODE=$(kubectl get nodes -l node-role=$NODE_ROLE -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[ -n "$GPU_NODE" ] || die "no node with node-role=$NODE_ROLE: scale llm-serving-g7e-4xl to 1 (infra/eks/gpu-nodegroup.sh --up --always-on)"
kubectl get nodes -l node-role=$LOAD_NODE_ROLE -o name | grep -q . || die "no node with node-role=$LOAD_NODE_ROLE: infra/eks/load-node.sh up"

# --- 2. Tear down the previous experiment (pods AND claims: the claims hold the slices) --
kubectl apply -f "$K8S/00-claims.yaml" >/dev/null
kubectl apply -f "$K8S/monitoring/servicemonitor.yaml" >/dev/null   # vLLM scrape (goal + constraints)
kubectl -n $NS delete statefulset vllm aiperf-client --ignore-not-found --wait=true
kubectl -n $NS wait --for=delete pod -l 'app in (vllm,aiperf-client)' --timeout=300s 2>/dev/null || true
T0=$SECONDS
while kubectl -n $NS get resourceclaims -o name 2>/dev/null | grep -q .; do
  [ $((SECONDS - T0)) -gt 300 ] && die "ResourceClaims still present after 300 s: $(kubectl -n $NS get resourceclaims -o name | tr '\n' ' ')"
  sleep 3
done
t "previous pods and claims gone"

# --- 3. Render and apply ---------------------------------------------------------------
read -r ACPU AMEM < <(kubectl get node "$GPU_NODE" -o jsonpath='{.status.allocatable.cpu} {.status.allocatable.memory}{"\n"}')
case $ACPU in *m) ACPU=${ACPU%m} ;; *) ACPU=$((ACPU * 1000)) ;; esac
AMEM_GI=$(( ${AMEM%Ki} / 1048576 ))
# Fair split of the node: allocatable minus 400m / 6 GiB for the DaemonSets, divided evenly.
CPU=$(( (ACPU - 400) / REPLICAS )); MEM=$(( (AMEM_GI - 6) / REPLICAS ))
t "per replica: cpu ${CPU}m, memory ${MEM}Gi (node allocatable ${ACPU}m / ${AMEM_GI}Gi)"
INDENT=$(grep -m1 '@TUNED_ARGS@' "$K8S/01-statefulset_template.yaml" | sed 's/@TUNED_ARGS@.*//')
[ -n "$INDENT" ] || die "01-statefulset_template.yaml has no @TUNED_ARGS@ line"
for a in "${ARGS[@]}"; do printf '%s- "%s"\n' "$INDENT" "$a"; done > "$K8S/tuned-args.tmp"
sed -e "s/@CLAIM@/$CLAIM/g" -e "s/@REPLICAS@/$REPLICAS/g" -e "s/@NODE_ROLE@/$NODE_ROLE/g" \
    -e "s/@CPU@/$CPU/g" -e "s/@MEM@/$MEM/g" \
    -e "/@TUNED_ARGS@/r $K8S/tuned-args.tmp" -e '/@TUNED_ARGS@/d' \
    "$K8S/01-statefulset_template.yaml" > "$K8S/01-statefulset.yaml"
rm -f "$K8S/tuned-args.tmp"
sed -e "s/@REPLICAS@/$REPLICAS/g" "$K8S/02-aiperf-clients_template.yaml" > "$K8S/02-aiperf-clients.yaml"
for f in 01-statefulset.yaml 02-aiperf-clients.yaml; do
  grep -q '@[A-Z_]*@' "$K8S/$f" && die "$f still has a token: $(grep -o '@[A-Z_]*@' "$K8S/$f" | sort -u | tr '\n' ' ')"
done
kubectl apply -f "$K8S/01-statefulset.yaml" || die "kubectl apply failed — refusing to benchmark a stale configuration" 3
kubectl apply -f "$K8S/02-aiperf-clients.yaml" || die "kubectl apply of the AIPerf clients failed" 3

# --- 4. Wait for vLLM, fail fast on a crash loop; print the logs either way -------------
set +e
ROLLOUT_EXIT=1
T0=$SECONDS
# 2400 s: cold node (10 GB image pull + 9 GB model, ~8 min for vllm-0) + 3 more sequential
# starts (~3 min each); Phase 0 measured 1063-1072 s for 4 replicas cold.
while [ $((SECONDS - T0)) -lt 2400 ]; do
  READY=$(kubectl -n $NS get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  if [ "${READY:-0}" = "$REPLICAS" ]; then ROLLOUT_EXIT=0; break; fi
  MAXR=$(kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort -n | tail -1)
  if [ "${MAXR:-0}" -ge 2 ]; then
    echo "error: a vLLM replica restarted ${MAXR} times during startup — crash loop, failing now" >&2
    ROLLOUT_EXIT=4; break
  fi
  # A claim that cannot be allocated (e.g. a slice left over) shows as a Pending pod.
  if [ $((SECONDS - T0)) -gt 300 ]; then
    UNSCHED=$(kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.metadata.name} {.status.conditions[?(@.type=="PodScheduled")].reason}{"\n"}{end}' 2>/dev/null | grep -c Unschedulable)
    if [ "${UNSCHED:-0}" -gt 0 ]; then
      echo "error: vLLM pod(s) unschedulable after 300 s:" >&2
      kubectl -n $NS describe pods -l app=vllm | grep -A6 '^Events' >&2
      ROLLOUT_EXIT=5; break
    fi
  fi
  sleep 15
done
[ $ROLLOUT_EXIT = 1 ] && echo "error: $REPLICAS replica(s) not Ready after 2400 s" >&2
t "vLLM rollout exit=$ROLLOUT_EXIT after $((SECONDS - T0)) s"
kubectl -n $NS get resourceclaims -o custom-columns=CLAIM:.metadata.name,DEVICE:.status.allocation.devices.results[*].device,POD:.status.reservedFor[*].name
for i in $(seq 0 $((REPLICAS - 1))); do
  echo "--- vllm-$i: startup summary ---"
  kubectl -n $NS logs vllm-$i --tail=-1 2>/dev/null | grep -E 'Model loading took|Available KV cache memory|GPU KV cache size|Error|Traceback' || true
done
for i in $(seq 0 $((REPLICAS - 1))); do
  echo "--- vllm-$i: full logs ---"
  kubectl -n $NS logs vllm-$i --all-containers --tail=-1 2>/dev/null || true
  if [ $ROLLOUT_EXIT -ne 0 ]; then
    echo "--- vllm-$i: previous container (if it crashed and restarted) ---"
    kubectl -n $NS logs vllm-$i --all-containers --tail=-1 --previous 2>/dev/null || true
  fi
done
set -e
[ $ROLLOUT_EXIT -eq 0 ] || exit $ROLLOUT_EXIT

# --- 5. dcgm-exporter re-reads the MIG layout only at start (Phase 0) -------------------
kubectl -n monitoring delete pod -l app.kubernetes.io/name=dcgm-exporter --field-selector spec.nodeName="$GPU_NODE" --wait=false >/dev/null || true
t "dcgm-exporter restarted on $GPU_NODE"

# --- 6. AIPerf clients ready (pip install + ShareGPT download, ~1 min) -------------------
if ! kubectl -n $NS rollout status statefulset/aiperf-client --timeout=600s; then
  for i in $(seq 0 $((REPLICAS - 1))); do kubectl -n $NS logs aiperf-client-$i --tail=-1 2>/dev/null || true; done
  die "AIPerf clients not ready after 600 s" 6
fi
t "done: $REPLICAS vLLM replica(s) and $REPLICAS AIPerf client(s) ready"
