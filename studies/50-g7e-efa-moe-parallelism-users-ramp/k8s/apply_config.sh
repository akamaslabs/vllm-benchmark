#!/bin/bash
# Apply-config step for 50-g7e-efa-moe-parallelism-users-ramp. Runs on the akamas-41 toolbox via
# the Akamas workflow's Executor task, after FileConfigurator has rendered params.env.
#
# One experiment = one layout of gpt-oss-120b over the two g7e.8xlarge nodes (lib_layout.sh) plus
# the vLLM flags, the same on both pods. The two StatefulSets (vllm-a on node a, the head; vllm-b
# on node b) are deleted and re-created at every experiment: the two pods form one distributed
# vLLM and must start together, the head first because the worker needs its pod IP. The AIPerf
# client pods are kept.
#
# Fail-fast by design: a half-applied configuration would benchmark the previous experiment's
# setup under the current one's name. Exit codes: 2 bad parameters or not a layout, 3 nodes
# missing / apply failed, 4 vLLM crash loop, not Ready or not answering, 6 AIPerf pods not Ready,
# 8 the running pods do not carry the configuration (args, cache_config_info, rank or expert
# parallelism lines), 9 NCCL is on the wrong transport.
#
# Environment (manual runs only, never set by the Akamas workflow):
#   DP_MODE=external|internal   load balancing of the two DP layouts (lib_layout.sh)
#   NCCL_ENV="NCCL_NET=Socket"  extra env of both pods: the TCP control runs; the transport
#                               check then expects Socket instead of Libfabric (EFA)
#   RENDER_ONLY=1               print the layout line and every flag ("a <flag>" / "b <flag>"),
#                               then exit: local tests (test_apply_config.sh)
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/study50/50-g7e-efa-moe-parallelism-users-ramp}   # overridable for manual tests
NS=moe-efa
NODE_ROLE=llm-serving-g7e-8xl-efa
K8S=$STUDY_DIR/k8s
PARAMS=$K8S/params.env
HERE=$(cd "$(dirname "$0")" && pwd)   # manifests live next to the script
DP_MODE=${DP_MODE:-external}
NCCL_ENV=${NCCL_ENV:-}
# shellcheck disable=SC1091
source "$HERE/lib_layout.sh"
# shellcheck disable=SC1091
source "$HERE/lib_render.sh"

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# --- 0. Parameters -------------------------------------------------------------------
[ -f "$PARAMS" ] || die "no $PARAMS: the Write config task did not run"
grep -q '\${' "$PARAMS" && die "params.env still has unsubstituted tokens: $(grep '\${' "$PARAMS")"
INTS="REPLICAS TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS"
VARS="$INTS ENABLE_EXPERT_PARALLEL GPU_MEMORY_UTILIZATION KV_CACHE_DTYPE MOE_BACKEND"
for v in $VARS; do
  grep -q "^$v=" "$PARAMS" || die "$v has no line in params.env (empty = default, absent = broken template)"
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
for v in $INTS; do
  [[ "${!v}" =~ ^([0-9]+)\.0+$ ]] && declare "$v=${BASH_REMATCH[1]}"
  [ -z "${!v}" ] || [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"
done
[ -z "$ENABLE_EXPERT_PARALLEL" ] || [[ "$ENABLE_EXPERT_PARALLEL" =~ ^(true|false)$ ]] \
  || die "enable_expert_parallel '$ENABLE_EXPERT_PARALLEL' is not true or false"
[ -z "$KV_CACHE_DTYPE" ] || [[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8|fp8_e4m3|fp8_e5m2)$ ]] \
  || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto, fp8, fp8_e4m3 or fp8_e5m2"
[ -z "$MOE_BACKEND" ] || [[ "$MOE_BACKEND" =~ ^[a-z0-9_]+$ ]] || die "moe_backend '$MOE_BACKEND' is not a backend name"
[ -z "$GPU_MEMORY_UTILIZATION" ] || awk -v g="$GPU_MEMORY_UTILIZATION" 'BEGIN { exit !(g ~ /^[0-9.]+$/ && g >= 0.5 && g <= 0.98) }' \
  || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not in [0.5, 0.98]"
if [ -n "$MAX_NUM_SEQS" ] && [ -n "$MAX_NUM_BATCHED_TOKENS" ] && [ "$MAX_NUM_BATCHED_TOKENS" -lt "$MAX_NUM_SEQS" ]; then
  die "max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS < max_num_seqs $MAX_NUM_SEQS (vLLM refuses it)"
fi
for kv in $NCCL_ENV; do
  [[ $kv =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_.,:/-]*$ ]] || die "NCCL_ENV word '$kv' is not NAME=value"
done

EP=${ENABLE_EXPERT_PARALLEL:-false}
resolve_layout "${REPLICAS:-1}" "${TENSOR_PARALLEL_SIZE:-1}" "${DATA_PARALLEL_SIZE:-1}" "${PIPELINE_PARALLEL_SIZE:-1}" "$EP" "$DP_MODE" \
  || die "the parameters do not make a layout"

# Empty = not rendered = no flag (vLLM's default). moe_backend auto is vLLM's default: no flag.
COMMON=()
[ -z "$GPU_MEMORY_UTILIZATION" ] || COMMON+=("--gpu-memory-utilization=$GPU_MEMORY_UTILIZATION")
[ -z "$MAX_NUM_SEQS" ] || COMMON+=("--max-num-seqs=$MAX_NUM_SEQS")
[ -z "$MAX_NUM_BATCHED_TOKENS" ] || COMMON+=("--max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS")
[ -z "$KV_CACHE_DTYPE" ] || COMMON+=("--kv-cache-dtype=$KV_CACHE_DTYPE")
if [ -n "$MOE_BACKEND" ] && [ "$MOE_BACKEND" != auto ]; then COMMON+=("--moe-backend=$MOE_BACKEND"); fi
[ ${#LAYOUT_ARGS[@]} -eq 0 ] || COMMON+=("${LAYOUT_ARGS[@]}")
# The head passes its own pod IP (Kubernetes expands $(POD_IP) from the container env), the worker
# the head's IP, known only once the head pod exists: HEAD_IP is replaced in section 2.
ARGS_A=(); ARGS_B=()
[ ${#COMMON[@]} -eq 0 ] || { ARGS_A=("${COMMON[@]}"); ARGS_B=("${COMMON[@]}"); }
if [ "$MULTINODE" = 1 ]; then
  ARGS_A+=(--nnodes=2 --node-rank=0 '--master-addr=$(POD_IP)')
  ARGS_B+=(--nnodes=2 --node-rank=1 --master-addr=HEAD_IP)
  [ "$WORKER_HEADLESS" = 0 ] || ARGS_B+=(--headless)
fi
# What the transport check expects: EFA (NCCL's OFI plugin logs "Using network Libfabric") unless
# the run forces TCP.
TRANSPORT=Libfabric
case " $NCCL_ENV " in *" NCCL_NET=Socket "*) TRANSPORT=Socket ;; esac
t "layout $LAYOUT (pods: $PODS; node b serving: $SERVING_B; NCCL check: $NCCL_CHECK, expecting $TRANSPORT; DP mode: $DP_MODE)"
t "flags a: ${ARGS_A[*]:-(none: vLLM defaults)}"
[ "$PODS" = a ] || t "flags b: ${ARGS_B[*]:-(none: vLLM defaults)}"
if [ "${RENDER_ONLY:-0}" = 1 ]; then
  echo "layout=$LAYOUT pods=${PODS// /,} serving_b=$SERVING_B nccl=$NCCL_CHECK dp_mode=$DP_MODE"
  [ ${#ARGS_A[@]} -eq 0 ] || printf 'a %s\n' "${ARGS_A[@]}"
  if [ "$PODS" != a ] && [ ${#ARGS_B[@]} -gt 0 ]; then printf 'b %s\n' "${ARGS_B[@]}"; fi
  [ -z "$NCCL_ENV" ] || echo "env $NCCL_ENV"
  exit 0
fi

# --- 1. Both nodes up (also for the one-pod layout: the telemetry reads both) ----------
for r in a b; do
  n=$(kubectl get nodes -l "node-role=$NODE_ROLE,study50/role=$r" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  [ -n "$n" ] || die "no node with node-role=$NODE_ROLE and study50/role=$r: infra/eks/nodegroups.sh up" 3
  declare "NODE_$r=$n"
done

# --- 2. Render and apply ---------------------------------------------------------------
kubectl apply -f "$HERE/00-base.yaml" >/dev/null || die "kubectl apply of the namespace and Services failed" 3
kubectl apply -f "$HERE/monitoring/servicemonitor.yaml" -f "$HERE/monitoring/efa-exporter.yaml" >/dev/null \
  || die "kubectl apply of the monitoring manifests failed" 3
kubectl apply -f "$HERE/03-aiperf-clients.yaml" >/dev/null || die "kubectl apply of the AIPerf clients failed" 3
# Both pods go away before the new ones come: the GPUs must be free and the distributed group
# cannot be joined by a pod of the previous experiment.
kubectl -n $NS delete statefulset vllm-a vllm-b --ignore-not-found >/dev/null
T0=$SECONDS; FORCED=0
while kubectl -n $NS get pods -l app=vllm -o name 2>/dev/null | grep -q .; do
  if [ $((SECONDS - T0)) -ge 180 ] && [ $FORCED = 0 ]; then
    t "vLLM pods still there after 180 s: force delete"
    kubectl -n $NS delete pods -l app=vllm --force --grace-period=0 >/dev/null 2>&1 || true; FORCED=1
  fi
  [ $((SECONDS - T0)) -lt 300 ] || die "the previous vLLM pods are still there after 300 s" 3
  sleep 3
done
APPLIED_AT=$(date -u +%Y%m%dT%H%M%SZ)
apply_pod() { # ROLE SERVING FLAG...
  local role=$1 serving=$2; shift 2
  render_vllm_manifest "$HERE/01-vllm_template.yaml" "$role" "$serving" "$LAYOUT" "$APPLIED_AT" "$NCCL_ENV" "$@" > "$K8S/01-vllm-$role.yaml" \
    || die "rendering of pod $role failed" 3
  check_rendered "$K8S/01-vllm-$role.yaml" "$@" || die "the rendered manifest of pod $role does not carry the experiment's flags" 3
  kubectl apply -f "$K8S/01-vllm-$role.yaml" || die "kubectl apply of pod $role failed: refusing to benchmark a stale configuration" 3
}
rm -f "$K8S/01-vllm-a.yaml" "$K8S/01-vllm-b.yaml"
if [ ${#ARGS_A[@]} -eq 0 ]; then apply_pod a true; else apply_pod a true "${ARGS_A[@]}"; fi
if [ "$PODS" != a ]; then
  if [ "$MULTINODE" = 1 ]; then
    HEAD_IP=""; T0=$SECONDS
    while [ -z "$HEAD_IP" ]; do
      [ $((SECONDS - T0)) -lt 300 ] || die "vllm-a-0 has no pod IP after 300 s" 3
      sleep 2
      HEAD_IP=$(kubectl -n $NS get pod vllm-a-0 -o jsonpath='{.status.podIP}' 2>/dev/null || true)
    done
    t "head pod IP $HEAD_IP"
    for i in "${!ARGS_B[@]}"; do [ "${ARGS_B[$i]}" != --master-addr=HEAD_IP ] || ARGS_B[$i]="--master-addr=$HEAD_IP"; done
  fi
  if [ ${#ARGS_B[@]} -eq 0 ]; then apply_pod b "$SERVING_B"; else apply_pod b "$SERVING_B" "${ARGS_B[@]}"; fi
fi

# --- 3. Wait for the pods, fail fast on a crash loop -----------------------------------
# A headless worker is Ready at once (its probes only look at the API server of a serving pod):
# what proves the layout is the head's /health, which answers only when every rank is up.
set +e
EXIT=1; T0=$SECONDS; NPODS=$(echo $PODS | wc -w | tr -d ' ')
# 2400 s: 65 GB of weights from the node's disk cache into one or two GPUs.
while [ $((SECONDS - T0)) -lt 2400 ]; do
  READY=0
  for r in $PODS; do
    [ "$(kubectl -n $NS get pod vllm-$r-0 -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" = True ] && READY=$((READY + 1))
  done
  if [ "$READY" = "$NPODS" ]; then EXIT=0; break; fi
  if kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Failed -o name 2>/dev/null | grep -q .; then
    echo "error: a vLLM pod failed during startup (evicted?):" >&2
    kubectl -n $NS get pods -l app=vllm -o custom-columns=POD:.metadata.name,PHASE:.status.phase,REASON:.status.reason,MESSAGE:.status.message >&2
    EXIT=4; break
  fi
  R=$(kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort -n | tail -1)
  if [ "${R:-0}" -ge 2 ]; then echo "error: a vLLM container restarted $R times during startup: crash loop" >&2; EXIT=4; break; fi
  sleep 15
done
[ $EXIT = 1 ] && { echo "error: vLLM not Ready after 2400 s" >&2; EXIT=4; }
t "vLLM rollout exit=$EXIT after $((SECONDS - T0)) s"
kubectl -n $NS get pods -l app=vllm -o wide

dump_logs() { # every pod: startup summary, full log, and on failure the previous container and the events
  local r
  for r in $PODS; do
    echo "--- vllm-$r-0: startup summary ---"
    kubectl -n $NS logs vllm-$r-0 --tail=-1 2>/dev/null | grep -E 'non-default args|is assigned as|Expert parallelism|MoE backend|Model loading took|Available KV cache memory|GPU KV cache size|Selected provider|Using network|Error|Traceback' | cut -c1-600 | awk '!seen[$0]++' || true
    echo "--- vllm-$r-0: full log ---"
    kubectl -n $NS logs vllm-$r-0 --tail=-1 2>/dev/null || true
    if [ "$1" != 0 ]; then
      echo "--- vllm-$r-0: previous container ---"
      kubectl -n $NS logs vllm-$r-0 --tail=-1 --previous 2>/dev/null || true
      kubectl -n $NS describe pod vllm-$r-0 2>/dev/null | sed -n '/^Events/,$p'
    fi
  done
}
if [ $EXIT -ne 0 ]; then dump_logs $EXIT; exit $EXIT; fi

# --- 4. The running pods carry the experiment (study 49: never trust the file alone) ----
VERIFY=0
fail() { [ $VERIFY != 0 ] || VERIFY=$1; }   # the first failure sets the exit code
BASE_N=$(grep -cE '^[[:space:]]*- "--' "$HERE/01-vllm_template.yaml")
for r in $PODS; do
  JSON=$(kubectl -n $NS get pod vllm-$r-0 -o jsonpath='{.spec.containers[0].args}')
  if [ $r = a ]; then WANT=(${ARGS_A[@]+"${ARGS_A[@]}"}); else WANT=(${ARGS_B[@]+"${ARGS_B[@]}"}); fi
  [ ${#WANT[@]} -eq 0 ] || args_carry_flags "$JSON" "${WANT[@]}" || fail 8
  args_count "$JSON" $((BASE_N + ${#WANT[@]})) || fail 8
  [ "$EP" = true ] || args_lack_flags "$JSON" --enable-expert-parallel || fail 8
  t "vllm-$r-0 args: $JSON"
done
SERVING_PODS=a; [ "$SERVING_B" != true ] || SERVING_PODS="a b"
for r in $SERVING_PODS; do
  CCI=$(kubectl -n $NS exec vllm-$r-0 -- curl -s -m 20 localhost:8000/metrics 2>/dev/null | grep '^vllm:cache_config_info')
  echo "$CCI" | cut -c1-700 | sed "s/^/vllm-$r-0 /"
  check_cache_config "$CCI" "$GPU_MEMORY_UTILIZATION" "$KV_CACHE_DTYPE" || fail 8
  # One tiny request: the layout must answer, and NCCL logs its transport at the first exchange.
  ANS=$(kubectl -n $NS exec vllm-$r-0 -- curl -sf -m 180 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}' 2>&1); RC=$?
  echo "vllm-$r-0 warm-up request rc=$RC: $(echo "$ANS" | cut -c1-400)"
  [ $RC = 0 ] || { echo "error: vllm-$r-0 is Ready but does not answer a chat completion" >&2; fail 4; }
done
if [ "$MULTINODE" = 1 ]; then
  LOGS=$(mktemp -d)
  for r in $PODS; do
    kubectl -n $NS logs vllm-$r-0 --tail=-1 > "$LOGS/$r.log" 2>/dev/null
    # Both pods in one distributed world of two ranks.
    if grep -q 'in world size 2 is assigned as' "$LOGS/$r.log"; then
      grep -m1 -o 'rank [0-9]* in world size 2 is assigned as.*' "$LOGS/$r.log" | cut -c1-200 | sed "s/^/vllm-$r-0 /"
    elif [ "$NCCL_CHECK" = required ]; then
      echo "error: vllm-$r-0 logged no rank line of a world of size 2: the two pods are not one vLLM" >&2; fail 8
    else
      echo "WARNING: vllm-$r-0 logged no rank line of a world of size 2"
    fi
    if grep -q 'Expert parallelism is enabled' "$LOGS/$r.log"; then
      grep -m1 -o 'Expert parallelism is enabled.*' "$LOGS/$r.log" | cut -c1-200 | sed "s/^/vllm-$r-0 /"
      [ "$EP" = true ] || echo "WARNING: vllm-$r-0 enabled expert parallelism although the experiment did not ask for it"
    elif [ "$EP" = true ]; then
      echo "error: vllm-$r-0 did not enable expert parallelism" >&2; fail 8
    fi
    SEEN=$(nccl_transport "$LOGS/$r.log")
    echo "vllm-$r-0 NCCL transport: ${SEEN:-none logged} (expected $TRANSPORT, check $NCCL_CHECK)"
    check_transport "$SEEN" "$TRANSPORT" "$NCCL_CHECK" || fail 9
  done
  rm -rf "$LOGS"
fi
dump_logs $VERIFY
set -e
[ $VERIFY -eq 0 ] || die "the running pods do not carry the experiment's configuration (see the errors above)" $VERIFY

# --- 5. AIPerf clients ready (pip install, ~1 min the first time) -----------------------
kubectl -n $NS wait --for=condition=Ready pod/aiperf-client-0 pod/aiperf-client-1 --timeout=600s \
  || { kubectl -n $NS logs aiperf-client-0 --tail=50 || true; kubectl -n $NS logs aiperf-client-1 --tail=50 || true; die "AIPerf client pods not Ready after 600 s" 6; }
t "done: layout $LAYOUT on $NODE_a (a) and $NODE_b (b), AIPerf clients ready"
