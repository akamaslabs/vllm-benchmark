#!/bin/bash
# Apply-config step for 41-l4-dsv2-lite-context-parallel (runs on the toolbox host via the
# Akamas workflow's Executor task, after FileConfigurator has rendered params.env).
#
# One experiment = one vLLM layout (TP / DP / DCP / PCP / EP) + engine settings on the four L4s
# of the g6.12xlarge node. Study 39's sequence, with GPU guards:
# validate -> node and GPU checks -> stop the old replica -> start one replica -> warm-up ->
# health check -> logs. Env overrides PARAMS / RENDERED for the calibration run.
#
# Fail-fast by design (set -e, explicit checks): a half-applied configuration would
# benchmark the previous experiment's vLLM under the current one's name.
# Exit codes: 0 ok; 2 invalid parameters (no cluster call); 3 cluster not ready (no node,
# wrong AZ, apply failed); 4 the replica did not come up healthy; 5 the GPUs are held by
# another pod (GPU-busy guard).
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/vllm-benchmark/studies/41-l4-dsv2-lite-context-parallel}   # overridable for manual tests
NS=llm-serving
APP=vllm-dsv2
POD=vllm-dsv2-0
NODE_ROLE=llm-serving-l4
# The model PVC (vllm-model-cache-dsv2lite) is an EBS volume in this AZ.
MODEL_AZ=us-east-2a
MODEL=dsv2-lite
PARAMS=${PARAMS:-$STUDY_DIR/k8s/params.env}
TEMPLATE=$STUDY_DIR/k8s/01-statefulset_template.yaml
RENDERED=${RENDERED:-$STUDY_DIR/k8s/01-statefulset.yaml}
RENDER=$STUDY_DIR/k8s/render_statefulset.sh
# 45 min: above the startupProbe (40 min) and VLLM_ENGINE_READY_TIMEOUT_S (30 min). Worst case
# of the whole script: pod delete (up to 180 s) + rollout (up to ROLLOUT_S) + warm-up (8
# requests on 4 threads, 600 s urlopen timeout each: up to ~1200 s if the engine stalls) + the
# log dump, ~4100 s plus the logs; the workflow's Apply config timeout (75 min = 4500 s) stays
# above it, so the log dump always reaches the Akamas output.
ROLLOUT_S=${ROLLOUT_S:-2700}
POLL_S=${AC_POLL_S:-20}
# shellcheck source=lib_health.sh
source "$STUDY_DIR/k8s/lib_health.sh"

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# --- 0. Parameters -------------------------------------------------------------------
# Validate everything BEFORE touching the cluster: render_statefulset.sh exits 2 on a
# leftover token, a missing key or a combination vLLM 0.32.0 refuses (or crashed on).
bash "$RENDER" "$PARAMS" "$TEMPLATE" "$RENDERED" || die "invalid parameters in $PARAMS"
t "parameters: $(grep -v '^#' "$PARAMS" | grep . | tr '\n' ' ')"
t "vLLM flags: $(grep -E '^ +- "--' "$RENDERED" | sed -e 's/^ *- "//' -e 's/"$//' | tr '\n' ' ')"

# --- Node and GPU checks ----------------------------------------------------------------
NODE=$(kubectl get nodes -l node-role=$NODE_ROLE -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[ -n "$NODE" ] || die "no node with node-role=$NODE_ROLE (node group llm-serving-l4 scaled to 0?)" 3
ZONE=$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}' 2>/dev/null || true)
# The ASG spans 2a/2b/2c; a node in another AZ can never mount the model PVC and the pod
# would sit Pending (volume node affinity conflict) until ROLLOUT_S.
[ "$ZONE" = "$MODEL_AZ" ] || die "node $NODE is in '${ZONE:-?}', the model PVC is in $MODEL_AZ: replace the node (infra/README.md)" 3
# GPU-busy guard: any other pod on the node that requests nvidia.com/gpu (another study's
# vLLM, the smoke pod dsv2-smoke, a probe) must be gone first; this study never deletes
# other studies' pods.
BUSY=$(kubectl get pods -A --field-selector "spec.nodeName=$NODE,status.phase!=Succeeded,status.phase!=Failed" \
  -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}:{.spec.containers[*].resources.requests.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null \
  | gpu_busy_pods "$NS/$APP-" | tr '\n' ' ')
[ -z "$BUSY" ] || die "the L4s on $NODE are held by other pod(s): $BUSY- stop them first" 5

# --- 1. Stop the previous replica -----------------------------------------------------
t "1/3 scaling vLLM to 0"
kubectl -n $NS scale sts $APP --replicas=0 2>/dev/null || true   # absent on the first run
# Delete the pod explicitly as well: with OrderedReady the controller stalls a scale-down
# while the replica is unready, and two engines must never share the GPUs.
kubectl -n $NS delete pod -l app=$APP --ignore-not-found --wait=true --timeout=180s >/dev/null
LEFT=$(kubectl -n $NS get pods -l app=$APP -o name | wc -l | tr -d ' ')
[ "$LEFT" = 0 ] || die "$LEFT vLLM pod(s) still present after scale-down" 3

# --- 2. Start the replica -------------------------------------------------------------
t "2/3 starting vLLM on $NODE ($ZONE)"
kubectl apply -f "$RENDERED" || die "kubectl apply failed — refusing to benchmark a stale configuration" 3
# `apply` alone does not restart a pod whose spec did not change; the delete above already
# removed it, so the controller creates a fresh one with this spec.
kubectl -n $NS scale sts $APP --replicas=1 >/dev/null

# Don't exit on a failed rollout before printing the logs: they land in the Akamas UI.
set +e
# Wait for Ready, but fail fast on a crash instead of sitting out the whole ROLLOUT_S (a
# layout vLLM refuses or that runs out of GPU memory would otherwise cost 45 min).
ROLLOUT_EXIT=1
T0=$SECONDS
UNSCHED_SINCE=""
while [ $((SECONDS - T0)) -lt "$ROLLOUT_S" ]; do
  READY=$(kubectl -n $NS get sts $APP -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  if [ "${READY:-0}" = 1 ]; then ROLLOUT_EXIT=0; break; fi
  RESTARTS=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  # Any restart already fails the trial (replicas_healthy after the warm-up), so stop at the
  # first one instead of sitting out the CrashLoopBackOff and a second full engine start.
  if [ "${RESTARTS:-0}" -ge 1 ]; then
    echo "error: $POD restarted during startup (${RESTARTS}) — failing now" >&2
    ROLLOUT_EXIT=4; break
  fi
  PHASE=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ "$PHASE" = Failed ]; then
    echo "error: $POD failed on $NODE" >&2
    ROLLOUT_EXIT=4; break
  fi
  # Not placed by the scheduler (volume node affinity conflict, GPUs taken, selector): give
  # a node coming up UNSCHED_S, then fail with the scheduler's message.
  SCHED=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].reason}' 2>/dev/null)
  if [ "$SCHED" = Unschedulable ]; then
    UNSCHED_SINCE=${UNSCHED_SINCE:-$SECONDS}
    if [ $((SECONDS - UNSCHED_SINCE)) -ge "${UNSCHED_S:-600}" ]; then
      echo "error: $POD unschedulable for $((SECONDS - UNSCHED_SINCE)) s: $(kubectl -n $NS get pod $POD -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].message}' 2>/dev/null)" >&2
      ROLLOUT_EXIT=4; break
    fi
  else
    UNSCHED_SINCE=""
  fi
  sleep "$POLL_S"
done
[ $ROLLOUT_EXIT = 1 ] && echo "error: $POD not Ready after $ROLLOUT_S s" >&2
t "rollout exit=$ROLLOUT_EXIT after $((SECONDS - T0)) s"
set -e

# --- 3. Warm-up ------------------------------------------------------------------------
# First-use kernels (Triton MLA / MoE autotune, CUDA graph replays of new sizes) of a fresh
# engine (study 25: TTFT p95 33.6 s on the first requests). They happen here, > 150 s (the
# p95 window of the constraints and the watchdog) before RunTest's measured run, which
# starts after pip install, the dataset generation and its own warm-up. Short prompts plus
# ~8k-token ones (the size of the load's shared prefixes), so both decode-sized and
# prefill-sized batches run, through every DP engine.
if [ $ROLLOUT_EXIT -eq 0 ]; then
  t "3/3 warm-up: 8 requests"
  WARM_FAIL=0
  kubectl -n $NS exec $POD -c vllm -- python3 -c '
import json, sys, time, urllib.request, concurrent.futures as cf
def one(i):
    text = "Write a short story about the number %d." % i
    if i % 2:
        text = " ".join("word%d" % ((j * (i + 1)) % 997) for j in range(2500)) + " Summarize the list above."
    body = {"model": sys.argv[1], "max_tokens": 64, "messages": [{"role": "user", "content": text}]}
    s = time.perf_counter()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read()
    return time.perf_counter() - s
with cf.ThreadPoolExecutor(4) as ex:
    print("warm-up e2e (s):", " ".join("%.2f" % x for x in ex.map(one, range(8))))
' "$MODEL" || { echo "error: warm-up requests failed on $POD" >&2; WARM_FAIL=1; }
  # Still Ready and never restarted: a replica that crashed on its first requests would
  # otherwise leave RunTest to discover it minutes later.
  READY=$(kubectl -n $NS get sts $APP -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  RESTARTS=$(kubectl -n $NS get pod $POD -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  if ! REASON=$(replicas_healthy 1 "$READY" "${RESTARTS:-0}" "$WARM_FAIL"); then
    echo "error: $REASON, failing the trial" >&2
    ROLLOUT_EXIT=4
  fi
fi

# Full logs, success or failure, so the Akamas task output alone is enough to debug an
# experiment. The summary first, because the full dump is long.
echo "--- $POD: startup summary ---"
kubectl -n $NS exec $POD -c vllm -- nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null || true
kubectl -n $NS logs $POD --all-containers --tail=-1 2>/dev/null \
  | grep -E 'Initializing a V1 LLM engine|Using .*[Bb]ackend|Available KV cache memory|GPU KV cache size|Maximum concurrency|Chunked prefill is enabled|enable_prefix_caching|[Cc]ontext parallel|CUDA graph|Model loading took|out of memory|NotImplementedError|ValueError|Error|Traceback' \
  | cut -c1-600 || true
echo "--- $POD: full logs ---"
kubectl -n $NS logs $POD --all-containers --tail=-1 2>/dev/null || true
if [ $ROLLOUT_EXIT -ne 0 ]; then
  echo "--- $POD: previous container (if it crashed and restarted) ---"
  kubectl -n $NS logs $POD --all-containers --tail=-1 --previous 2>/dev/null || true
  kubectl -n $NS describe pod $POD 2>/dev/null | tail -30 || true
fi
exit $ROLLOUT_EXIT
