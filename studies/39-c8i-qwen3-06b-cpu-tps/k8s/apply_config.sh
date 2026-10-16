#!/bin/bash
# Apply-config step for 39-c8i-qwen3-06b-cpu-tps (runs on the toolbox host via the Akamas
# workflow's Executor task, after FileConfigurator has rendered params.env).
#
# One experiment = one vLLM CPU configuration on the c8i.12xlarge node. Study 30's sequence:
# validate -> stop the old replica -> start one replica -> warm-up -> health check -> logs.
# Env overrides PARAMS / RENDERED for the probe (../probe/probe.sh) and the manual smoke run.
#
# Fail-fast by design (set -e, explicit checks): a half-applied configuration would
# benchmark the previous experiment's vLLM under the current one's name.
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/vllm-benchmark/studies/39-c8i-qwen3-06b-cpu-tps}   # overridable for manual tests
NS=llm-cpu
NODE_ROLE=llm-serving-cpu
MODEL=qwen3-06b
PARAMS=${PARAMS:-$STUDY_DIR/k8s/params.env}
TEMPLATE=$STUDY_DIR/k8s/01-statefulset_template.yaml
RENDERED=${RENDERED:-$STUDY_DIR/k8s/01-statefulset.yaml}
RENDER=$STUDY_DIR/k8s/render_statefulset.sh
# 45 min: above the startupProbe (40 min) and VLLM_ENGINE_READY_TIMEOUT_S (30 min). Worst case
# of the whole script: pod delete (up to 180 s) + rollout (up to ROLLOUT_S) + warm-up (8
# requests on 4 threads, 600 s urlopen timeout each: up to ~1200 s if the engine stalls just
# after a late Ready) + the log dump, ~4100 s plus the logs; the workflows' Apply config
# timeout (75 min = 4500 s) stays above it, so the log dump always reaches the Akamas output.
ROLLOUT_S=${ROLLOUT_S:-2700}
POLL_S=${AC_POLL_S:-20}
# shellcheck source=lib_health.sh
source "$STUDY_DIR/k8s/lib_health.sh"

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# --- 0. Parameters -------------------------------------------------------------------
# Validate everything BEFORE touching the cluster: render_statefulset.sh exits 2 on a
# leftover token, an empty value or a value vLLM 0.31.0 would refuse at startup.
bash "$RENDER" "$PARAMS" "$TEMPLATE" "$RENDERED" || die "invalid parameters in $PARAMS"
t "parameters: $(grep -v '^#' "$PARAMS" | grep . | tr '\n' ' ')"
t "vLLM flags: $(grep -E '^ +- "--' "$RENDERED" | sed -e 's/^ *- "//' -e 's/"$//' | tr '\n' ' ')"

NODE=$(kubectl get nodes -l node-role=$NODE_ROLE -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[ -n "$NODE" ] || die "no node with node-role=$NODE_ROLE (node group llm-serving-c8i scaled to 0?)" 3

# --- 1. Stop the previous replica -----------------------------------------------------
t "1/3 scaling vLLM to 0"
kubectl -n $NS scale sts vllm --replicas=0 2>/dev/null || true   # absent on the first run
# Delete the pod explicitly as well: with OrderedReady the controller stalls a scale-down
# while the replica is unready, and two engines must never share the node's cores (the
# second could not even be admitted: 2 x 44 CPUs > the node's allocatable).
kubectl -n $NS delete pod -l app=vllm --ignore-not-found --wait=true --timeout=180s >/dev/null
LEFT=$(kubectl -n $NS get pods -l app=vllm -o name | wc -l | tr -d ' ')
[ "$LEFT" = 0 ] || die "$LEFT vLLM pod(s) still present after scale-down" 3

# --- 2. Start the replica -------------------------------------------------------------
t "2/3 starting vLLM on $NODE"
kubectl apply -f "$RENDERED" || die "kubectl apply failed — refusing to benchmark a stale configuration" 3
# `apply` alone does not restart a pod whose spec did not change; the delete above already
# removed it, so the controller creates a fresh one with this spec.
kubectl -n $NS scale sts vllm --replicas=1 >/dev/null

# Don't exit on a failed rollout before printing the logs: they land in the Akamas UI.
set +e
# Wait for Ready, but fail fast on a crash loop instead of sitting out the whole ROLLOUT_S
# (a configuration vLLM refuses at the KV allocation, or that is OOMKilled, would otherwise
# cost 45 min per failure).
ROLLOUT_EXIT=1
T0=$SECONDS
UNSCHED_SINCE=""
while [ $((SECONDS - T0)) -lt "$ROLLOUT_S" ]; do
  READY=$(kubectl -n $NS get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  if [ "${READY:-0}" = 1 ]; then ROLLOUT_EXIT=0; break; fi
  RESTARTS=$(kubectl -n $NS get pod vllm-0 -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  # Any restart already fails the trial (replicas_healthy after the warm-up), so stop at the
  # first one instead of sitting out the CrashLoopBackOff and a second full engine start.
  if [ "${RESTARTS:-0}" -ge 1 ]; then
    echo "error: vllm-0 restarted during startup (${RESTARTS}) — failing now" >&2
    ROLLOUT_EXIT=4; break
  fi
  # An unschedulable pod (TopologyAffinityError, SMTAlignmentError, Insufficient cpu) never
  # restarts: fail on it at once, the next experiment would hit the same wall.
  PHASE_REASON=$(kubectl -n $NS get pod vllm-0 -o jsonpath='{.status.phase}/{.status.reason}' 2>/dev/null)
  case "$PHASE_REASON" in
    Failed/*|*/TopologyAffinityError|*/SMTAlignmentError|*/OutOfcpu|*/OutOfmemory)
      echo "error: vllm-0 was not admitted on $NODE ($PHASE_REASON)" >&2
      ROLLOUT_EXIT=4; break ;;
  esac
  # Not placed by the scheduler (Insufficient cpu/memory, taint, selector): give the
  # autoscaler or a node coming up UNSCHED_S, then fail instead of waiting ROLLOUT_S.
  SCHED=$(kubectl -n $NS get pod vllm-0 -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].reason}' 2>/dev/null)
  if [ "$SCHED" = Unschedulable ]; then
    UNSCHED_SINCE=${UNSCHED_SINCE:-$SECONDS}
    if [ $((SECONDS - UNSCHED_SINCE)) -ge "${UNSCHED_S:-300}" ]; then
      echo "error: vllm-0 unschedulable for $((SECONDS - UNSCHED_SINCE)) s: $(kubectl -n $NS get pod vllm-0 -o jsonpath='{.status.conditions[?(@.type=="PodScheduled")].message}' 2>/dev/null)" >&2
      ROLLOUT_EXIT=4; break
    fi
  else
    UNSCHED_SINCE=""
  fi
  sleep "$POLL_S"
done
[ $ROLLOUT_EXIT = 1 ] && echo "error: vllm-0 not Ready after $ROLLOUT_S s" >&2
t "rollout exit=$ROLLOUT_EXIT after $((SECONDS - T0)) s"
set -e

# --- 3. Warm-up ------------------------------------------------------------------------
# First-use kernels and inductor shapes of a fresh engine (study 25: TTFT p95 33.6 s on the
# first requests). They happen here, > 150 s (the p95 window of the constraints and the
# watchdog) before RunTest's measured run starts after pip install and its own warm-up.
# Short prompts plus ~1.5k-token ones, so both decode-sized and prefill-sized batches run.
if [ $ROLLOUT_EXIT -eq 0 ]; then
  t "3/3 warm-up: 8 requests"
  WARM_FAIL=0
  kubectl -n $NS exec vllm-0 -c vllm -- python3 -c '
import json, sys, time, urllib.request, concurrent.futures as cf
def one(i):
    text = "Write a short story about the number %d." % i
    if i % 2:
        text = " ".join("word%d" % (j % 997) for j in range(1200)) + " Summarize the list above."
    body = {"model": sys.argv[1], "max_tokens": 64, "messages": [{"role": "user", "content": text}]}
    s = time.perf_counter()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read()
    return time.perf_counter() - s
with cf.ThreadPoolExecutor(4) as ex:
    print("warm-up e2e (s):", " ".join("%.2f" % x for x in ex.map(one, range(8))))
' "$MODEL" || { echo "error: warm-up requests failed on vllm-0" >&2; WARM_FAIL=1; }
  # Still Ready and never restarted: a replica OOMKilled by its first requests would
  # otherwise leave RunTest to discover it minutes later.
  READY=$(kubectl -n $NS get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  RESTARTS=$(kubectl -n $NS get pod vllm-0 -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)
  if ! REASON=$(replicas_healthy 1 "$READY" "${RESTARTS:-0}" "$WARM_FAIL"); then
    echo "error: $REASON, failing the trial" >&2
    ROLLOUT_EXIT=4
  fi
fi

# Full logs, success or failure, so the Akamas task output alone is enough to debug an
# experiment. The summary first, because the full dump is long. The OpenMP binding message
# is multi-line (core ids / reserved_cpus on the lines after it).
echo "--- vllm-0: startup summary ---"
# What the container sees (CPU, NUMA, its exclusive cpuset, the memory limit the KV cache is
# sized from), so the Akamas task output records it for every experiment, Smoke included.
kubectl -n $NS exec vllm-0 -c vllm -- sh -c 'lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA)"; grep Cpus_allowed_list /proc/self/status; echo "memory.max $(cat /sys/fs/cgroup/memory.max) memory.current $(cat /sys/fs/cgroup/memory.current)"' 2>/dev/null || true
# Per-process RSS next to memory.current: on CPU the pod runs the API server, the EngineCore
# process and the worker, and vLLM's KV check subtracts only the worker's RSS (see
# params.env.template). No procps in the image: read /proc.
kubectl -n $NS exec vllm-0 -c vllm -- python3 -c '
import os
for pid in sorted(filter(str.isdigit, os.listdir("/proc")), key=int):
    try:
        st = dict(l.split(":", 1) for l in open("/proc/%s/status" % pid) if ":" in l)
        cmd = open("/proc/%s/cmdline" % pid).read().replace("\0", " ").strip()[:100]
    except OSError:
        continue
    rss = int(st.get("VmRSS", "0 kB").split()[0])
    if rss >= 102400:
        print("rss %d MiB pid %s %s: %s" % (rss // 1024, pid, st.get("Name", "?").strip(), cmd))
' 2>/dev/null || true
kubectl -n $NS logs vllm-0 --all-containers --tail=-1 2>/dev/null \
  | grep -E -A4 'OpenMP thread binding info' || true
kubectl -n $NS logs vllm-0 --all-containers --tail=-1 2>/dev/null \
  | grep -E 'Auto set|Explicitly set|non-kv usages|KV cache size|Maximum concurrency|Using .*[Bb]ackend|Chunked prefill is enabled|LD_PRELOAD|get_mempolicy|Operation not permitted|not in available memory nodes|Model loading took|Error|Traceback' || true
echo "--- vllm-0: full logs ---"
kubectl -n $NS logs vllm-0 --all-containers --tail=-1 2>/dev/null || true
if [ $ROLLOUT_EXIT -ne 0 ]; then
  echo "--- vllm-0: previous container (if it crashed and restarted) ---"
  kubectl -n $NS logs vllm-0 --all-containers --tail=-1 --previous 2>/dev/null || true
  kubectl -n $NS describe pod vllm-0 2>/dev/null | tail -30 || true
fi
exit $ROLLOUT_EXIT
