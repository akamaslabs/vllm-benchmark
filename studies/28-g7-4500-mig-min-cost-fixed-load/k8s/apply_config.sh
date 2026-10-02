#!/bin/bash
# Apply-config step for 28-g7-4500-mig-min-cost-fixed-load (runs on the toolbox host via the
# Akamas workflow's Executor task, after FileConfigurator has rendered params.env).
#
# One experiment = one MIG layout (gpu0.mig_profile) + the pod's CPU / memory + one vLLM
# configuration. Study 26's switching sequence, unchanged:
#   none     MIG off, one replica on the whole GPU
#   1g.16gb  MIG on, both 1g.16gb instances, two replicas: vllm-0 is the tenant under
#            test, vllm-1 the busy neighbour (same configuration, same traffic)
# Changes against study 26: the parameters are validated and the StatefulSet rendered by
# render_statefulset.sh (step 0 validates before anything touches the GPU); every replica
# gets 8 warm-up requests at the end (first-use kernel JIT, see the README); env overrides
# PARAMS / RENDERED / REPLICAS_OVERRIDE for the kernel probe.
#
# Fail-fast by design (set -e, explicit checks): a half-applied mode would benchmark the
# previous experiment's configuration under the current one's name.
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load}   # overridable for manual tests
NS=gpu-sharing
PARAMS=${PARAMS:-$STUDY_DIR/k8s/params.env}
TEMPLATE=$STUDY_DIR/k8s/01-statefulset_template.yaml
RENDERED=${RENDERED:-$STUDY_DIR/k8s/01-statefulset.yaml}
RENDER=$STUDY_DIR/k8s/render_statefulset.sh
# shellcheck source=lib_health.sh
source "$STUDY_DIR/k8s/lib_health.sh"

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# --- 0. Parameters -------------------------------------------------------------------
# Validate everything BEFORE touching the GPU: render once with one replica into a scratch
# file (render_statefulset.sh exits 2 on a leftover token, an empty value or a value
# outside the study's domains), then source the values for the steps below.
bash "$RENDER" "$PARAMS" "$TEMPLATE" "$RENDERED.check" 1 || die "invalid parameters in $PARAMS"
rm -f "$RENDERED.check"
# shellcheck disable=SC1090
source "$PARAMS"
t "mig_profile=$MIG_PROFILE cpu=${CPU_LIMIT%m}m memory=${MEMORY_LIMIT%M}M gpu_memory_utilization=$GPU_MEMORY_UTILIZATION kv_cache_dtype=$KV_CACHE_DTYPE max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS linear_backend=$LINEAR_BACKEND attention_backend=$ATTENTION_BACKEND"

NODE=$(kubectl get nodes -l node-role=llm-serving-g7-4500 -o jsonpath='{.items[0].metadata.name}')
[ -n "$NODE" ] || die "no node with node-role=llm-serving-g7-4500 (node group scaled to 0?)" 3
# Host-level nvidia-smi through the study's privileged gpu-admin DaemonSet
# (infra/gpu-sharing/gpu-admin.yaml).
H() { kubectl -n $NS exec ds/gpu-admin -- nsenter -t 1 -m -u -n -i -- "$@"; }

# --- 1. Free the GPU -----------------------------------------------------------------
t "1/6 scaling vLLM to 0"
kubectl -n $NS scale sts vllm --replicas=0 2>/dev/null || true   # absent on the first run
# Delete the pods explicitly as well: with OrderedReady the controller stalls a
# scale-down while a replica is unready, and MIG must never change under a live engine.
kubectl -n $NS delete pod -l app=vllm --ignore-not-found --wait=true --timeout=180s >/dev/null
LEFT=$(kubectl -n $NS get pods -l app=vllm -o name | wc -l | tr -d ' ')
[ "$LEFT" = 0 ] || die "$LEFT vLLM pod(s) still present after scale-down — refusing to touch the GPU" 3

# --- 2. Neutral device-plugin config -------------------------------------------------
# Defensive, for a node left in MPS by study 25: the MPS control daemon holds a GPU context and
# keeps the GPU in Exclusive_Process compute mode, and both must go before MIG changes.
t "2/6 neutral device-plugin config"
kubectl label node "$NODE" nvidia.com/device-plugin.config=exclusive --overwrite >/dev/null
kubectl label node "$NODE" nvidia.com/mps.capable- >/dev/null 2>&1 || true
# Wait by NAME: the chart gives the MPS daemon's pods the same labels as the plugin's
# (only app.kubernetes.io/{name,instance}), so a label selector cannot tell them apart.
# Then make sure no MPS server process survives on the host: `nvidia-smi -mig` fails
# with "In use by another client" while one holds the GPU. The [n] keeps pgrep -f from
# matching the `sh -c` command line that carries the pattern itself.
MPS_LEFT=1
for _ in $(seq 1 60); do
  MPS_LEFT=$(kubectl -n $NS get pods --field-selector spec.nodeName="$NODE" -o name | grep -c mps-control-daemon || true)
  [ "$MPS_LEFT" = 0 ] && H sh -c '! pgrep -f "[n]vidia-cuda-mps" >/dev/null' && break
  sleep 2
done
[ "$MPS_LEFT" = 0 ] || die "MPS control daemon still running on $NODE after 2 min" 4
H sh -c '! pgrep -f "[n]vidia-cuda-mps" >/dev/null' || die "an nvidia-cuda-mps process is still alive on $NODE" 4

# --- 3. MIG layout -------------------------------------------------------------------
t "3/6 MIG layout"
CUR=$(H nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | tr -d ' ')
# Destroy every existing instance first: GPU/compute instances do not survive a reboot
# (MIG mode does), and a leftover layout from a previous experiment must not leak in.
if [ "$CUR" = Enabled ]; then
  H nvidia-smi mig -dci >/dev/null 2>&1 || true
  H nvidia-smi mig -dgi >/dev/null 2>&1 || true
fi
if [ "$MIG_PROFILE" = none ]; then
  [ "$CUR" = Enabled ] && H nvidia-smi -i 0 -mig 0
  INSTANCES=1; PLUGIN_CONFIG=exclusive; WANT_STATE=Disabled,Disabled
else
  [ "$CUR" = Enabled ] || H nvidia-smi -i 0 -mig 1
  # `nvidia-smi mig -lgip` rows: | 0  MIG <name>  <id>  <free>/<total>  <mem> ...
  read -r PID FREE < <(H nvidia-smi mig -lgip | awk -v n="$MIG_PROFILE" '$3=="MIG" && $4==n {split($6,a,"/"); print $5, a[1]; exit}') || true
  [ -n "${PID:-}" ] || die "MIG profile $MIG_PROFILE is not offered by this GPU (nvidia-smi mig -lgip)" 4
  [ "${FREE:-0}" -ge 1 ] || die "no free $MIG_PROFILE instance on an empty GPU" 4
  IDS=$(printf "$PID,%.0s" $(seq 1 "$FREE")); IDS=${IDS%,}
  H nvidia-smi mig -cgi "$IDS" -C
  INSTANCES=$(H nvidia-smi -L | grep -c "MIG $MIG_PROFILE")
  [ "$INSTANCES" = "$FREE" ] || die "created $INSTANCES $MIG_PROFILE instances, expected $FREE" 4
  PLUGIN_CONFIG=mig; WANT_STATE=Enabled,Enabled
fi
H nvidia-smi -c DEFAULT >/dev/null
STATE=$(H nvidia-smi --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader | tr -d ' ')
[ "$STATE" = "$WANT_STATE" ] || die "MIG state '$STATE', expected $WANT_STATE for $MIG_PROFILE (a pending change would need a GPU reset; refusing to benchmark the wrong layout)" 4
# One replica per instance (the GPU always fully used); the kernel probe asks for one.
REPLICAS=${REPLICAS_OVERRIDE:-$INSTANCES}
[[ "$REPLICAS" =~ ^[12]$ ]] && [ "$REPLICAS" -le "$INSTANCES" ] || die "REPLICAS_OVERRIDE=$REPLICAS with $INSTANCES instance(s)" 2
t "   $INSTANCES x $MIG_PROFILE -> $REPLICAS replica(s)"

# --- 4. Device-plugin config ----------------------------------------------------------
t "4/6 device-plugin config $PLUGIN_CONFIG"
kubectl label node "$NODE" nvidia.com/device-plugin.config=$PLUGIN_CONFIG --overwrite >/dev/null
# The config-manager sidecar restarts the plugin on a label change (15-40 s in phase 0).
# Give it time to re-register before trusting the count: the previous layout's value can
# match by accident.
sleep 15
N=""
for _ in $(seq 1 30); do
  N=$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
  [ "$N" = "$INSTANCES" ] && break
  sleep 5
done
[ "$N" = "$INSTANCES" ] || die "node advertises nvidia.com/gpu=$N, expected $INSTANCES for $MIG_PROFILE" 5

# --- 5. dcgm-exporter re-reads the GPU layout ----------------------------------------
# After a MIG change the exporter has to be restarted to see (or stop seeing) the
# slices. Only the pod on THIS node is touched; a no-op while dcgm-exporter is still
# pinned to another node group (see README "Before starting").
t "5/6 restarting dcgm-exporter on $NODE (if present)"
kubectl -n monitoring delete pod -l app.kubernetes.io/name=dcgm-exporter --field-selector spec.nodeName="$NODE" --wait=false 2>/dev/null || true

# --- 6. vLLM replicas ----------------------------------------------------------------
t "6/6 starting $REPLICAS vLLM replica(s)"
bash "$RENDER" "$PARAMS" "$TEMPLATE" "$RENDERED" "$REPLICAS" || die "rendering the StatefulSet failed"
kubectl apply -f "$RENDERED" || die "kubectl apply failed — refusing to benchmark a stale configuration" 3

# Don't exit on a failed rollout before printing the logs: they land in the Akamas UI.
set +e
# Wait for all replicas Ready, but fail fast on a crash loop instead of sitting out the
# whole 2400 s (a config that OOMs at startup would otherwise cost 40 min per failure).
# 2400 s covers two sequential starts (OrderedReady), ~3.5 min each warm, up to ~12 min on
# a cold node (10 GB image pull).
ROLLOUT_EXIT=1
T0=$SECONDS
while [ $((SECONDS - T0)) -lt 2400 ]; do
  READY=$(kubectl -n $NS get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  if [ "${READY:-0}" = "$REPLICAS" ]; then ROLLOUT_EXIT=0; break; fi
  MAXR=$(kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort -n | tail -1)
  if [ "${MAXR:-0}" -ge 2 ]; then
    echo "error: a vLLM replica restarted ${MAXR} times during startup — crash loop, failing now" >&2
    ROLLOUT_EXIT=4; break
  fi
  sleep 30
done
[ $ROLLOUT_EXIT = 1 ] && echo "error: $REPLICAS replica(s) not Ready after 2400 s" >&2
t "rollout exit=$ROLLOUT_EXIT after $((SECONDS - T0)) s"
set -e
# Warm-up (README "Load"): the first requests a fresh vLLM serves hit first-use kernel JIT
# (TTFT p95 33.6 s in study 25's phase 0). They happen here, so they are > 150 s old (the
# p95 window) when RunTest's measured run starts after pip install and its warm-up.
if [ $ROLLOUT_EXIT -eq 0 ]; then
  WARM_FAIL=0
  for i in $(seq 0 $((REPLICAS - 1))); do
    echo "--- vllm-$i: 8 warm-up requests ---"
    kubectl -n $NS exec vllm-$i -c vllm -- python3 -c '
import json, time, urllib.request, concurrent.futures as cf
def one(i):
    body = {"model": "qwen3-8b-mig", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Write a short story about the number %d." % i}]}
    s = time.perf_counter()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read()
    return time.perf_counter() - s
with cf.ThreadPoolExecutor(4) as ex:
    print("warm-up e2e (s):", " ".join("%.2f" % x for x in ex.map(one, range(8))))
' || { echo "error: warm-up requests failed on vllm-$i" >&2; WARM_FAIL=$((WARM_FAIL + 1)); }
  done
  # Every replica still Ready and never restarted: a replica OOMKilled by its first requests
  # (the memory floor) would leave the tenant with an idle neighbour, a falsely VALID trial.
  READY=$(kubectl -n $NS get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
  MAXR=$(kubectl -n $NS get pods -l app=vllm -o jsonpath='{range .items[*]}{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort -n | tail -1)
  if ! REASON=$(replicas_healthy "$REPLICAS" "$READY" "${MAXR:-0}" "$WARM_FAIL"); then
    echo "error: $REASON, failing the trial" >&2
    ROLLOUT_EXIT=4
  fi
fi
# Full logs of every replica, success or failure, so the Akamas task output alone is
# enough to debug an experiment. The summary first, because the full dump is long.
for i in $(seq 0 $((REPLICAS - 1))); do
  echo "--- vllm-$i: startup summary ---"
  kubectl -n $NS logs vllm-$i --all-containers --tail=-1 2>/dev/null | grep -E 'Selected .*Kernel|Using .*[Bb]ackend|attention backend|linear|Model loading took|Available KV cache memory|GPU KV cache size|Error|Traceback' || true
done
for i in $(seq 0 $((REPLICAS - 1))); do
  echo "--- vllm-$i: full logs ---"
  kubectl -n $NS logs vllm-$i --all-containers --tail=-1 2>/dev/null || true
  if [ $ROLLOUT_EXIT -ne 0 ]; then
    echo "--- vllm-$i: previous container (if it crashed and restarted) ---"
    kubectl -n $NS logs vllm-$i --all-containers --tail=-1 --previous 2>/dev/null || true
  fi
done
exit $ROLLOUT_EXIT
