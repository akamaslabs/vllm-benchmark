#!/bin/bash
# Apply-config step for 26-g7-4500-gpu-slice-right-sizing (runs on the toolbox host via the
# Akamas workflow's Executor task, after FileConfigurator has rendered params.env).
#
# One experiment = one GPU sharing mode + how many of its pieces are used + one vLLM
# configuration. Study 26 (right-sizing) differs from study 25 only here: the replica count
# is a parameter (vllm_workload.replicas) instead of following from the mode, so one MIG
# slice / one MPS client can serve alone while the other piece stays idle. The switching
# sequence is study 25's, validated by hand in its phase 0 (2026-09-29) for every
# transition between the four modes, including MIG on/off with dcgm-exporter and the
# device plugin running (no reboot was ever needed).
#
# Fail-fast by design (set -e, explicit checks): a half-applied mode would benchmark the
# previous experiment's configuration under the current one's name — the failure study
# 17's apply script was hardened against.
set -euo pipefail

STUDY_DIR=${STUDY_DIR:-/work/vllm-benchmark/studies/26-g7-4500-gpu-slice-right-sizing}   # overridable for manual tests
NS=gpu-sharing
PARAMS=$STUDY_DIR/k8s/params.env
TEMPLATE=$STUDY_DIR/k8s/01-statefulset_template.yaml
RENDERED=$STUDY_DIR/k8s/01-statefulset.yaml

t() { echo "$(date -u +%T) apply_config: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# --- 0. Parameters -------------------------------------------------------------------
grep -q '\${' "$PARAMS" && die "params.env still has unsubstituted tokens — a parameter is missing from the study's parametersSelection: $(grep '\${' "$PARAMS")"
# shellcheck disable=SC1090
source "$PARAMS"
for v in SHARING_MODE REPLICAS GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS STREAM_INTERVAL; do
  [ -n "${!v:-}" ] || die "$v is empty in params.env (doNotRenderParameters renders an empty string, not the token)"
done

# Plugin config names use a hyphen (time-slicing), the Akamas category an underscore.
# PIECES = how many units the device plugin advertises in this mode; REPLICAS (<= PIECES)
# is how many of them serve. The study's parameterConstraints forbid the combinations
# rejected here; this is the last line of defence.
case "$SHARING_MODE" in
  exclusive)    PIECES=1; PLUGIN_CONFIG=exclusive ;;
  mig)          PIECES=2; PLUGIN_CONFIG=mig ;;
  time_slicing) PIECES=2; PLUGIN_CONFIG=time-slicing ;;
  mps)          PIECES=2; PLUGIN_CONFIG=mps ;;
  *) die "unknown sharing_mode '$SHARING_MODE'" ;;
esac
case "$SHARING_MODE:$REPLICAS" in
  exclusive:1|mig:1|mig:2|mps:1|mps:2|time_slicing:2) ;;
  *) die "sharing_mode=$SHARING_MODE with replicas=$REPLICAS is not a valid combination" ;;
esac

# gpu_memory_utilization is the fraction of the memory the REPLICA owns. vLLM measures
# it against what CUDA reports as total, which phase 0 showed differs per mode:
#   exclusive     whole GPU (31.38 GiB)                   -> as is
#   mig           the 1g.16gb slice (15.66 GiB)            -> as is
#   time_slicing  whole GPU, shared by two engines         -> / 2
#   mps           whole GPU as total, 15.79 GiB FREE (the  -> / 2; 0.90 un-halved failed
#                 MPS limit lowers free memory, not total)      with "Free memory ... less
#                                                               than desired" in phase 0
# The study domain tops out at 0.90, so the halved value never exceeds 0.45 (14.1 GiB),
# inside the 15.79 GiB MPS clients get.
case "$SHARING_MODE" in
  time_slicing|mps) GMU_EFFECTIVE=$(awk -v g="$GPU_MEMORY_UTILIZATION" 'BEGIN{printf "%.4f", g/2}') ;;
  *)                GMU_EFFECTIVE=$GPU_MEMORY_UTILIZATION ;;
esac
t "mode=$SHARING_MODE replicas=$REPLICAS/$PIECES gpu_memory_utilization=$GPU_MEMORY_UTILIZATION (effective $GMU_EFFECTIVE) max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS stream_interval=$STREAM_INTERVAL"

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
# Stops the MPS control daemon if the previous mode was mps: it holds a GPU context and
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

# --- 3. MIG state --------------------------------------------------------------------
t "3/6 MIG state"
CUR=$(H nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | tr -d ' ')
if [ "$SHARING_MODE" = mig ]; then
  [ "$CUR" = Enabled ] || H nvidia-smi -i 0 -mig 1
  # Recreate the two slices every time: GPU/compute instances do not survive a reboot
  # (MIG mode does), and a leftover layout from a manual test must not leak in.
  H nvidia-smi mig -dci >/dev/null 2>&1 || true
  H nvidia-smi mig -dgi >/dev/null 2>&1 || true
  H nvidia-smi mig -cgi 5,5 -C      # profile 5 = 1g.16gb, the only two-way split
else
  if [ "$CUR" = Enabled ]; then
    H nvidia-smi mig -dci >/dev/null 2>&1 || true
    H nvidia-smi mig -dgi >/dev/null 2>&1 || true
    H nvidia-smi -i 0 -mig 0
  fi
fi
H nvidia-smi -c DEFAULT >/dev/null
STATE=$(H nvidia-smi --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader | tr -d ' ')
case "$SHARING_MODE:$STATE" in
  mig:Enabled,Enabled|exclusive:Disabled,Disabled|time_slicing:Disabled,Disabled|mps:Disabled,Disabled) ;;
  *) die "MIG state '$STATE' does not match mode $SHARING_MODE (a pending change would need a GPU reset; refusing to benchmark the wrong mode)" 4 ;;
esac

# --- 4. Device-plugin config for the mode --------------------------------------------
t "4/6 device-plugin config $PLUGIN_CONFIG"
[ "$SHARING_MODE" = mps ] && kubectl label node "$NODE" nvidia.com/mps.capable=true --overwrite >/dev/null
kubectl label node "$NODE" nvidia.com/device-plugin.config=$PLUGIN_CONFIG --overwrite >/dev/null
# The config-manager sidecar restarts the plugin on a label change (15-40 s in phase 0).
# Give it time to re-register before trusting the count: the previous mode's value can
# match by accident (exclusive -> exclusive, or mig -> time_slicing, both 2).
sleep 15
N=""
for _ in $(seq 1 30); do
  N=$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
  [ "$N" = "$PIECES" ] && break
  sleep 5
done
[ "$N" = "$PIECES" ] || die "node advertises nvidia.com/gpu=$N, expected $PIECES for $SHARING_MODE" 5
if [ "$SHARING_MODE" = mps ]; then
  kubectl -n $NS rollout status ds/nvdp-g7-nvidia-device-plugin-mps-control-daemon --timeout=180s
fi

# --- 5. dcgm-exporter re-reads the GPU layout ----------------------------------------
# After a MIG change the exporter has to be restarted to see (or stop seeing) the
# slices. Only the pod on THIS node is touched; a no-op while dcgm-exporter is still
# pinned to another node group (see README "Before starting").
t "5/6 restarting dcgm-exporter on $NODE (if present)"
kubectl -n monitoring delete pod -l app.kubernetes.io/name=dcgm-exporter --field-selector spec.nodeName="$NODE" --wait=false 2>/dev/null || true

# --- 6. vLLM replicas ----------------------------------------------------------------
t "6/6 starting $REPLICAS vLLM replica(s)"
# sed, not envsubst: the toolbox image has no gettext (audit 2026-09-29). Values are
# checked numeric first, so they are safe as sed replacements.
for v in REPLICAS GMU_EFFECTIVE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS STREAM_INTERVAL; do
  [[ "${!v}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "$v='${!v}' is not a number"
done
sed -e "s/\$REPLICAS/$REPLICAS/g" \
    -e "s/\$GMU_EFFECTIVE/$GMU_EFFECTIVE/g" \
    -e "s/\$MAX_NUM_SEQS/$MAX_NUM_SEQS/g" \
    -e "s/\$MAX_NUM_BATCHED_TOKENS/$MAX_NUM_BATCHED_TOKENS/g" \
    -e "s/\$STREAM_INTERVAL/$STREAM_INTERVAL/g" \
    "$TEMPLATE" > "$RENDERED"
grep -q '\$[A-Z_]\{3,\}' "$RENDERED" && die "rendered StatefulSet still has a \$VARIABLE: $(grep -o '\$[A-Z_]\{3,\}' "$RENDERED" | sort -u | tr '\n' ' ')"
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
# Full logs of every replica, success or failure, so the Akamas task output alone is
# enough to debug an experiment. The summary first, because the full dump is long.
for i in $(seq 0 $((REPLICAS - 1))); do
  echo "--- vllm-$i: startup summary ---"
  kubectl -n $NS logs vllm-$i --all-containers --tail=-1 2>/dev/null | grep -E 'Selected .*Kernel|attention backend|Model loading took|Available KV cache memory|GPU KV cache size|Error|Traceback' || true
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
