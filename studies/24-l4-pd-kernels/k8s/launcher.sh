#!/bin/bash
# Study 24 engine launcher (study 20's, copied): starts the vLLM processes of one trial's topology inside ONE
# container that sees all 4 L4s, each process pinned with CUDA_VISIBLE_DEVICES.
#
# This is the layout vLLM's own CI uses for NixlConnector P/D on 4x L4
# (tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh, v0.29.0). Keeping every
# process in one container keeps them in one PID namespace, and UCX's cuda_ipc transport
# refuses legacy CUDA IPC handles across PID namespaces (ucx src/uct/cuda/cuda_ipc/
# cuda_ipc_iface.c). Separate pods would silently fall back to host staging, which is
# what preset kv_buffer_device=cpu tests on purpose.
#
# Inputs (ConfigMap pd-config, rendered per trial, mounted at /pd-config):
#   topology.env   PD_PREFILL_INSTANCES PD_DECODE_INSTANCES PD_KV_CONNECTOR PD_KV_BUFFER_DEVICE
#                  PD_TUNED_KERNEL_CONFIGS (study 24: "true" installs the tuned kernel configs
#                  from /pd-tuned-configs/tuned-configs.tgz, optional ConfigMap pd-tuned-configs)
#   common.args    flags shared by every instance (one per line)
#   prefill.args   flags of prefill instances    (one per line)
#   decode.args    flags of decode instances     (one per line; aggregated replicas use these)
#
# Layout (the router uses the same convention):
#   prefill i -> GPU i,     HTTP 8100+i, NIXL side channel 5600+i, VLLM_PORT 20000+100*i
#   decode  j -> GPU P+j,   HTTP 8200+j, NIXL side channel 5700+j, VLLM_PORT 30000+100*j
#   served names: <base>-prefill / <base>-decode first (metrics label model_name = first
#   name, vllm/config/model.py get_served_model_name), then <base> so requests for the plain
#   model name are accepted by every instance.
#
# Every instance's output is prefixed with its role so `kubectl logs -c engine` stays
# readable. If ANY instance exits, the launcher kills the rest and exits non-zero: the
# container restarts and apply_config.sh / run_test_tps.sh fail the trial on the restart.
set -uo pipefail

MODEL=${PD_MODEL:-Qwen/Qwen3-8B-FP8}
SERVED_BASE=${PD_SERVED_BASE:-qwen3-8b}
NUM_GPUS=${PD_NUM_GPUS:-4}
CFG=${PD_CONFIG_DIR:-/pd-config}

set -a
# shellcheck disable=SC1091
source "$CFG/topology.env"
set +a
P=${PD_PREFILL_INSTANCES:-0}
D=${PD_DECODE_INSTANCES:-1}
CONNECTOR=${PD_KV_CONNECTOR:-NixlConnector}
BUFFER=${PD_KV_BUFFER_DEVICE:-cuda}
TUNED=${PD_TUNED_KERNEL_CONFIGS:-false}
TUNED_ARCHIVE=${PD_TUNED_ARCHIVE:-/pd-tuned-configs/tuned-configs.tgz}

read_args() { grep -vE '^[[:space:]]*(#|$)' "$1" 2>/dev/null | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'; }
mapfile -t COMMON < <(read_args "$CFG/common.args")
mapfile -t PREFILL_ARGS < <(read_args "$CFG/prefill.args")
mapfile -t DECODE_ARGS < <(read_args "$CFG/decode.args")

if (( D < 1 )); then echo "launcher: PD_DECODE_INSTANCES must be >= 1 (got $D)"; exit 1; fi
if (( P + D > NUM_GPUS )); then
  echo "launcher: topology ${P}P${D}D needs $((P + D)) GPUs, the pod has $NUM_GPUS"; exit 1
fi
if (( P > 0 )); then MODE=disaggregated; else MODE=aggregated; fi
echo "launcher: model=$MODEL mode=$MODE prefill=$P decode=$D connector=$CONNECTOR kv_buffer_device=$BUFFER"
echo "launcher: common: ${COMMON[*]}"
echo "launcher: prefill: ${PREFILL_ARGS[*]}"
echo "launcher: decode: ${DECODE_ARGS[*]}"
nvidia-smi -L || true
# PCIe P2P between the L4s decides whether cuda_ipc can be used at all (unverified on
# g6.12xlarge as of 2026-09-23 — the first trial's log answers it).
nvidia-smi topo -m 2>/dev/null || true
nvidia-smi topo -p2p r 2>/dev/null || true

# Download the checkpoint once, before N processes race for the same HF cache.
python3 -c "from huggingface_hub import snapshot_download; print('launcher: weights at', snapshot_download('$MODEL'))" \
  || { echo "launcher: model download failed"; exit 1; }

# Tuned kernel configs (study 24, vllm_decode.tuned_kernel_configs). vLLM reads the Triton
# FP8 block configs only from its own package folder (fp8_utils.py
# get_w8a8_block_fp8_configs), so they are installed there before any instance starts. The
# file names (N=...,K=...,device_name=...,block_shape=[128,128].json) are not valid
# ConfigMap keys, so apply_config.sh ships them as one archive. The container's filesystem
# is fresh on every start, so a "false" trial never sees files from an earlier "true" one.
if [[ $TUNED == true ]]; then
  CONFIG_DIR=$(python3 -c "import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), 'model_executor/layers/quantization/utils/configs'))") \
    || { echo "launcher: cannot locate vLLM's kernel config folder"; exit 1; }
  if [[ ! -s $TUNED_ARCHIVE ]]; then
    # Fail instead of running on default configs: the trial would be labelled "tuned".
    echo "launcher: tuned_kernel_configs=true but $TUNED_ARCHIVE is missing or empty"; exit 1
  fi
  TUNED_FILES=$(tar -xzvf "$TUNED_ARCHIVE" -C "$CONFIG_DIR" --no-same-owner) \
    || { echo "launcher: extracting $TUNED_ARCHIVE failed"; exit 1; }
  echo "launcher: tuned kernel configs installed in $CONFIG_DIR:"
  echo "$TUNED_FILES" | sed 's/^/launcher:   /'
else
  echo "launcher: tuned_kernel_configs=$TUNED: vLLM's bundled/default kernel configs"
fi

PIDS=()
start_instance() {  # role idx gpu http_port nixl_port vllm_port kv_role
  local role=$1 idx=$2 gpu=$3 port=$4 nixl=$5 vport=$6 kvrole=$7
  local -a args=("$MODEL" --host 0.0.0.0 --port "$port" --served-model-name "${SERVED_BASE}-${role}" "$SERVED_BASE" "${COMMON[@]}")
  if [[ $role == prefill ]]; then args+=("${PREFILL_ARGS[@]}"); else args+=("${DECODE_ARGS[@]}"); fi
  if [[ -n $kvrole ]]; then
    args+=(--kv-transfer-config "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"$kvrole\",\"kv_buffer_device\":\"$BUFFER\",\"kv_load_failure_policy\":\"fail\"}")
  fi
  # Per-role environment from pd-config (<role>.env, KEY=VALUE per line). The file may be
  # absent: older rendered configs have none.
  local -a roleenv=()
  mapfile -t roleenv < <(read_args "$CFG/$role.env")
  echo "launcher: [$role-$idx] GPU $gpu port $port nixl $nixl env: ${roleenv[*]:-(none)}: vllm serve ${args[*]}"
  (
    for kv in "${roleenv[@]}"; do export "${kv?}"; done
    export CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl VLLM_PORT=$vport
    exec vllm serve "${args[@]}" 2>&1 | sed -u "s/^/[$role-$idx] /"
  ) &
  PIDS+=($!)
}

for ((i = 0; i < P; i++)); do
  start_instance prefill "$i" "$i" $((8100 + i)) $((5600 + i)) $((20000 + 100 * i)) kv_producer
done
for ((j = 0; j < D; j++)); do
  if (( P > 0 )); then kv=kv_consumer; else kv=""; fi
  start_instance decode "$j" $((P + j)) $((8200 + j)) $((5700 + j)) $((30000 + 100 * j)) "$kv"
done

shutdown() { echo "launcher: stopping all instances"; kill "${PIDS[@]}" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; wait; }
trap 'shutdown; exit 0' TERM INT

wait -n "${PIDS[@]}"
RC=$?
echo "launcher: an instance exited (rc=$RC) — stopping the others and failing the container"
shutdown
exit 1
