#!/bin/bash
# Study 44 engine entrypoint: ONE vLLM process per pod, one GPU, one role. Study 27's
# launcher.sh started every process in one container (one PID namespace, so UCX could use
# cuda_ipc); here prefill and decode are on different nodes, so the KV goes over the network
# and each role gets its own pod.
#
# Inputs:
#   PD_ROLE        prefill | decode | aggregated          (pod env)
#   PD_ARGS_NAME   optional: read <name>.args / <name>.env instead of <role>.* (tappa C: the
#                  GPU, l40s | a10g, so one component per GPU whatever its role)
#   PD_SERVED_NAME optional: first served model name (tappa C: qwen3-8b-l40s | qwen3-8b-a10g),
#                  the model_name label of this process's metrics
#   POD_IP         downward API status.podIP              (pod env)
#   /pd-config/    ConfigMap pd-config: topology.env, common.args, <role>.args, <role>.env
#   /pd-tuned-configs/tuned-configs.tgz   optional ConfigMap pd-tuned-configs
#
# NIXL: the prefill advertises VLLM_NIXL_SIDE_CHANNEL_HOST in the kv_transfer_params it
# returns (vllm v0.29.0 nixl/pull_scheduler.py: remote_host=self.side_channel_host), and the
# decode connects there to pull the KV. So it must be the pod IP, reachable from the other
# node, not study 27's 127.0.0.1. The KV data itself moves over UCX (tcp with
# kv_buffer_device=cpu, the only path without RDMA); UCX_PROTO_INFO=y logs the chosen
# transport at the first transfer.
set -uo pipefail
MODEL=${PD_MODEL:-Qwen/Qwen3-8B-FP8}
SERVED_BASE=${PD_SERVED_BASE:-qwen3-8b}
CFG=${PD_CONFIG_DIR:-/pd-config}
ROLE=${PD_ROLE:?PD_ROLE must be prefill, decode or aggregated}
PORT=${PD_PORT:-8000}
TUNED_ARCHIVE=${PD_TUNED_ARCHIVE:-/pd-tuned-configs/tuned-configs.tgz}

set -a
# shellcheck disable=SC1091
source "$CFG/topology.env"
set +a
CONNECTOR=${PD_KV_CONNECTOR:-NixlConnector}
BUFFER=${PD_KV_BUFFER_DEVICE:-cpu}
TUNED=${PD_TUNED_KERNEL_CONFIGS:-false}

read_args() { grep -vE '^[[:space:]]*(#|$)' "$1" 2>/dev/null | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'; }
mapfile -t COMMON < <(read_args "$CFG/common.args")
ARGS_NAME=${PD_ARGS_NAME:-$ROLE}
mapfile -t ROLE_ARGS < <(read_args "$CFG/$ARGS_NAME.args")
mapfile -t ROLE_ENV < <(read_args "$CFG/$ARGS_NAME.env")

case $ROLE in
  prefill) KV_ROLE=kv_producer; NAME=prefill ;;
  decode) KV_ROLE=kv_consumer; NAME=decode ;;
  aggregated) KV_ROLE=""; NAME=decode ;;   # same served name as study 27's aggregated replicas
  *) echo "serve: unknown PD_ROLE '$ROLE'"; exit 1 ;;
esac

nvidia-smi -L || true
echo "serve: role=$ROLE pod_ip=${POD_IP:-?} node=${NODE_NAME:-?} connector=$CONNECTOR kv_buffer_device=$BUFFER tuned=$TUNED"

if [[ $TUNED == true ]]; then
  CONFIG_DIR=$(python3 -c "import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), 'model_executor/layers/quantization/utils/configs'))") \
    || { echo "serve: cannot locate vLLM's kernel config folder"; exit 1; }
  [[ -s $TUNED_ARCHIVE ]] || { echo "serve: tuned_kernel_configs=true but $TUNED_ARCHIVE is missing"; exit 1; }
  tar -xzvf "$TUNED_ARCHIVE" -C "$CONFIG_DIR" --no-same-owner | sed 's/^/serve: tuned config /' \
    || { echo "serve: extracting $TUNED_ARCHIVE failed"; exit 1; }
fi

SERVED=${PD_SERVED_NAME:-${SERVED_BASE}-${NAME}}
args=("$MODEL" --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED" "$SERVED_BASE" "${COMMON[@]}" "${ROLE_ARGS[@]}")
if [[ -n $KV_ROLE ]]; then
  # PD_KV_EXTRA_CONFIG (topology.env, optional): JSON object passed as kv_connector_extra_config,
  # e.g. {"num_threads":16} (NIXL UCX worker threads, vllm v0.29.0 nixl/base_worker.py, default 4).
  extra=""
  [[ -n ${PD_KV_EXTRA_CONFIG:-} ]] && extra=",\"kv_connector_extra_config\":$PD_KV_EXTRA_CONFIG"
  args+=(--kv-transfer-config "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"$KV_ROLE\",\"kv_buffer_device\":\"$BUFFER\",\"kv_load_failure_policy\":\"fail\"$extra}")
fi
for kv in "${ROLE_ENV[@]}"; do export "${kv?}"; done
export VLLM_NIXL_SIDE_CHANNEL_HOST=${POD_IP:?POD_IP not set} VLLM_NIXL_SIDE_CHANNEL_PORT=${PD_NIXL_PORT:-5600}
echo "serve: env: ${ROLE_ENV[*]:-(none)} VLLM_NIXL_SIDE_CHANNEL_HOST=$VLLM_NIXL_SIDE_CHANNEL_HOST VLLM_NIXL_SIDE_CHANNEL_PORT=$VLLM_NIXL_SIDE_CHANNEL_PORT"
echo "serve: vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
