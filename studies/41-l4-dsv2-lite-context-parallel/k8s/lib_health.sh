# Pure decisions for apply_config.sh (sourced; no kubectl): the health check after the warm-up
# requests and the GPU-busy guard before the start.
# shellcheck shell=bash

replicas_healthy() {  # $1 replicas asked for, $2 ready replicas, $3 highest restart count, $4 warm-up failures.
  # 0 if healthy; otherwise prints the reason and returns 1. A replica that restarted (a
  # CUDA out-of-memory at the KV-allocation edge, most likely: the optimizer explores
  # gpu_memory_utilization up to 0.94 of each L4) or failed its warm-up would be
  # benchmarked in a state that is not the configuration under test, or not at all.
  if [ "${2:-0}" != "$1" ]; then echo "${2:-0} of $1 replica(s) ready after the warm-up"; return 1; fi
  if [ "${3:-0}" -gt 0 ]; then echo "a replica restarted during startup or the warm-up (restartCount ${3})"; return 1; fi
  if [ "${4:-0}" -gt 0 ]; then echo "${4} replica(s) failed the warm-up requests"; return 1; fi
  return 0
}

gpu_busy_pods() {  # stdin: "namespace/name:gpus" lines of the pods on the GPU node; $1 our "namespace/pod-prefix".
  # Prints the pods that hold GPUs and are not ours (empty output = the node's GPUs are free
  # for this study). Two engines on the same L4s would share their memory: the second one
  # fails at the KV allocation, or both run and the measurement is meaningless.
  local line ns_name gpus
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    ns_name=${line%%:*}; gpus=${line#*:}
    [[ "$gpus" =~ [1-9] ]] || continue
    [[ "$ns_name" == "$1"* ]] && continue
    echo "$ns_name"
  done
}
