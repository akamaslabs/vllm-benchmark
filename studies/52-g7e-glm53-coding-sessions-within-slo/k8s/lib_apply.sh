# shellcheck shell=bash
# Helpers of apply_config.sh (study 52), sourced; kept apart so test_apply_config.sh can test them
# without a cluster. Needs NODE_ROLE (default llm-serving-g7e-24xl-spot).
NODE_ROLE=${NODE_ROLE:-llm-serving-g7e-24xl-spot}

# wait_for_node TIMEOUT_S: 0 as soon as a node with node-role=$NODE_ROLE reports Ready=True; 3 after
# TIMEOUT_S without one. A reclaimed spot node is replaced by the ASG within minutes, so Apply
# waits instead of failing in seconds (study 42's spot interruptions put the study in ERROR).
# NODE_POLL_S (default 30) is the poll interval.
wait_for_node() {
  local timeout=$1 t0=$SECONDS
  while :; do
    if kubectl get nodes -l node-role="$NODE_ROLE" --no-headers \
         -o custom-columns='N:.metadata.name,R:.status.conditions[?(@.type=="Ready")].status' 2>/dev/null \
         | awk '$2 == "True" { f = 1 } END { exit !f }'; then
      return 0
    fi
    [ $((SECONDS - t0)) -ge "$timeout" ] && { echo "no Ready node with node-role=$NODE_ROLE after ${timeout}s" >&2; return 3; }
    sleep "${NODE_POLL_S:-30}"
  done
}

# startup_failure_reason LOGFILE: the vLLM log line that explains a failed start (Mamba blocks
# below max_num_seqs, no KV block left after warmup, max_model_len above the cache, CUDA OOM),
# or nothing. Seen on the 2026-11-17 spike (runs 5-9).
startup_failure_reason() {
  grep -m1 -E 'exceeds available Mamba cache blocks|No available memory for the cache blocks|To serve at least one request with the model|CUDA out of memory|OutOfMemoryError' "$1" \
    | sed -E 's/^.*(ValueError|RuntimeError|torch\.OutOfMemoryError|OutOfMemoryError): //' | cut -c1-400
}
