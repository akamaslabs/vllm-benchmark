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

# gmu_matches CACHE_CONFIG_INFO_LINE GMU: 0 when the line's gpu_memory_utilization label equals GMU
# as a number. vLLM writes the label as Python str(float) at full precision (0.9017631142560059) and
# Akamas renders optimizer values at full precision too, so a printf-%g string match fails on most
# optimize experiments (final review C1).
gmu_matches() {
  echo "$1" | awk -v g="$2" '{ if (match($0, /gpu_memory_utilization="[^"]*"/)) {
      v = substr($0, RSTART + 24, RLENGTH - 25); d = v - g; if (d < 0) d = -d; exit !(d < 1e-9) } exit 1 }'
}

# max_seqs_hint LOGFILE: the max_num_seqs vLLM says would fit ("lower max_num_seqs to at most N"), or
# nothing. The smoke pins the baseline's max_num_seqs from it (final review I2).
max_seqs_hint() {
  grep -m1 -oE 'max_num_seqs to at most [0-9]+' "$1" | grep -oE '[0-9]+$'
}

# crash_reason POD: 0 (and the reason line) when the pod's previous container log names a start-up
# failure that will repeat on every restart, so Apply stops at the first crash instead of the third
# (final review I4).
crash_reason() {
  local f r; f=$(mktemp)
  kubectl -n "${NS:-glm53-coding}" logs "$1" --previous > "$f" 2>/dev/null
  r=$(startup_failure_reason "$f"); rm -f "$f"
  [ -n "$r" ] && { echo "$r"; return 0; }
  return 1
}

# node_lost POD: 0 when the failed pod went down with its node (node gone or NotReady, or a node
# shutdown / termination reason): a reclaimed spot node, not a configuration that does not start
# (final review I5).
node_lost() {
  local nr node reason ready
  nr=$(kubectl -n "${NS:-glm53-coding}" get pod "$1" -o jsonpath='{.spec.nodeName}/{.status.reason}' 2>/dev/null)
  node=${nr%%/*}; reason=${nr#*/}
  case "$reason" in NodeShutdown|Shutdown|Terminated|NodeLost) return 0 ;; esac
  [ -n "$node" ] || return 1
  ready=$(kubectl get node "$node" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null) || return 0
  [ "$ready" != True ]
}
