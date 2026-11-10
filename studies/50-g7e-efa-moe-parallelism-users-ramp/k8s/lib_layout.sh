# shellcheck shell=bash
# Layouts of study 50, sourced by apply_config.sh. One layout = how gpt-oss-120b is spread over the
# two GPU nodes (pod vllm-a-0 on node a, the head; vllm-b-0 on node b).
#
#   layout    replicas tp dp pp ep     pods  node b                      NCCL between the nodes
#   single        1     1  1  1 false  a     idle (no pod)               none
#   replicas      2     1  1  1 false  a b   its own vLLM, serving       none
#   dp2           1     1  2  1 false  a b   DP rank 1                   maybe (checked if seen)
#   dp2ep         1     1  2  1 true   a b   DP rank 1, half the experts required
#   tp2           1     2  1  1 false  a b   headless, TP rank 1         required
#   tp2ep         1     2  1  1 true   a b   headless, half the experts  required
#   pp2           1     1  1  2 false  a b   headless, second half of the layers  required
#
# The two DP layouts follow DP_MODE (one switch for both, so that they stay comparable):
#   external (default)  --data-parallel-external-lb: each node runs its own API server on its
#                       local engine, the load generator splits the users (one client per node);
#   internal            one API server on node a balancing both engines, node b headless.
# Hybrid load balancing needs more than one GPU per node (EFA spike, 2026-11-10).

# resolve_layout REPLICAS TP DP PP EP [DP_MODE]: sets
#   LAYOUT           the name above
#   PODS             "a" or "a b"
#   SERVING_B        true if pod b runs an API server (the clients then take one node each)
#   MULTINODE        1 if the two pods form one vLLM (--nnodes=2 --node-rank --master-addr)
#   WORKER_HEADLESS  1 if pod b gets --headless
#   NCCL_CHECK       required | warn | none: what apply_config.sh asks of the NCCL transport lines
#   LAYOUT_ARGS      the parallelism flags, the same on both pods (array, may be empty)
# Returns 1 (message on stderr) for any other combination.
resolve_layout() {
  local replicas=$1 tp=$2 dp=$3 pp=$4 ep=$5 dp_mode=${6:-external}
  LAYOUT="" PODS="a b" SERVING_B=false MULTINODE=1 WORKER_HEADLESS=1 NCCL_CHECK=required
  LAYOUT_ARGS=()
  case "$replicas/$tp/$dp/$pp/$ep" in
    1/1/1/1/false) LAYOUT=single   PODS="a" MULTINODE=0 WORKER_HEADLESS=0 NCCL_CHECK=none ;;
    2/1/1/1/false) LAYOUT=replicas SERVING_B=true MULTINODE=0 WORKER_HEADLESS=0 NCCL_CHECK=none ;;
    1/1/2/1/false) LAYOUT=dp2      NCCL_CHECK=warn LAYOUT_ARGS=(--data-parallel-size=2) ;;
    1/1/2/1/true)  LAYOUT=dp2ep    LAYOUT_ARGS=(--data-parallel-size=2 --enable-expert-parallel) ;;
    1/2/1/1/false) LAYOUT=tp2      LAYOUT_ARGS=(--tensor-parallel-size=2) ;;
    1/2/1/1/true)  LAYOUT=tp2ep    LAYOUT_ARGS=(--tensor-parallel-size=2 --enable-expert-parallel) ;;
    1/1/1/2/false) LAYOUT=pp2      LAYOUT_ARGS=(--pipeline-parallel-size=2) ;;
    *) echo "error: replicas=$replicas tp=$tp dp=$dp pp=$pp ep=$ep is not a layout of study 50 (two GPUs: replicas x tp x dp x pp <= 2 with one of them at 2, expert parallel only with tp or dp at 2)" >&2
       return 1 ;;
  esac
  case "$LAYOUT" in dp2|dp2ep)
    case "$dp_mode" in
      external) SERVING_B=true; WORKER_HEADLESS=0; LAYOUT_ARGS+=(--data-parallel-external-lb) ;;
      internal) ;;
      *) echo "error: DP_MODE '$dp_mode' is not external or internal" >&2; return 1 ;;
    esac ;;
  esac
}
