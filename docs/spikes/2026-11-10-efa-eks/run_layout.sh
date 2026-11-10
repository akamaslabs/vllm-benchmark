#!/usr/bin/env bash
# EFA spike: start one multi-node vLLM layout on the two spike pods, check it answers, run load.py
# at two concurrencies and print the NCCL transport lines of both nodes.
#   ./run_layout.sh <single|tp2|tp2ep|pp2|dp2|dp2ep|dp2x|dp2xep> [TAG] [ENV=VALUE ...]
# Example (TCP control): ./run_layout.sh tp2 tp2-socket NCCL_NET=Socket
# Node 0 runs the API server (--node-rank 0), node 1 the headless workers (--node-rank 1 --headless);
# both point --master-addr at node 0. dp2x / dp2xep use --data-parallel-external-lb instead: each
# node runs its own API server (no --headless on node 1) and the load is split over both.
# Logs are copied to $OUT (default ./logs, not committed).
set -euo pipefail
NS=efa-spike; MODEL=openai/gpt-oss-20b
LAYOUT=${1:?layout}; TAG=${2:-$LAYOUT}; shift; shift || true; EXTRA_ENV="$*"
OUT=${OUT:-./logs}; mkdir -p "$OUT"
case "$LAYOUT" in
  single) PAR="" ;;
  tp2)    PAR="--tensor-parallel-size 2" ;;
  tp2ep)  PAR="--tensor-parallel-size 2 --enable-expert-parallel" ;;
  pp2)    PAR="--pipeline-parallel-size 2" ;;
  dp2)    PAR="--data-parallel-size 2" ;;
  dp2ep)  PAR="--data-parallel-size 2 --enable-expert-parallel" ;;
  dp2x)   PAR="--data-parallel-size 2 --data-parallel-external-lb" ;;
  dp2xep) PAR="--data-parallel-size 2 --data-parallel-external-lb --enable-expert-parallel" ;;
  *) echo "unknown layout $LAYOUT" >&2; exit 2 ;;
esac
x() { local p=$1; shift; kubectl -n "$NS" exec "efa-spike-$p" -- bash -c "$*"; }
P0=$(kubectl -n "$NS" get pod efa-spike-0 -o jsonpath='{.status.podIP}')
COMMON="--served-model-name spike --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000"
[ "$LAYOUT" = single ] || PAR="$PAR --nnodes 2 --master-addr $P0"

for p in 0 1; do x $p 'pkill -f "[v]llm serve" || true; for i in $(seq 30); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 1; done; pkill -9 -f "[v]llm|[m]ultiproc" || true; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader'; done
x 0 "nohup env NCCL_DEBUG=INFO $EXTRA_ENV vllm serve $MODEL $COMMON $PAR ${PAR:+--node-rank 0} > /tmp/vllm-$TAG.log 2>&1 &"
case "$LAYOUT" in dp2x*) HEADLESS="" ;; *) HEADLESS="--headless" ;; esac
[ "$LAYOUT" = single ] || x 1 "nohup env NCCL_DEBUG=INFO $EXTRA_ENV vllm serve $MODEL $COMMON $PAR --node-rank 1 $HEADLESS > /tmp/vllm-$TAG.log 2>&1 &"

T0=$(date +%s); READY=0
for i in $(seq 120); do
  if x 0 'curl -sf -o /dev/null http://localhost:8000/health' 2>/dev/null \
     && { [ -n "$HEADLESS" ] || x 1 'curl -sf -o /dev/null http://localhost:8000/health' 2>/dev/null; }; then READY=1; break; fi
  for p in 0 1; do
    [ "$LAYOUT" = single ] && [ $p = 1 ] && continue
    x $p 'pgrep -f "[v]llm serve" >/dev/null' || { echo "vllm died on node $p"; break 2; }
  done
  sleep 5
done
echo "LAYOUT $TAG ready=$READY after $(( $(date +%s) - T0 )) s"
if [ $READY = 1 ]; then
  kubectl -n "$NS" cp "$(dirname "$0")/load.py" efa-spike-0:/tmp/load.py >/dev/null
  if [ -z "$HEADLESS" ]; then
    kubectl -n "$NS" cp "$(dirname "$0")/load.py" efa-spike-1:/tmp/load.py >/dev/null
    for p in 0 1; do echo "--- node $p alone, conc 1"; x $p 'python3 /tmp/load.py http://localhost:8000 1 20'; done
    echo "--- both nodes, conc 16 each"
    x 0 'python3 /tmp/load.py http://localhost:8000 16 40' & x 1 'python3 /tmp/load.py http://localhost:8000 16 40' & wait
    for p in 0 1; do x $p 'curl -s http://localhost:8000/metrics | grep -E "^vllm:(cache_config_info|request_success_total)" | cut -c1-400'; done
  else
    x 0 'python3 /tmp/load.py http://localhost:8000 1 20; python3 /tmp/load.py http://localhost:8000 32 40'
    x 0 'curl -s http://localhost:8000/metrics | grep -E "^vllm:cache_config_info" | cut -c1-400'
  fi
fi
for p in 0 1; do
  x $p "cat /tmp/vllm-$TAG.log 2>/dev/null" > "$OUT/vllm-$TAG-node$p.log" || true
  echo "--- node $p transport"; grep -E 'NET/OFI Selected provider|Using network|via NET/' "$OUT/vllm-$TAG-node$p.log" | sed -E 's/^.*NCCL INFO //' | sort | uniq -c | sort -rn | head -6 || true
  echo "--- node $p errors"; grep -E 'Error|Traceback|ValueError|RuntimeError|AssertionError' "$OUT/vllm-$TAG-node$p.log" | tail -4 | cut -c1-300 || true
done
