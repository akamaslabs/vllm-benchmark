#!/bin/bash
# Stop AIPerf step of study 54 (toolbox, Akamas Executor task after RunTest, alwaysRun: also after a
# failed RunTest, so the load always stops; critical false).
# For each client: SIGINT to the replay's top aiperf process (it cancels the requests in flight and
# writes its records), up to STOP_WAIT_S for its exit file, then anything left is killed; the run's
# per-request records (profile_export.jsonl) are saved gzipped on the toolbox as
# RUNS_DIR/<run>/c<i>.jsonl.gz, with the AIPerf log tail and the CPU / memory samples. Runs older than
# the last three are pruned from the pods by run_test.sh.
# Usage: bash stop_aiperf.sh [RUN]   (default: RUNS_DIR/current). Exit 0 both exports saved, 1 otherwise.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OV_RUNS_DIR=${RUNS_DIR-}
# shellcheck disable=SC1091
source "$HERE/workload.env"
[ -z "$OV_RUNS_DIR" ] || RUNS_DIR=$OV_RUNS_DIR
# shellcheck disable=SC1091
source "$HERE/lib_clients.sh"
POLL_S=${POLL_S:-5}; STOP_WAIT_S=${STOP_WAIT_S:-300}
t() { echo "$(date -u +%T) stop_aiperf: $*"; }
RUN=${1:-$(cat "$RUNS_DIR/current" 2>/dev/null)}
[ -n "$RUN" ] || { echo "error: no run name (RUNS_DIR/current missing)" >&2; exit 1; }
D=$RUNS_DIR/$RUN; mkdir -p "$D"
saved=0
for i in 0 1; do
  c=aiperf-client-$i
  PIDS=$(kx "$c" python3 -c "$STOPPER" 2>/dev/null | tr '\n' ' ')
  if [ -n "$PIDS" ]; then
    t "$c: SIGINT sent to aiperf (pid $PIDS)"
    T0=$SECONDS
    until [ -n "$(kx "$c" cat "/tmp/a13/runs/$RUN/exit" 2>/dev/null)" ] || [ $((SECONDS - T0)) -ge "$STOP_WAIT_S" ]; do sleep "$POLL_S"; done
    t "$c: replay exit '$(kx "$c" cat "/tmp/a13/runs/$RUN/exit" 2>/dev/null)' $((SECONDS - T0)) s after the SIGINT"
  else
    t "$c: no aiperf process running"
  fi
  kx "$c" python3 -c "$REAPER" 2>&1 | tail -1
  E=$(kx "$c" sh -c "find /tmp/a13/runs/$RUN -name profile_export.jsonl 2>/dev/null | head -1")
  if [ -n "$E" ] && kx "$c" gzip -c "$E" > "$D/c$i.jsonl.gz" && [ -s "$D/c$i.jsonl.gz" ]; then
    t "$c: records saved to $D/c$i.jsonl.gz ($(gzip -dc "$D/c$i.jsonl.gz" | wc -l | tr -d ' ') requests)"
    saved=$((saved + 1))
  else
    echo "error: $c: no profile_export.jsonl for run $RUN" >&2
    rm -f "$D/c$i.jsonl.gz"
  fi
  kx "$c" tail -n 300 "/tmp/a13/runs/$RUN/aiperf.log" > "$D/c$i-aiperf.log" 2>/dev/null || true
  kx "$c" cat "/tmp/a13/runs/$RUN/cgroup.log" > "$D/c$i-cgroup.log" 2>/dev/null || true
done
[ "$saved" = 2 ]
