#!/bin/bash
# Ramp task of study 48: find the most users this configuration serves within the SLO. Walks
# LEVELS in order, one fresh AIPerf run per level (lib_load.sh run_level), and judges each level
# client-side over its steady window (lib_checks.sh check_level): TTFT p95 <= SLO_TTFT_MS,
# errors < 1 %, >= 95 % completed, >= 95 % of the expected requests sent. Stops at the first
# failing level (an AIPerf failure counts as failing). Writes the last passing level to
# LEVEL_FILE (default k8s/level.current) for run_hold.sh; if no level passes, the first level
# (Hold then runs there and Akamas' constraints mark the experiment).
# Prints RAMP level=<L> pass|fail <reason> per level and RESULT ramp best=<L|none> hold=<L>.
# Exit 5 if vLLM restarted or was replaced (an OOM at a high level must not silently score a
# lower one), 3 if there is no Running vLLM pod.
# Env overrides for smoke runs: LEVELS, LEVEL_SECONDS, STEADY_FROM; LIB_LOAD and LEVEL_FILE for
# the local tests (test_ramp.sh).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export SCRIPT_NAME=run_ramp
OV_LEVELS=${LEVELS:-} OV_LEVEL_SECONDS=${LEVEL_SECONDS:-} OV_STEADY_FROM=${STEADY_FROM:-}
# shellcheck disable=SC1091
source "$HERE/workload.env"
LEVELS=${OV_LEVELS:-$LEVELS} LEVEL_SECONDS=${OV_LEVEL_SECONDS:-$LEVEL_SECONDS} STEADY_FROM=${OV_STEADY_FROM:-$STEADY_FROM}
# shellcheck disable=SC1091
source "$HERE/lib_checks.sh"
# shellcheck disable=SC1090
source "${LIB_LOAD:-$HERE/lib_load.sh}"
LEVEL_FILE=${LEVEL_FILE:-$HERE/level.current}

R0=$(restarts)
t "levels: $LEVELS; ${LEVEL_SECONDS} s each (+60 s grace), steady from ${STEADY_FROM} s; SLO client TTFT p95 <= ${SLO_TTFT_MS} ms"
BEST="" FIRST=""
for L in $LEVELS; do
  FIRST=${FIRST:-$L}
  OUT=$(run_level "$L" "$LEVEL_SECONDS" "ramp-$L"); RC=$?
  echo "$OUT"
  [ "$(restarts)" = "$R0" ] || die "vLLM restarted or was replaced during level $L: $R0 -> $(restarts)" 5
  if [ "$RC" != 0 ]; then
    echo "RAMP level=$L fail aiperf exited $RC"
    break
  fi
  if WHY=$(check_level "$OUT" "$(expected_sent "$L")" "$SLO_TTFT_MS"); then
    echo "RAMP level=$L pass $WHY"
    BEST=$L
  else
    echo "RAMP level=$L fail $WHY"
    break
  fi
done
HOLD=${BEST:-$FIRST}
echo "$HOLD" > "$LEVEL_FILE"
[ -n "$BEST" ] && [ "$BEST" = "${LEVELS##* }" ] && t "WARNING: the top level $BEST passed: the score is capped by the ramp, not by the SLO"
echo "RESULT ramp best=${BEST:-none} hold=$HOLD"
t "done"
