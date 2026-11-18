#!/bin/bash
# Ramp task of study 52: find the most coding sessions this configuration serves within the SLO.
# Bisects LEVELS (sorted, ascending), one fresh AIPerf run per probed level (lib_load.sh run_level),
# and judges each level client-side over its steady window (lib_checks.sh check_level): TTFT p95 <=
# SLO_TTFT_MS and ITL p95 <= SLO_ITL_MS (each the worse of the two clients), errors < 1 %, >= 95 %
# completed, >= 95 % of the expected requests sent. A pass moves up, a fail moves down (AIPerf
# exiting non-zero with a RESULT line counts as failing; killed or with no RESULT, exit 6), assuming
# a configuration that fails a level fails every higher one. With the 7 levels of workload.env that
# is 3 levels at most (24, then 48 or 12, then the level between): about 54 min instead of up to
# 2 h 20 for the walk from the bottom that experiments 1-2 ran (2026-11-18, the 90-minute cap per
# experiment); the result of a configuration is the same level the walk would have found.
# Writes the highest passing level to LEVEL_FILE (default k8s/level.current) for run_hold.sh; if no
# level passes, the lowest level plus "best=none" (run_hold.sh exits 7).
# Prints RAMP level=<L> pass|fail <reason> per level and RESULT ramp best=<L|none> hold=<L>.
# Exit 6 if AIPerf was killed or crashed; 5 if vLLM restarted or was replaced (an OOM, or a
# reclaimed spot node, at a high level must not silently score a lower one); 3 if there is no
# Running vLLM pod.
# Env overrides for smoke runs: LEVELS (one level runs that level only), LEVEL_SECONDS, STEADY_FROM;
# LIB_LOAD and LEVEL_FILE for the local tests (test_ramp.sh).
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
read -r -a LV <<< "$LEVELS"
t "levels: $LEVELS sessions, bisected (${#LV[@]} levels: at most $(awk -v n="${#LV[@]}" 'BEGIN { k = 0; while (2 ^ k < n + 1) k++; print k }') runs); ${LEVEL_SECONDS} s each (+60 s grace), steady from ${STEADY_FROM} s; SLO client TTFT p95 <= ${SLO_TTFT_MS} ms and ITL p95 <= ${SLO_ITL_MS} ms"
BEST="" FIRST=${LV[0]}
LO=-1 HI=${#LV[@]}   # LV[LO] passed (-1: none yet), LV[HI] failed (past the top: not run)
while [ $((HI - LO)) -gt 1 ]; do
  MID=$(( (LO + HI) / 2 )); L=${LV[$MID]}
  OUT=$(run_level "$L" "$LEVEL_SECONDS" "ramp-$L"); RC=$?
  echo "$OUT"
  [ "$(restarts)" = "$R0" ] || die "vLLM restarted or was replaced during level $L: $R0 -> $(restarts)" 5
  # AIPerf killed (signal, e.g. OOM: rc >= 128) or crashed with no RESULT: a load-generator
  # failure, not an SLO verdict. Fail the task instead of scoring a lower level.
  if [ "$RC" -ge 128 ] || { [ "$RC" != 0 ] && ! echo "$OUT" | grep -q '^RESULT'; }; then
    die "AIPerf failed at level $L (exit $RC): load-generator failure, not an SLO verdict" 6
  fi
  if [ "$RC" != 0 ]; then
    echo "RAMP level=$L fail aiperf exited $RC"
    HI=$MID
  elif WHY=$(check_level "$OUT" "$(expected_sent "$L")" "$SLO_TTFT_MS" "$SLO_ITL_MS"); then
    echo "RAMP level=$L pass $WHY"
    LO=$MID
  else
    echo "RAMP level=$L fail $WHY"
    HI=$MID
  fi
done
[ "$LO" -ge 0 ] && BEST=${LV[$LO]}
HOLD=${BEST:-$FIRST}
# Line 1: the level to hold; line 2 "best=none" if no level passed (run_hold.sh then fails: a
# configuration that passes no level must not score as if it held the first one).
{ echo "$HOLD"; [ -z "$BEST" ] && echo "best=none"; } > "$LEVEL_FILE"
[ -n "$BEST" ] && [ "$BEST" = "${LEVELS##* }" ] && t "WARNING: the top level $BEST passed: the score is capped by the ramp, not by the SLO"
echo "RESULT ramp best=${BEST:-none} hold=$HOLD"
t "done"
