#!/bin/bash
# Hold task of study 52: re-run, fresh, the level the Ramp task chose (k8s/level.current) for
# HOLD_SECONDS (+60 s grace). The study windows on this task (trim 520s / 90s: AIPerf start-up +
# STEADY_FROM warm-up, then grace and log dump), and Akamas scores sessions = 30 x completed
# requests/s from Prometheus; server TTFT p95 <= 6 s and ITL p95 <= 60 ms are Akamas constraints.
# This script fails the trial (exit 6) only if errors >= 1 %, completed < 95 % or the load fell
# behind (the SLO itself was judged by the Ramp), exit 5 if vLLM restarted or was replaced, 4 if
# level.current is missing, 7 if no Ramp level met the SLO.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export SCRIPT_NAME=run_hold
OV_LEVEL_SECONDS=${LEVEL_SECONDS:-} OV_STEADY_FROM=${STEADY_FROM:-}
# shellcheck disable=SC1091
source "$HERE/workload.env"
# expected_sent reads LEVEL_SECONDS: the hold's own length.
LEVEL_SECONDS=${OV_LEVEL_SECONDS:-$HOLD_SECONDS} STEADY_FROM=${OV_STEADY_FROM:-$STEADY_FROM}
# shellcheck disable=SC1091
source "$HERE/lib_checks.sh"
# shellcheck disable=SC1090
source "${LIB_LOAD:-$HERE/lib_load.sh}"
LEVEL_FILE=${LEVEL_FILE:-$HERE/level.current}

[ -s "$LEVEL_FILE" ] || die "no $LEVEL_FILE: the Ramp task did not complete" 4
grep -qx 'best=none' "$LEVEL_FILE" && die "no Ramp level met the SLO: nothing to hold (the experiment must not score)" 7
L=$(head -1 "$LEVEL_FILE")
R0=$(restarts)
t "holding $L sessions for ${LEVEL_SECONDS} s (+60 s grace)"
OUT=$(run_level "$L" "$LEVEL_SECONDS" "hold-$L"); RC=$?
echo "$OUT"
[ "$(restarts)" = "$R0" ] || die "vLLM restarted or was replaced during the hold: $R0 -> $(restarts)" 5
[ "$RC" = 0 ] || die "aiperf exited $RC" 6
CHECK=$(check_result "$OUT" "$(expected_sent "$L")") || die "client-side check failed ($CHECK)" 6
# Reporting only: the score and the constraints are server-side in Akamas.
echo "$OUT" | awk -v slo="$SLO_TTFT_MS" -v slo_itl="$SLO_ITL_MS" '/^RESULT/ { for (k = 2; k <= NF; k++) { split($k, a, "="); v[a[1]] = a[2] }
  if (v["ttft_p95_ms"] + 0 > slo + 0) print "NOTE: hold client TTFT p95 " v["ttft_p95_ms"] " ms above the " slo " ms SLO (Ramp passed this level)"
  if (v["itl_p95_ms"] + 0 > slo_itl + 0) print "NOTE: hold client ITL p95 " v["itl_p95_ms"] " ms above the " slo_itl " ms SLO (Ramp passed this level)" }'
echo "RESULT hold sessions=$L"
t "done"
