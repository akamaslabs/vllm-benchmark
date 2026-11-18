#!/bin/bash
# Load task of study 53: one fixed load of long, mostly idle coding sessions (workload.env: SESSIONS
# of ~125-130k tokens, one turn each every SECONDS_PER_TURN s) for LOAD_SECONDS (+60 s grace), from
# the two AIPerf clients (lib_load.sh run_level, label load-<SESSIONS>: the same seed in every
# experiment). The score is server-side in Akamas (TTFT average over the trimmed window, request rate
# and TPOT as constraints); this script prints the client view for the README:
#   the clients' summaries, the merged RESULT, TURNS (first turns apart from resumed turns, each count
#   and TTFT p50 / p95 over requests started >= STEADY_FROM s, the worse client), resumed-turn TTFT per
#   5 minutes, and RESULT load sessions=<n> sent=<n> completed=<n> resumed_ttft_p95_ms=<x>.
# A slow configuration (every resumed turn recomputed) or one that falls behind still exits 0: it
# must score badly, not fail (design). Exit 6 if AIPerf was killed, crashed with no RESULT, or nothing
# completed in the window; 5 if vLLM restarted or was replaced; 3 if there is no Running vLLM pod.
# Env overrides for smoke runs: LOAD_SECONDS, STEADY_FROM; LIB_LOAD for the local tests
# (test_run_load.sh).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export SCRIPT_NAME=run_load
OV_LOAD_SECONDS=${LOAD_SECONDS:-} OV_STEADY_FROM=${STEADY_FROM:-}
# shellcheck disable=SC1091
source "$HERE/workload.env"
LOAD_SECONDS=${OV_LOAD_SECONDS:-$LOAD_SECONDS} STEADY_FROM=${OV_STEADY_FROM:-$STEADY_FROM}
# expected_sent (lib_load.sh) reads LEVEL_SECONDS: the load's own length.
LEVEL_SECONDS=$LOAD_SECONDS
# shellcheck disable=SC1090
source "${LIB_LOAD:-$HERE/lib_load.sh}"

R0=$(restarts)
t "load: $SESSIONS sessions for ${LOAD_SECONDS} s (+60 s grace), summarized from ${STEADY_FROM} s"
OUT=$(run_level "$SESSIONS" "$LOAD_SECONDS" "load-$SESSIONS"); RC=$?
echo "$OUT"
[ "$(restarts)" = "$R0" ] || die "vLLM restarted or was replaced during the load: $R0 -> $(restarts)" 5
if [ "$RC" -ge 128 ] || ! echo "$OUT" | grep -q '^RESULT'; then
  die "AIPerf failed (exit $RC): load-generator failure, not a result" 6
fi
[ "$RC" = 0 ] || echo "NOTE: AIPerf exited $RC with a RESULT line"
RES=$(echo "$OUT" | grep '^RESULT' | tail -1)
TURNS=$(echo "$OUT" | merge_turns)
echo "${TURNS:-TURNS missing: a client printed no TURNS line}"
val() { echo "$1" | tr ' ' '\n' | awk -F= -v k="$2" '$1 == k { print $2 }'; }
SENT=$(val "$RES" sent) COMP=$(val "$RES" completed) RP95=$(val "$TURNS" resumed_ttft_p95_ms)
[ "${COMP:-0}" -gt 0 ] || die "nothing completed in the window (sent ${SENT:-0})" 6
awk -v c="$COMP" -v s="$SENT" 'BEGIN { exit !(c < 0.95 * s) }' && echo "NOTE: completed $COMP of $SENT requests in the window: the load fell behind"
[ -n "$RP95" ] && [ "$RP95" != NA ] && awk -v x="$RP95" -v slo="$SLO_TTFT_MS" 'BEGIN { exit !(x > slo) }' \
  && echo "NOTE: resumed turns TTFT p95 $RP95 ms above the $SLO_TTFT_MS ms SLO"
echo "RESULT load sessions=$SESSIONS sent=$SENT completed=$COMP resumed_ttft_p95_ms=${RP95:-NA}"
t "done"
