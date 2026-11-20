# shellcheck shell=bash
# Pure watchdog helpers of run_test.sh (study 38's, the threshold now WD_TTFT_MS and armed on the run's
# first completed request instead of a log marker). Sourced; no kubectl, no network. The caller sets
# WD_TTFT_MS (ms), WD_HOLD_S and WD_ARM_DELAY_S (s).

wd_over() {  # $1 server TTFT p95 in ms (empty or non-integer = no data). 0 if over.
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt "$WD_TTFT_MS" ]
}

wd_next_since() {  # $1 now, $2 current over-since ("" = under), $3 0 if over now. Prints the new over-since.
  if [ "$3" = 0 ]; then echo "${2:-$1}"; else echo ""; fi
}

wd_fired() {  # $1 now, $2 over-since. 0 once over for at least WD_HOLD_S.
  [ -n "${2:-}" ] && [ $(( $1 - $2 )) -ge "$WD_HOLD_S" ]
}

wd_ready() {  # $1 now, $2 when the first request completed ("" = not yet). 0 once the arm delay has passed.
  [ -n "${2:-}" ] && [ $(( $1 - $2 )) -ge "$WD_ARM_DELAY_S" ]
}
