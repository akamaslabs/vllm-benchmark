# Pure watchdog helpers for run_test.sh (sourced; no kubectl, no network).
# The caller sets WD_TTFT_MS, WD_ITL_MS (ms), WD_HOLD_S and WD_ARM_DELAY_S (s).
# shellcheck shell=bash
WD_MARKER="MEASURED RUN START"

wd_over() {  # $1 TTFT p95 ms, $2 ITL p95 ms (empty or non-integer = no data). 0 if either is over.
  { [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt "$WD_TTFT_MS" ]; } || \
  { [[ "${2:-}" =~ ^[0-9]+$ ]] && [ "$2" -gt "$WD_ITL_MS" ]; }
}

wd_next_since() {  # $1 now, $2 current over-since ("" = under), $3 0 if over now. Prints the new over-since.
  if [ "$3" = 0 ]; then echo "${2:-$1}"; else echo ""; fi
}

wd_fired() {  # $1 now, $2 over-since. 0 once over for at least WD_HOLD_S.
  [ -n "${2:-}" ] && [ $(( $1 - $2 )) -ge "$WD_HOLD_S" ]
}

wd_armed() {  # $1 the vllm-0 Job's log text. 0 once the measured run has started.
  grep -qF "$WD_MARKER" <<<"$1"
}

wd_ready() {  # $1 now, $2 when the marker was first seen ("" = not yet). 0 once the arm delay has passed.
  [ -n "${2:-}" ] && [ $(( $1 - $2 )) -ge "$WD_ARM_DELAY_S" ]
}
