# shellcheck shell=bash
# Client-side trial checks for run_ramp.sh / run_hold.sh (study 48, from study 47). Sourced, not executed.
# check_result SUMMARY EXPECTED_SENT: 0 if the steady window is healthy, 1 otherwise (reason on
# stdout). Fails when no request fell in the window, errors >= 1 %, completed < 95 % of sent, or
# sent < 95 % of EXPECTED_SENT = (DURATION - STEADY_FROM) x RATE: AIPerf's user-centric mode makes
# each user wait for its previous answer, so a configuration that falls behind sends fewer
# requests instead of leaving them uncompleted (final review 2026-11-05, finding 3).
check_result() {
  echo "$1" | awk -v exp_sent="$2" '
    /^RESULT/ { found = 1
      for (k = 2; k <= NF; k++) { split($k, a, "="); v[a[1]] = a[2] }
      if (v["sent"] == 0) { print "no request in the steady window"; bad = 1; exit }
      if (v["errors"] / v["sent"] >= 0.01) { print "errors " v["errors"] " / " v["sent"] " >= 1 %"; bad = 1; exit }
      if (v["completed"] / v["sent"] < 0.95) { print "completed " v["completed"] " / " v["sent"] " < 95 %"; bad = 1; exit }
      if (v["sent"] < 0.95 * exp_sent) { print "sent " v["sent"] " < 95 % of the expected " exp_sent ": the load fell behind"; bad = 1; exit }
    }
    END { if (!found) { print "no RESULT line"; exit 1 } exit bad }'
}

# check_level SUMMARY EXPECTED_SENT SLO_MS (study 48 Ramp): the trial checks above, then client
# TTFT p95 <= SLO_MS. 0 = the level passes; 1 = it fails (reason on stdout). A missing or
# non-numeric TTFT fails.
check_level() {
  local why
  why=$(check_result "$1" "$2") || { echo "$why"; return 1; }
  echo "$1" | awk -v slo="$3" '
    /^RESULT/ { for (k = 2; k <= NF; k++) { split($k, a, "="); v[a[1]] = a[2] } }
    END {
      x = v["ttft_p95_ms"]
      if (x !~ /^[0-9]+(\.[0-9]+)?$/) { print "client TTFT p95 not available (" x ")"; exit 1 }
      if (x + 0 > slo + 0) { print "client TTFT p95 " x " ms > " slo " ms"; exit 1 }
      print "client TTFT p95 " x " ms <= " slo " ms"
    }'
}
