# shellcheck shell=bash
# Client-side trial checks of study 49's Load task (run_load.sh). Sourced, not executed.
# check_clients SUMMARY: SUMMARY has one line per AIPerf client (lib_load.sh run_all):
#   C<i> role=<customer|neighbour> users=<u> ttft_p95=<ms> itl_p95=<ms> requests=<ok> errors=<n>
# rc 0 if the trial is usable, 1 otherwise (reasons on stdout):
#   - customer: missing, no result, or completed (requests / (requests + errors)) < 95 %;
#     its TTFT/ITL are NOT checked here: they are the Akamas constraints;
#   - neighbour: no result, or errors >= 1 % or completed < 95 %: a neighbour that is not
#     really busy would make the customer's slice look better than it is (design anti-goal).
check_clients() {
  echo "$1" | awk '
    /^C[0-9]+ / {
      delete v
      for (k = 2; k <= NF; k++) { split($k, a, "="); v[a[1]] = a[2] }
      tot = v["requests"] + v["errors"]
      if (v["role"] == "customer") seen = 1
      if (tot == 0 || v["ttft_p95"] == "NA") { print $1 " (" v["role"] "): no result"; bad = 1; next }
      if (v["requests"] / tot < 0.95) { print $1 " (" v["role"] "): completed " v["requests"] " / " tot " < 95 %"; bad = 1; next }
      if (v["role"] == "neighbour" && v["errors"] / tot >= 0.01) { print $1 " (neighbour): errors " v["errors"] " / " tot " >= 1 %"; bad = 1 }
    }
    END {
      if (!seen) { print "no customer client line"; bad = 1 }
      exit bad
    }'
}
