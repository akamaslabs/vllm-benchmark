#!/bin/bash
# Load task of study 49: the customer at 128 closed-loop users and every neighbour at its own load,
# all at once (lib_load.sh run_all), for a 20 s warm-up + LOAD_SECONDS. The study windows on this
# task: Akamas reads the customer's TTFT/ITL p95 (constraints) and its slice's framebuffer (goal)
# from Prometheus. This script only decides whether the trial is usable (lib_checks.sh):
#   exit 5  a vLLM pod (customer or neighbour) restarted or was replaced during the load
#   exit 6  the customer completed < 95 % or has no result, or a neighbour was not really busy
#           (errors >= 1 %, completed < 95 %, no result): measured next to an idle neighbour
# The customer missing its SLA is NOT a script failure: Akamas marks it CONSTRAINTS_VIOLATED.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$HERE/lib_load.sh"
# shellcheck disable=SC1091
source "$HERE/lib_checks.sh"

R0=$(restarts)
t "layout $PROFILE: customer $CUSTOMER_USERS users, $NEIGHBOURS neighbour(s) at $NEIGHBOUR_USERS users each, ${LOAD_SECONDS} s after a 20 s warm-up"
OUT=$(run_all)
echo "$OUT"
R1=$(restarts)
# Full AIPerf output of every client, so the Akamas task log is enough to debug.
for i in $(seq 0 $((CLIENTS - 1))); do
  echo "--- aiperf-client-$i: load log ---"
  kubectl -n $NS exec aiperf-client-$i -- cat /work/out/load.log 2>/dev/null | LC_ALL=C tr '\r' '\n' | grep -v 'Profiling:\|Grace Period' | tail -80 || true
done
[ "$R1" = "$R0" ] || die "a vLLM pod restarted or was replaced during the load: $R0 -> $R1" 5
WHY=$(check_clients "$OUT") || die "trial not usable: $(echo "$WHY" | tr '\n' ';')" 6
echo "$OUT" | awk -v tm=$TTFT_MAX -v im=$ITL_MAX '/role=customer/ { for (k = 2; k <= NF; k++) { split($k, a, "="); v[a[1]] = a[2] }
  sla = (v["ttft_p95"] + 0 <= tm && v["itl_p95"] + 0 <= im) ? "met" : "MISSED"
  print "RESULT customer users=" v["users"] " ttft_p95_ms=" v["ttft_p95"] " itl_p95_ms=" v["itl_p95"] " client SLA " sla }'
echo "$OUT" | awk '/role=neighbour/ { n++; for (k = 2; k <= NF; k++) { split($k, a, "="); if (a[1] == "ttft_p95" && a[2] + 0 > m) m = a[2] + 0 } }
  END { if (n) print "RESULT neighbours n=" n " worst_ttft_p95_ms=" m; else print "RESULT neighbours n=0" }'
t "done"
