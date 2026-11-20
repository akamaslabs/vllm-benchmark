#!/bin/bash
# prom_query.sh 'PROMQL': study 54's instant Prometheus query (run_test.sh's watchdog). Prints the first
# result's value rounded to an integer, or nothing (no series, NaN, Inf, unreachable). Tries PROM_URL
# (in-cluster address, the toolbox), then the API server's service proxy (kubectl get --raw, a laptop).
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$HERE/workload.env"
Q=$(python3 -I -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$1")
R=$(curl -s -m 10 "$PROM_URL/api/v1/query?query=$Q" 2>/dev/null)
[ -n "$R" ] || R=$(kubectl get --raw "/api/v1/namespaces/${PROM_SVC%%/*}/services/${PROM_SVC#*/}/proxy/api/v1/query?query=$Q" 2>/dev/null)
echo "$R" | python3 -I -c '
import json, math, sys
try:
    v = json.load(sys.stdin)["data"]["result"]
    x = float(v[0]["value"][1]) if v else float("nan")
except (ValueError, KeyError, IndexError):
    x = float("nan")
if math.isfinite(x):
    print(int(round(x)))'
