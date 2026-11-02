#!/bin/bash
# Tappa B sequence, run from the toolbox. Each step logs to results/B<n>.out.
cd /tmp/study44/44-l4-pd-multinode
step() { local n=$1; shift; echo "=== $n start $(date -u +%T)"; "$@" > results/$n.out 2>&1; echo "=== $n exit=$? $(date -u +%T)"; }
step B0-iperf k8s/run.sh iperf l40s:a10g 120
step B2-ramp k8s/run.sh ramp 2.4 7200
step B4-up k8s/run.sh up l40s agg
step B4-ramp k8s/run.sh ramp 2.4 7200
PD_EXTRA_ARGS_AGGREGATED=--linear-backend=marlin step B5-up k8s/run.sh up a10g agg
step B5-ramp k8s/run.sh ramp 1.2 3600
k8s/run.sh down
echo "=== ALL DONE $(date -u +%T)"
