#!/bin/bash

BENCH_FILE=/work/vllm-benchmark/studies/17-g7e-speculative-decoding-goodput/k8s/05-job.yaml

# No ConfigMap to apply separately — the load pattern is just CLI flags on the Job's
# own command, so re-applying the Job manifest each run is enough for a manual edit
# to 05-job.yaml (e.g. a recalibrated concurrency list) to take effect on the next trial.
kubectl delete -f "$BENCH_FILE" ; kubectl apply -f "$BENCH_FILE"

# Same rationale as apply_config.sh's vLLM log dump: don't exit immediately on a failed
# wait — print the job's own container logs first, so they land in this task's stdout
# and show up in the Akamas UI without needing separate kubectl access.
#
# --timeout=4200s (70m), RECOMPUTED for study 17 (was 5700s/95m in the study this was
# copied from): up to 900s (15min) one-time dataset-prep on a cold cache + this study's
# 8 x 300s levels (40min, down from 12 x 300s = 60min) + ~15min buffer for pip-install,
# per-level dataset-file generation and misc overhead = worst case under 70m.
#
# This MUST stay below the Akamas RunTest task's own timeout (80m, see
# akamas/17-G7e-Speculative-Decoding-Goodput-Workflow.yaml). If Akamas kills the task
# first, the `kubectl logs` dump below never runs and the trial fails with no evidence
# of why — the whole point of the dump. Keep a margin when changing either number.
set +e
kubectl wait --for=condition=complete job/aiperf-benchmark -n llm-benchmark --timeout=4200s
WAIT_EXIT=$?
set -e

echo "--- wait-for-vllm init container logs ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c wait-for-vllm --tail=200 || true
echo "--- aiperf container logs ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c aiperf --tail=500 || true

exit $WAIT_EXIT
