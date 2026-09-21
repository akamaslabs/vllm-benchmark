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
# --timeout=4800s (80m), recomputed 2026-09-21 with the model swap and the longer ramp:
#   up to 900 s (15 min) one-time ShareGPT dataset prep on a cold cache — not expected,
#     since study 16 already generated inputs-qwen3-30b-a3b.json for this same
#     served-model-name and it is on the shared aiperf-results volume
# + 9 x 300 s levels (45 min, the 8 -> 2048 sweep)
# + ~15 min buffer for pip install, per-level dataset-file generation and misc overhead
#   = worst case comfortably under 80 min.
#
# This MUST stay below the Akamas RunTest task's own timeout, which is 95m in
# akamas/17-G7e-Speculative-Decoding-Goodput-Workflow.yaml. If Akamas kills the task
# first, the `kubectl logs` dump below never runs and the trial fails with no evidence
# of why — which is the whole point of the dump. Keep a margin when changing either.
set +e
kubectl wait --for=condition=complete job/aiperf-benchmark -n llm-benchmark --timeout=4800s
WAIT_EXIT=$?
set -e

echo "--- wait-for-vllm init container logs ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c wait-for-vllm --tail=200 || true
echo "--- aiperf container logs ---"
kubectl logs job/aiperf-benchmark -n llm-benchmark -c aiperf --tail=500 || true

exit $WAIT_EXIT
