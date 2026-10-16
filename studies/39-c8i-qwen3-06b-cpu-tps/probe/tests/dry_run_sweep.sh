#!/bin/bash
# Local dry run of the probe's sweep Job: renders sweep_job_template.yaml, extracts the
# container's shell script exactly as Kubernetes would run it (yq), points it at
# ../../k8s/tests/mock_openai.py and a local AIPerf 0.11.0, runs it, then feeds its log to
# summarize.py --sweep (Prometheus unreachable: prom_* columns empty) and to the summary.
# Checks the in-YAML shell (loop, heredoc, early stop) and the STEP_RESULT parsing on a
# real AIPerf export. Usage: DRY_VENV=<venv with aiperf==0.11.0> bash probe/tests/dry_run_sweep.sh
# (network the first time: pip and the Qwen3-0.6B tokenizer).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); P=$(dirname "$HERE"); K8S=$(dirname "$P")/k8s
W=$(mktemp -d /tmp/sweep-dry-XXXXXX)
VENV=${DRY_VENV:-$W/venv}
if [ ! -x "$VENV/bin/aiperf" ]; then python3 -m venv "$VENV"; "$VENV/bin/pip" install --quiet aiperf==0.11.0; fi
python3 "$K8S/tests/mock_openai.py" 18001 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
# Rates 2 and 4 for 60 s; a 2x-SLO stop threshold of 1 ms TTFT forces the early stop after
# the first step, so both the loop and the stop branch run.
KP_STOP_TTFT_MS=${DRY_STOP_TTFT_MS:-1} bash "$P/render_sweep_job.sh" "2 4" 60 "$W/job.yaml"
yq '.spec.template.spec.containers[0].args[0]' "$W/job.yaml" \
  | sed -e '/pip install/d' \
        -e "s|http://vllm-0.vllm-headless.llm-cpu.svc.cluster.local:8000|http://127.0.0.1:18001|" \
        -e "s|/benchmarks/|$W/benchmarks/|" \
        -e 's|--num-dataset-entries 8000|--num-dataset-entries 200|' > "$W/script.sh"
mkdir -p "$W/benchmarks"
PATH="$VENV/bin:$PATH" sh "$W/script.sh" > "$W/sweep-job.log" 2>&1 || { tail -30 "$W/sweep-job.log"; exit 1; }
grep -E '^STEP (START|FAILED|STOP)|SWEEP DONE' "$W/sweep-job.log"
mkdir -p "$W/out"
python3 "$P/summarize.py" --sweep "$W/sweep-job.log" http://127.0.0.1:9 > "$W/out/sweep.jsonl"
python3 "$P/summarize.py" "$W/out"
FAILS=0
[ "$(wc -l < "$W/out/sweep.jsonl" | tr -d ' ')" = 1 ] && echo "ok   one step (early stop after rate 2)" || { echo "FAIL steps: $(wc -l < "$W/out/sweep.jsonl")"; FAILS=1; }
grep -q '"ttft_p95_ms": [0-9]' "$W/out/sweep.jsonl" && echo "ok   AIPerf TTFT p95 parsed" || { echo "FAIL TTFT p95"; FAILS=1; }
grep -q '"tpot_p95_ms": [0-9]' "$W/out/sweep.jsonl" && echo "ok   AIPerf TPOT p95 parsed" || { echo "FAIL TPOT p95"; FAILS=1; }
grep -q '"total_tok_s": [0-9]' "$W/out/sweep.jsonl" && echo "ok   total tok/s parsed" || { echo "FAIL total tok/s"; FAILS=1; }
[ $FAILS = 0 ]
