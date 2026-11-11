#!/bin/bash
# ab.sh <label> [users=448] [seconds=300]
# Runs in the aiperf pod: one fresh AIPerf run at a saturating level, the same seed every time, to
# compare vLLM with the Nsight service and the CUDA 13.5 libcuda (label "nsight") against plain
# vLLM on the host's libcuda with no service (label "plain"). Output /work/out/ab/<label>/run.
set -u
L=$1; U=${2:-448}; D=${3:-300}; out=/work/out/ab/$L; mkdir -p "$out"
rate=$(awk -v u="$U" 'BEGIN{printf "%.4f", u/64}')
echo "$(date -u +%H:%M:%S) $L start: $U users, $D s"
aiperf profile --model gpt-oss-120b --tokenizer openai/gpt-oss-120b \
  --url http://vllm.nsight-probe.svc:8000 --endpoint-type chat --streaming \
  --isl 600 --isl-stddev 0 --osl 200 --osl-stddev 0 --extra-inputs ignore_eos:true reasoning_effort:low \
  --conversation-turn-mean 10 --conversation-turn-stddev 0 --conversation-num 600 --random-seed 4242 \
  --user-centric-rate "$rate" --num-users "$U" --benchmark-duration "$D" --benchmark-grace-period 30 \
  --slice-duration 60 --ui simple --output-artifact-dir "$out/run" > "$out/run.log" 2>&1
echo "$(date -u +%H:%M:%S) $L done rc=$?"
