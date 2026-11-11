#!/bin/bash
# ramp.sh <HH:MM:SS UTC start> [step-seconds] [cohorts] [users-per-cohort]
# Runs in the aiperf pod. Adds one AIPerf cohort of chat users every step, so the users on the
# server grow 64, 128, ... up to cohorts x users; all cohorts end together. Same per-user load as
# study 51 (10 turns, 600 input and 200 output tokens per turn, one turn every 64 s, history re-sent).
set -u
T0=$(date -u -d "today $1" +%s); STEP=${2:-240}; N=${3:-7}; U=${4:-64}
rate=$(awk -v u="$U" 'BEGIN{printf "%.4f", u/64}')
while [ "$(date +%s)" -lt "$T0" ]; do sleep 1; done
for i in $(seq 1 "$N"); do
  dur=$(( (N - i + 1) * STEP )); out=/work/out/ramp/c$i; mkdir -p "$out"
  nohup aiperf profile --model gpt-oss-120b --tokenizer openai/gpt-oss-120b \
    --url http://vllm.nsight-probe.svc:8000 --endpoint-type chat --streaming \
    --isl 600 --isl-stddev 0 --osl 200 --osl-stddev 0 --extra-inputs ignore_eos:true reasoning_effort:low \
    --conversation-turn-mean 10 --conversation-turn-stddev 0 --conversation-num 600 --random-seed $((1000 + i)) \
    --user-centric-rate "$rate" --num-users "$U" --benchmark-duration "$dur" --benchmark-grace-period 30 \
    --slice-duration 60 --ui simple --output-artifact-dir "$out/run" > "$out/run.log" 2>&1 &
  echo "$(date -u +%H:%M:%S) cohort $i started: $((i * U)) users, $dur s"
  next=$(( T0 + i * STEP )); while [ "$(date +%s)" -lt "$next" ]; do sleep 1; done
done
wait; echo "$(date -u +%H:%M:%S) ramp done"
