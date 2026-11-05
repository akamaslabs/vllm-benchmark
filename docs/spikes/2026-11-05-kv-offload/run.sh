#!/bin/bash
# run.sh TAG RATE [USERS] [DURATION]: one multi-turn user-centric AIPerf run against the spike vLLM.
set -u
TAG=$1 RATE=$2 USERS=${3:-64} DUR=${4:-900}
NS=kv-offload-spike
snap() { kubectl -n $NS exec aiperf -- python3 -c "
import urllib.request,re
t=urllib.request.urlopen('http://vllm.$NS.svc:8000/metrics').read().decode()
for l in t.splitlines():
    if re.match(r'vllm:(prefix_cache_(hits|queries)|external_prefix_cache_(hits|queries)|num_preemptions|prompt_tokens|generation_tokens|kv_offload[a-z_]*|request_success)(_total)?\{', l) or l.startswith('vllm:kv_cache_usage_perc'):
        print(l)
"; }
echo "== $TAG rate=$RATE users=$USERS dur=$DUR start $(date -u +%T)"
snap > $TAG.m0
kubectl -n $NS exec aiperf -- sh -c "rm -rf /work/out/$TAG; mkdir -p /work/out/$TAG; cd /work; \
  aiperf profile --model qwen3-8b --tokenizer Qwen/Qwen3-8B-FP8 --url http://vllm.$NS.svc:8000 \
    --endpoint-type chat --streaming \
    --isl 600 --isl-stddev 0 --osl 200 --osl-stddev 0 --extra-inputs ignore_eos:true \
    --conversation-turn-mean 10 --conversation-turn-stddev 0 --conversation-num 3000 \
    --user-centric-rate $RATE --num-users $USERS --benchmark-duration $DUR --benchmark-grace-period 60 \
    --slice-duration 60 --ui simple --output-artifact-dir /work/out/$TAG > /work/out/$TAG.log 2>&1"
echo "aiperf exit=$? end $(date -u +%T)"
snap > $TAG.m1
kubectl -n $NS exec aiperf -- sh -c "tail -60 /work/out/$TAG.log" > $TAG.log
kubectl -n $NS exec aiperf -- sh -c "cat \$(ls /work/out/$TAG/**/*aiperf*.csv /work/out/$TAG/*aiperf*.csv 2>/dev/null | grep -v timeslice | head -1)" > $TAG.csv 2>/dev/null
kubectl -n $NS exec aiperf -- sh -c "cat \$(ls /work/out/$TAG/**/*timeslice*.csv /work/out/$TAG/*timeslice*.csv 2>/dev/null | head -1)" > $TAG.slices.csv 2>/dev/null
echo "--- metrics delta"; paste <(sed 's/ .*//' $TAG.m1) <(awk '{print $NF}' $TAG.m0) <(awk '{print $NF}' $TAG.m1) | awk '{printf "%s start=%s end=%s delta=%.0f\n",$1,$2,$3,$3-$2}'
grep -E '^(Time to First Token|Inter Token Latency|Input Sequence Length|Request Latency|Output Token Throughput|Request Count|Error Request Count)' $TAG.csv | cut -c1-200
