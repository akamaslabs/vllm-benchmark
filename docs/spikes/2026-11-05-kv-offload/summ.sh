#!/bin/bash
# summ.sh TAG: counters at end minus start (by metric name), CSV headline, per-minute TTFT
T=$1
python3 - "$T" <<'PY'
import sys,re
t=sys.argv[1]
def load(f):
    d={}
    for l in open(f):
        l=l.strip()
        if not l: continue
        k,v=l.rsplit(' ',1); d[k]=float(v)
    return d
a,b=load(t+'.m0'),load(t+'.m1')
for k in sorted(b):
    if re.search(r'(prefix_cache_(hits|queries)_total|num_preemptions_total|prompt_tokens_total|kv_offload_total_bytes_total)',k):
        print(re.sub(r'\{.*transfer_type="([^"]+)".*',r'[\1]',re.sub(r'\{engine.*\}','',k)), int(b[k]-a.get(k,0)))
PY
grep -E '^(Time to First Token|Inter Token Latency|Output Token Throughput \(|Request Count)' $T.csv | awk -F, '{print $1, "avg="$2, "p50="$10, "p95="$13}'
awk -F, '$2=="Time to First Token" && ($4=="p50"||$4=="p95") {v[$1","$4]=$5} ($2=="Input Sequence Length"&&$4=="avg"){v[$1","$2]=$5} END{for(i=0;i<20;i++) if((i",p50") in v) printf "min %2d ttft p50 %7.0f p95 %7.0f isl %5.0f\n", i, v[i",p50"], v[i",p95"], v[i",Input Sequence Length"]}' $T.slices.csv
