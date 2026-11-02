#!/bin/bash
# Study 44 driver (tappe A and B) — manual runs, from the toolbox or a laptop with kubectl on vllm-bench.
#
# A "size" names a node group: 4xl, xl (L4, tappa A), l40s (g6e.2xlarge), a10g (g5.2xlarge).
# A "pair" is <prefill size>:<decode size>; a single size means the same group for both roles.
#
#   ./run.sh prep                       Services, ServiceMonitor, ConfigMaps (config, scripts, tuned configs)
#   ./run.sh iperf <pair> [seconds]     iperf3 client on the prefill group -> server on the decode group (default 600 s)
#   ./run.sh up <pair> pd               deploy P1D1: prefill on the first group, decode + router on the second
#   ./run.sh up <size> agg              deploy the P0D1 aggregated control on one node of <size>
#   ./run.sh smoke                      one request through the router + NIXL/UCX log lines
#   ./run.sh ramp [rate_max] [ramp_s]   AIPerf open-loop ramp with study 27's watchdog (default 1.2 req/s in 3600 s)
#   ./run.sh bench [n] [conc] [label]   KV-transfer micro-benchmark (default 40 requests x 4)
#   ./run.sh score <start> <end>        study-27 score from Prometheus over [start,end] (UTC, e.g. 2026-10-29T11:00:00Z)
#   ./run.sh logs                       last lines of every pod
#   ./run.sh down                       delete the serving pods (the node group is scaled by ../infra/nodegroups.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RESULTS=$HERE/../results
NS=llm-serving
PROM_PROXY=/api/v1/namespaces/monitoring/services/kube-prometheus-stack-prometheus:9090/proxy
mkdir -p "$RESULTS"
ts() { date -u +%Y%m%dT%H%M%SZ; }

node_role() {
  case "$1" in
    4xl) echo llm-serving-l4-pd ;; xl) echo llm-serving-l4-pd-xl ;;
    l40s) echo llm-serving-l40s-pd ;; a10g) echo llm-serving-a10g-pd ;;
    *) echo "size: 4xl|xl|l40s|a10g" >&2; exit 2 ;;
  esac
}
# Per-pod resources. 4xl reproduces study 27's per-GPU share (16 vCPU / 64 GiB per L4).
# xl: 4 vCPU / 16 GiB per node, ~14 GiB allocatable. With kv_buffer_device=cpu NIXL keeps a
# HOST copy of the KV region, so a 0.9 gmu KV cache (~10 GiB on an L4) may not fit: that is
# one of the things the xl run is meant to show.
# Tappa B: l40s = g6e.2xlarge (8 vCPU, 64 GiB), a10g = g5.2xlarge (8 vCPU, 32 GiB).
sizes() {  # size -> CPU_REQ MEM_REQ MEM_LIM SHM
  case "$1" in
    4xl) CPU_REQ=12; MEM_REQ=40Gi; MEM_LIM=56Gi; SHM=8Gi ;;
    xl) CPU_REQ=3; MEM_REQ=11Gi; MEM_LIM=13Gi; SHM=2Gi ;;
    l40s) CPU_REQ=6; MEM_REQ=40Gi; MEM_LIM=56Gi; SHM=8Gi ;;
    a10g) CPU_REQ=6; MEM_REQ=20Gi; MEM_LIM=27Gi; SHM=4Gi ;;
  esac
}
pair() {  # "<p>:<d>" or "<s>" -> P_SIZE D_SIZE
  P_SIZE=${1%%:*}; D_SIZE=${1##*:}
  node_role "$P_SIZE" >/dev/null; node_role "$D_SIZE" >/dev/null
}
role_sed() {  # sed expressions for the __P_*__ / __D_*__ placeholders
  local pre size
  for pre in P D; do
    if [[ $pre == P ]]; then size=$P_SIZE; else size=$D_SIZE; fi
    sizes "$size"
    printf -- '-e s/__%s_NODE_ROLE__/%s/g -e s/__%s_CPU_REQ__/%s/g -e s/__%s_MEM_REQ__/%s/g -e s/__%s_MEM_LIM__/%s/g -e s/__%s_SHM__/%s/g ' \
      $pre "$(node_role "$size")" $pre $CPU_REQ $pre $MEM_REQ $pre $MEM_LIM $pre $SHM
  done
}

# Variants (KV-transfer tuning, 2026-10-29): PD_EXTRA_COMMON (flags appended to common.args, so
# both roles get them: block size must match for NIXL) and PD_EXTRA_ENV (KEY=VALUE pairs
# appended to every role's env), both space-separated, e.g.
#   PD_EXTRA_COMMON="--block-size=64" PD_EXTRA_ENV="UCX_TCP_SNDBUF=8M" ./run.sh up 4xl pd
render_config() {
  python3 - "$HERE/20-pd-config.yaml" <<'EOF'
import os, sys
s = open(sys.argv[1]).read()
common = os.environ.get("PD_EXTRA_COMMON", "").split()
env = os.environ.get("PD_EXTRA_ENV", "").split()
if common:
    s = s.replace("  common.args: |\n", "  common.args: |\n" + "".join(f"    {a}\n" for a in common), 1)
kv_extra = os.environ.get("PD_KV_EXTRA_CONFIG", "")
if kv_extra:
    s = s.replace("  topology.env: |\n", f"  topology.env: |\n    PD_KV_EXTRA_CONFIG='{kv_extra}'\n", 1)
for role in ("prefill", "decode", "aggregated"):
    if env:
        s = s.replace(f"  {role}.env: |\n", f"  {role}.env: |\n" + "".join(f"    {e}\n" for e in env), 1)
    # PD_EXTRA_ARGS_<ROLE>: flags appended at the END of that role's args (argparse: the last
    # value wins, so this overrides the base config, e.g. the linear backend on an A10G).
    extra = os.environ.get(f"PD_EXTRA_ARGS_{role.upper()}", "").split()
    if extra:
        head = f"  {role}.args: |\n"
        i = s.index(head) + len(head)
        j = i
        while j < len(s) and s[j:j + 4] == "    ":
            j = s.index("\n", j) + 1
        s = s[:j] + "".join(f"    {a}\n" for a in extra) + s[j:]
print(s)
EOF
}

prep() {
  render_config | kubectl apply -f -
  kubectl apply -f "$HERE/02-service.yaml"
  kubectl apply -f "$HERE/monitoring/servicemonitor.yaml"
  kubectl create configmap pd-mn-scripts -n $NS --from-file="$HERE/serve.sh" --from-file="$HERE/pd_router.py" \
    --dry-run=client -o yaml | kubectl apply -f -
  local tgz; tgz=$(mktemp -t tuned.XXXXXX)
  (cd "$HERE/tuned-configs" && COPYFILE_DISABLE=1 tar -czf "$tgz" ./*.json)   # no macOS ._ files
  kubectl create configmap pd-tuned-configs -n $NS --from-file=tuned-configs.tgz="$tgz" --dry-run=client -o yaml | kubectl apply -f -
  rm -f "$tgz"
}

yaml_doc() {  # print the Nth (0-based) document of a multi-document YAML on stdin
  python3 -c 'import sys; d=[x for x in sys.stdin.read().split("\n---\n")]; print(d[int(sys.argv[1])])' "$1"
}

iperf() {
  local secs log ip srole crole
  pair "${1:?pair}"; secs=${2:-600}; log=$RESULTS/iperf3-${1/:/-}-$(ts).log
  crole=$(node_role "$P_SIZE"); srole=$(node_role "$D_SIZE")
  kubectl delete pod iperf3-server -n $NS --ignore-not-found; kubectl delete job iperf3-client -n $NS --ignore-not-found
  sed -e "s/__S_NODE_ROLE__/$srole/g" -e "s/__C_NODE_ROLE__/$crole/g" -e "s/__SECONDS__/$secs/g" -e "s/__SERVER_IP__/PENDING/" "$HERE/10-iperf3_template.yaml" \
    | yaml_doc 0 | kubectl apply -f -
  kubectl wait pod/iperf3-server -n $NS --for=condition=Ready --timeout=300s
  ip=$(kubectl get pod iperf3-server -n $NS -o jsonpath='{.status.podIP}')
  sed -e "s/__S_NODE_ROLE__/$srole/g" -e "s/__C_NODE_ROLE__/$crole/g" -e "s/__SECONDS__/$secs/g" -e "s/__SERVER_IP__/$ip/" "$HERE/10-iperf3_template.yaml" \
    | yaml_doc 1 | kubectl apply -f -
  kubectl get pod -n $NS -l app=iperf3 -o wide
  echo "iperf3 for ${secs} s, log: $log"
  kubectl wait pod -n $NS -l side=client --for=condition=Ready --timeout=300s || true
  kubectl logs -f job/iperf3-client -n $NS | tee "$log"
  grep SUM "$log" | tail -3
  kubectl delete pod iperf3-server -n $NS --wait=false; kubectl delete job iperf3-client -n $NS --wait=false
}

up() {
  local mode=${2:?pd|agg} tpl sha
  pair "${1:?pair}"
  case $mode in pd) tpl=21-serving_template.yaml ;; agg) tpl=22-aggregated_template.yaml ;; *) echo "mode: pd|agg"; exit 2 ;; esac
  prep
  sha=$( { render_config; cat "$HERE/serve.sh" "$HERE/pd_router.py"; } | { sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12)
  kubectl delete deployment -n $NS -l app=vllm-pd-mn --ignore-not-found --wait=true
  # shellcheck disable=SC2046
  sed $(role_sed) -e "s/__CONFIG_SHA__/$sha/g" "$HERE/$tpl" | kubectl apply -f -
  echo "$(date -u +%T) waiting for the engines (model download + compile, ~10-20 min)..."
  for d in $(kubectl get deployment -n $NS -l app=vllm-pd-mn -o name); do
    kubectl rollout status "$d" -n $NS --timeout=25m
  done
  kubectl get pod -n $NS -l app=vllm-pd-mn -o wide
}

smoke() {
  local log=$RESULTS/smoke-$(ts).log
  {
    echo "--- one chat request through the router ---"
    kubectl exec deploy/vllm-pd-mn-router -n $NS -- python3 -c '
import json, time, urllib.request
body = {"model": "qwen3-8b", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 32}
t = time.time()
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=300)
print(round(time.time() - t, 3), "s", json.loads(r.read())["choices"][0]["message"]["content"])'
    echo "--- router /health ---"
    kubectl exec deploy/vllm-pd-mn-router -n $NS -- python3 -c 'import urllib.request;print(urllib.request.urlopen("http://127.0.0.1:8000/health").read().decode())'
    for r in prefill decode; do
      kubectl get deploy vllm-pd-mn-$r -n $NS >/dev/null 2>&1 || continue
      echo "--- $r: NIXL / UCX / KV lines ---"
      kubectl logs deploy/vllm-pd-mn-$r -n $NS --tail=3000 | grep -iE 'nixl|ucx|ucp|kv.?transfer|side.?channel|handshake|serve:' | tail -40
    done
  } 2>&1 | tee "$log"
}

ramp() {
  local rate=${1:-1.2} ramp_s=${2:-3600} job log t0 over_since="" ttft itl
  local wd_ttft=20000 wd_itl=225 wd_hold=120
  log=$RESULTS/ramp-$(kubectl get deploy -n $NS -l app=vllm-pd-mn -o jsonpath='{.items[0].spec.template.metadata.labels.topology}')-$(ts).log
  job=$(mktemp -t aiperf.XXXXXX)
  sed -e "s/__RATE_MAX__/$rate/g" -e "s/__RAMP_S__/$ramp_s/g" "$HERE/05-job_template.yaml" > "$job"
  kubectl delete job aiperf-pd-mn -n llm-benchmark --ignore-not-found
  kubectl apply -f "$job"
  t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "ramp 0 -> $rate req/s over $ramp_s s started $t0; watchdog TTFT p95 > $wd_ttft ms or ITL p95 > $wd_itl ms for $wd_hold s" | tee "$log"
  while true; do
    s=$(kubectl get job aiperf-pd-mn -n llm-benchmark -o jsonpath='{.status.succeeded}{"/"}{.status.failed}')
    [[ $s == 1/* ]] && { echo "$(date -u +%T) AIPerf completed" | tee -a "$log"; break; }
    [[ $s == */1 ]] && { echo "$(date -u +%T) AIPerf FAILED" | tee -a "$log"; break; }
    ttft=$(prom_now 'histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{model_name="qwen3-8b-router"}[150s])))*1000')
    itl=$(prom_now 'histogram_quantile(0.95, sum by(le)(rate(vllm:inter_token_latency_seconds_bucket{model_name="qwen3-8b-router"}[150s])))*1000')
    echo "$(date -u +%T) TTFT p95 ${ttft:-n/a} ms  ITL p95 ${itl:-n/a} ms" >> "$log"
    if { [[ $ttft =~ ^[0-9]+$ ]] && (( ttft > wd_ttft )); } || { [[ $itl =~ ^[0-9]+$ ]] && (( itl > wd_itl )); }; then
      [[ -z $over_since ]] && over_since=$SECONDS
      if (( SECONDS - over_since >= wd_hold )); then
        echo "$(date -u +%T) watchdog: TTFT p95 $ttft ms, ITL p95 $itl ms past the limits for $wd_hold s, ending the ramp" | tee -a "$log"; break
      fi
    else over_since=""; fi
    sleep 15
  done
  local t1; t1=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  kubectl logs job/aiperf-pd-mn -n llm-benchmark -c aiperf --tail=80 >> "$log" 2>&1 || true
  kubectl delete job aiperf-pd-mn -n llm-benchmark --wait=false || true
  kubectl get pod -n $NS -l app=vllm-pd-mn -o custom-columns=POD:.metadata.name,RESTARTS:.status.containerStatuses[*].restartCount | tee -a "$log"
  echo "window: $t0 -> $t1" | tee -a "$log"
  score "$t0" "$t1" | tee -a "$log"
}

prom_now() {  # instant query, integer value or empty
  kubectl get --raw "$PROM_PROXY/api/v1/query?query=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$1")" 2>/dev/null \
    | python3 -c 'import json,sys
r=json.load(sys.stdin)["data"]["result"]
v=r[0]["value"][1] if r else ""
print(int(float(v)) if v not in ("","NaN","+Inf") else "")' 2>/dev/null || true
}

score() {  # study 27's goal recomputed: 30 s samples, 6-sample windows, max valid window mean / GPUs
  python3 "$HERE/score.py" "${1:?start}" "${2:?end}"
}

# KV-transfer micro-benchmark: N long prompts (~4000 tokens of random words, no prefix reuse)
# at fixed concurrency C, 16 output tokens, through the router. Then the decode's
# "KV Transfer metrics" lines over the same interval: transfer time, MB, descriptors.
bench() {
  local n=${1:-40} c=${2:-4} label=${3:-run} log since
  log=$RESULTS/bench-$label-$(ts).log
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  {
    echo "bench $label: $n requests, concurrency $c, extra common='${PD_EXTRA_COMMON:-}' env='${PD_EXTRA_ENV:-}'"
    kubectl exec -i deploy/vllm-pd-mn-router -n $NS -- python3 - "$n" "$c" <<'EOF'
import json, random, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
n, c = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(44)
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
def one(i):
    prompt = " ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(1400))
    body = {"model": "qwen3-8b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 16,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); ttft = None; ptok = None
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if ttft is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                ttft = time.time() - t0
            if d.get("usage"):
                ptok = d["usage"]["prompt_tokens"]
    return ttft, time.time() - t0, ptok
t = time.time()
with ThreadPoolExecutor(c) as ex:
    res = list(ex.map(one, range(n)))
ttfts = sorted(r[0] for r in res if r[0])
print(f"wall {time.time() - t:.1f} s, prompt tokens ~{statistics.mean(r[2] for r in res if r[2]):.0f}, "
      f"TTFT p50 {statistics.median(ttfts):.3f} s, p90 {ttfts[int(0.9 * len(ttfts)) - 1]:.3f} s")
EOF
    echo "--- decode KV Transfer metrics since $since ---"
    kubectl logs deploy/vllm-pd-mn-decode -n $NS --since-time="$since" | grep "KV Transfer metrics" | python3 -c '
import re, sys
rows = []
for l in sys.stdin:
    g = lambda k: float(re.search(k + r"=([0-9.]+)", l).group(1))
    rows.append((g(r"Num successful transfers"), g(r"Avg xfer time \(ms\)"), g(r"Avg MB per transfer"), g(r"Avg number of descriptors")))
if not rows:
    print("no KV Transfer metrics lines"); sys.exit()
nt = sum(r[0] for r in rows)
avg = lambda i: sum(r[0] * r[i] for r in rows) / nt
print(f"transfers {nt:.0f}, avg xfer {avg(1):.0f} ms, avg {avg(2):.1f} MB, {avg(2) / avg(1) * 1000:.0f} MB/s per transfer, avg descriptors {avg(3):.0f}")'
  } 2>&1 | tee "$log"
}

logs() {
  for d in $(kubectl get deployment -n $NS -l app=vllm-pd-mn -o name); do echo "=== $d"; kubectl logs "$d" -n $NS --tail=40; done
}

down() { kubectl delete deployment -n $NS -l app=vllm-pd-mn --ignore-not-found; }

cmd=${1:-}; shift || true
case $cmd in
  prep|iperf|up|smoke|ramp|bench|score|logs|down) "$cmd" "$@" ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
