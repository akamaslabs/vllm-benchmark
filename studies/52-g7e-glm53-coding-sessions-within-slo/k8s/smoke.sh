#!/bin/bash
# Smoke run in the toolbox before the study (design, "Prerequisites" 2): answers what the study
# needs from the node before any Akamas resource is created, one step at a time, each printing
# "SMOKE <step> PASS|FAIL <detail>":
#   baseline  the target's arguments (gmu 0.95, MTP 5) start at TP4: cold start-up time, KV tokens,
#             Mamba blocks, the MoE backend candidates ("potential backends") from the log;
#   answer    one coding prompt and one tool call answer correctly (glm47 parsers);
#   mtp       one 4-session level of 300 s: MTP acceptance (accepted / draft tokens) > 0, and the
#             server ITL p95 (histogram) against the client one;
#   warm      the same configuration again: start-up time with warm compile caches;
#   dcp       decode_context_parallel_size 4 starts (or the reason it does not);
#   offload   kv_offloading_size 256 starts and its counters move under a 4-session level of 300 s.
# Then the go / no-go verdict of the design: go only if baseline and offload pass (the FP8 checkpoint
# fits at TP4 with --max-model-len 262144 >= 131072, and offload works on this hybrid model).
#   bash smoke.sh [step ...]      default: all, in the order above
# Needs the GPU node (infra/eks/nodegroups.sh up) and the m8a-2xlarge load node.
K=${STUDY_DIR:-/work/study52/52-g7e-glm53-coding-sessions-within-slo}/k8s
HERE=$(cd "$(dirname "$0")" && pwd)
NS=glm53-coding
declare -A R

# params KEY=VALUE ...: writes params.env with every variable (empty unless given).
params() {
  local v kv
  : > "$K/params.env"
  for v in KV_OFFLOADING_SIZE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS SPEC_TOKENS \
           DECODE_CONTEXT_PARALLEL_SIZE MOE_BACKEND MAX_CUDAGRAPH_CAPTURE_SIZE ASYNC_SCHEDULING \
           PERFORMANCE_MODE OPTIMIZATION_LEVEL; do
    echo "$v=" >> "$K/params.env"
  done
  for kv in "$@"; do sed -i "s|^${kv%%=*}=.*|$kv|" "$K/params.env"; done
}
apply() { local t0=$SECONDS; bash "$HERE/apply_config.sh" > "$K/smoke-$1.log" 2>&1; local rc=$?; echo "$rc $((SECONDS - t0))"; }
pod() { kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}'; }
level() { LEVELS=4 LEVEL_SECONDS=300 STEADY_FROM=60 bash "$HERE/run_ramp.sh" > "$K/smoke-$1-load.log" 2>&1; }
report() { R[$1]=$2; echo "SMOKE $1 $2 ${*:3}"; }
# Server ITL p95 from vllm:inter_token_latency_seconds bucket deltas between two /metrics snapshots.
HIST='
import re, sys
def b(p):
    d = {}
    for l in open(p):
        m = re.match(r"vllm:inter_token_latency_seconds_bucket\{.*le=\"([^\"]+)\".*\} (\S+)$", l)
        if m: d[float(m.group(1))] = d.get(float(m.group(1)), 0) + float(m.group(2))
    return d
a, z = b(sys.argv[1]), b(sys.argv[2])
c = sorted((le, z[le] - a.get(le, 0)) for le in z)
n = c[-1][1] if c else 0
print(next(("%.1f ms" % (le * 1000) for le, v in c if n and v >= 0.95 * n), "n/a"))
'
metrics() { kubectl -n $NS exec "$(pod)" -- python3 -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8000/metrics").read().decode())' > "$1"; }
ASK='
import json, sys, urllib.request
def chat(body):
    r = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
d = chat({"model": "zai-org/GLM-5.3-Flash", "messages": [{"role": "user", "content": "Write a Python function is_palindrome(s) that ignores case and punctuation. Only the code."}], "max_tokens": 4000, "temperature": 0})
code = d["choices"][0]["message"].get("content") or ""
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
d = chat({"model": "zai-org/GLM-5.3-Flash", "messages": [{"role": "user", "content": "Open src/main.py and tell me what it does."}], "tools": tools, "max_tokens": 2000, "temperature": 0})
tc = d["choices"][0]["message"].get("tool_calls") or []
ok = "def is_palindrome" in code and tc and tc[0]["function"]["name"] == "read_file"
print(("ok" if ok else "bad") + " code=%d chars tool_calls=%s" % (len(code), json.dumps(tc)[:120]))
'

for S in ${@:-baseline answer mtp warm dcp offload}; do
  echo "===== SMOKE $S $(date -u +%T)"
  case "$S" in
    baseline)
      params GPU_MEMORY_UTILIZATION=0.95 SPEC_TOKENS=5
      read -r rc secs <<< "$(apply baseline)"
      [ "$rc" = 0 ] || { report baseline FAIL "apply exit $rc after ${secs}s: $(grep -A1 'why it did not start' "$K/smoke-baseline.log" | tail -1)"; continue; }
      report baseline PASS "cold start ${secs}s; $(kubectl -n $NS logs "$(pod)" | grep -E 'GPU KV cache size|Maximum concurrency' | tail -1 | sed 's/.*\] //')"
      kubectl -n $NS logs "$(pod)" | grep -E 'potential backends|Mamba|attention block size|Using .* backend' | sed 's/^.*\] /  log: /' | sort -u ;;
    answer)
      out=$(kubectl -n $NS exec "$(pod)" -- python3 -c "$ASK" 2>&1 | tail -1)
      [[ "$out" == ok* ]] && report answer PASS "$out" || report answer FAIL "$out" ;;
    mtp)
      metrics /tmp/smoke-m0; level mtp; metrics /tmp/smoke-m1
      acc=$(grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total ' "$K/smoke-mtp-load.log" | awk '{v[$1]=$2} END { d=v["vllm:spec_decode_num_draft_tokens_total"]; a=v["vllm:spec_decode_num_accepted_tokens_total"]; if (d > 0) printf "%.1f%%", 100 * a / d; else print "no drafts" }')
      res=$(grep '^RESULT sent' "$K/smoke-mtp-load.log" | tail -1)
      itl=$(python3 -c "$HIST" /tmp/smoke-m0 /tmp/smoke-m1)
      [[ "$acc" == *% ]] && [ "${acc%%.*}" -gt 0 ] && report mtp PASS "acceptance $acc; server ITL p95 $itl; client: $res" \
        || report mtp FAIL "acceptance $acc (vllm#59724?); server ITL p95 $itl; client: $res" ;;
    warm)
      read -r rc secs <<< "$(apply warm)"
      [ "$rc" = 0 ] && report warm PASS "warm start ${secs}s" || report warm FAIL "apply exit $rc" ;;
    dcp)
      params GPU_MEMORY_UTILIZATION=0.95 SPEC_TOKENS=5 DECODE_CONTEXT_PARALLEL_SIZE=4
      read -r rc secs <<< "$(apply dcp)"
      [ "$rc" = 0 ] && report dcp PASS "started in ${secs}s; $(kubectl -n $NS logs "$(pod)" | grep 'GPU KV cache size' | tail -1 | sed 's/.*\] //')" \
        || report dcp FAIL "apply exit $rc: $(grep -A1 'why it did not start' "$K/smoke-dcp.log" | tail -1)" ;;
    offload)
      params GPU_MEMORY_UTILIZATION=0.95 SPEC_TOKENS=5 KV_OFFLOADING_SIZE=256
      read -r rc secs <<< "$(apply offload)"
      [ "$rc" = 0 ] || { report offload FAIL "apply exit $rc: $(grep -A1 'why it did not start' "$K/smoke-offload.log" | tail -1)"; continue; }
      level offload
      moved=$(grep -E '^vllm:(kv_offload_total_bytes_total|external_prefix_cache_hits_total)' "$K/smoke-offload-load.log" | awk '$2 > 0' | tr '\n' ' ')
      [ -n "$moved" ] && report offload PASS "counters: $moved" || report offload FAIL "no offload counter moved (KDA state not offloaded?)" ;;
    *) echo "unknown step $S" >&2; exit 2 ;;
  esac
done
if [ "${R[baseline]:-}" = PASS ] && [ "${R[offload]:-}" = PASS ]; then
  echo "SMOKE VERDICT go (baseline and offload pass; MTP ${R[mtp]:-not run}, DCP ${R[dcp]:-not run})"
else
  echo "SMOKE VERDICT no-go (baseline ${R[baseline]:-not run}, offload ${R[offload]:-not run}): bring it back to the user before creating the study"
fi
