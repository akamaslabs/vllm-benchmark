#!/bin/bash
# Smoke run in the toolbox before the study (design, "Prerequisites" 2): answers what the study
# needs from the node before any Akamas resource is created, one step at a time, each printing
# "SMOKE <step> PASS|FAIL <detail>":
#   baseline  the target's arguments (gmu 0.95, MTP 5) start at TP4: cold start-up time, KV tokens,
#             Mamba blocks, the MoE backend candidates ("potential backends") from the log. If vLLM
#             refuses its default max_num_seqs ("at most N" Mamba blocks), retry once with
#             min(N, 128) and carry that pin into every later step (final review I2);
#   answer    one coding prompt and one tool call answer correctly (glm47 parsers);
#   mtp       one 16-session level of 300 s: MTP acceptance (accepted / draft tokens) > 0, and the
#             client ITL p95 against the server ITL p95 (per engine step) and TPOT p95 (per request),
#             interpolated inside the buckets (hist_quantile.py; final review I1);
#   warm      the same configuration again: start-up time with warm compile caches;
#   dcp       decode_context_parallel_size 4 starts (or the reason it does not);
#   offload   kv_offloading_size 256 starts and its counters move under a 4-session level of 300 s;
#   edges     the optimize domain's edges start: gmu 0.85 with max_num_seqs 128, max_num_batched_tokens
#             2048, every MoE candidate of apply_config.sh, DCP 2 (final review I4).
# Then the go / no-go verdict of the design: go only if baseline and offload pass (the FP8 checkpoint
# fits at TP4 with --max-model-len 262144 >= 131072, and offload works on this hybrid model).
#   bash smoke.sh [step ...]      default: all, in the order above
# Needs the GPU node (infra/eks/nodegroups.sh up) and the m8a-2xlarge load node.
K=${STUDY_DIR:-/work/study52/52-g7e-glm53-coding-sessions-within-slo}/k8s
HERE=$(cd "$(dirname "$0")" && pwd)
NS=glm53-coding
# shellcheck disable=SC1091
source "$HERE/lib_apply.sh"
declare -A R
MNS=""   # max_num_seqs pin found by the baseline step (empty = vLLM's default)
# The baseline's gpu_memory_utilization: the target's 0.95 by default. SMOKE_GMU=1.0 is the design's
# upper bound (v0.34.0's default with the extensible KV cache) when 0.95 leaves no KV block.
GMU=${SMOKE_GMU:-0.95}
MOE=$(grep -m1 '^MOE_ALLOWED=' "$HERE/apply_config.sh" | cut -d'"' -f2)

# params KEY=VALUE ...: writes params.env with every variable (empty unless given), plus the
# max_num_seqs pin unless the step sets its own.
params() {
  local v kv
  : > "$K/params.env"
  for v in KV_OFFLOADING_SIZE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS SPEC_TOKENS \
           DECODE_CONTEXT_PARALLEL_SIZE MOE_BACKEND MAX_CUDAGRAPH_CAPTURE_SIZE ASYNC_SCHEDULING \
           PERFORMANCE_MODE OPTIMIZATION_LEVEL; do
    echo "$v=" >> "$K/params.env"
  done
  # -i.bak works with GNU and BSD sed (the smoke can run from a laptop with kubectl, not only the toolbox).
  [ -z "$MNS" ] || sed -i.bak "s|^MAX_NUM_SEQS=.*|MAX_NUM_SEQS=$MNS|" "$K/params.env"
  # SMOKE_EXTRA="KEY=VALUE ...": applied to every step, before the step's own values.
  for kv in ${SMOKE_EXTRA:-} "$@"; do sed -i.bak "s|^${kv%%=*}=.*|$kv|" "$K/params.env"; done
  rm -f "$K/params.env.bak"
}
apply() { local t0=$SECONDS; bash "$HERE/apply_config.sh" > "$K/smoke-$1.log" 2>&1; local rc=$?; echo "$rc $((SECONDS - t0))"; }
why() { grep -A1 'why it did not start\|error:' "$K/smoke-$1.log" | grep -v '^--$' | tail -2 | tr '\n' ' '; }
pod() { kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}'; }
level() { LEVELS=$2 LEVEL_SECONDS=300 STEADY_FROM=60 bash "$HERE/run_ramp.sh" > "$K/smoke-$1-load.log" 2>&1; }
report() { R[$1]=$2; echo "SMOKE $1 $2 ${*:3}"; }
metrics() { kubectl -n $NS exec "$(pod)" -- python3 -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8000/metrics").read().decode())' > "$1"; }
# try NAME KEY=VALUE ...: one start with these parameters, PASS/FAIL with the reason.
try() {
  local n=$1 rc secs; shift
  params GPU_MEMORY_UTILIZATION=$GMU SPEC_TOKENS=5 "$@"
  read -r rc secs <<< "$(apply "$n")"
  if [ "$rc" = 0 ]; then report "$n" PASS "started in ${secs}s; $(kubectl -n $NS logs "$(pod)" | grep 'GPU KV cache size' | tail -1 | sed 's/.*\] //')"
  else report "$n" FAIL "apply exit $rc: $(why "$n")"; fi
}
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

for S in "${@:-baseline answer mtp warm dcp offload edges}"; do
 for S in $S; do
  echo "===== SMOKE $S $(date -u +%T)"
  case "$S" in
    baseline)
      params GPU_MEMORY_UTILIZATION=$GMU SPEC_TOKENS=5
      read -r rc secs <<< "$(apply baseline)"
      if [ "$rc" != 0 ]; then
        N=$(max_seqs_hint "$K/smoke-baseline.log")
        if [ -n "$N" ]; then
          MNS=$(( N < 128 ? N : 128 ))
          echo "SMOKE baseline: default max_num_seqs refused (at most $N Mamba blocks): retrying with $MNS"
          params GPU_MEMORY_UTILIZATION=$GMU SPEC_TOKENS=5
          read -r rc secs <<< "$(apply baseline)"
        fi
      fi
      [ "$rc" = 0 ] || { report baseline FAIL "apply exit $rc after ${secs}s: $(why baseline)"; continue; }
      report baseline PASS "start ${secs}s; max_num_seqs pin: ${MNS:-none (vLLM default fits)}; $(kubectl -n $NS logs "$(pod)" | grep -E 'GPU KV cache size' | tail -1 | sed 's/.*\] //')"
      kubectl -n $NS logs "$(pod)" | grep -E 'potential backends|Mamba|attention block size|Using .* backend' | sed 's/^.*\] /  log: /' | sort -u ;;
    answer)
      out=$(kubectl -n $NS exec "$(pod)" -- python3 -c "$ASK" 2>&1 | tail -1)
      [[ "$out" == ok* ]] && report answer PASS "$out" || report answer FAIL "$out" ;;
    mtp)
      metrics /tmp/smoke-m0; level mtp 16; metrics /tmp/smoke-m1
      acc=$(grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens_total ' "$K/smoke-mtp-load.log" | awk '{v[$1]=$2} END { d=v["vllm:spec_decode_num_draft_tokens_total"]; a=v["vllm:spec_decode_num_accepted_tokens_total"]; if (d > 0) printf "%.1f%%", 100 * a / d; else print "no drafts" }')
      res=$(grep '^RESULT sent' "$K/smoke-mtp-load.log" | tail -1)
      itl=$(python3 "$HERE/hist_quantile.py" vllm:inter_token_latency_seconds /tmp/smoke-m0 /tmp/smoke-m1 0.95)
      tpot=$(python3 "$HERE/hist_quantile.py" vllm:time_per_output_token_seconds /tmp/smoke-m0 /tmp/smoke-m1 0.95)
      detail="acceptance $acc; server ITL p95 $itl (per step), TPOT p95 $tpot (per request); client: $res"
      [[ "$acc" == *% ]] && [ "${acc%%.*}" -gt 0 ] && report mtp PASS "$detail" || report mtp FAIL "$detail (vllm#59724?)" ;;
    warm)
      read -r rc secs <<< "$(apply warm)"
      [ "$rc" = 0 ] && report warm PASS "warm start ${secs}s" || report warm FAIL "apply exit $rc" ;;
    dcp)
      try dcp DECODE_CONTEXT_PARALLEL_SIZE=4 ;;
    offload)
      params GPU_MEMORY_UTILIZATION=$GMU SPEC_TOKENS=5 KV_OFFLOADING_SIZE=256
      read -r rc secs <<< "$(apply offload)"
      [ "$rc" = 0 ] || { report offload FAIL "apply exit $rc: $(why offload)"; continue; }
      level offload 4
      moved=$(grep -E '^vllm:(kv_offload_total_bytes_total|external_prefix_cache_hits_total)' "$K/smoke-offload-load.log" | awk '$2 > 0' | tr '\n' ' ')
      [ -n "$moved" ] && report offload PASS "counters: $moved" || report offload FAIL "no offload counter moved (KDA state not offloaded?)" ;;
    edges)
      try edge-gmu085-mns128 GPU_MEMORY_UTILIZATION=0.85 MAX_NUM_SEQS=128
      try edge-mnbt2048 MAX_NUM_BATCHED_TOKENS=2048
      try edge-dcp2 DECODE_CONTEXT_PARALLEL_SIZE=2
      for m in $MOE; do [ "$m" = auto ] || try "edge-moe-$m" MOE_BACKEND="$m"; done ;;
    *) echo "unknown step $S" >&2; exit 2 ;;
  esac
 done
done
if [ "${R[baseline]:-}" = PASS ] && [ "${R[offload]:-}" = PASS ]; then
  echo "SMOKE VERDICT go (baseline and offload pass; max_num_seqs pin ${MNS:-none}; MTP ${R[mtp]:-not run}, DCP ${R[dcp]:-not run})"
else
  echo "SMOKE VERDICT no-go (baseline ${R[baseline]:-not run}, offload ${R[offload]:-not run}): bring it back to the user before creating the study"
fi
echo "Edges and MoE candidates: narrow the study's domains to what passed (apply_config.sh MOE_ALLOWED, the study YAML)."
