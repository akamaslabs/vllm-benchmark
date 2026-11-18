#!/bin/bash
# Spike 2026-11-18 (feasibility of study 53): does vLLM's CPU KV offload RELOAD work for
# GLM-5.3-Flash (hybrid KDA + sparse MLA) at TP4 on the g7e.24xlarge, and how does it compare with
# no offload when the sessions' history is ~2x the GPU KV cache? Study 52's experiment 2 (offload
# 256 GiB) wrote 1.3 TB to RAM and read back 0 bytes: its sessions always fit on the GPU.
#
# Runs from the toolbox while study 52 is finished (paused), with study 52's own k8s scripts:
#   K=/work/study52/52-g7e-glm53-coding-sessions-within-slo/k8s nohup bash spike.sh [OFF ...] > spike.log 2>&1 &
# For each offload size (default "512 0"): apply_config.sh with study 52's baseline (gmu 0.95, MTP 5)
# plus that offload, then one AIPerf level of long, mostly idle coding sessions (lib_load.sh
# run_level), then the Prometheus deltas over the level. Prints "SPIKE <off> ..." summary lines.
#
# Workload (assumed, not the target's data): SESSIONS sessions, each a 20k shared system prompt +
# USER_CONTEXT tokens of its own context, then TURNS turns of 4k in / 400 out, one turn every
# SECONDS_PER_TURN s. 64 x ~135k = ~8.6M tokens of history vs 3.77M GPU KV tokens; 0.53 turns/s,
# about half of the turn rate where study 52 broke the ITL SLO.
set -uo pipefail
K=${K:-/work/study52/52-g7e-glm53-coding-sessions-within-slo/k8s}
NS=glm53-coding
SESSIONS=${SESSIONS:-64}
export SCRIPT_NAME=spike53
SHARED_PROMPT=20000 USER_CONTEXT=${USER_CONTEXT:-100000} TURNS=${TURNS:-8} ISL=4000 OSL=400
SECONDS_PER_TURN=${SECONDS_PER_TURN:-120}
CONVERSATIONS=$SESSIONS   # per client: SESSIONS/2 users, each <= 2 conversations in the run
# AIPerf 0.11.0 writes --num-dataset-entries whole conversations (every turn's full history) to
# inputs.json, whatever --conversation-num says: 2000 entries of ~1M tokens = 8.6 GB, over the clients'
# 8Gi emptyDir (both clients evicted in the first run, 17:54 and 18:02). 64 entries: 277 MB, < 10 s.
DATASET_ENTRIES=${DATASET_ENTRIES:-$CONVERSATIONS} EXTRA_INPUTS="ignore_eos:true"
LEVEL_SECONDS=${LEVEL_SECONDS:-900} STEADY_FROM=${STEADY_FROM:-480} SLO_TTFT_MS=5000 SLO_ITL_MS=50
PROM=http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090/api/v1/query
q() { curl -s --data-urlencode "query=$1" --data-urlencode "time=$2" "$PROM" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(" ".join("%s=%.4g" % (",".join(v for k, v in x["metric"].items() if k == "transfer_type") or "v", float(x["value"][1])) for x in r) or "none")'; }

for OFF in ${@:-512 0}; do
  echo "=== $(date -u +%T) offload $OFF GiB: apply"
  { for v in KV_OFFLOADING_SIZE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS SPEC_TOKENS \
             MOE_BACKEND MAX_CUDAGRAPH_CAPTURE_SIZE ASYNC_SCHEDULING PERFORMANCE_MODE OPTIMIZATION_LEVEL; do
      case $v in KV_OFFLOADING_SIZE) echo "$v=$OFF" ;; GPU_MEMORY_UTILIZATION) echo "$v=0.95" ;; SPEC_TOKENS) echo "$v=5" ;; *) echo "$v=" ;; esac
    done; } > "$K/params.env"
  T0=$SECONDS
  bash "$K/apply_config.sh" > "$K/spike53-apply-$OFF.log" 2>&1; RC=$?
  echo "SPIKE $OFF apply exit=$RC in $((SECONDS - T0)) s"
  [ "$RC" = 0 ] || { tail -20 "$K/spike53-apply-$OFF.log"; continue; }
  POD=$(kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
  kubectl -n $NS logs "$POD" | grep -E 'Available KV cache memory|GPU KV cache size|offload|Offload' | sed 's/^/vllm: /' | tail -12
  # shellcheck disable=SC1091
  source "$K/lib_load.sh"
  # inputs.json size and AIPerf start-up, sampled while the level runs (the run directory is removed at the end).
  ( sleep 240; for c in 0 1; do echo "SPIKE $OFF client $c inputs.json: $(kubectl -n $NS exec aiperf-client-$c -- sh -c 'du -h /work/out/run/inputs.json 2>/dev/null; df -h /work | tail -1')" | tr '\n' ' '; echo; done ) &
  # Keep each client's per-request export (run_level deletes /work/out at the end): revisit TTFT
  # (turn_index >= 1), restarted sessions (turn_index 0 after the creation phase).
  ( sleep $((LEVEL_SECONDS + 45)); for c in 0 1; do kubectl -n $NS exec aiperf-client-$c -- sh -c "cp /work/out/run/profile_export.jsonl /work/keep-${TAG:-run}-$OFF.jsonl" && echo "SPIKE $OFF client $c export kept in /work/keep-${TAG:-run}-$OFF.jsonl"; done ) &
  S0=$(date +%s)
  run_level "$SESSIONS" "$LEVEL_SECONDS" "spike-$OFF"; RL=$?
  S1=$(date +%s); W=$((S1 - S0)); WS=$((S1 - S0 - STEADY_FROM - 100))
  wait
  echo "SPIKE $OFF run_level exit=$RL, ${W} s"
  echo "SPIKE $OFF prefix_hit_level=$(q "sum(increase(vllm:prefix_cache_hits_total{namespace=\"$NS\"}[${W}s]))/sum(increase(vllm:prefix_cache_queries_total{namespace=\"$NS\"}[${W}s]))" "$S1")" \
       "prefix_hit_steady=$(q "sum(increase(vllm:prefix_cache_hits_total{namespace=\"$NS\"}[${WS}s]))/sum(increase(vllm:prefix_cache_queries_total{namespace=\"$NS\"}[${WS}s]))" "$S1")"
  # CPU (external) hits per prompt token: > 0 only if the offload is read back (the reload this spike is about).
  echo "SPIKE $OFF cpu_hit_steady=$(q "(sum(increase(vllm:external_prefix_cache_hits_total{namespace=\"$NS\"}[${WS}s])) or vector(0))/sum(increase(vllm:prompt_tokens_total{namespace=\"$NS\"}[${WS}s]))" "$S1")" \
       "gpu_hit_per_prompt_steady=$(q "sum(increase(vllm:prefix_cache_hits_total{namespace=\"$NS\"}[${WS}s]))/sum(increase(vllm:prompt_tokens_total{namespace=\"$NS\"}[${WS}s]))" "$S1")"
  echo "SPIKE $OFF offload_GB $(q "sum by (transfer_type) (increase(vllm:kv_offload_total_bytes_total{namespace=\"$NS\"}[${W}s]))/1e9" "$S1")" \
       "cpu_cache_usage_max=$(q "max(max_over_time(vllm:kv_offload_cpu_cache_usage_perc{namespace=\"$NS\"}[${W}s]))" "$S1")"
  echo "SPIKE $OFF kv_usage_max=$(q "max(max_over_time(vllm:kv_cache_usage_perc{namespace=\"$NS\"}[${W}s]))" "$S1")" \
       "running_max=$(q "max(max_over_time(vllm:num_requests_running{namespace=\"$NS\"}[${W}s]))" "$S1")" \
       "waiting_max=$(q "max(max_over_time(vllm:num_requests_waiting{namespace=\"$NS\"}[${W}s]))" "$S1")" \
       "preemptions=$(q "sum(increase(vllm:num_preemptions_total{namespace=\"$NS\"}[${W}s]))" "$S1")"
  # Prefill rate: prompt tokens computed (not served from a cache) per second, per minute of the level.
  for m in $(seq 1 $((W / 60))); do
    echo "SPIKE $OFF minute $m prompt_tok_s=$(q "sum(rate(vllm:prompt_tokens_total{namespace=\"$NS\"}[60s]))" $((S0 + 60 * m))) computed_tok_s=$(q "sum(rate(vllm:prompt_tokens_total{namespace=\"$NS\"}[60s])) - sum(rate(vllm:prefix_cache_hits_total{namespace=\"$NS\"}[60s])) - (sum(rate(vllm:external_prefix_cache_hits_total{namespace=\"$NS\"}[60s])) or vector(0))" $((S0 + 60 * m))) gen_tok_s=$(q "sum(rate(vllm:generation_tokens_total{namespace=\"$NS\"}[60s]))" $((S0 + 60 * m)))"
  done
done
echo "=== $(date -u +%T) spike done"
