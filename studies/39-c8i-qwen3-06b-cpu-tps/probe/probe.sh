#!/bin/bash
# Calibration probe for 39-c8i-qwen3-06b-cpu-tps (README "Probe"). Runs outside Akamas, with
# the c8i node to itself (no study running), from the toolbox or the workstation (kubectl and
# python3 only):
#   mkdir -p /tmp/probe39 && KP_OUT=/tmp/probe39/results setsid nohup bash probe/probe.sh > /tmp/probe39/probe.log 2>&1 &
# It answers, before any experiment budget is spent:
#   Phase A, startup edges (one ../k8s/apply_config.sh per combination, then node_facts.sh
#     inside vllm-0 and the vLLM log lines):
#     - what the pod sees: lscpu / SMT siblings / NUMA nodes / AMX, the exclusive cpuset,
#       memory.max (= the 48Gi limit vLLM sizes the KV cache from);
#     - vLLM's own report: "Auto set (... GiB) for KV cache", "... GiB memory was consumed by
#       non-kv usages" (the engine RSS that sets the gpu_memory_utilization lower bound),
#       "KV cache size: N tokens", and the OpenMP binding (expected 20 core ids: 22
#       cores - VLLM_CPU_NUM_OF_RESERVED_CPU 2);
#     - whether the domain edges start: 0.90 (the baseline and the domain top) and 0.92
#       (vLLM's default: can the top extend to it?), 0.15 with the largest batch at the bottom, fp8 KV, block 32, the corner of the
#       scheduler domains; C-nochunk must be refused by render_statefulset.sh (exit 2, no
#       cluster call): chunked prefill off is impossible below max_num_batched_tokens 16384.
#   Phase B, the SLO knee of the baseline: apply the baseline again, then one AIPerf Job
#     (sweep_job_template.yaml) at constant rates KP_RATES for KP_STEP_S each, with the
#     study's workload; per step AIPerf's p95s and, if Prometheus answers (KP_PROM), vLLM's
#     150 s p95s (the watchdog's view), prefix-cache hit rate, KV usage, preemptions.
# summarize.py prints both tables and the RT_RATE / RT_RAMP_S proposal (results/summary.txt).
# KP_COMBOS overrides phase A (name, then KEY=VALUE overrides; "-" for none); KP_SKIP_A=1 or
# KP_SKIP_B=1 skips a phase.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=$(dirname "$HERE"); export STUDY_DIR
OUT=${KP_OUT:-$HERE/results}; mkdir -p "$OUT"
NS=llm-cpu
RATES=${KP_RATES:-"1 2 4 6 8 12 16 24"}
STEP_S=${KP_STEP_S:-240}
PROM=${KP_PROM:-http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090}
# The study's baseline step: vLLM 0.31.0 CPU online-serving defaults except
# gpu_memory_utilization 0.90 (akamas/gen_study.py BASE).
BASELINE='GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=128
KV_CACHE_DTYPE=auto
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true
LONG_PREFILL_TOKEN_THRESHOLD=0
STREAM_INTERVAL=1'
COMBOS=${KP_COMBOS:-"
B-default      -
E-gmu-high     GPU_MEMORY_UTILIZATION=0.92
E-gmu-low      GPU_MEMORY_UTILIZATION=0.15 MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192
E-gmu-low-fp8  GPU_MEMORY_UTILIZATION=0.15 KV_CACHE_DTYPE=fp8
K-fp8          KV_CACHE_DTYPE=fp8
S-block32      BLOCK_SIZE=32
S-block96      BLOCK_SIZE=96
E-corner       MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 LONG_PREFILL_TOKEN_THRESHOLD=4096 ENABLE_PREFIX_CACHING=false
C-nochunk      ENABLE_CHUNKED_PREFILL=false
"}

write_params() {  # $1 file, $2 overrides
  printf '%s\n' "$BASELINE" > "$1"
  local kv
  for kv in $2; do
    [ "$kv" = - ] && continue
    if grep -q "^${kv%%=*}=" "$1"; then
      python3 - "$1" "$kv" <<'PY'
import sys
p, kv = sys.argv[1:3]; k = kv.split('=', 1)[0]
lines = [kv if l.startswith(k + '=') else l for l in open(p).read().splitlines()]
open(p, 'w').write('\n'.join(lines) + '\n')
PY
    else
      echo "$kv" >> "$1"
    fi
  done
}

apply() {  # $1 name, $2 overrides. Sets RC and START_S.
  write_params "$OUT/$1.params.env" "$2"
  local t0=$SECONDS
  PARAMS=$OUT/$1.params.env RENDERED=$OUT/$1.sts.yaml bash "$STUDY_DIR/k8s/apply_config.sh" > "$OUT/$1.log" 2>&1
  RC=$?; START_S=$((SECONDS - t0))
}

# --- Phase A: startup edges -------------------------------------------------------------
if [ "${KP_SKIP_A:-0}" != 1 ]; then
  while read -r NAME OVR; do
    [ -n "${NAME:-}" ] || continue
    echo "=== $NAME: $OVR ($(date -u +%T))"
    apply "$NAME" "$OVR"
    if [ "$RC" -eq 0 ]; then
      kubectl -n $NS exec -i vllm-0 -c vllm -- bash -s < "$HERE/node_facts.sh" > "$OUT/$NAME.facts.txt" 2>&1
    fi
    # The vLLM lines the decisions rest on (apply_config.sh dumped the full log).
    { grep -A4 'OpenMP thread binding info' "$OUT/$NAME.log" | head -8
      grep -E 'Auto set|Explicitly set|non-kv usages|KV cache size|Maximum concurrency|Using .*[Bb]ackend|LD_PRELOAD|get_mempolicy|Operation not permitted|Model loading took|warm-up e2e' "$OUT/$NAME.log" \
        | grep -v 'apply_config' | sort -u | head -16
    } > "$OUT/$NAME.vllm.txt"
    python3 "$HERE/summarize.py" --parse "$NAME" "$OVR" "$RC" "$START_S" "$OUT" > "$OUT/$NAME.json"
    if [ "$RC" -ne 0 ]; then
      grep -E '^error:|Error|Traceback|ValueError|RuntimeError|OOMKilled' "$OUT/$NAME.log" | grep -v '^--- ' | tail -5
    else
      cat "$OUT/$NAME.vllm.txt"
    fi
  done <<< "$COMBOS"
fi

# --- Phase B: the baseline's SLO knee ----------------------------------------------------
if [ "${KP_SKIP_B:-0}" != 1 ]; then
  echo "=== sweep: baseline, rates $RATES, $STEP_S s each ($(date -u +%T))"
  apply B-sweep -
  if [ "$RC" -ne 0 ]; then
    echo "error: the baseline did not start for the sweep (see $OUT/B-sweep.log)"
  else
    kubectl -n $NS delete job -l app=aiperf-cpu --ignore-not-found --wait=true
    bash "$HERE/render_sweep_job.sh" "$RATES" "$STEP_S" "$OUT/sweep-job.yaml" || exit 2
    kubectl apply -f "$OUT/sweep-job.yaml"
    N=$(wc -w <<< "$RATES")
    DEADLINE=$((SECONDS + 900 + N * (STEP_S + 120)))
    while [ $SECONDS -lt $DEADLINE ]; do
      S=$(kubectl -n $NS get job aiperf-cpu-sweep -o jsonpath='{.status.succeeded}' 2>/dev/null)
      F=$(kubectl -n $NS get job aiperf-cpu-sweep -o jsonpath='{.status.failed}' 2>/dev/null)
      [ "${S:-0}" -ge 1 ] || [ "${F:-0}" -ge 1 ] && break
      sleep 30
    done
    kubectl -n $NS logs job/aiperf-cpu-sweep --tail=-1 > "$OUT/sweep-job.log" 2>&1
    grep -E '^STEP (START|FAILED|STOP)|SWEEP DONE' "$OUT/sweep-job.log"
    # One line per step: AIPerf's summary plus Prometheus over the step's last 150 s.
    python3 "$HERE/summarize.py" --sweep "$OUT/sweep-job.log" "$PROM" > "$OUT/sweep.jsonl"
    kubectl -n $NS delete job aiperf-cpu-sweep --ignore-not-found --wait=false
    kubectl -n $NS logs vllm-0 --tail=-1 > "$OUT/B-sweep.vllm-after.log" 2>&1
  fi
fi
kubectl -n $NS scale sts vllm --replicas=0
python3 "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
