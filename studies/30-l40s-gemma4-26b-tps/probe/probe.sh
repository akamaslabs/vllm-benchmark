#!/bin/bash
# Startup probe for 30-l40s-gemma4-26b-tps (README "Startup probe"). Runs outside Akamas, with
# the L40S node to itself, from the toolbox or the workstation (it only needs kubectl):
#   mkdir -p /tmp/probe30 && KP_OUT=/tmp/probe30/results setsid nohup bash probe/probe.sh > /tmp/probe30/probe.log 2>&1 &
# For each combination: a params.env (the baseline plus the combination's overrides), then
# ../k8s/apply_config.sh, then bench_in_pod.py inside vllm-0. It answers, before any
# experiment budget is spent:
#   - does Gemma 4 26B-A4B FP8 load and serve on the L40S with vLLM 0.29.0 (B-default, which
#     also downloads the model onto the node and prints the KV cache size);
#   - which attention backends start (A-*): vLLM forces TRITON_ATTN for this model on SM 8.9;
#   - fp8 KV with TRITON_ATTN (K-fp8);
#   - which FP8 linear kernels start and how fast they are (L-*);
#   - the edges of the study's domains (E-*): the memory corner, eager/O0, O3/throughput;
#   - Gemma 4's MTP speculative decoding (M-*, drafter google/gemma-4-26B-A4B-it-assistant):
#     does it start on Ada, and does it gain at batch 1 and at 64 concurrent requests;
#     acceptance on English and Italian prompts.
# KP_COMBOS overrides the list (name, then KEY=VALUE overrides; "-" for none).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=$(dirname "$HERE"); export STUDY_DIR
OUT=${KP_OUT:-$HERE/results}; mkdir -p "$OUT"
NS=llm-l40s
BASELINE='GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_DTYPE=auto
PERFORMANCE_MODE=balanced
OPTIMIZATION_LEVEL=2
ENFORCE_EAGER=false
SCHEDULING_POLICY=fcfs
ASYNC_SCHEDULING=true
MAX_CUDAGRAPH_CAPTURE_SIZE=512
BLOCK_SIZE=16'
COMBOS=${KP_COMBOS:-"
B-default     -
K-fp8         KV_CACHE_DTYPE=fp8
A-flashinfer  ATTENTION_BACKEND=FLASHINFER
A-fa          ATTENTION_BACKEND=FLASH_ATTN
A-triton      ATTENTION_BACKEND=TRITON_ATTN
L-cutlass     LINEAR_BACKEND=cutlass
L-triton      LINEAR_BACKEND=triton
L-marlin      LINEAR_BACKEND=marlin
L-torch       LINEAR_BACKEND=torch
E-memory      GPU_MEMORY_UTILIZATION=0.94 MAX_NUM_SEQS=512 MAX_NUM_BATCHED_TOKENS=16384 KV_CACHE_DTYPE=fp8
E-eager       ENFORCE_EAGER=true OPTIMIZATION_LEVEL=0 ASYNC_SCHEDULING=false SCHEDULING_POLICY=priority BLOCK_SIZE=128 PERFORMANCE_MODE=interactivity MAX_NUM_SEQS=16 MAX_NUM_BATCHED_TOKENS=512
E-o3          OPTIMIZATION_LEVEL=3 PERFORMANCE_MODE=throughput BLOCK_SIZE=48 MAX_CUDAGRAPH_CAPTURE_SIZE=16 GPU_MEMORY_UTILIZATION=0.80
M-mtp2        SPEC_METHOD=mtp SPEC_TOKENS=2
M-mtp4        SPEC_METHOD=mtp SPEC_TOKENS=4
M-mtp2-fp8    SPEC_METHOD=mtp SPEC_TOKENS=2 KV_CACHE_DTYPE=fp8
"}
while read -r NAME OVR; do
  [ -n "${NAME:-}" ] || continue
  P=$OUT/$NAME.params.env
  printf '%s\n' "$BASELINE" > "$P"
  for kv in $OVR; do
    [ "$kv" = - ] && continue
    if grep -q "^${kv%%=*}=" "$P"; then
      python3 - "$P" "$kv" <<'PY'
import sys
p, kv = sys.argv[1:3]; k = kv.split('=', 1)[0]
lines = [kv if l.startswith(k + '=') else l for l in open(p).read().splitlines()]
open(p, 'w').write('\n'.join(lines) + '\n')
PY
    else
      echo "$kv" >> "$P"
    fi
  done
  echo "=== $NAME: $OVR ($(date -u +%T))"
  T0=$SECONDS
  RENDER_ALLOW_FA_FP8=1 PARAMS=$P RENDERED=$OUT/$NAME.sts.yaml bash "$STUDY_DIR/k8s/apply_config.sh" > "$OUT/$NAME.log" 2>&1
  RC=$?
  START_S=$((SECONDS - T0))
  if [ $RC -ne 0 ]; then
    printf '{"name":"%s","overrides":"%s","started":false,"apply_exit":%d,"startup_s":%d}\n' \
      "$NAME" "$OVR" "$RC" "$START_S" > "$OUT/$NAME.json"
    grep -E 'Error|Traceback|not supported|ValueError|RuntimeError' "$OUT/$NAME.log" | grep -v '^--- ' | tail -5
    continue
  fi
  R=$(kubectl -n $NS exec -i vllm-0 -c vllm -- python3 - < "$HERE/bench_in_pod.py" 2>>"$OUT/$NAME.log" \
      | grep '^BENCH_RESULT ' | cut -d' ' -f2-)
  python3 - "$NAME" "$OVR" "$START_S" "${R:-null}" > "$OUT/$NAME.json" <<'PY'
import json, sys
name, ovr, start, res = sys.argv[1:5]
bench = json.loads(res)
print(json.dumps({"name": name, "overrides": ovr, "started": bench is not None,
                  "startup_s": int(start), "bench": bench}))
PY
  grep -E 'Selected .*Kernel|Using .*[Bb]ackend|default MoE config|heterogeneous head|TRITON_ATTN|Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency' \
    "$OUT/$NAME.log" | grep -v 'apply_config' | sort -u | head -12 > "$OUT/$NAME.kernels.txt"
  cat "$OUT/$NAME.kernels.txt"
done <<< "$COMBOS"
kubectl -n $NS scale sts vllm --replicas=0
python3 "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
