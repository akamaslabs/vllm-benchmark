#!/bin/bash
# Startup probe for 32-l40s-gemma4-26b-awq-thinking (README "Runbook"; study 30's probe). Runs
# outside Akamas, with the L40S node to itself, from the toolbox or the workstation (it only
# needs kubectl):
#   mkdir -p /tmp/probe32 && KP_OUT=/tmp/probe32/results setsid nohup bash probe/probe.sh > /tmp/probe32/probe.log 2>&1 &
# For each combination: a params.env (the baseline plus the combination's overrides), then
# ../k8s/apply_config.sh, then bench_in_pod.py inside vllm-0. It answers, before any
# experiment budget is spent:
#   - does the customer's int4 checkpoint (cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit rev 0ef577a)
#     load and serve on the L40S with vLLM 0.29.0 and their flags (B-compose: their compose's
#     values; it also downloads the checkpoint onto the node and prints the KV cache size at
#     max_model_len 96000, with the vision tower loaded);
#   - fp8 KV on this checkpoint (K-fp8);
#   - which mixed-precision (W4A16) linear kernels start and how fast they are (L-*): auto takes
#     Marlin on SM 8.9; triton and humming are the alternatives that can run here (exllama
#     needs float16 activations, conch a package the image lacks);
#   - the edges of the study's domains (E-*): the memory corner, O3/throughput at gmu 0.80;
#   - Gemma 4's MTP drafter on an int4 target (M-*): does it start, does it gain, acceptance;
#   - study 30's best configuration, the "study 30 best" preset, starts (S30-best).
# Every bench request turns thinking off (bench_in_pod.py), so the rows compare with study 30's
# probe.
# KP_COMBOS overrides the list (name, then KEY=VALUE overrides; "-" for none).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=$(dirname "$HERE"); export STUDY_DIR
OUT=${KP_OUT:-$HERE/results}; mkdir -p "$OUT"
NS=llm-l40s
# The study's baseline: the compose's two values (and the always-rendered batch budget and
# enforce_eager); every other line empty = no flag = vLLM's default, as doNotRenderParameters
# renders it. A combination's overrides fill some of them in.
BASELINE='GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_DTYPE=
PERFORMANCE_MODE=
OPTIMIZATION_LEVEL=
ENFORCE_EAGER=false
SCHEDULING_POLICY=
ASYNC_SCHEDULING=
MAX_CUDAGRAPH_CAPTURE_SIZE=
BLOCK_SIZE=
LINEAR_BACKEND=
SPEC_METHOD=
SPEC_TOKENS='
COMBOS=${KP_COMBOS:-"
B-compose     -
K-fp8         KV_CACHE_DTYPE=fp8
L-triton      LINEAR_BACKEND=triton
L-humming     LINEAR_BACKEND=humming
E-memory      GPU_MEMORY_UTILIZATION=0.94 MAX_NUM_SEQS=512 MAX_NUM_BATCHED_TOKENS=16384 KV_CACHE_DTYPE=fp8
E-o3          OPTIMIZATION_LEVEL=3 PERFORMANCE_MODE=throughput BLOCK_SIZE=48 MAX_CUDAGRAPH_CAPTURE_SIZE=16 GPU_MEMORY_UTILIZATION=0.80
M-mtp2        SPEC_METHOD=mtp SPEC_TOKENS=2
M-mtp2-fp8    SPEC_METHOD=mtp SPEC_TOKENS=2 KV_CACHE_DTYPE=fp8
S30-best      GPU_MEMORY_UTILIZATION=0.94 MAX_NUM_SEQS=451 MAX_NUM_BATCHED_TOKENS=16384 KV_CACHE_DTYPE=fp8 OPTIMIZATION_LEVEL=1 MAX_CUDAGRAPH_CAPTURE_SIZE=179 SPEC_METHOD=mtp SPEC_TOKENS=3
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
  grep -E 'Selected .*Kernel|Using .*[Bb]ackend|LinearKernel|Marlin|default MoE config|heterogeneous head|TRITON_ATTN|Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency' \
    "$OUT/$NAME.log" | grep -v 'apply_config' | sort -u | head -12 > "$OUT/$NAME.kernels.txt"
  # What the engine resolved (the baseline leaves these flags out): the values the baseline
  # records and the compose-based presets render must match B-compose's (README "Runbook").
  grep -o -E "'max_cudagraph_capture_size': [0-9]+|'cudagraph_mode': <[^>]*>|enable_prefix_caching=[A-Za-z]+|enable_chunked_prefill=[A-Za-z]+" \
    "$OUT/$NAME.log" | sort -u >> "$OUT/$NAME.kernels.txt"
  cat "$OUT/$NAME.kernels.txt"
done <<< "$COMBOS"
kubectl -n $NS scale sts vllm --replicas=0
python3 "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
