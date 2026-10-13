#!/bin/bash
# W4A16 kernel probe (Gemma 4 26B-A4B W4A16 on one L40S). Outside Akamas, from the
# workstation (it only needs kubectl on the vllm-bench-gatling cluster):
#   KP_OUT=/tmp/kb32 setsid nohup bash kernel-bench/probe.sh > /tmp/kb32.log 2>&1 &
# For each combination: a params.env (the customer's flags plus the combination's), then
# ../k8s/apply_config.sh, then bench_in_pod.py inside vllm-w4a16-0. summarize.py tabulates.
# KP_COMBOS overrides the list. One combination per line, three '|'-separated fields:
#   name | extra vLLM flags | extra environment (NAME=value ...)
# KP_BASE overrides the base params (defaults: the customer's compose on vLLM 0.29.0).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=$(dirname "$HERE")
export KUBE_CONTEXT=${KUBE_CONTEXT:-lab-vllm-bench-gatling}
kubectl() { command kubectl --context "$KUBE_CONTEXT" "$@"; }
OUT=${KP_OUT:-$HERE/results}; mkdir -p "$OUT"
NS=llm-serving
POD=vllm-w4a16-0
BASE=${KP_BASE:-'GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=96000
MAX_NUM_SEQS=64
KV_CACHE_DTYPE=auto'}
# Stage 1: every MoE and linear kernel vLLM 0.29.0 can run for this checkpoint on SM 8.9, and
# their environment variants (vLLM 0.29.0 envs.py). Expected default: Marlin for both.
COMBOS=${KP_COMBOS:-"
B-default         |                                                 |
X-marlin          | --moe-backend=marlin                            |
X-marlin-atomic   | --moe-backend=marlin                            | VLLM_MARLIN_USE_ATOMIC_ADD=1
X-marlin-a8fp8    | --moe-backend=marlin                            | VLLM_MARLIN_INPUT_DTYPE=fp8
X-marlin-a8int8   | --moe-backend=marlin                            | VLLM_MARLIN_INPUT_DTYPE=int8
X-triton          | --moe-backend=triton                            |
X-humming         | --moe-backend=humming                           |
X-humming-indexed | --moe-backend=humming                           | VLLM_HUMMING_MOE_GEMM_TYPE=indexed
X-humming-grouped | --moe-backend=humming                           | VLLM_HUMMING_MOE_GEMM_TYPE=grouped
X-humming-a8fp8   | --moe-backend=humming                           | VLLM_HUMMING_INPUT_QUANT_CONFIG={\"dtype\":\"float8e4m3\"}
X-humming-a8int8  | --moe-backend=humming                           | VLLM_HUMMING_INPUT_QUANT_CONFIG={\"dtype\":\"int8\"}
X-humming-f16acc  | --moe-backend=humming                           | VLLM_HUMMING_USE_F16_ACCUM=1
X-fi-trtllm       | --moe-backend=flashinfer_trtllm                 |
L-marlin          | --linear-backend=marlin                         |
L-humming         | --linear-backend=humming                        |
L-triton          | --linear-backend=triton                         |
L-conch           | --linear-backend=conch                          |
H-all             | --moe-backend=humming --linear-backend=humming  |
H-all-a8fp8       | --moe-backend=humming --linear-backend=humming  | VLLM_HUMMING_INPUT_QUANT_CONFIG={\"dtype\":\"float8e4m3\"}
"}
trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; echo "${s%"${s##*[![:space:]]}"}"; }
while IFS='|' read -r NAME ARGS ENVS; do
  NAME=$(trim "${NAME:-}"); ARGS=$(trim "${ARGS:-}"); ENVS=$(trim "${ENVS:-}")
  [ -n "$NAME" ] || continue
  [ -f "$OUT/$NAME.json" ] && { echo "=== $NAME: already done, skipping"; continue; }
  P=$OUT/$NAME.params.env
  { printf '%s\n' "$BASE"; printf 'VLLM_ARGS=%q\n' "$ARGS"; printf 'KERNEL_ENV=%q\n' "$ENVS"; } > "$P"
  echo "=== $NAME: args [$ARGS] env [$ENVS] ($(date -u +%T))"
  T0=$SECONDS
  PARAMS=$P RENDERED=$OUT/$NAME.sts.yaml bash "$STUDY_DIR/k8s/apply_config.sh" > "$OUT/$NAME.log" 2>&1
  RC=$?
  START_S=$((SECONDS - T0))
  grep -E 'Selected .*Kernel|Using .*(Kernel|Experts|[Bb]ackend)|MoE backend|[Mm]oE config|configuration from|TRITON_ATTN|Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|W4A8|input_dtype|Overrid' \
    "$OUT/$NAME.log" | grep -v 'apply_config' | sed 's/^.*\] //' | sort -u | head -20 > "$OUT/$NAME.kernels.txt"
  if [ $RC -ne 0 ]; then
    printf '{"name":"%s","args":"%s","started":false,"apply_exit":%d,"startup_s":%d}\n' \
      "$NAME" "$ARGS" "$RC" "$START_S" > "$OUT/$NAME.json"
    grep -E 'Error|error:|not supported|ValueError|RuntimeError|AssertionError' "$OUT/$NAME.log" | grep -v '^--- ' | tail -4
    continue
  fi
  cat "$OUT/$NAME.kernels.txt"
  LONGENV=""; [ "$NAME" = B-default ] && LONGENV="BENCH_LONG=1"
  R=$(kubectl -n $NS exec -i $POD -c vllm -- env $LONGENV python3 - < "$HERE/bench_in_pod.py" 2>>"$OUT/$NAME.log" \
      | grep '^BENCH_RESULT ' | cut -d' ' -f2-)
  python3 - "$NAME" "$ARGS" "$ENVS" "$START_S" "${R:-null}" > "$OUT/$NAME.json" <<'PY'
import json, sys
name, args, envs, start, res = sys.argv[1:6]
bench = json.loads(res)
print(json.dumps({"name": name, "args": args, "env": envs, "started": True, "bench_ok": bench is not None,
                  "startup_s": int(start), "bench": bench}))
PY
  python3 "$HERE/summarize.py" "$OUT" | grep -E "^(name|$NAME) "
done <<< "$COMBOS"
kubectl -n $NS scale sts vllm-w4a16 --replicas=0
python3 "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
