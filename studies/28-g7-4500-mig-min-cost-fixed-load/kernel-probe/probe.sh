#!/bin/bash
# Kernel probe for 28-g7-4500-mig-min-cost-fixed-load (README "Kernel probe"). Runs on the
# toolbox, outside Akamas, with the GPU node to itself (no study running on it):
#   setsid nohup bash kernel-probe/probe.sh > kernel-probe/probe.log 2>&1 &
# For each combination: write a params.env, run ../k8s/apply_config.sh with ONE replica on a
# 1g.16gb slice (the other slice idle: this ranks kernels, it does not measure capacity),
# then bench_in_pod.py inside vllm-0. KP_COMBOS overrides the list (same 4 columns).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=$(dirname "$HERE"); export STUDY_DIR
OUT=${KP_OUT:-$HERE/results}; mkdir -p "$OUT"
NS=gpu-sharing
# name          linear_backend      attention_backend  kv_cache_dtype
COMBOS=${KP_COMBOS:-"
L-auto          auto                FLASHINFER         auto
L-cutlass       cutlass             FLASHINFER         auto
L-fi-cutlass    flashinfer_cutlass  FLASHINFER         auto
L-deepgemm      deep_gemm           FLASHINFER         auto
L-marlin        marlin              FLASHINFER         auto
L-humming       humming             FLASHINFER         auto
L-triton        triton              FLASHINFER         auto
A-fa            auto                FLASH_ATTN         auto
A-triton        auto                TRITON_ATTN        auto
A-auto          auto                auto               auto
K-fi-fp8        auto                FLASHINFER         fp8
K-triton-fp8    auto                TRITON_ATTN        fp8
K-fa-fp8        auto                FLASH_ATTN         fp8
"}
while read -r NAME LB AB KV; do
  [ -n "${NAME:-}" ] || continue
  P=$OUT/$NAME.params.env
  cat > "$P" <<EOF
MIG_PROFILE=1g.16gb
CPU_LIMIT=7000
MEMORY_LIMIT=28000
GPU_MEMORY_UTILIZATION=0.90
KV_CACHE_DTYPE=$KV
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
LINEAR_BACKEND=$LB
ATTENTION_BACKEND=$AB
EOF
  echo "=== $NAME: linear=$LB attention=$AB kv=$KV ($(date -u +%T))"
  T0=$SECONDS
  RENDER_ALLOW_FA_FP8=1 PARAMS=$P RENDERED=$OUT/$NAME.sts.yaml REPLICAS_OVERRIDE=1 \
    bash "$STUDY_DIR/k8s/apply_config.sh" > "$OUT/$NAME.log" 2>&1
  RC=$?
  START_S=$((SECONDS - T0))
  if [ $RC -ne 0 ]; then
    printf '{"name":"%s","linear":"%s","attention":"%s","kv":"%s","started":false,"apply_exit":%d,"startup_s":%d}\n' \
      "$NAME" "$LB" "$AB" "$KV" "$RC" "$START_S" > "$OUT/$NAME.json"
    grep -E 'Error|Traceback|not supported|ValueError' "$OUT/$NAME.log" | tail -5
    continue
  fi
  R=$(kubectl -n $NS exec -i vllm-0 -c vllm -- python3 - < "$HERE/bench_in_pod.py" 2>>"$OUT/$NAME.log" \
      | grep '^BENCH_RESULT ' | cut -d' ' -f2-)
  python3 - "$NAME" "$LB" "$AB" "$KV" "$START_S" "${R:-null}" > "$OUT/$NAME.json" <<'PY'
import json, sys
name, lb, ab, kv, start, res = sys.argv[1:7]
bench = json.loads(res)
print(json.dumps({"name": name, "linear": lb, "attention": ab, "kv": kv, "started": bench is not None,
                  "startup_s": int(start), "bench": bench}))
PY
  grep -E 'Selected .*Kernel|Using .*[Bb]ackend|attention backend' "$OUT/$NAME.log" | head -5 > "$OUT/$NAME.kernels.txt"
  cat "$OUT/$NAME.kernels.txt"
done <<< "$COMBOS"
kubectl -n $NS scale sts vllm --replicas=0
python3 "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
