#!/bin/bash
# Smoke test for 13-gpt-oss-20b-tp-goodput-per-gpu — run ONCE on the toolbox host, BEFORE
# `akamas start study`, and only while no other study owns Deployment/vllm (study 12 was
# still running against it on 2026-09-15). Study 12's five configurations (TP1 auto/fp8/
# block 96, TP2+EP, DP4) all passed on this node on 2026-09-15, so the kernel path
# (TRITON_ATTN + Marlin MXFP4 MoE) is no longer in question. This study's six
# configurations cover its own TP >= 2 domain instead:
#   tp2-auto / tp2-ep-fp8 / tp4 / tp2-dp2-ep   the three valid topologies (TP2/DP1, TP4,
#                                              TP2/DP2), each with a different KV dtype,
#                                              compile mode and expert-parallel setting;
#   tp2-guard-090 / tp2-guard-085              the EMPIRICAL CHECK of the sampler-warmup
#       memory guard in the study manifest: both sit just inside the line
#       gpu_memory_utilization x 22.03 + max_num_seqs x 0.00375 <= 21.63 (0.90/470 and
#       0.85/760 against the 0.90/1024 that OOMed in study 12's experiment 12).
#       tp2-guard-090 also replays that experiment's other settings (fp8_e4m3, eager,
#       block 80, EP) at TP2. If either fails with a CUDA OOM in warmup_kernels, the guard's
#       assumption that TP does not change the per-GPU sampler footprint is wrong: lower
#       the 21.63 constant (or raise 0.00375) in the manifest BEFORE starting the study.
#
# It renders the template with each configuration, applies it, waits for the rollout,
# sends one chat completion, and prints the backend/kernel lines from vLLM's log. Every
# configuration must come up; if one does not, fix the study's domains or
# parameterConstraints before starting it (each such failure inside the study costs up to
# the 20-minute rollout deadline — apply_config.sh's crash-loop guard shortens it, the
# smoke test does not).
#
# Usage: bash smoke_test.sh                # all six configs
#        bash smoke_test.sh tp2-guard-090  # a single config by name
set -u
STUDY=/work/vllm-benchmark/studies/13-gpt-oss-20b-tp-goodput-per-gpu
TEMPLATE=$STUDY/k8s/01-deployment_template.yaml
OUT=/tmp/vllm-smoke.yaml

# name | gmu | seqs | batched | kv dtype | perf | opt | block | eager | policy | async | cudagraph | TP | DP | EP
# (names must be valid Kubernetes object names: lowercase, hyphens only)
# Every row satisfies the study's five parameterConstraints (batched >= seqs, TP x DP <= 4,
# TP != 3, weights-shard fit, sampler-warmup guard) — check again after editing a row.
CONFIGS='
tp2-auto       0.85 256 4096 auto     balanced      2 16 false fcfs     true  256  2 1 false
tp2-ep-fp8     0.85 512 8192 fp8_e4m3 balanced      2 32 false fcfs     true  256  2 1 true
tp4            0.85 512 8192 fp8      throughput    3 64 false priority true  512  4 1 false
tp2-dp2-ep     0.85 256 4096 auto     interactivity 1 16 true  fcfs     false 256  2 2 true
tp2-guard-090  0.90 470 4096 fp8_e4m3 balanced      3 80 true  fcfs     false 1022 2 1 true
tp2-guard-085  0.85 760 4096 auto     balanced      2 16 false fcfs     true  256  2 1 false
'
ONLY=${1:-}
FAILED=0
while read -r name gmu seqs batched kv perf opt block eager policy async cg tp dp ep; do
  [ -z "$name" ] && continue
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  echo "=================================================================="
  echo "== smoke config: $name (TP=$tp DP=$dp EP=$ep kv=$kv gmu=$gmu seqs=$seqs)"
  echo "=================================================================="
  sed -e "s#\${vLLM.gpu_memory_utilization}#$gmu#" -e "s#\${vLLM.max_num_seqs}#$seqs#" \
      -e "s#\${vLLM.max_num_batched_tokens}#$batched#" -e "s#\${vLLM.kv_cache_dtype}#$kv#" \
      -e "s#\${vLLM.performance_mode}#$perf#" -e "s#\${vLLM.optimization_level}#$opt#" \
      -e "s#\${vLLM.block_size}#$block#" -e "s#\${vLLM.enforce_eager}#$eager#" \
      -e "s#\${vLLM.scheduling_policy}#$policy#" -e "s#\${vLLM.async_scheduling}#$async#" \
      -e "s#\${vLLM.max_cudagraph_capture_size}#$cg#" -e "s#\${vLLM.tensor_parallel_size}#$tp#" \
      -e "s#\${vLLM.data_parallel_size}#$dp#" -e "s#\${vLLM.enable_expert_parallel}#$ep#" \
      "$TEMPLATE" > "$OUT"
  for flag in enforce-eager async-scheduling enable-expert-parallel; do
    sed -i "s/--${flag}=true/--${flag}/; s/--${flag}=false/--no-${flag}/" "$OUT"
  done
  sed -i -E '/\$\{vLLM\./d' "$OUT"
  kubectl apply -f "$OUT" -n llm-serving
  if ! kubectl rollout status deployment/vllm -n llm-serving --timeout=1500s; then
    echo "!! $name: rollout FAILED"; FAILED=$((FAILED+1))
    kubectl logs deployment/vllm -n llm-serving -c vllm --tail=200 || true
    kubectl logs deployment/vllm -n llm-serving -c vllm --tail=200 --previous 2>/dev/null || true
    continue
  fi
  echo "-- backend / kernel selection:"
  kubectl logs deployment/vllm -n llm-serving -c vllm --tail=-1 | grep -iE 'attention backend|mxfp4|marlin|quantization|expert parallel|Loading weights took|GPU KV cache size|Maximum concurrency|reasoning parser' || true
  echo "-- one chat completion (reasoning_effort low):"
  kubectl run smoke-curl-$name --rm -i --restart=Never --image=curlimages/curl:8.10.1 -n llm-benchmark -- \
    curl -s -m 120 http://vllm.llm-serving.svc.cluster.local:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"In one sentence, what is tensor parallelism?"}],"max_tokens":128,"reasoning_effort":"low"}' \
    </dev/null | head -c 1500; echo   # </dev/null: keep kubectl -i from eating the CONFIGS here-string
  echo "-- DCGM view (GPUs holding weights = active_gpus):"
  kubectl exec -n llm-serving deployment/vllm -c vllm -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
done <<< "$CONFIGS"
echo "=================================================================="
echo "smoke test finished: $FAILED failed configuration(s)"
exit $FAILED
