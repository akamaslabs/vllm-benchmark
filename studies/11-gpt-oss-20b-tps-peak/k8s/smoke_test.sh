#!/bin/bash
# Smoke test for 11-gpt-oss-20b-tps-peak — run ONCE on the toolbox host, BEFORE
# `akamas start study`, as soon as the GPU node is back. Nothing in this study has run on a
# real L4 yet (2026-09-14): the vLLM recipe lists Ada Lovelace as "actively working on", the
# kernel path (TRITON_ATTN + Marlin MXFP4 MoE) was verified only from the v0.29.0 source.
#
# It renders the template with five representative configurations, applies each, waits for
# the rollout, sends one chat completion, and prints the backend/kernel lines from vLLM's
# log. Every configuration must come up; if one does not, fix the study's domains or
# parameterConstraints before starting it (each such failure inside the study costs the
# 20-minute rollout deadline).
#
# Usage: bash smoke_test.sh            # all five configs
#        bash smoke_test.sh tp1-auto   # a single config by name
set -u
STUDY=/work/vllm-benchmark/studies/11-gpt-oss-20b-tps-peak
TEMPLATE=$STUDY/k8s/01-deployment_template.yaml
OUT=/tmp/vllm-smoke.yaml

# name | gmu | seqs | batched | kv dtype | perf | opt | block | eager | policy | async | cudagraph | TP | DP | EP
# (names must be valid Kubernetes object names: lowercase, hyphens only)
# tp1-block96 exercises a block_size that does not divide gpt-oss's 128-token sliding
# window (48/80/96/112 in the study domain) against the hybrid KV-cache manager.
CONFIGS='
tp1-auto     0.85 256 4096 auto     balanced      2 16  false fcfs     true  256 1 1 false
tp1-fp8      0.85 512 8192 fp8      throughput    3 64  false priority true  512 1 1 false
tp1-block96  0.85 256 4096 auto     balanced      2 96  false fcfs     true  256 1 1 false
tp2-ep       0.85 512 8192 fp8_e4m3 balanced      2 32  false fcfs     true  256 2 1 true
dp4          0.85 256 4096 auto     interactivity 1 16  true  fcfs     false 256 1 4 false
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
