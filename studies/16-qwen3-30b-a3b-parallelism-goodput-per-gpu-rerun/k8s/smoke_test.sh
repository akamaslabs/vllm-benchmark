#!/bin/bash
# Smoke test for 15-qwen3-30b-a3b-parallelism-goodput-per-gpu — run ONCE on the toolbox
# host, BEFORE `akamas start study`, and only while no other study owns Deployment/vllm
# (study 14 was still running against this node on 2026-09-17).
#
# Nothing about this model has been run on this node before, so this smoke test is not a
# formality: it is where three assumptions written into the study manifest get checked.
#
#   1. FP8 BLOCK-QUANT ON ADA. Qwen3-30B-A3B-Instruct-2507-FP8 is quantized with
#      weight_block_size [128, 128]. On SM 8.9 vLLM runs such checkpoints through FP8
#      Marlin as w8a16, keeping the weights 8-bit in VRAM (29.03 GiB total). If instead
#      they were dequantized to bf16 the model would need 56.9 GiB and NOTHING would fit
#      — every configuration below would fail at load. Watch `Loading weights took` and
#      the memory figures printed at the end of each config.
#   2. THE EXPERT-PARALLEL CONSTRAINT. moe_intermediate_size is 768. Sharded over 4 ranks
#      that is 192, which is not a multiple of the 128-wide quantization block, so a
#      TP x DP = 4 layout with expert parallelism OFF is expected to fail at
#      create_weights. `tp4-noep` tests exactly that, and `tp2-noep` (768/2 = 384, a clean
#      multiple) is its control. If tp4-noep comes up fine, the manifest's constraint
#      "expert parallelism required at 4 GPUs" is too strict and should be dropped.
#   3. PIPELINE PARALLELISM AT ALL, and with data parallelism. Only PP on its own is
#      clearly documented upstream; PP combined with DP on vLLM V1 was NOT verified from
#      the source, so `pp2` and `pp4` check the plain case here. If they start, the PP
#      cells of the study are sound; if a later PP+DP trial fails, add the constraint
#      `pipeline_parallel_size == 1 || data_parallel_size == 1` to the manifest.
#      Note async_scheduling is false on every PP row: vLLM does not support the
#      combination (upstream issue #32701), which the manifest also enforces.
#   4. THE SAMPLER-WARMUP GUARD, re-derived for this model's 151936-token vocabulary:
#      gpu_memory_utilization x 22.03 + max_num_seqs x 0.00283 <= 21.63.
#      `tp2-guard` sits just inside it (0.88 / 780). A CUDA OOM in warmup_kernels there
#      means the constant is wrong — fix the manifest BEFORE starting the study.
#
# It renders the template with each configuration, applies it, waits for the rollout,
# sends one chat completion, and prints the backend/kernel lines from vLLM's log. Every
# configuration except `tp4-noep` must come up. Budget ~10 x 15-30 min.
#
# Usage: bash smoke_test.sh            # all ten configs
#        bash smoke_test.sh tp2-guard  # a single config by name
set -u
STUDY=/work/vllm-benchmark/studies/15-qwen3-30b-a3b-parallelism-goodput-per-gpu
TEMPLATE=$STUDY/k8s/01-deployment_template.yaml
OUT=/tmp/vllm-smoke.yaml

# name | gmu | seqs | batched | kv dtype | perf | opt | block | eager | policy | async | cudagraph | TP | DP | EP | PP
# (names must be valid Kubernetes object names: lowercase, hyphens only)
# Every row satisfies the study's seven parameterConstraints (batched >= seqs, sampler
# warmup, TP != 3, TP x DP x PP <= 4, weights-shard fit, EP at 4 expert-sharding ranks,
# no async scheduling with PP > 1) EXCEPT tp4-noep, which is here precisely to check that
# the EP one is needed — see the header.
# gmu 0.88 throughout: at 0.85 the 2-GPU layouts have only 4.6 GiB left after the 14.5 GiB
# weight shard, and the manifest's fit constraint puts their floor at ~0.845.
# disable_custom_all_reduce is not a column: it is rendered as "false" (vLLM's default) in
# every row, since the smoke test checks that configurations start, not how fast they are.
# Its log line ("Custom allreduce is disabled ...", or its absence) is worth reading here
# anyway — it tells you whether that parameter can do anything at all on this node.
CONFIGS='
tp4-ep      0.88 768 8192 auto     balanced   2 16 false fcfs true  512 4 1 true  1
tp2dp2-ep   0.88 768 8192 auto     balanced   2 16 false fcfs true  512 2 2 true  1
tp1dp4-ep   0.88 768 8192 auto     balanced   2 16 false fcfs true  512 1 4 true  1
tp2-ep      0.88 768 8192 auto     balanced   2 16 false fcfs true  512 2 1 true  1
tp1dp2-ep   0.88 768 8192 auto     balanced   2 16 false fcfs true  512 1 2 true  1
tp2-noep    0.88 768 8192 fp8_e4m3 balanced   2 32 false fcfs true  512 2 1 false 1
tp4-noep    0.88 768 8192 fp8_e4m3 balanced   2 32 false fcfs true  512 4 1 false 1
tp2-guard   0.88 780 8192 fp8_e4m3 throughput 3 96 false fcfs true  512 2 1 true  1
pp2         0.88 768 8192 auto     balanced   2 16 false fcfs false 512 1 1 false 2
pp4         0.88 768 8192 auto     balanced   2 16 false fcfs false 512 1 1 false 4
'
ONLY=${1:-}
FAILED=0
while read -r name gmu seqs batched kv perf opt block eager policy async cg tp dp ep pp; do
  [ -z "$name" ] && continue
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  echo "=================================================================="
  echo "== smoke config: $name (TP=$tp DP=$dp PP=$pp EP=$ep kv=$kv gmu=$gmu seqs=$seqs)"
  echo "=================================================================="
  sed -e "s#\${vLLM.gpu_memory_utilization}#$gmu#" -e "s#\${vLLM.max_num_seqs}#$seqs#" \
      -e "s#\${vLLM.max_num_batched_tokens}#$batched#" -e "s#\${vLLM.kv_cache_dtype}#$kv#" \
      -e "s#\${vLLM.performance_mode}#$perf#" -e "s#\${vLLM.optimization_level}#$opt#" \
      -e "s#\${vLLM.block_size}#$block#" -e "s#\${vLLM.enforce_eager}#$eager#" \
      -e "s#\${vLLM.scheduling_policy}#$policy#" -e "s#\${vLLM.async_scheduling}#$async#" \
      -e "s#\${vLLM.max_cudagraph_capture_size}#$cg#" -e "s#\${vLLM.tensor_parallel_size}#$tp#" \
      -e "s#\${vLLM.data_parallel_size}#$dp#" -e "s#\${vLLM.enable_expert_parallel}#$ep#" \
      -e "s#\${vLLM.disable_custom_all_reduce}#false#" \
      -e "s#\${vLLM.pipeline_parallel_size}#$pp#" \
      "$TEMPLATE" > "$OUT"
  for flag in enforce-eager async-scheduling enable-expert-parallel disable-custom-all-reduce; do
    sed -i "s/--${flag}=true/--${flag}/; s/--${flag}=false/--no-${flag}/" "$OUT"
  done
  sed -i -E '/\$\{vLLM\./d' "$OUT"
  kubectl apply -f "$OUT" -n llm-serving
  # 2100s: the 1800s progressDeadlineSeconds of the template plus margin for the 31 GB
  # cold-cache download on the very first configuration.
  if ! kubectl rollout status deployment/vllm -n llm-serving --timeout=2100s; then
    echo "!! $name: rollout FAILED"; FAILED=$((FAILED+1))
    kubectl logs deployment/vllm -n llm-serving -c vllm --tail=200 || true
    kubectl logs deployment/vllm -n llm-serving -c vllm --tail=200 --previous 2>/dev/null || true
    continue
  fi
  echo "-- backend / kernel selection (expect on L4: TRITON_ATTN + FP8 Marlin w8a16, NOT a bf16 dequant):"
  kubectl logs deployment/vllm -n llm-serving -c vllm --tail=-1 | grep -iE 'attention backend|marlin|fp8|block_?size|quantization|expert parallel|Loading weights took|GPU KV cache size|Maximum concurrency' || true
  echo "-- one chat completion:"
  kubectl run smoke-curl-$name --rm -i --restart=Never --image=curlimages/curl:8.10.1 -n llm-benchmark -- \
    curl -s -m 120 http://vllm.llm-serving.svc.cluster.local:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"In one sentence, what is tensor parallelism?"}],"max_tokens":128}' \
    </dev/null | head -c 1500; echo   # </dev/null: keep kubectl -i from eating the CONFIGS here-string
  echo "-- DCGM view (GPUs holding weights = active_gpus):"
  kubectl exec -n llm-serving deployment/vllm -c vllm -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
done <<< "$CONFIGS"
echo "=================================================================="
echo "smoke test finished: $FAILED failed configuration(s)"
echo "REMINDER: tp4-noep is EXPECTED to fail (768 / 4 ranks = 192, not a multiple of the"
echo "128-wide quantization block). If it passed, drop the EP-at-4-GPUs constraint from"
echo "the study manifest; if something else failed, fix the manifest before starting."
exit $FAILED
