#!/bin/bash
# Local tests of apply_config.sh's parameter section (RENDER_ONLY=1: no cluster needed), of the
# vLLM template's startup budget, and of lib_apply.sh (wait_for_node with a stub kubectl,
# startup_failure_reason on a fixture log).
# Usage: bash k8s/test_apply_config.sh   → "N passed, M failed", exit 1 on any failure.
HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
VARS="KV_OFFLOADING_SIZE GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS SPEC_TOKENS MOE_BACKEND MAX_CUDAGRAPH_CAPTURE_SIZE ASYNC_SCHEDULING PERFORMANCE_MODE OPTIMIZATION_LEVEL"
FIXED="--served-model-name zai-org/GLM-5.3-Flash --kv-cache-dtype fp8 --max-model-len 262144 --tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --tensor-parallel-size 4 --language-model-only"

# render NAME "VAR=value ..." → sets OUT (all output), ARGS1 (the args line: the one before "shm=") and RC
render() {
  local tmp; tmp=$(mktemp -d); mkdir -p "$tmp/k8s"
  for v in $VARS; do echo "$v=" >> "$tmp/k8s/params.env"; done
  for kv in $2; do sed -i.bak "s|^${kv%%=*}=.*|$kv|" "$tmp/k8s/params.env"; done
  OUT=$(RENDER_ONLY=1 STUDY_DIR="$tmp" bash "$HERE/apply_config.sh" 2>&1); RC=$?
  ARGS1=$(RENDER_ONLY=1 STUDY_DIR="$tmp" bash "$HERE/apply_config.sh" 2>/dev/null | tail -2 | head -1)
  rm -rf "$tmp"
}
ok()   { PASS=$((PASS + 1)); }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL: $1"; echo "$OUT" | sed 's/^/    /'; }
is()   { [ "$ARGS1" = "$2" ] && ok || bad "$1: args line '$ARGS1', expected '$2'"; }
has()  { [[ "$ARGS1" == *"$2"* ]] && ok || bad "$1: expected '$2' in the args line"; }
hasnt(){ [[ "$ARGS1" == *"$2"* ]] && bad "$1: unexpected '$2'" || ok; }
line() { echo "$OUT" | grep -qx -- "$2" && ok || bad "$1: expected line '$2'"; }
rc()   { [ "$RC" = "$2" ] && ok || bad "$1: exit $RC, expected $2"; }

render "all empty" ""
rc "all empty" 0; is "all empty" "$FIXED"; line "all empty" "shm=2 mem=96"

render "baseline" "GPU_MEMORY_UTILIZATION=0.95 SPEC_TOKENS=5"
rc "baseline" 0
is "baseline" "$FIXED --gpu-memory-utilization 0.95 --speculative-config.method mtp --speculative-config.num_speculative_tokens 5"

render "mtp 0" "SPEC_TOKENS=0"
rc "mtp 0" 0; hasnt "mtp 0" "--speculative-config"
render "mtp 6" "SPEC_TOKENS=6"
rc "mtp 6" 2

render "offload 256" "KV_OFFLOADING_SIZE=256"
rc "offload 256" 0; has "offload 256" "--kv-offloading-size 256 --kv-offloading-backend native"; line "offload 256" "shm=257 mem=352"
render "offload 0" "KV_OFFLOADING_SIZE=0"
rc "offload 0" 0; hasnt "offload 0" "--kv-offloading"; line "offload 0" "shm=2 mem=96"
render "offload 513" "KV_OFFLOADING_SIZE=513"
rc "offload 513" 2

# DCP is not a parameter any more (no MLA backend supports it on SM120, smoke 2026-11-17).
render "no dcp" ""
hasnt "no dcp" "--decode-context-parallel-size"
grep -q DECODE_CONTEXT_PARALLEL_SIZE "$HERE/params.env.template" && bad "params.env.template still renders DCP" || ok

render "mnbt < mns" "MAX_NUM_BATCHED_TOKENS=64 MAX_NUM_SEQS=128"
rc "mnbt < mns" 2
render "integers as reals" "MAX_NUM_SEQS=12.0 MAX_NUM_BATCHED_TOKENS=8192.0"
rc "integers as reals" 0; has "integers as reals" "--max-num-seqs 12 --max-num-batched-tokens 8192"

render "gmu scientific" "GPU_MEMORY_UTILIZATION=9.5e-1"
rc "gmu scientific" 0; has "gmu scientific" "--gpu-memory-utilization 0.950000"
render "gmu 1.5" "GPU_MEMORY_UTILIZATION=1.5"
rc "gmu 1.5" 2

render "moe nonsense" "MOE_BACKEND=nonsense"
rc "moe nonsense" 2
render "moe auto" "MOE_BACKEND=auto"
rc "moe auto" 0; hasnt "moe auto" "--moe-backend"
render "moe triton" "MOE_BACKEND=triton"
rc "moe triton" 0; has "moe triton" "--moe-backend triton"
# W4A16 checkpoint (2026-11-18): the WNA16 MoE candidates, not the FP8 ones.
render "moe marlin" "MOE_BACKEND=marlin"
rc "moe marlin" 0; has "moe marlin" "--moe-backend marlin"
render "moe deep_gemm" "MOE_BACKEND=deep_gemm"
rc "moe deep_gemm" 2

render "async false" "ASYNC_SCHEDULING=false"
rc "async false" 0; has "async false" "--no-async-scheduling"
render "async true" "ASYNC_SCHEDULING=true"
has "async true" "--async-scheduling"; hasnt "async true" "--no-async-scheduling"

render "perf mode" "PERFORMANCE_MODE=throughput OPTIMIZATION_LEVEL=3 MAX_CUDAGRAPH_CAPTURE_SIZE=64"
rc "perf mode" 0; has "perf mode" "--max-cudagraph-capture-size 64"; has "perf mode" "--performance-mode throughput"; has "perf mode" "--optimization-level 3"
render "perf mode bogus" "PERFORMANCE_MODE=fast"
rc "perf mode bogus" 2

# A token Akamas did not substitute (broken template) must stop the experiment.
tmp=$(mktemp -d); mkdir -p "$tmp/k8s"; for v in $VARS; do echo "$v=" >> "$tmp/k8s/params.env"; done
echo 'MAX_NUM_SEQS=${vllm.max_num_seqs}' >> "$tmp/k8s/params.env"
OUT=$(RENDER_ONLY=1 STUDY_DIR="$tmp" bash "$HERE/apply_config.sh" 2>&1); RC=$?; rm -rf "$tmp"
rc "unsubstituted token" 2

# Startup budget: a first start on an empty node downloads 328 GB, then ~4 min of JIT warmup and the
# weight load; the startupProbe must allow at least 3600 s once the container runs.
OUT=$(awk '/startupProbe/ { for (k = 1; k <= NF; k++) { if ($k ~ /periodSeconds/) p = $(k+1); if ($k ~ /failureThreshold/) f = $(k+1) } }
  END { gsub(/[^0-9]/, "", p); gsub(/[^0-9]/, "", f); print p * f }' "$HERE/01-vllm_template.yaml")
[ "${OUT:-0}" -ge 3600 ] && ok || bad "startupProbe budget ${OUT:-none} s < 3600 s"

# lib_apply.sh
# shellcheck disable=SC1091
source "$HERE/lib_apply.sh"
OUT=$(startup_failure_reason "$HERE/fixtures/mamba-blocks.log")
[[ "$OUT" == *"exceeds available Mamba cache blocks"* ]] && ok || bad "startup_failure_reason: '$OUT'"
OUT=$(startup_failure_reason "$HERE/fixtures/no-kv-blocks.log")
[[ "$OUT" == *"No available memory for the cache blocks"* ]] && ok || bad "startup_failure_reason no KV: '$OUT'"
STUB=$(mktemp -d)
printf '#!/bin/bash\nexit 0\n' > "$STUB/kubectl"; chmod +x "$STUB/kubectl"
( PATH="$STUB:$PATH" NODE_POLL_S=0 wait_for_node 1 >/dev/null 2>&1 ); RC=$?
rc "wait_for_node no node" 3
printf '#!/bin/bash\necho "ip-10-0-0-1 True"\n' > "$STUB/kubectl"
( PATH="$STUB:$PATH" NODE_POLL_S=0 wait_for_node 1 >/dev/null 2>&1 ); RC=$?
rc "wait_for_node ready" 0
rm -rf "$STUB"

# --- final review fixes ------------------------------------------------------------------
# C1: gpu_memory_utilization compared as numbers (vLLM writes Python str(float), full precision).
L='vllm:cache_config_info{block_size="64",gpu_memory_utilization="0.9017631142560059",num_gpu_blocks="812"} 1.0'
gmu_matches "$L" 0.9017631142560059; RC=$?; rc "gmu full precision" 0
gmu_matches 'vllm:cache_config_info{gpu_memory_utilization="0.95",num_gpu_blocks="9"} 1.0' 0.950000; RC=$?; rc "gmu 0.95 vs 0.950000" 0
gmu_matches 'vllm:cache_config_info{gpu_memory_utilization="0.9",num_gpu_blocks="9"} 1.0' 0.95; RC=$?; rc "gmu mismatch" 1
gmu_matches 'vllm:cache_config_info{num_gpu_blocks="9"} 1.0' 0.95; RC=$?; rc "gmu label missing" 1
# I2: the max_num_seqs the log says would fit.
OUT=$(max_seqs_hint "$HERE/fixtures/mamba-blocks.log"); [ "$OUT" = 114 ] && ok || bad "max_seqs_hint: '$OUT'"
OUT=$(max_seqs_hint "$HERE/fixtures/no-kv-blocks.log"); [ -z "$OUT" ] && ok || bad "max_seqs_hint without the hint: '$OUT'"
# I4 / I5 with a stub kubectl: the previous container's log names the failure; a pod lost with its node.
STUB=$(mktemp -d)
cat > "$STUB/kubectl" <<'STUBEOF'
#!/bin/bash
case "$*" in
  *"logs vllm-x --previous"*) cat "${STUB_PREV:-/dev/null}" ;;
  *"get pod vllm-x"*) printf '%s' "${STUB_POD:-ip-a/}" ;;
  *"get node ip-a"*) [ "${STUB_NODE:-}" = gone ] && exit 1; printf '%s' "${STUB_NODE:-True}" ;;
esac
STUBEOF
chmod +x "$STUB/kubectl"
( PATH="$STUB:$PATH" STUB_PREV="$HERE/fixtures/mamba-blocks.log" crash_reason vllm-x >/dev/null ); RC=$?; rc "crash_reason found" 0
( PATH="$STUB:$PATH" STUB_PREV=/dev/null crash_reason vllm-x >/dev/null ); RC=$?; rc "crash_reason none" 1
( PATH="$STUB:$PATH" STUB_POD="ip-a/" STUB_NODE=gone node_lost vllm-x >/dev/null ); RC=$?; rc "node_lost: node gone" 0
( PATH="$STUB:$PATH" STUB_POD="ip-a/" STUB_NODE=False node_lost vllm-x >/dev/null ); RC=$?; rc "node_lost: node NotReady" 0
( PATH="$STUB:$PATH" STUB_POD="ip-a/NodeShutdown" STUB_NODE=True node_lost vllm-x >/dev/null ); RC=$?; rc "node_lost: NodeShutdown" 0
( PATH="$STUB:$PATH" STUB_POD="ip-a/Evicted" STUB_NODE=True node_lost vllm-x >/dev/null ); RC=$?; rc "node_lost: evicted on a live node" 1
rm -rf "$STUB"
# Smoke 2026-11-17: the previous experiment's pod (crash-looping or terminating) must not count as the
# new one: only pods created after this apply are checked for restarts and crash reasons.
STUB=$(mktemp -d)
cat > "$STUB/kubectl" <<'STUBEOF'
#!/bin/bash
printf '%s\n' "${STUB_PODS:-}"
STUBEOF
chmod +x "$STUB/kubectl"
OUT=$(PATH="$STUB:$PATH" STUB_PODS=$'2026-11-17T15:20:00Z vllm-old 2\n2026-11-17T15:30:05Z vllm-new 0' newest_pod_since 2026-11-17T15:30:00Z)
[ "$OUT" = "vllm-new 0" ] && ok || bad "newest_pod_since: the new pod only ('$OUT')"
OUT=$(PATH="$STUB:$PATH" STUB_PODS='2026-11-17T15:20:00Z vllm-old 2' newest_pod_since 2026-11-17T15:30:00Z)
[ -z "$OUT" ] && ok || bad "newest_pod_since: no new pod yet ('$OUT')"
OUT=$(PATH="$STUB:$PATH" STUB_PODS='2026-11-17T15:30:05Z vllm-new ' newest_pod_since 2026-11-17T15:30:00Z)
[ "$OUT" = "vllm-new 0" ] && ok || bad "newest_pod_since: no restart count yet = 0 ('$OUT')"
rm -rf "$STUB"
OUT=$(startup_failure_reason "$HERE/fixtures/dcp-no-backend.log")
[[ "$OUT" == *"No valid attention backend"* ]] && ok || bad "startup_failure_reason DCP: '$OUT'"
# I6: checkpoint and tokenizer pinned to one revision.
# 2026-11-18: an anonymous download stalled for 8+ minutes (8 connections, no byte); a restart resumed
# at ~690 MB/s. Every attempt is bounded and retried (hf download resumes the .incomplete files).
# The hub-cache layout gave every attempt new .incomplete names (no resume across attempts: 312 GB of
# partial files, no large file complete); --local-dir keeps fixed names and resumes.
grep -qE 'for attempt in \$\(seq 1 20\); do timeout 900 hf download' "$HERE/01-vllm_template.yaml" && ok || bad "template: download attempts not bounded and retried"
grep -qE 'hf download canada-quant/GLM-5.3-Flash-W4A16-MTP --revision [0-9a-f]+ --local-dir /hf-cache/canada-quant-w4a16-[0-9a-f]+' "$HERE/01-vllm_template.yaml" && ok || bad "template: checkpoint not downloaded with --local-dir"
grep -qE 'hf download canada-quant/GLM-5.3-Flash-W4A16-MTP --revision [0-9a-f]+ --local-dir /hf-cache/canada-quant-w4a16-[0-9a-f]+' "$HERE/03-download-checkpoint.yaml" && ok || bad "download Job: same --local-dir as the template"
# Authenticated downloads (anonymous ones stalled): HF_TOKEN from the namespace's Secret hf-token,
# optional so a missing Secret falls back to anonymous; never a literal token in the repo.
for f in 01-vllm_template.yaml 03-download-checkpoint.yaml 02-aiperf-clients.yaml; do
  grep -q 'secretKeyRef: {name: hf-token, key: token, optional: true}' "$HERE/$f" && ok || bad "$f: HF_TOKEN not read from the Secret hf-token"
done
grep -rqE 'hf_[A-Za-z0-9]{30,}' "$HERE/.." && bad "a literal Hugging Face token is in the study folder" || ok
# The 4-bit checkpoint (decided 2026-11-18): canada-quant/GLM-5.3-Flash-W4A16-MTP, pinned.
REV=5681f7c2dc6436d63f82ab6fb892b4f5062b8304
grep -q -- "hf download canada-quant/GLM-5.3-Flash-W4A16-MTP --revision $REV" "$HERE/01-vllm_template.yaml" && ok || bad "template: checkpoint download not pinned to $REV"
grep -q -- "ln -sfn /hf-cache/canada-quant-w4a16-$REV /mnt/models" "$HERE/01-vllm_template.yaml" && ok || bad "template: /mnt/models not the pinned local dir"
grep -q -- "zai-org/GLM-5.3-Flash --revision" "$HERE/01-vllm_template.yaml" && bad "template: still downloads the FP8 checkpoint" || ok
grep -q -- "hf download canada-quant/GLM-5.3-Flash-W4A16-MTP --revision $REV" "$HERE/02-aiperf-clients.yaml" && ok || bad "clients: tokenizer not pinned to $REV"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
