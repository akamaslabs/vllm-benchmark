#!/bin/bash
# Local tests of apply_config.sh's parameter section (RENDER_ONLY=1: no cluster needed).
# Usage: bash k8s/test_apply_config.sh   -> "N passed, M failed", exit 1 on any failure.
HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
VARS="REPLICAS TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE ENABLE_EXPERT_PARALLEL GPU_MEMORY_UTILIZATION MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS KV_CACHE_DTYPE MOE_BACKEND"
# The vLLM flags of every preset of the study.
P="GPU_MEMORY_UTILIZATION=0.90 MAX_NUM_SEQS=512 MAX_NUM_BATCHED_TOKENS=8192 KV_CACHE_DTYPE=auto MOE_BACKEND=auto"
lay() { echo "REPLICAS=$1 TENSOR_PARALLEL_SIZE=$2 DATA_PARALLEL_SIZE=$3 PIPELINE_PARALLEL_SIZE=$4 ENABLE_EXPERT_PARALLEL=$5"; }

# render NAME "VAR=value ..." [ENV=value ...] -> sets OUT and RC
render() {
  local tmp kv params=$2; tmp=$(mktemp -d); mkdir -p "$tmp/k8s"; shift 2
  for v in $VARS; do echo "$v=" >> "$tmp/k8s/params.env"; done
  for kv in $params; do sed -i.bak "s|^${kv%%=*}=.*|$kv|" "$tmp/k8s/params.env"; done
  OUT=$(env -u DP_MODE -u NCCL_ENV RENDER_ONLY=1 STUDY_DIR="$tmp" "$@" bash "$HERE/apply_config.sh" 2>&1); RC=$?
  rm -rf "$tmp"
}
ok()   { PASS=$((PASS + 1)); }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL: $1"; echo "$OUT" | sed 's/^/    /'; }
has()  { echo "$OUT" | grep -qxF -- "$2" && ok || bad "$1: expected line '$2'"; }
hasnt(){ echo "$OUT" | grep -q -- "$2" && bad "$1: unexpected '$2'" || ok; }
rc()   { [ "$RC" = "$2" ] && ok || bad "$1: exit $RC, expected $2"; }

render "all empty" ""
rc "all empty" 0; has "all empty" "layout=single pods=a serving_b=false nccl=none dp_mode=external"; hasnt "all empty" "^[ab] "

render "single" "$(lay 1 1 1 1 false) $P"
rc "single" 0; has "single" "layout=single pods=a serving_b=false nccl=none dp_mode=external"
has "single" "a --gpu-memory-utilization=0.90"; has "single" "a --max-num-seqs=512"; has "single" "a --max-num-batched-tokens=8192"
has "single" "a --kv-cache-dtype=auto"; hasnt "single" "moe-backend"; hasnt "single" "^b "; hasnt "single" "nnodes"; hasnt "single" "master-addr"

render "replicas" "$(lay 2 1 1 1 false) $P"
rc "replicas" 0; has "replicas" "layout=replicas pods=a,b serving_b=true nccl=none dp_mode=external"
has "replicas" "b --max-num-seqs=512"; has "replicas" "b --gpu-memory-utilization=0.90"; hasnt "replicas" "nnodes"; hasnt "replicas" "master-addr"; hasnt "replicas" "headless"

render "dp2" "$(lay 1 1 2 1 false) $P"
rc "dp2" 0; has "dp2" "layout=dp2 pods=a,b serving_b=true nccl=warn dp_mode=external"
has "dp2" "a --data-parallel-size=2"; has "dp2" "a --data-parallel-external-lb"; has "dp2" "a --nnodes=2"; has "dp2" "a --node-rank=0"
has "dp2" 'a --master-addr=$(POD_IP)'; has "dp2" "b --data-parallel-size=2"; has "dp2" "b --data-parallel-external-lb"
has "dp2" "b --nnodes=2"; has "dp2" "b --node-rank=1"; has "dp2" "b --master-addr=HEAD_IP"
hasnt "dp2" "headless"; hasnt "dp2" "enable-expert-parallel"

render "dp2ep" "$(lay 1 1 2 1 true) $P"
rc "dp2ep" 0; has "dp2ep" "layout=dp2ep pods=a,b serving_b=true nccl=required dp_mode=external"
has "dp2ep" "a --enable-expert-parallel"; has "dp2ep" "b --enable-expert-parallel"; has "dp2ep" "b --data-parallel-external-lb"; hasnt "dp2ep" "headless"

render "dp2 internal" "$(lay 1 1 2 1 false) $P" DP_MODE=internal
rc "dp2 internal" 0; has "dp2 internal" "layout=dp2 pods=a,b serving_b=false nccl=warn dp_mode=internal"
has "dp2 internal" "b --headless"; hasnt "dp2 internal" "external-lb"; hasnt "dp2 internal" "^a --headless"

render "dp2ep internal" "$(lay 1 1 2 1 true) $P" DP_MODE=internal
rc "dp2ep internal" 0; has "dp2ep internal" "layout=dp2ep pods=a,b serving_b=false nccl=required dp_mode=internal"; has "dp2ep internal" "b --headless"

render "dp mode hybrid" "$(lay 1 1 2 1 false) $P" DP_MODE=hybrid
rc "dp mode hybrid" 2

render "tp2" "$(lay 1 2 1 1 false) $P"
rc "tp2" 0; has "tp2" "layout=tp2 pods=a,b serving_b=false nccl=required dp_mode=external"
has "tp2" "a --tensor-parallel-size=2"; has "tp2" "b --tensor-parallel-size=2"; has "tp2" 'a --master-addr=$(POD_IP)'; has "tp2" "b --master-addr=HEAD_IP"
has "tp2" "b --headless"; hasnt "tp2" "^a --headless"; hasnt "tp2" "expert"; hasnt "tp2" "external-lb"; hasnt "tp2" "^env "

render "tp2ep" "$(lay 1 2 1 1 true) $P"
rc "tp2ep" 0; has "tp2ep" "layout=tp2ep pods=a,b serving_b=false nccl=required dp_mode=external"
has "tp2ep" "a --enable-expert-parallel"; has "tp2ep" "b --enable-expert-parallel"; has "tp2ep" "b --headless"

render "pp2" "$(lay 1 1 1 2 false) $P"
rc "pp2" 0; has "pp2" "layout=pp2 pods=a,b serving_b=false nccl=required dp_mode=external"
has "pp2" "a --pipeline-parallel-size=2"; has "pp2" "b --pipeline-parallel-size=2"; has "pp2" "b --headless"

render "tp as 2.0" "$(lay 1 2.0 1 1 false) $P"
rc "tp as 2.0" 0; has "tp as 2.0" "a --tensor-parallel-size=2"

# Not a layout: more than two GPUs' worth, or expert parallelism with nothing to spread over.
render "replicas x tp" "$(lay 2 2 1 1 false) $P";  rc "replicas x tp" 2
render "tp x dp" "$(lay 1 2 2 1 false) $P";        rc "tp x dp" 2
render "three replicas" "$(lay 3 1 1 1 false) $P"; rc "three replicas" 2
render "ep alone" "$(lay 1 1 1 1 true) $P";        rc "ep alone" 2
render "ep on replicas" "$(lay 2 1 1 1 true) $P";  rc "ep on replicas" 2
render "ep on pp2" "$(lay 1 1 1 2 true) $P";       rc "ep on pp2" 2
render "ep yes" "$(lay 1 2 1 1 yes) $P";           rc "ep yes" 2
render "tp x" "$(lay 1 x 1 1 false) $P";           rc "tp x" 2

render "tcp control" "$(lay 1 2 1 1 false) $P" "NCCL_ENV=NCCL_NET=Socket NCCL_SOCKET_NTHREADS=4"
rc "tcp control" 0; has "tcp control" "env NCCL_NET=Socket NCCL_SOCKET_NTHREADS=4"
echo "$OUT" | grep -q 'expecting Socket' && ok || bad "tcp control: transport check must expect Socket"
render "efa default" "$(lay 1 2 1 1 false) $P"
echo "$OUT" | grep -q 'expecting Libfabric' && ok || bad "efa default: transport check must expect Libfabric"
render "bad nccl env" "$(lay 1 2 1 1 false) $P" "NCCL_ENV=nccl_net=Socket"
rc "bad nccl env" 2

render "gmu sci" "$(lay 1 1 1 1 false) GPU_MEMORY_UTILIZATION=9.2E-1"
rc "gmu sci" 0; has "gmu sci" "a --gpu-memory-utilization=0.920000"
render "gmu no exponent" "$(lay 1 1 1 1 false) GPU_MEMORY_UTILIZATION=2.19"; rc "gmu no exponent" 2
render "mnbt < mns" "$(lay 1 1 1 1 false) MAX_NUM_SEQS=128 MAX_NUM_BATCHED_TOKENS=64"; rc "mnbt < mns" 2
render "kv dtype fp16" "$(lay 1 1 1 1 false) KV_CACHE_DTYPE=fp16"; rc "kv dtype fp16" 2
render "kv dtype fp8" "$(lay 1 1 1 1 false) KV_CACHE_DTYPE=fp8"
rc "kv dtype fp8" 0; has "kv dtype fp8" "a --kv-cache-dtype=fp8"
render "moe marlin" "$(lay 1 2 1 1 true) $P MOE_BACKEND=marlin"
rc "moe marlin" 0; has "moe marlin" "a --moe-backend=marlin"; has "moe marlin" "b --moe-backend=marlin"
render "moe bad name" "$(lay 1 1 1 1 false) MOE_BACKEND=Marlin-2"; rc "moe bad name" 2

# A broken template: a variable with no line, a token FileConfigurator did not substitute.
tmp=$(mktemp -d); mkdir -p "$tmp/k8s"
for v in $VARS; do [ $v = MOE_BACKEND ] || echo "$v=" >> "$tmp/k8s/params.env"; done
OUT=$(RENDER_ONLY=1 STUDY_DIR="$tmp" bash "$HERE/apply_config.sh" 2>&1); RC=$?; rc "missing line" 2
echo 'MOE_BACKEND=${vllm.moe_backend}' >> "$tmp/k8s/params.env"
OUT=$(RENDER_ONLY=1 STUDY_DIR="$tmp" bash "$HERE/apply_config.sh" 2>&1); RC=$?; rc "unsubstituted token" 2
rm -rf "$tmp"
# The template itself names every variable the script requires.
for v in $VARS; do grep -q "^$v=" "$HERE/params.env.template" && ok || { FAIL=$((FAIL + 1)); echo "FAIL: params.env.template has no $v line"; }; done

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
