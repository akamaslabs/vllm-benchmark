#!/bin/bash
# Nsight trace captures of study 51's layouts, run by hand in the toolbox after the last experiment.
#   bash capture_all.sh <layout>:<users> [<layout>:<users> ...]      e.g. pp2:512 single:64
# For each item:
#   1. apply the layout with the study's own apply_config.sh at the presets' flags, on the NSIGHT
#      VARIANT of the pod template (W/k8s is a copy of the study's k8s folder with the template of
#      this folder), so every vLLM process loads the CUDA 13.5 compatibility libcuda and reaches
#      the Nsight service of its node. The services (nsight-a, nsight-b) must already be up.
#   2. run one load level of LEVEL_SECONDS at <users> with the study's run_ramp.sh;
#   3. CAP_AT seconds into the level, capture CAP_MS of trace on every node that runs a vLLM pod;
#   4. after the level, export each capture to a Perfetto zip in the Nsight pod
#      (/work/traces/<layout>/, that is /var/lib/nsight/traces on the node). The consumer names
#      its TSF directory <pod hostname>.<timestamp> (nsight-a.20261111_225820).
# Logs go to W/logs/<layout>-{apply,load,capture-<role>,export-<role>}.log. A layout whose apply
# fails is skipped. The export waits for the end of the level so that it does not take CPU from
# the node while AIPerf measures.
set -uo pipefail
W=${W:-/work/study51-nsight}; K=$W/k8s; NS=moe-tcp
LEVEL_SECONDS=${LEVEL_SECONDS:-780}; CAP_AT=${CAP_AT:-690}; CAP_MS=${CAP_MS:-10000}
export STUDY_DIR=$W
mkdir -p "$W/logs"
say() { echo "$(date -u +%T) $*"; }

for item in "$@"; do
  L=${item%%:*}; U=${item##*:}
  case "$L" in
    single)   P="1 1 1 1 false" ;;
    replicas) P="2 1 1 1 false" ;;
    dp2)      P="1 1 2 1 false" ;;
    dp2ep)    P="1 1 2 1 true" ;;
    tp2)      P="1 2 1 1 false" ;;
    tp2ep)    P="1 2 1 1 true" ;;
    pp2)      P="1 1 1 2 false" ;;
    *) say "unknown layout $L, skipped"; continue ;;
  esac
  set -- $P
  cat > "$K"/params.env <<EOF
REPLICAS=$1
TENSOR_PARALLEL_SIZE=$2
DATA_PARALLEL_SIZE=$3
PIPELINE_PARALLEL_SIZE=$4
ENABLE_EXPERT_PARALLEL=$5
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=512
MAX_NUM_BATCHED_TOKENS=8192
KV_CACHE_DTYPE=auto
MOE_BACKEND=auto
EOF
  say "===== $L: apply"
  bash "$K"/apply_config.sh > "$W/logs/$L-apply.log" 2>&1; rc=$?
  say "===== $L: apply rc=$rc"
  [ $rc = 0 ] || { say "$L skipped: apply failed (see logs/$L-apply.log)"; continue; }
  ROLES=$(kubectl -n $NS get pods -l app=vllm -o name | sed -n 's|^pod/vllm-\([ab]\)-0$|\1|p' | tr '\n' ' ')
  for r in $ROLES; do
    # Proof that the pod runs on the compatibility libcuda: the libcuda files mapped by its processes.
    say "vllm-$r-0 libcuda: $(kubectl -n $NS exec vllm-$r-0 -- sh -c 'cat /proc/[0-9]*/maps 2>/dev/null | grep -o "/[^ ]*libcuda\.so[^ ]*" | sort | uniq -c | tr "\n" " "' 2>&1 | cut -c1-300)"
  done

  say "===== $L: load at $U users for $LEVEL_SECONDS s, capture at +$CAP_AT s on node(s) $ROLES"
  LEVELS=$U LEVEL_SECONDS=$LEVEL_SECONDS STEADY_FROM=600 LEVEL_FILE=$W/level.$L bash "$K"/run_ramp.sh > "$W/logs/$L-load.log" 2>&1 &
  LOAD=$!
  sleep "$CAP_AT"
  if ! kill -0 $LOAD 2>/dev/null; then say "$L: the load ended before the capture (see logs/$L-load.log)"; wait $LOAD; continue; fi
  for r in $ROLES; do
    if kubectl -n $NS get pod vllm-$r-0 -o go-template='{{index .metadata.labels "study51/serving"}}' 2>/dev/null | grep -q true; then
      say "vllm-$r-0 at capture: $(kubectl -n $NS exec vllm-$r-0 -- curl -s -m 10 localhost:8000/metrics 2>/dev/null | grep -E '^vllm:(num_requests_running|num_requests_waiting\{|kv_cache_usage_perc)' | sed 's/{[^}]*}//' | tr '\n' ' ')"
    fi
  done
  CAPS=()
  for r in $ROLES; do
    kubectl -n $NS exec nsight-$r -- bash -c "rm -rf /work/traces/$L; mkdir -p /work/traces/$L; cd /work/sdk/build/bin; date -u +'capture start %H:%M:%S'; ./nvtrace_streaming_consumer --output-dir /work/traces/$L --duration-ms $CAP_MS 2>&1 | tail -3; date -u +'capture end %H:%M:%S'; du -sm /work/traces/$L/*/ | sed 's/^/tsf MB /'; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader | sed 's/^/gpu: /'" > "$W/logs/$L-capture-$r.log" 2>&1 &
    CAPS+=($!)
  done
  for p in "${CAPS[@]}"; do wait "$p"; done
  for r in $ROLES; do say "capture $r: $(tr '\n' ' ' < "$W/logs/$L-capture-$r.log" | cut -c1-300)"; done
  wait $LOAD; say "===== $L: load rc=$?"
  grep -E '^(RESULT|RAMP)|first request|cpu used' "$W/logs/$L-load.log" | sed "s/^/$L: /"
  sed -n '/^--- vLLM counters/,/^--- client 0: AIPerf/p' "$W/logs/$L-load.log" | grep '^vllm:' | sed "s/^/$L: /"
  for r in $ROLES; do
    kubectl -n $NS exec nsight-$r -- bash -c "cd /work/sdk/build/bin; d=\$(ls -d /work/traces/$L/*/ | head -1); d=\${d%/}; s=\$(date +%s); ./nvtrace_perfetto_trace_exporter --tsf-database-dir \$d 2>&1 | tail -2; mv \$d.perfetto-trace.zip /work/traces/$L/$L-${U}users-node-$r.perfetto-trace.zip; echo \"export \$(( \$(date +%s) - s )) s\"; ls -l /work/traces/$L/*.zip" > "$W/logs/$L-export-$r.log" 2>&1
    say "export $r: $(tail -2 "$W/logs/$L-export-$r.log" | tr '\n' ' ' | cut -c1-300)"
  done
done
say "===== all done"
