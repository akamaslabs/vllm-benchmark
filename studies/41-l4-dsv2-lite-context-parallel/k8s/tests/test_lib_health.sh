#!/bin/bash
# Tests for ../lib_health.sh: apply_config.sh's decision after the warm-up requests and its
# GPU-busy guard.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../lib_health.sh
source "$(dirname "$HERE")/lib_health.sh"
FAILS=0
t() { local name=$1; shift; if "$@" >/dev/null; then echo "ok   $name"; else echo "FAIL $name"; FAILS=$((FAILS + 1)); fi; }
n() { local name=$1; shift; if "$@" >/dev/null; then echo "FAIL $name"; FAILS=$((FAILS + 1)); else echo "ok   $name"; fi; }
t "two replicas ready, no restart, warm-up ok"   replicas_healthy 2 2 0 0
t "one replica ready, no restart, warm-up ok"    replicas_healthy 1 1 0 0
n "neighbour not ready"                          replicas_healthy 2 1 0 0
n "nothing ready (empty jsonpath)"               replicas_healthy 1 "" 0 0
n "a replica restarted once during startup"      replicas_healthy 2 2 1 0
n "a replica failed the warm-up"                 replicas_healthy 2 2 0 1
REASON=$(replicas_healthy 2 1 0 0); [[ "$REASON" == *"1 of 2"* ]] && echo "ok   reason names the counts" || { echo "FAIL reason names the counts: $REASON"; FAILS=$((FAILS + 1)); }
# GPU-busy guard: pods on the GPU node that hold GPUs and are not this study's.
busy() { printf '%s\n' "$@" | gpu_busy_pods llm-serving/vllm-dsv2-; }
eqs() { [ "$1" = "$2" ]; }
t "no pod on the node"                           eqs "$(busy '')" ""
t "only our replica holds GPUs"                  eqs "$(busy 'llm-serving/vllm-dsv2-0:4' 'monitoring/dcgm-exporter-x:')" ""
t "DaemonSets without GPU requests ignored"      eqs "$(busy 'kube-system/nvidia-device-plugin-daemonset-a:' 'kube-system/aws-node-b:')" ""
t "the smoke pod is reported"                    eqs "$(busy 'llm-serving/dsv2-smoke:4' 'llm-serving/vllm-dsv2-0:4')" "llm-serving/dsv2-smoke"
t "another study's vllm is reported"             eqs "$(busy 'llm-serving/vllm-5b9d8db784-abc:4')" "llm-serving/vllm-5b9d8db784-abc"
t "a one-GPU probe is reported"                  eqs "$(busy 'llm-l40s/probe:1')" "llm-l40s/probe"
t "gpu request 0 ignored"                        eqs "$(busy 'default/x:0')" ""
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
