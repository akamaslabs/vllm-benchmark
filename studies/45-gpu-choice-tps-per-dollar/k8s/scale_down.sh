#!/bin/bash
# Study 45, last workflow task (alwaysRun): remove the serving pods and the AIPerf job, and
# scale EVERY study-45 GPU node group to 0. Runs after every experiment, failed ones included,
# so nothing stays up when the study stops or errors out over the weekend. The next
# experiment's apply_config.sh scales up what it needs.
K8S=${STUDY45_K8S:-/work/study45/45-gpu-choice-tps-per-dollar/k8s}
# shellcheck disable=SC1091
source "$K8S/nodes.sh"
kubectl delete deployment -n llm-serving -l app=vllm-pd-mn --ignore-not-found --wait=false
kubectl delete job aiperf-pd-mn -n llm-benchmark --ignore-not-found --wait=false
RC=0
for g in $ALL_GROUPS; do
  [ "$(desired_of "$g")" = "0" ] && continue
  scale_group "$g" 0 || RC=1
done
for g in $ALL_GROUPS; do echo "$g desired=$(desired_of "$g")"; done
exit $RC
