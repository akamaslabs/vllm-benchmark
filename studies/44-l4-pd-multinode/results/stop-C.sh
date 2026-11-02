#!/bin/bash
# Study 44 tappa C: stop after the first experiment that finishes after 16:30 UTC on 2026-10-30
# (or at 18:30 UTC at the latest), then remove the pods and scale every study-44 node group to 0.
cd "$(dirname "$0")/.."
TB="kubectl -n akamas-41 exec deploy/toolbox -c toolbox --"
start=$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "2026-10-30T16:30:00" +%s); hard=$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "2026-10-30T18:30:00" +%s)
logged_in() { local o; o=$($TB akamas whoami 2>&1); echo "$o" | grep -q "Logged User" && ! echo "$o" | grep -qiE "not currently logged|unauthorized"; }
until [ "$(date -u +%s)" -ge "$start" ]; do sleep 60; done
until logged_in || [ "$(date -u +%s)" -ge "$hard" ]; do echo "$(date -u +%T) LOGIN NEEDED: akamas CLI logged out"; sleep 300; done
fin0=$($TB akamas list experiment 44-C-L40S-A10G 2>/dev/null | grep -cE "FINISHED|FAILED|ERROR")
echo "$(date -u +%T) waiting for an experiment to end (finished/failed now: $fin0)"
while [ "$(date -u +%s)" -lt "$hard" ]; do
  if ! logged_in; then echo "$(date -u +%T) LOGIN NEEDED: akamas CLI logged out"; sleep 300; continue; fi
  n=$($TB akamas list experiment 44-C-L40S-A10G 2>/dev/null | grep -cE "FINISHED|FAILED|ERROR")
  [ -n "$n" ] && [ "$n" -gt "$fin0" ] && break
  sleep 60
done
echo "$(date -u +%T) finishing the study"
$TB akamas list experiment 44-C-L40S-A10G 2>&1 | grep -E "^[0-9]+ " | cut -c1-140
logged_in || echo "$(date -u +%T) WARNING: akamas CLI logged out, the study cannot be finished: scaling the nodes down anyway (its next experiments fail without nodes)"
$TB akamas finish study 44-C-L40S-A10G --workspace default 2>&1 | tail -3
sleep 60
$TB akamas describe study 44-C-L40S-A10G 2>&1 | grep -iE "^status|status message"
kubectl delete deployment -n llm-serving -l app=vllm-pd-mn --ignore-not-found
kubectl delete job aiperf-pd-mn -n llm-benchmark --ignore-not-found
AWS_PROFILE=lab infra/nodegroups.sh down all
sleep 300
AWS_PROFILE=lab infra/nodegroups.sh status
AWS_PROFILE=lab aws ec2 describe-instances --region us-east-2 --filters Name=tag:Study,Values=44 Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId' --output text
echo "$(date -u +%T) STOP-C DONE"
