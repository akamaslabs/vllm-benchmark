#!/bin/bash
# Stop study 45 once experiment LAST (default 24) has ended (FINISHED or ERROR), then make
# sure every study-45 GPU node group is at 0. Decided with the user on 2026-10-31: 5 more
# experiments counting from the one running at 08:30 UTC (experiment 20).
LAST=${LAST:-24}
tb() { kubectl -n akamas-41 exec deploy/toolbox -c toolbox -- "$@"; }
login() { tb akamas whoami 2>/dev/null | grep -qiE "Logged User *\| *akamas" || tb akamas login -u Akamas -p Akamas.Akamas >/dev/null 2>&1; }
cd "$(dirname "$0")/.."
echo "$(date -u +%T) waiting for experiment $LAST to end"
while true; do
  login
  st=$(tb sh -c "akamas list experiment 45-GPU-Choice-TPS-per-Dollar-v2 > /tmp/stop45.txt 2>&1; grep -E '^$LAST ' /tmp/stop45.txt" 2>/dev/null | grep -oE "FINISHED|ERROR" | head -1)
  [ -n "$st" ] && break
  sleep 120
done
echo "$(date -u +%T) experiment $LAST: $st; finishing the study"
tb sh -c 'cat /tmp/stop45.txt' 2>/dev/null | grep -E "^[0-9]+ " | cut -c1-120
login
tb akamas finish study 45-GPU-Choice-TPS-per-Dollar-v2 --workspace default 2>&1 | tail -2 | cut -c1-150
sleep 60
tb akamas describe study 45-GPU-Choice-TPS-per-Dollar-v2 2>&1 | grep -iE "^status"
# Akamas stops the workflow; make sure nothing keeps running.
kubectl delete deployment -n llm-serving -l app=vllm-pd-mn --ignore-not-found
kubectl delete job aiperf-pd-mn -n llm-benchmark --ignore-not-found
AWS_PROFILE=lab infra/nodegroups.sh down
sleep 360
AWS_PROFILE=lab infra/nodegroups.sh status
kubectl get nodes -L node-role,node.kubernetes.io/instance-type --no-headers | grep -E "l4-pd-xl|g6-2xl|g5-xl|a10g-pd|g6e-xl|l40s-pd" || echo "no study-45 GPU nodes left"
AWS_PROFILE=lab aws ec2 describe-instances --region us-east-2 --filters Name=instance-state-name,Values=pending,running \
  Name=instance-type,Values=g5.xlarge,g5.2xlarge,g6.xlarge,g6.2xlarge,g6e.xlarge,g6e.2xlarge --query 'Reservations[].Instances[].[InstanceId,InstanceType]' --output text
echo "$(date -u +%T) STOP-45 DONE"
