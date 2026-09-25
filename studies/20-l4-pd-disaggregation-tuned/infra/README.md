# infra/ — what study 18 needs from the cluster

Study 18 runs on the existing `vllm-bench` cluster (us-east-2, AWS account 916205288457,
`AWS_PROFILE=lab`). It needs no new node group. `eks/cluster.yaml` describes the three
node groups it uses; `k8s-bootstrap/` holds the namespaces and StorageClasses, identical
to earlier studies and already applied on this cluster.

## State on 2026-09-24

| Node group | Instance | Role here | State |
|---|---|---|---|
| `llm-serving-l4` | g6.12xlarge, 4x L4 | vLLM (all instances) + router | ACTIVE, **0 nodes** |
| `system-2b` | m8a.xlarge | Prometheus, Grafana, AIPerf Job | ACTIVE, 1 node (created 2026-09-23) |
| `akamas` | r6i.xlarge | Akamas + toolbox | ACTIVE, 1 node |

## Bringing the GPU node up (smoke test / study run)

1. **Capacity probe first.** It costs nothing: a capacity reservation is refused at once
   when the pool is empty, and cancelled at once when it succeeds.
   ```bash
   export AWS_PROFILE=lab
   END=$(date -u -v+1H '+%Y-%m-%dT%H:%M:%SZ')
   for az in us-east-2a us-east-2b; do
     aws ec2 create-capacity-reservation --region us-east-2 --instance-type g6.12xlarge \
       --instance-platform Linux/UNIX --availability-zone $az --instance-count 1 \
       --instance-match-criteria targeted --end-date-type limited --end-date $END \
       --query CapacityReservation.CapacityReservationId --output text   # then cancel it:
     # aws ec2 cancel-capacity-reservation --region us-east-2 --capacity-reservation-id cr-...
   done
   ```
   `studies/17-.../infra/eks/gpu-capacity-fallback.sh probe` does the same, but it stops at
   the first refusal (seen 2026-09-23), so loop by hand when probing several types.
2. **Scale up:**
   `eksctl scale nodegroup --cluster vllm-bench --region us-east-2 --name llm-serving-l4 --nodes 1 --nodes-max 1`
   (g6.12xlarge on-demand: 4.60 USD/h).
3. **Re-point DCGM Exporter** at this node group. There is one `dcgm-exporter` release,
   pointed at `llm-serving-l4-single` on 2026-09-23. Without this, the GPU series and
   `active_gpus` stay empty and nothing errors:
   `helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring --reuse-values --set nodeSelector.node-role=llm-serving-l4`
4. **Tag the nodes `AlwaysOn=true` before leaving the study running past 17:00 UTC.** The
   lab account stops every running instance without that tag at 17:00 UTC daily (EventBridge
   `StopEC2Instances`). Tag both the GPU node and the `system-2b` node, and their ASGs with
   propagation:
   ```bash
   export AWS_PROFILE=lab AWS_REGION=us-east-2
   GPU=$(kubectl get nodes -l node-role=llm-serving-l4 -o jsonpath='{.items[0].spec.providerID}' | awk -F/ '{print $NF}')
   SYS=$(kubectl get nodes -l eks.amazonaws.com/nodegroup=system-2b -o jsonpath='{.items[0].spec.providerID}' | awk -F/ '{print $NF}')
   aws ec2 create-tags --resources $GPU $SYS --tags Key=AlwaysOn,Value=true
   for ng in llm-serving-l4 system-2b; do ASG=$(aws eks describe-nodegroup --cluster-name vllm-bench --nodegroup-name $ng --query 'nodegroup.resources.autoScalingGroups[0].name' --output text); aws autoscaling create-or-update-tags --tags "ResourceId=$ASG,ResourceType=auto-scaling-group,Key=AlwaysOn,Value=true,PropagateAtLaunch=true"; done
   ```
5. **Scale back to 0 when done,** and remove the tag from the GPU ASG so a forgotten GPU node
   is stopped at night again:
   `eksctl scale nodegroup --cluster vllm-bench --region us-east-2 --name llm-serving-l4 --nodes 0`

## Cost of the study

| Phase | Node time | Cost |
|---|---|---|
| Smoke test (phase B) | ~1-1.5 h | ~5-7 USD |
| 10 experiments x ~65-70 min (60 min load + startup) | ~11-12 h | ~50-55 USD |
