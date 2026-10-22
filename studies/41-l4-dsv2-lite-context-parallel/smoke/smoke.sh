#!/usr/bin/env bash
# Smoke-test DeepSeek-V2-Lite-Chat on the 4x L4 node, one vLLM pod per layout.
# usage: smoke.sh <layout-name> <extra vllm args...>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=$1; shift
NS=llm-serving
POD=dsv2-smoke
OUT=$HERE/results; mkdir -p "$OUT"
LOG=$OUT/$NAME.log
IMAGE=vllm/vllm-openai:v0.32.0

kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: vllm-model-cache-dsv2lite, namespace: $NS}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: gp3-ephemeral
  resources: {requests: {storage: 60Gi}}
EOF

ARGS=""
for a in "$@"; do ARGS="$ARGS          - \"$a\"
"; done

kubectl delete pod -n $NS $POD --ignore-not-found --wait=true >/dev/null
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata: {name: $POD, namespace: $NS, labels: {app: dsv2-smoke}}
spec:
  restartPolicy: Never
  enableServiceLinks: false
  nodeSelector: {node-role: llm-serving-l4}
  tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
  containers:
    - name: vllm
      image: $IMAGE
      command: ["vllm", "serve"]
      args:
          - "deepseek-ai/DeepSeek-V2-Lite-Chat"
          - "--served-model-name=dsv2-lite"
          - "--trust-remote-code"
          - "--max-model-len=32768"
          - "--gpu-memory-utilization=0.90"
          - "--no-enable-prefix-caching"
$ARGS      env: [{name: VLLM_LOGGING_LEVEL, value: INFO}]
      resources:
        requests: {nvidia.com/gpu: "4", cpu: "16", memory: "64Gi"}
        limits: {nvidia.com/gpu: "4", cpu: "40", memory: "160Gi"}
      readinessProbe: {httpGet: {path: /health, port: 8000}, periodSeconds: 10}
      volumeMounts:
        - {name: shm, mountPath: /dev/shm}
        - {name: model-cache, mountPath: /root/.cache/huggingface}
  volumes:
    - {name: shm, emptyDir: {medium: Memory, sizeLimit: 32Gi}}
    - {name: model-cache, persistentVolumeClaim: {claimName: vllm-model-cache-dsv2lite}}
EOF

echo "== $NAME: $* (start $(date -u +%H:%M:%S))" | tee "$LOG"
T0=$(date +%s); STATUS=TIMEOUT
while [ $(( $(date +%s) - T0 )) -lt 2700 ]; do
  PH=$(kubectl get pod -n $NS $POD -o jsonpath='{.status.phase}' 2>/dev/null)
  RD=$(kubectl get pod -n $NS $POD -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
  if [ "$PH" = "Failed" ] || [ "$PH" = "Succeeded" ]; then STATUS=FAILED; break; fi
  if [ "$RD" = "true" ]; then STATUS=READY; break; fi
  sleep 15
done
echo "status=$STATUS after $(( $(date +%s) - T0 ))s" | tee -a "$LOG"
kubectl logs -n $NS $POD > "$OUT/$NAME.vllm.log" 2>&1
grep -E "GPU KV cache size|Maximum concurrency|[Bb]ackend|NotImplementedError|ValueError|Error|out of memory|model weights took|Available KV cache memory" "$OUT/$NAME.vllm.log" | grep -v "^INFO.*Route" | head -40 | tee -a "$LOG"
if [ "$STATUS" = "READY" ]; then
  kubectl cp "$HERE/bench.py" $NS/$POD:/tmp/bench.py -c vllm
  kubectl exec -n $NS $POD -c vllm -- sh -c "nohup python3 /tmp/bench.py dsv2-lite > /tmp/bench.out 2>&1 &"
  T1=$(date +%s)
  while [ $(( $(date +%s) - T1 )) -lt 2400 ]; do
    sleep 30
    if kubectl exec -n $NS $POD -c vllm -- sh -c "grep -q -e SUMMARY -e Traceback /tmp/bench.out" 2>/dev/null; then break; fi
  done
  kubectl exec -n $NS $POD -c vllm -- cat /tmp/bench.out 2>&1 | tee -a "$LOG"
fi
kubectl delete pod -n $NS $POD --wait=true >/dev/null
echo "== done $NAME $(date -u +%H:%M:%S)" | tee -a "$LOG"
