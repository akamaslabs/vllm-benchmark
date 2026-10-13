#!/bin/bash
# Stop AIPerf step for 35-l40s-gemma4-26b-awq-replay-users (toolbox, Akamas Executor task after RunTest,
# alwaysRun: it also runs when RunTest failed).
#
# 1. Stops AIPerf gracefully: SIGINT to the top aiperf process (study 34: the child of the
#    customer's run_benchmark.sh, piped into tee; studies 32/33: of the Job's sh). On its
#    first SIGINT AIPerf 0.11.0 stops sending, cancels the requests in flight and writes its files
#    (controller/system_controller.py _handle_signal); the per-request records go out 100 at a time
#    and the last partial batch is flushed on stop (common/mixins/buffered_jsonl_writer_mixin.py).
#    `kubectl delete job` alone SIGTERMs the Job's sh, which forwards nothing, and SIGKILLs aiperf
#    after the grace period: that batch and the summary json are lost. Checked locally (AIPerf
#    0.11.0 under dash against k8s/tests/mock_openai.py, 2026-10-12): exit 0 about 1 s after the
#    SIGINT, 291 records written (the partial batch included), profile_export_aiperf.json too; the
#    ~50 requests in flight at the SIGINT are not in the records.
# 2. Deletes the Job (run_test.sh leaves it running when the workflow sets RT_STOP_AIPERF=0).
# 3. Saves the measured run's per-request records (profile_export.jsonl) on the toolbox, gzipped, as
#    SA_DEST/<epoch>.jsonl.gz, <epoch> being the measured run's start (their script's folder
#    /benchmarks/study35/replay-benchmark/data/results/run_<YYYYMMDDHHMM>_<epoch>: the Job passes
#    NAME=<epoch>). The AIPerf pod
#    is gone by then, so a short-lived pod reads the results PVC (read-only); the folder stays there
#    until the next Job's rm -rf. The file travels gzipped through `kubectl exec` and is checked
#    against the pod's sha256, up to SA_COPY_TRIES times: a plain `kubectl exec cat` of a 13 MB
#    export arrived truncated and corrupted mid-line (2026-10-13, study 33's 1791892169.jsonl).
# The toolbox's /work is a 1 GiB volume that the FileConfigurator task also writes to: the copy is
# skipped when it would leave less than SA_MIN_FREE_MB free.
# Exit 1 if the Job could not be deleted, or the records could not be saved (copy failed, no
# room); 0 otherwise, also when no measured run started (nothing to save). The task is critical:
# false, so either way the trial's result stands.
set -euo pipefail
NS=llm-replay
# vllm-bench-gatling through the toolbox's `gatling` context (the default context is Akamas's cluster).
KUBE_CONTEXT=${KUBE_CONTEXT:-gatling}
kubectl() { command kubectl --context "$KUBE_CONTEXT" "$@"; }
JOB=aiperf-replay
HELPER=aiperf-export
RESULTS=/benchmarks/study35
DEST=${SA_DEST:-/work/aiperf-exports/35-l40s-gemma4-26b-awq-replay-users}
MIN_FREE_MB=${SA_MIN_FREE_MB:-150}
STOP_WAIT_S=${SA_STOP_WAIT_S:-300}
POD_WAIT_S=${SA_POD_WAIT_S:-300}
POLL_S=${SA_POLL_S:-5}
COPY_TRIES=${SA_COPY_TRIES:-3}
RC=0

# --- 1. Stop AIPerf ---
POD=$(kubectl -n "$NS" get pods -l "app=$JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -z "$POD" ]; then
  echo "no AIPerf pod: job/$JOB already gone"
else
  PHASE=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)
  echo "AIPerf pod $POD: ${PHASE:-unknown}"
  if [ "$PHASE" = Running ]; then
    # The top aiperf process renames itself "aiperf system_controller"; its parent is the
    # customer's script (bash), its own children (aiperf workers) ignore SIGINT. The image has no
    # pgrep/pkill: scan /proc for the aiperf processes whose parent is not one.
    PID=$(kubectl -n "$NS" exec "$POD" -c aiperf -- python3 -c '
import os, signal
procs = {}
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    try:
        s = open("/proc/%s/stat" % p).read()
    except OSError:
        continue
    procs[p] = (s[s.index("(") + 1:s.rindex(")")], s[s.rindex(")") + 2:].split()[1])
for p, (comm, ppid) in procs.items():
    if comm.startswith("aiperf") and not procs.get(ppid, ("",))[0].startswith("aiperf"):
        os.kill(int(p), signal.SIGINT)
        print(p)
' 2>/dev/null || true)
    if [ -n "$PID" ]; then
      T_SIG=$SECONDS
      echo "$(date -u +%T) SIGINT sent to aiperf (pid $PID): it cancels the requests in flight and writes its files"
      while [ "$PHASE" = Running ] && [ $((SECONDS - T_SIG)) -lt "$STOP_WAIT_S" ]; do
        sleep "$POLL_S"
        PHASE=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null || true)
      done
      if [ "$PHASE" = Running ]; then
        echo "warning: aiperf still running $((SECONDS - T_SIG)) s after the SIGINT: deleting the Job anyway (the last records may be lost)"
      else
        echo "$(date -u +%T) AIPerf pod ${PHASE:-gone} $((SECONDS - T_SIG)) s after the SIGINT"
      fi
      echo "--- $POD: logs since the SIGINT ---"
      kubectl -n "$NS" logs "$POD" -c aiperf --since="$((SECONDS - T_SIG + 10))s" || true
    else
      echo "no aiperf process in $POD (installing, or between the warm-up and the run): nothing to cancel"
    fi
  fi
fi

# --- 2. Delete the Job ---
# Foreground: the pod is gone when this returns, so the results volume (RWO) is free.
if ! kubectl -n "$NS" delete job -l "app=$JOB" --ignore-not-found --cascade=foreground --wait=true --timeout=180s; then
  echo "error: job/$JOB not deleted: it may still be loading vllm-0 (the next RunTest deletes it at its start)"
  RC=1
fi

# --- 3. Save the records ---
cleanup() { kubectl -n "$NS" delete pod "$HELPER" --ignore-not-found --wait=false >/dev/null 2>&1 || true; }
trap cleanup EXIT
kubectl -n "$NS" delete pod "$HELPER" --ignore-not-found --wait=true   # left by an interrupted step
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $HELPER
  namespace: $NS
  labels:
    app: $HELPER
spec:
  restartPolicy: Never
  nodeSelector:
    node-role: system
  containers:
    - name: export
      image: alpine:3
      command: ["sleep", "1800"]
      volumeMounts:
        - name: results
          mountPath: /benchmarks
          readOnly: true
  volumes:
    - name: results
      persistentVolumeClaim:
        claimName: aiperf-results
        readOnly: true
EOF
if ! kubectl -n "$NS" wait --for=condition=Ready "pod/$HELPER" --timeout="${POD_WAIT_S}s"; then
  echo "error: pod/$HELPER not ready in $POD_WAIT_S s: records not saved"
  kubectl -n "$NS" describe pod "$HELPER" | tail -20 || true
  exit 1
fi
RUN_DIR=$(kubectl -n "$NS" exec "$HELPER" -- sh -c "ls -d $RESULTS/replay-benchmark/data/results/run_* 2>/dev/null | sort | tail -1" || true)
if [ -z "$RUN_DIR" ]; then
  echo "no measured run on the volume (stopped before it started): nothing to save"
  exit "$RC"
fi
echo "--- $RUN_DIR ---"
kubectl -n "$NS" exec "$HELPER" -- ls -la "$RUN_DIR" || true
SIZE=$(kubectl -n "$NS" exec "$HELPER" -- stat -c %s "$RUN_DIR/profile_export.jsonl" 2>/dev/null || true)
if ! [[ "$SIZE" =~ ^[0-9]+$ ]]; then
  echo "error: no $RUN_DIR/profile_export.jsonl: records not saved"
  exit 1
fi
# Kept gzipped (~4x smaller: ~9 MB against ~38 MB a trial at 200 users): /work is a shared 1 GiB volume.
OUT="$DEST/${RUN_DIR##*_}.jsonl.gz"
mkdir -p "$DEST"
if [ -f "$OUT" ] && [ "$(gzip -dc "$OUT" 2>/dev/null | wc -c | tr -d ' ')" -ge "$SIZE" ]; then
  echo "$OUT already saved ($SIZE bytes): the volume still holds the previous run"
  exit "$RC"
fi
FREE_KB=$(df -Pk "$DEST" | awk 'NR == 2 {print $4}')
# The gzipped copy is ~1/4 of SIZE: count 1/3 to stay on the safe side.
if [ $(( (FREE_KB * 1024 - SIZE / 3) / 1048576 )) -lt "$MIN_FREE_MB" ]; then
  echo "error: ~$((SIZE / 3)) gzipped bytes would leave less than $MIN_FREE_MB MB free in $DEST ($((FREE_KB / 1024)) MB free): records not saved"
  df -h "$DEST" || true
  exit 1
fi
WANT=$(kubectl -n "$NS" exec "$HELPER" -- sha256sum "$RUN_DIR/profile_export.jsonl" 2>/dev/null | cut -d' ' -f1 || true)
if ! [[ "$WANT" =~ ^[0-9a-f]{64}$ ]]; then
  echo "error: no sha256 of $RUN_DIR/profile_export.jsonl from pod/$HELPER: records not saved"
  exit 1
fi
SAVED=""
for TRY in $(seq 1 "$COPY_TRIES"); do
  rm -f "$OUT.tmp"
  kubectl -n "$NS" exec "$HELPER" -- gzip -c "$RUN_DIR/profile_export.jsonl" > "$OUT.tmp" 2>/dev/null || true
  # Checked on the decompressed content: a truncated stream fails gzip's own check too.
  GOT=$(gzip -dc "$OUT.tmp" 2>/dev/null | sha256sum | cut -d' ' -f1 || true)  # pipefail: a truncated gzip fails here
  if [ "$GOT" = "$WANT" ]; then SAVED=1; break; fi
  echo "warning: copy $TRY of $COPY_TRIES: $({ gzip -dc "$OUT.tmp" 2>/dev/null || true; } | wc -c | tr -d ' ') of $SIZE bytes, sha256 mismatch"
done
if [ -z "$SAVED" ]; then
  echo "error: no intact copy in $COPY_TRIES tries: records not saved"
  rm -f "$OUT.tmp"; exit 1
fi
mv "$OUT.tmp" "$OUT"
python3 - "$OUT" <<'PY' || true
import gzip, json, sys
n = errors = 0
for line in gzip.open(sys.argv[1], "rt"):
    n += 1
    errors += bool(json.loads(line).get("error"))
print("records: %d, with an error: %d" % (n, errors))
PY
echo "saved $OUT ($SIZE bytes, $(wc -c < "$OUT" | tr -d ' ') gzipped); $(df -h "$DEST" | awk 'NR == 2 {print $4}') left on the toolbox volume"
exit "$RC"
