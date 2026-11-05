#!/bin/bash
# Load task of study 47: one AIPerf multi-turn chat run (user-centric rate: USERS users, each
# sending a turn every USERS/RATE seconds, TURNS turns per conversation with the history re-sent)
# against the vLLM pod, from the long-lived AIPerf pod. The study windows on this task (trim
# 360s / 90s), so the goal (vLLM's TTFT p95) is read in steady state.
#
# Prints, for the Akamas task log and the README:
#   RESULT sent=<n> completed=<n> errors=<n> ttft_p50_ms=<x> ttft_p95_ms=<x> itl_p95_ms=<x>
#     client-side, requests started >= STEADY_FROM s after the first request
#   the TTFT p50/p95 per minute, the vLLM prefix-cache/offload counter deltas, the AIPerf log tail.
# Exit 5 if vLLM restarted during the run, 6 if errors >= 1 % or completed < 95 % of sent (the
# load is open-loop: a configuration that falls behind fails the trial instead of scoring).
# Env overrides for manual runs: DURATION, USERS, RATE.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NS=kv-offload
OV_DURATION=${DURATION:-} OV_USERS=${USERS:-} OV_RATE=${RATE:-}
# shellcheck disable=SC1091
source "$HERE/workload.env"
# shellcheck disable=SC1091
source "$HERE/lib_checks.sh"
DURATION=${OV_DURATION:-$DURATION} USERS=${OV_USERS:-$USERS} RATE=${OV_RATE:-$RATE}
t() { echo "$(date -u +%T) run_load: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# Only a Running pod counts: an evicted pod stays as Failed with the same label (final review).
POD=$(kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[ -n "$POD" ] || die "no Running vLLM pod in $NS: the Apply config task did not complete" 3
# Phase, restarts and uid: an OOM-killed container bumps the restart count, an evicted pod turns
# Failed (restart count unchanged), a replaced pod disappears.
restarts() { kubectl -n $NS get pod "$POD" -o jsonpath='{.status.phase}/{.status.containerStatuses[0].restartCount}/{.metadata.uid}' 2>/dev/null || echo gone; }
R0=$(restarts)

# vLLM counters (prefix cache, offload) before and after, read from inside the AIPerf pod.
SNAP='
import urllib.request, re
t = urllib.request.urlopen("http://vllm.kv-offload.svc:8000/metrics").read().decode()
for l in t.splitlines():
    m = re.match(r"(?P<n>vllm:(?:prefix_cache_(?:hits|queries)|external_prefix_cache_(?:hits|queries)|num_preemptions|prompt_tokens)_total|vllm:kv_offload_total_bytes_total)(?P<l>\{[^}]*\})? (?P<v>\S+)$", l)
    if m:
        tt = re.search(r"transfer_type=\"([^\"]+)\"", m.group("l") or "")
        print(m.group("n") + ("[" + tt.group(1) + "]" if tt else ""), m.group("v"))
'
snap() { kubectl -n $NS exec aiperf -- python3 -c "$SNAP" 2>/dev/null | sort; }

# A Load task killed by Akamas (finish/abort/timeout) drops its kubectl exec but the AIPerf
# process keeps running in the pod: on 2026-11-05 a survivor of an aborted experiment doubled the
# load of the next one (TTFT p95 73 s instead of 0.9 s). Kill everything in the pod except PID 1
# (sleep infinity) and this command, then check nothing is left.
REAPER='
import os, signal, time
me = {os.getpid(), os.getppid()}
def others():
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != 1 and int(p) not in me:
            try:
                c = open("/proc/%s/cmdline" % p, "rb").read().replace(b"\0", b" ").decode().strip()
            except OSError:
                continue
            if c: out.append((int(p), c))
    return out
found = others()
for pid, c in found:
    print("killing stale process %d: %s" % (pid, c[:100]))
    try: os.kill(pid, signal.SIGKILL)
    except OSError: pass
time.sleep(2)
left = others()
print("stale processes killed: %d, left: %d" % (len(found), len(left)))
raise SystemExit(1 if left else 0)
'
kubectl -n $NS exec aiperf -- python3 -c "$REAPER" || die "could not clear stale processes in the AIPerf pod" 6
M0=$(snap)

TASK_START=$(date +%s)
t "users=$USERS rate=$RATE req/s (one turn per user every $(awk -v u=$USERS -v r=$RATE 'BEGIN{print u/r}') s) turns=$TURNS isl=$ISL osl=$OSL duration=${DURATION}s"
kubectl -n $NS exec aiperf -- sh -c "rm -rf /work/out; mkdir -p /work/out; cd /work; \
  aiperf profile --model qwen3-8b --tokenizer Qwen/Qwen3-8B-FP8 --url http://vllm.$NS.svc:8000 \
    --endpoint-type chat --streaming \
    --isl $ISL --isl-stddev 0 --osl $OSL --osl-stddev 0 --extra-inputs ignore_eos:true \
    --conversation-turn-mean $TURNS --conversation-turn-stddev 0 --conversation-num 5000 \
    --user-centric-rate $RATE --num-users $USERS --benchmark-duration $DURATION --benchmark-grace-period 60 \
    --slice-duration 60 --ui simple --output-artifact-dir /work/out/run > /work/out/run.log 2>&1"
AIPERF_RC=$?
t "aiperf exit=$AIPERF_RC"
M1=$(snap)

# Client-side summary from the per-request export.
PARSER='
import json, sys
steady = float(sys.argv[1]); task_start = float(sys.argv[2])
rows = []
for l in open("/work/out/run/profile_export.jsonl"):
    j = json.loads(l); m = j["metadata"]
    if m.get("benchmark_phase") != "profiling": continue
    rows.append(j)
if not rows:
    print("RESULT sent=0 completed=0 errors=0 ttft_p50_ms=NA ttft_p95_ms=NA itl_p95_ms=NA"); sys.exit()
t0 = min(r["metadata"]["request_start_ns"] for r in rows) / 1e9
# Akamas trims 360 s from the start of the Load task, the RESULT line from the first request:
# this is the gap between the two windows (AIPerf start-up).
print("first request %.0f s after the Load task started" % (t0 - task_start))
def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
st = [r for r in rows if r["metadata"]["request_start_ns"] / 1e9 - t0 >= steady]
ok = [r for r in st if "time_to_first_token" in r.get("metrics", {}) and not r.get("error")]
err = [r for r in st if r.get("error") or r["metadata"].get("was_cancelled")]
ttft = [r["metrics"]["time_to_first_token"]["value"] for r in ok]
itl = [r["metrics"]["inter_token_latency"]["value"] for r in ok if "inter_token_latency" in r["metrics"]]
print("RESULT sent=%d completed=%d errors=%d ttft_p50_ms=%.0f ttft_p95_ms=%.0f itl_p95_ms=%.1f" %
      (len(st), len(ok), len(err), q(ttft, .5), q(ttft, .95), q(itl, .95)))
by = {}
for r in rows:
    if "time_to_first_token" in r.get("metrics", {}):
        by.setdefault(int((r["metadata"]["request_start_ns"] / 1e9 - t0) // 60), []).append(r["metrics"]["time_to_first_token"]["value"])
print("minute  requests  ttft_p50_ms  ttft_p95_ms")
for k in sorted(by): print("%6d  %8d  %11.0f  %11.0f" % (k, len(by[k]), q(by[k], .5), q(by[k], .95)))
'
SUMMARY=$(kubectl -n $NS exec aiperf -- python3 -c "$PARSER" "${STEADY_FROM_OVERRIDE:-$STEADY_FROM}" "$TASK_START" 2>&1)
echo "$SUMMARY"
echo "--- vLLM counters during the run (end - start) ---"
join <(echo "$M0") <(echo "$M1") | awk '{printf "%s %.0f\n", $1, $3 - $2}'
echo "--- AIPerf log (tail) ---"
kubectl -n $NS exec aiperf -- sh -c 'tail -c 4000 /work/out/run.log' 2>/dev/null | LC_ALL=C tr '\r' '\n' | grep -v 'Grace Period\|Profiling:' | tail -40
kubectl -n $NS exec aiperf -- rm -rf /work/out >/dev/null 2>&1 || true

[ "$(restarts)" = "$R0" ] || die "vLLM restarted or was replaced during the run: $R0 -> $(restarts)" 5
[ "$AIPERF_RC" = 0 ] || die "aiperf exited $AIPERF_RC" 6
EXPECTED_SENT=$(awk -v d="$DURATION" -v s="${STEADY_FROM_OVERRIDE:-$STEADY_FROM}" -v r="$RATE" 'BEGIN { printf "%d", (d - s) * r }')
CHECK=$(check_result "$SUMMARY" "$EXPECTED_SENT") || die "client-side check failed ($CHECK): the configuration does not keep up with the load" 6
t "done"
