# shellcheck shell=bash
# Load library of study 48, sourced by run_ramp.sh and run_hold.sh (after workload.env). It is
# study 47's run_load.sh turned into a function: run_level runs one AIPerf multi-turn chat load
# (user-centric rate) from the long-lived AIPerf pod against the vLLM pod and prints
#   first request <n> s after the level started
#   RESULT sent=<n> completed=<n> errors=<n> ttft_p50_ms=<x> ttft_p95_ms=<x> itl_p95_ms=<x>
#     client-side, requests started >= STEADY_FROM s after the first request
#   the TTFT p50/p95 per minute, the vLLM prefix-cache/offload counter deltas, the AIPerf log tail.
# On source it selects the Running vLLM pod (exit 3 if none).
NS=kv-offload
t() { echo "$(date -u +%T) ${SCRIPT_NAME:-load}: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

# Only a Running pod counts: an evicted pod stays as Failed with the same label (study 47).
POD=$(kubectl -n $NS get pods -l app=vllm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[ -n "$POD" ] || die "no Running vLLM pod in $NS: the Apply config task did not complete" 3
# Phase, restarts and uid: an OOM-killed container bumps the restart count, an evicted pod turns
# Failed (restart count unchanged), a replaced pod disappears.
restarts() { kubectl -n $NS get pod "$POD" -o jsonpath='{.status.phase}/{.status.containerStatuses[0].restartCount}/{.metadata.uid}' 2>/dev/null || echo gone; }

# Requests a level must send in its steady window if the server keeps up.
expected_sent() { awk -v u="$1" -v d="$LEVEL_SECONDS" -v s="$STEADY_FROM" -v p="$SECONDS_PER_TURN" 'BEGIN { printf "%d", (d - s) * u / p }'; }

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

# A task killed by Akamas (finish/abort/timeout) drops its kubectl exec but the AIPerf process
# keeps running in the pod (study 47, 2026-11-05: a survivor doubled the next load). Kill
# everything in the pod except PID 1 (sleep infinity) and this command, then check nothing is left.
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
# Akamas trims the Hold task from its start, the RESULT line from the first request: this is the
# gap between the two windows (AIPerf start-up).
print("first request %.0f s after the level started" % (t0 - task_start))
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

# run_level USERS SECONDS LABEL: one AIPerf run of USERS users (one turn each every
# SECONDS_PER_TURN s) for SECONDS s + 60 s grace. Prints the summary above; returns AIPerf's exit
# code. Exits 6 only if stale processes cannot be cleared from the AIPerf pod.
run_level() {
  local users=$1 secs=$2 label=$3 rate m0 m1 rc start summary
  rate=$(awk -v u="$users" -v p="$SECONDS_PER_TURN" 'BEGIN { printf "%.4f", u / p }')
  kubectl -n $NS exec aiperf -- python3 -c "$REAPER" || die "could not clear stale processes in the AIPerf pod" 6
  m0=$(snap)
  start=$(date +%s)
  t "[$label] users=$users rate=$rate req/s (one turn per user every $SECONDS_PER_TURN s) turns=$TURNS isl=$ISL osl=$OSL duration=${secs}s"
  kubectl -n $NS exec aiperf -- sh -c "rm -rf /work/out; mkdir -p /work/out; cd /work; \
    aiperf profile --model qwen3-8b --tokenizer Qwen/Qwen3-8B-FP8 --url http://vllm.$NS.svc:8000 \
      --endpoint-type chat --streaming \
      --isl $ISL --isl-stddev 0 --osl $OSL --osl-stddev 0 --extra-inputs ignore_eos:true \
      --conversation-turn-mean $TURNS --conversation-turn-stddev 0 --conversation-num 5000 \
      --user-centric-rate $rate --num-users $users --benchmark-duration $secs --benchmark-grace-period 60 \
      --slice-duration 60 --ui simple --output-artifact-dir /work/out/run > /work/out/run.log 2>&1"
  rc=$?
  t "[$label] aiperf exit=$rc"
  m1=$(snap)
  summary=$(kubectl -n $NS exec aiperf -- python3 -c "$PARSER" "$STEADY_FROM" "$start" 2>&1)
  echo "$summary"
  echo "--- vLLM counters during the run (end - start) ---"
  join <(echo "$m0") <(echo "$m1") | awk '{printf "%s %.0f\n", $1, $3 - $2}'
  echo "--- AIPerf log (tail) ---"
  kubectl -n $NS exec aiperf -- sh -c 'tail -c 4000 /work/out/run.log' 2>/dev/null | LC_ALL=C tr '\r' '\n' | grep -v 'Grace Period\|Profiling:' | tail -40
  kubectl -n $NS exec aiperf -- rm -rf /work/out >/dev/null 2>&1 || true
  return $rc
}
