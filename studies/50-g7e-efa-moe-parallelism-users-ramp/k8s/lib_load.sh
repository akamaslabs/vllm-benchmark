# shellcheck shell=bash
# Load library of study 50, sourced by run_ramp.sh and run_hold.sh (after workload.env). run_level
# runs one AIPerf multi-turn chat load (user-centric rate) from the two long-lived AIPerf client
# pods, half the users each, and prints
#   client <i>: ...   each client's own summary (first request, RESULT, tokens per answer, CPU,
#                     TTFT per minute); never starts with RESULT
#   RESULT sent=<n> completed=<n> errors=<n> ttft_p50_ms=<x> ttft_p95_ms=<x> itl_p95_ms=<x>
#     the two clients merged over requests started >= STEADY_FROM s after each client's first
#     request: counts are summed, latencies are the worse (max) of the two clients
#   the vLLM counter deltas (summed over the serving pods), the AIPerf log tails.
# Where the clients point: when pod b runs its own API server (layouts replicas, dp2 and dp2ep
# with external load balancing) client 0 loads node a and client 1 node b, each through its
# per-pod Service; otherwise both clients load node a. Deviation from the design, which named one
# client per node only: two clients are kept on every layout so that the load generator's own
# CPU budget is the same everywhere.
# On source it requires a Running vllm-a-0 (exit 3).
NS=moe-efa
t() { echo "$(date -u +%T) ${SCRIPT_NAME:-load}: $*"; }
die() { echo "error: $*" >&2; exit "${2:-2}"; }

[ "$(kubectl -n $NS get pod vllm-a-0 -o jsonpath='{.status.phase}' 2>/dev/null)" = Running ] \
  || die "vllm-a-0 is not Running in $NS: the Apply config task did not complete" 3
SERVING_B=$(kubectl -n $NS get pod vllm-b-0 -o go-template='{{index .metadata.labels "study50/serving"}}' 2>/dev/null || true)
URL_A=http://vllm-a.$NS.svc:8000
if [ "$SERVING_B" = true ]; then URLS=("$URL_A" "http://vllm-b.$NS.svc:8000"); else URLS=("$URL_A" "$URL_A"); fi

# Name, phase, restarts and uid of every vLLM pod: an OOM-killed container bumps the restart
# count, an evicted pod turns Failed, a replaced pod changes uid, a deleted one disappears.
restarts() {
  kubectl -n $NS get pods -l app=vllm --sort-by=.metadata.name \
    -o jsonpath='{range .items[*]}{.metadata.name}={.status.phase}/{.status.containerStatuses[0].restartCount}/{.metadata.uid}{" "}{end}' 2>/dev/null || echo gone
}

# Requests a level must send in its steady window (both clients together) if the server keeps up.
expected_sent() { awk -v u="$1" -v d="$LEVEL_SECONDS" -v s="$STEADY_FROM" -v p="$SECONDS_PER_TURN" 'BEGIN { printf "%d", (d - s) * u / p }'; }

# vLLM counters before and after, read from inside client 0 and summed over the API servers given
# as arguments (duplicates dropped) and over every label set (engine, finished_reason).
SNAP='
import urllib.request, re, sys
tot = {}
for u in sorted(set(sys.argv[1:])):
    text = urllib.request.urlopen(u + "/metrics", timeout=10).read().decode()
    for l in text.splitlines():
        m = re.match(r"(vllm:(?:prefix_cache_hits|prefix_cache_queries|num_preemptions|prompt_tokens|generation_tokens|request_success)_total)(\{[^}]*\})? (\S+)$", l)
        if m: tot[m.group(1)] = tot.get(m.group(1), 0.0) + float(m.group(3))
for k in sorted(tot): print(k, "%.0f" % tot[k])
'
snap() { kubectl -n $NS exec aiperf-client-0 -- python3 -c "$SNAP" "${URLS[@]}" 2>/dev/null | sort; }

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

# One client's summary from its per-request export. Also reports what AIPerf counted per answer
# (gpt-oss streams a reasoning channel first: the smoke checks read here whether ignore_eos gave
# OSL tokens) and the CPU the client pod used and was throttled for (cgroup cpu.stat).
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
if err:
    e = err[0].get("error"); print("first error: %s" % (json.dumps(e)[:300] if e else "cancelled"))
keys = sorted({k for r in ok for k in r["metrics"] if "token" in k or "sequence_length" in k})
for k in keys:
    v = [r["metrics"][k]["value"] for r in ok if k in r["metrics"] and isinstance(r["metrics"][k].get("value"), (int, float))]
    if v: print("per answer %s: mean %.1f min %.1f max %.1f (n=%d)" % (k, sum(v) / len(v), min(v), max(v), len(v)))
def cpu(path):
    try: return dict((a, int(b)) for a, b in (l.split() for l in open(path)))
    except (OSError, ValueError): return None
c0, c1 = cpu("/work/out/cpu0"), cpu("/work/out/cpu1")
if c0 and c1:
    wall = max(1.0, float(sys.argv[3]))
    used = (c1.get("usage_usec", 0) - c0.get("usage_usec", 0)) / 1e6
    print("cpu used %.0f s (%.2f cores on average), throttled %.1f s in %d periods" %
          (used, used / wall, (c1.get("throttled_usec", 0) - c0.get("throttled_usec", 0)) / 1e6,
           c1.get("nr_throttled", 0) - c0.get("nr_throttled", 0)))
by = {}
for r in rows:
    if "time_to_first_token" in r.get("metrics", {}):
        by.setdefault(int((r["metadata"]["request_start_ns"] / 1e9 - t0) // 60), []).append(r["metrics"]["time_to_first_token"]["value"])
print("minute  requests  ttft_p50_ms  ttft_p95_ms")
for k in sorted(by): print("%6d  %8d  %11.0f  %11.0f" % (k, len(by[k]), q(by[k], .5), q(by[k], .95)))
'

# merge_results: stdin = the two clients' prefixed summaries; prints the merged RESULT line, or
# nothing unless both clients printed one. A latency that is not a number in either client is NA.
merge_results() {
  awk '
    /^client [0-9]+: RESULT / { n++
      for (k = 4; k <= NF; k++) { split($k, a, "="); key = a[1]; val = a[2]
        if (key == "sent" || key == "completed" || key == "errors") sum[key] += val
        else if (val !~ /^[0-9]+(\.[0-9]+)?$/) na[key] = 1
        else if (!(key in mx) || val + 0 > mx[key] + 0) mx[key] = val } }
    function lat(k) { return ((k in na) || !(k in mx)) ? "NA" : mx[k] }
    END { if (n == 2) printf "RESULT sent=%d completed=%d errors=%d ttft_p50_ms=%s ttft_p95_ms=%s itl_p95_ms=%s\n",
            sum["sent"], sum["completed"], sum["errors"], lat("ttft_p50_ms"), lat("ttft_p95_ms"), lat("itl_p95_ms") }'
}

# run_level USERS SECONDS LABEL: one AIPerf run of USERS users (one turn each every
# SECONDS_PER_TURN s) for SECONDS s + 60 s grace, USERS / 2 from each client pod. Prints the
# summary above; returns the worse of the two AIPerf exit codes (1 if they exited 0 but a client
# gave no RESULT line). Exits 6 only if stale processes cannot be cleared from a client pod.
run_level() {
  local users=$1 secs=$2 label=$3 half rate m0 m1 rc=0 r start i merged pids=() out=()
  half=$((users / 2))
  [ $((half * 2)) = "$users" ] || die "level $users is odd: the two clients take half the users each"
  rate=$(awk -v u="$half" -v p="$SECONDS_PER_TURN" 'BEGIN { printf "%.4f", u / p }')
  for i in 0 1; do
    kubectl -n $NS exec aiperf-client-$i -- python3 -c "$REAPER" | sed "s/^/client $i: /"
    [ "${PIPESTATUS[0]}" = 0 ] || die "could not clear stale processes in aiperf-client-$i" 6
  done
  m0=$(snap)
  start=$(date +%s)
  t "[$label] users=$users: 2 clients x $half users, $rate req/s each (one turn per user every $SECONDS_PER_TURN s) turns=$TURNS isl=$ISL osl=$OSL extra=[$EXTRA_INPUTS] duration=${secs}s; client 0 -> ${URLS[0]}, client 1 -> ${URLS[1]}"
  for i in 0 1; do
    kubectl -n $NS exec aiperf-client-$i -- sh -c "rm -rf /work/out; mkdir -p /work/out; cd /work; \
      cat /sys/fs/cgroup/cpu.stat > /work/out/cpu0 2>/dev/null; \
      aiperf profile --model gpt-oss-120b --tokenizer openai/gpt-oss-120b --url ${URLS[$i]} \
        --endpoint-type chat --streaming \
        --isl $ISL --isl-stddev 0 --osl $OSL --osl-stddev 0 --extra-inputs $EXTRA_INPUTS \
        --conversation-turn-mean $TURNS --conversation-turn-stddev 0 --conversation-num 5000 \
        --user-centric-rate $rate --num-users $half --benchmark-duration $secs --benchmark-grace-period 60 \
        --slice-duration 60 --ui simple --output-artifact-dir /work/out/run > /work/out/run.log 2>&1; rc=\$?; \
      cat /sys/fs/cgroup/cpu.stat > /work/out/cpu1 2>/dev/null; exit \$rc" &
    pids+=($!)
  done
  for i in 0 1; do
    wait "${pids[$i]}"; r=$?
    t "[$label] client $i aiperf exit=$r"
    [ "$r" -le "$rc" ] || rc=$r
  done
  m1=$(snap)
  for i in 0 1; do
    out[$i]=$(kubectl -n $NS exec aiperf-client-$i -- python3 -c "$PARSER" "$STEADY_FROM" "$start" "$(( $(date +%s) - start ))" 2>&1 | sed "s/^/client $i: /")
    echo "${out[$i]}"
  done
  merged=$(printf '%s\n%s\n' "${out[0]}" "${out[1]}" | merge_results)
  if [ -n "$merged" ]; then echo "$merged"; else
    echo "no merged RESULT: a client printed no RESULT line"
    [ "$rc" != 0 ] || rc=1
  fi
  echo "--- vLLM counters during the run (end - start, summed over the serving pods) ---"
  join <(echo "$m0") <(echo "$m1") | awk '{printf "%s %.0f\n", $1, $3 - $2}'
  for i in 0 1; do
    echo "--- client $i: AIPerf log (tail) ---"
    kubectl -n $NS exec aiperf-client-$i -- sh -c 'tail -c 4000 /work/out/run.log' 2>/dev/null | LC_ALL=C tr '\r' '\n' | grep -v 'Grace Period\|Profiling:' | tail -30
    kubectl -n $NS exec aiperf-client-$i -- rm -rf /work/out >/dev/null 2>&1 || true
  done
  return $rc
}
