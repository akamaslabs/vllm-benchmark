#!/bin/bash
# RunTest step of study 54 (toolbox, Akamas Executor task, after Apply config).
#
# One TraceLab replay per AIPerf client pod (client_run.sh, detached): users ramp 1 -> USERS_PER_CLIENT
# per client over RAMP_S s and never leave; the replay lasts RAMP_S + 300 s (workload.env).
# Before the launch, for each client: AIPerf must be 0.13.0 (else exit 6); the pool and the shared
# system prompt in /tmp must be the calibration's (sha256 / md5), else they are copied from LOAD_DIR
# (load/prepare_load.sh) and checked again (exit 6 if still wrong); client_run.sh is copied; anything
# left running in the pod is killed (study 47).
# The trial FAILS (exit 1) if the vLLM pod restarts or is replaced, a client's replay exits non-zero,
# vLLM completes nothing for STALL_S (or nothing at all for FIRST_OK_S: dataset build + first prefill),
# or the deadline (RAMP_S + DEADLINE_EXTRA_S) passes. The watchdog ENDS the test with SUCCESS (exit 0)
# once the server TTFT p95 over 150 s (vLLM's histogram, Prometheus) stays above WD_TTFT_MS for
# WD_HOLD_S, armed WD_ARM_DELAY_S after vLLM completes the run's first request: past the knee latency
# only grows with users, and the CSV Akamas reads ends at the knee anyway (aiperf_to_csv.py). Both
# replays ending (the ramp held to the end) is a success too, with a note to raise the users.
# The replays are left running for the Stop AIPerf task (stop_aiperf.sh: SIGINT, records saved), on
# failure too. The run's name (s54-<epoch>) goes to RUNS_DIR/current. Logs: each client's AIPerf log
# tail and CPU / memory samples, and on failure the vLLM log tail.
# Env overrides (smoke runs and tests): USERS_PER_CLIENT RAMP_S WD_TTFT_MS WD_HOLD_S WD_ARM_DELAY_S
# STALL_S DEADLINE_EXTRA_S LOAD_DIR RUNS_DIR, plus POLL_S FIRST_OK_S PROGRESS_EVERY_S PROM_QUERY.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OVR="USERS_PER_CLIENT RAMP_S WD_TTFT_MS WD_HOLD_S WD_ARM_DELAY_S STALL_S DEADLINE_EXTRA_S LOAD_DIR RUNS_DIR"
for v in $OVR; do eval "OV_$v=\${$v-}"; done
# shellcheck disable=SC1091
source "$HERE/workload.env"
for v in $OVR; do eval "[ -z \"\$OV_$v\" ] || $v=\$OV_$v"; done
# shellcheck disable=SC1091
source "$HERE/lib_watchdog.sh"
# shellcheck disable=SC1091
source "$HERE/lib_clients.sh"
POLL_S=${POLL_S:-15}; FIRST_OK_S=${FIRST_OK_S:-1500}; PROGRESS_EVERY_S=${PROGRESS_EVERY_S:-60}
PROM_QUERY=${PROM_QUERY:-$HERE/prom_query.sh}
DUR_S=$((RAMP_S + 300)); DEADLINE_S=$((RAMP_S + DEADLINE_EXTRA_S))
RUN=s54-$(date -u +%s)
t() { echo "$(date -u +%T) run_test: $*"; }
die() { echo "error: $*" >&2; exit "${2:-1}"; }

POD=$(kubectl -n "$NS" get pods -l app=vllm --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[ -n "$POD" ] || die "no Running vLLM pod in $NS: Apply config did not leave it up" 2
state() { kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}/{.status.containerStatuses[0].restartCount}/{.metadata.uid}' 2>/dev/null || echo gone; }
successes() {  # requests vLLM completed so far (empty if unreadable)
  kubectl -n "$NS" exec "$POD" -c main -- python3 -c \
    "import urllib.request;print(int(sum(float(l.split()[-1]) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith('vllm:request_success_total'))))" 2>/dev/null || true
}
STATE0=$(state)
t "vLLM pod $POD ($STATE0); run $RUN: 2 x 1 -> $USERS_PER_CLIENT users over $RAMP_S s, $DUR_S s; watchdog: server TTFT p95 (150 s) > $WD_TTFT_MS ms for $WD_HOLD_S s, armed $WD_ARM_DELAY_S s after the first completed request; deadline $DEADLINE_S s"

# --- 1. Clients: AIPerf version, pool, prompt, run script; nothing left running -------------------
for i in 0 1; do
  c=aiperf-client-$i
  v=$(kx "$c" python3 -c 'import importlib.metadata as m; print(m.version("aiperf"))' 2>/dev/null)
  [ "$v" = 0.13.0 ] || die "$c runs AIPerf '${v:-none}', not 0.13.0: apply k8s/02-aiperf-clients.yaml" 6
  want=POOL${i}_SHA
  pool_sha() { kx "$c" sh -c 'gzip -dc /tmp/pool.jsonl.gz 2>/dev/null | sha256sum | cut -c1-64'; }
  if [ "$(pool_sha)" != "${!want}" ]; then
    [ -s "$LOAD_DIR/pool_$i.jsonl.gz" ] || die "$c has not pool $i and $LOAD_DIR/pool_$i.jsonl.gz is missing: run load/prepare_load.sh" 6
    t "$c: copying pool $i from $LOAD_DIR"
    kubectl cp "$LOAD_DIR/pool_$i.jsonl.gz" "$NS/$c:/tmp/pool.jsonl.gz" -c aiperf >/dev/null 2>&1
    [ "$(pool_sha)" = "${!want}" ] || die "$c: pool $i sha256 still differs from workload.env's after the copy" 6
  fi
  if [ "$(kx "$c" sh -c 'md5sum /tmp/sys8k.txt 2>/dev/null | cut -c1-32')" != "$SYS_MD5" ]; then
    kubectl cp "$HERE/../load/sys8k.txt" "$NS/$c:/tmp/sys8k.txt" -c aiperf >/dev/null 2>&1
    [ "$(kx "$c" sh -c 'md5sum /tmp/sys8k.txt 2>/dev/null | cut -c1-32')" = "$SYS_MD5" ] || die "$c: system prompt md5 differs after the copy" 6
  fi
  kubectl cp "$HERE/client_run.sh" "$NS/$c:/tmp/client_run.sh" -c aiperf >/dev/null 2>&1 || die "$c: could not copy client_run.sh" 6
  kx "$c" python3 -c "$REAPER" || die "$c: processes left after the reaper" 6
done

# --- 2. Launch both replays (detached) ---------------------------------------------------------------
S0=$(successes); [[ "$S0" =~ ^[0-9]+$ ]] || S0=0
# Before the launch, so Stop AIPerf targets this run even if a launch fails (final review M2).
mkdir -p "$RUNS_DIR" && echo "$RUN" > "$RUNS_DIR/current"
for c in $CLIENTS; do
  kx "$c" sh -c "ls -dt /tmp/a13/runs/s54-* 2>/dev/null | tail -n +4 | xargs rm -rf; setsid nohup sh /tmp/client_run.sh /tmp/pool.jsonl.gz $RUN $USERS_PER_CLIENT $RAMP_S $DUR_S $URL $MODEL > /dev/null 2>&1 < /dev/null &" \
    || die "$c: launch failed" 1
done
t "replays launched"

# --- 3. Watch -------------------------------------------------------------------------------------
FAIL=""; WATCHDOG=""
T0=$SECONDS; ARMED_AT=""; OVER_SINCE=""; LAST_OK=$S0; LAST_PROGRESS=$SECONDS; NEXT_PROGRESS=$SECONDS; TTFT=""
while :; do
  # A changed state is read again up to 3 times, 5 s apart: one failed API call ("gone") must not fail a
  # trial that runs over an hour (final review M1); a real restart or replacement persists.
  NOW_STATE=$(state); k=0
  while [ "$NOW_STATE" != "$STATE0" ] && [ $k -lt 3 ]; do sleep $((POLL_S > 0 ? 5 : 0)); NOW_STATE=$(state); k=$((k + 1)); done
  [ "$NOW_STATE" = "$STATE0" ] || { FAIL="the vLLM pod restarted or was replaced during the test ($STATE0 -> $NOW_STATE)"; break; }
  n_done=0
  for i in 0 1; do
    e=$(kx "aiperf-client-$i" cat "/tmp/a13/runs/$RUN/exit" 2>/dev/null)
    [ -n "$e" ] || continue
    [ "${e%% *}" = 0 ] || { FAIL="client $i: the replay exited ${e%% *} after ${e#* } s, before the watchdog fired"; break 2; }
    n_done=$((n_done + 1))
  done
  [ "$n_done" = 2 ] && break   # both replays ended: the ramp completed
  if [ "$SECONDS" -ge "$NEXT_PROGRESS" ]; then
    NEXT_PROGRESS=$((SECONDS + PROGRESS_EVERY_S))
    OK=$(successes)
    if [[ "$OK" =~ ^[0-9]+$ ]]; then
      [ "$OK" != "$LAST_OK" ] && { LAST_OK=$OK; LAST_PROGRESS=$SECONDS; }
      if [ -z "$ARMED_AT" ] && [ "$OK" -gt "$S0" ]; then
        ARMED_AT=$SECONDS; t "first request completed $((SECONDS - T0)) s after the launch: watchdog armed in $WD_ARM_DELAY_S s"
      fi
      if [ -n "$ARMED_AT" ] && [ $((SECONDS - LAST_PROGRESS)) -ge "$STALL_S" ]; then
        FAIL="stalled: vLLM completed no request for $((SECONDS - LAST_PROGRESS)) s (at $((OK - S0)))"; break
      fi
      if [ -z "$ARMED_AT" ] && [ $((SECONDS - T0)) -ge "$FIRST_OK_S" ]; then
        FAIL="vLLM completed no request $((SECONDS - T0)) s after the launch"; break
      fi
      t "progress: $((OK - S0)) requests completed, server TTFT p95 (150 s) ${TTFT:-n/a} ms"
    fi
  fi
  if wd_ready "$SECONDS" "$ARMED_AT"; then
    TTFT=$("$PROM_QUERY" "histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{namespace=\"$NS\"}[150s])))*1000")
    if wd_over "$TTFT"; then NOW_OVER=0; else NOW_OVER=1; fi
    NEW_SINCE=$(wd_next_since "$SECONDS" "$OVER_SINCE" "$NOW_OVER")
    [ -z "$OVER_SINCE" ] && [ -n "$NEW_SINCE" ] && t "watchdog: over (server TTFT p95 ${TTFT} ms)"
    [ -n "$OVER_SINCE" ] && [ -z "$NEW_SINCE" ] && t "watchdog: back under (server TTFT p95 ${TTFT:-n/a} ms)"
    OVER_SINCE=$NEW_SINCE
    if wd_fired "$SECONDS" "$OVER_SINCE"; then WATCHDOG="server TTFT p95 ${TTFT} ms for $((SECONDS - OVER_SINCE)) s"; break; fi
  fi
  [ $((SECONDS - T0)) -ge "$DEADLINE_S" ] && { FAIL="test deadline (${DEADLINE_S} s)"; break; }
  sleep "$POLL_S"
done

# --- 4. Report --------------------------------------------------------------------------------------
if [ -n "$FAIL" ]; then
  echo "error: $FAIL, after $((SECONDS - T0)) s. Dumping logs, then failing the trial."
elif [ -n "$WATCHDOG" ]; then
  echo "Watchdog ended the test after $((SECONDS - T0)) s: $WATCHDOG. The test is complete (Akamas scores the best valid window before the knee)."
else
  echo "Both replays ended: the ramp completed after $((SECONDS - T0)) s ($USERS_PER_CLIENT users per client without the watchdog firing: raise USERS_PER_CLIENT)."
fi
for i in 0 1; do
  echo "--- aiperf-client-$i: AIPerf log tail (run $RUN) ---"
  kx "aiperf-client-$i" tail -n 40 "/tmp/a13/runs/$RUN/aiperf.log" 2>/dev/null || true
  echo "--- aiperf-client-$i: CPU / memory samples (last 3) ---"
  kx "aiperf-client-$i" tail -n 3 "/tmp/a13/runs/$RUN/cgroup.log" 2>/dev/null || true
done
if [ -n "$FAIL" ]; then
  echo "--- vLLM pod at failure ---"
  kubectl -n "$NS" get pods -l app=vllm -o wide || true
  kubectl -n "$NS" logs "$POD" -c main --tail=200 2>/dev/null || true
  exit 1
fi
exit 0
