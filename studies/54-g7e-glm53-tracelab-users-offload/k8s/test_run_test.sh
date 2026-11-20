#!/bin/bash
# Local tests of run_test.sh, stop_aiperf.sh and aiperf_csv.sh against a stub kubectl and a stub
# Prometheus query (no cluster). The stub answers from files in $ST (state that the test changes)
# and logs every call to $ST/calls.
# Usage: bash k8s/test_run_test.sh   → "N passed, M failed", exit 1 on any failure.
HERE=$(cd "$(dirname "$0")" && pwd)
PASS=0; FAIL=0
ok()  { PASS=$((PASS + 1)); }
bad() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; echo "$OUT" | tail -15 | sed 's/^/    /'; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# shellcheck disable=SC1091
source "$HERE/workload.env"
export POOL0_SHA POOL1_SHA SYS_MD5   # read by the stub kubectl

mkdir -p "$TMP/bin"
cat > "$TMP/bin/kubectl" <<'STUB'
#!/bin/bash
echo "kubectl $*" >> "$ST/calls"
a="$*"
cnt() { local n; n=$(cat "$ST/$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$ST/$1"; echo $((n + 1)); }
case "$a" in
  *"get pods -l app=vllm --field-selector=status.phase=Running"*) echo vllm-abc ;;
  *"get pod vllm-abc -o jsonpath"*) n=$(cnt state_calls); if [ -n "${UID_CHANGE_AT:-}" ] && [ "$n" -ge "$UID_CHANGE_AT" ]; then echo "Running/0/uid-B"; else echo "Running/0/uid-A"; fi ;;
  *"exec aiperf-client-"*"importlib.metadata"*) echo "${AIPERF_VERSION:-0.13.0}" ;;
  *"exec aiperf-client-0"*"sha256sum"*) echo "${POOL0_GOT:-$POOL0_SHA}" ;;
  *"exec aiperf-client-1"*"sha256sum"*) echo "${POOL1_GOT:-$POOL1_SHA}" ;;
  *"exec aiperf-client-"*"md5sum"*) echo "$SYS_MD5" ;;
  *"exec aiperf-client-"*"stale processes"*) echo "stale processes killed: 1, left: 0" ;;
  *"exec aiperf-client-"*"SIGINT"*) echo 4242 ;;
  *"exec aiperf-client-"*"client_run.sh"*) : ;;
  *"exec aiperf-client-0"*"/exit"*) [ -f "$ST/exit0" ] && [ "$(cnt exit_calls0)" -ge "${EXIT_AFTER:-1}" ] && cat "$ST/exit0" ;;
  *"exec aiperf-client-1"*"/exit"*) [ -f "$ST/exit1" ] && [ "$(cnt exit_calls1)" -ge "${EXIT_AFTER:-1}" ] && cat "$ST/exit1" ;;
  *"exec aiperf-client-"*"find /tmp/a13/runs"*) echo /tmp/a13/runs/x/aiperf/profile_export.jsonl ;;
  *"exec aiperf-client-"*"gzip -c"*) printf 'records' | gzip -c ;;
  *"exec aiperf-client-"*"tail"*) echo "aiperf log line" ;;
  *"exec vllm-abc"*"request_success_total"*) n=$(cnt succ); echo $((n * 10)) ;;
  *"logs"*) echo "vllm log line" ;;
  *) : ;;
esac
exit 0
STUB
chmod +x "$TMP/bin/kubectl"
printf '#!/bin/bash\necho "prom $*" >> "$ST/calls"; echo "${PROM_TTFT:-1000}"\n' > "$TMP/bin/promq"; chmod +x "$TMP/bin/promq"

run() {  # run SCRIPT [env...] -> OUT, RC
  local s=$1; shift
  rm -rf "$TMP/st"; mkdir -p "$TMP/st" "$TMP/runs"
  [ -n "${EXIT0:-}" ] && echo "$EXIT0" > "$TMP/st/exit0"; [ -n "${EXIT1:-}" ] && echo "$EXIT1" > "$TMP/st/exit1"
  OUT=$(env PATH="$TMP/bin:$PATH" ST="$TMP/st" PROM_QUERY="$TMP/bin/promq" RUNS_DIR="$TMP/runs" LOAD_DIR="$TMP/load" \
        POLL_S=0 PROGRESS_EVERY_S=0 "$@" bash "$HERE/$s" 2>&1); RC=$?
}
calls() { cat "$TMP/st/calls" 2>/dev/null; }

# 1. Both clients finish the ramp: reaped before the launch, launched with the workload's values, exit 0.
EXIT0="0 4600" EXIT1="0 4601" run run_test.sh
[ "$RC" = 0 ] && ok || bad "ramp completed: exit $RC"
L=$(calls | grep -n 'setsid nohup' | head -1 | cut -d: -f1); R=$(calls | grep -n 'stale processes' | head -1 | cut -d: -f1)
[ -n "$L" ] && [ -n "$R" ] && [ "$R" -lt "$L" ] && ok || bad "the reaper must run before the launch (reaper line ${R:-none}, launch ${L:-none})"
[ "$(calls | grep -c 'stale processes')" -ge 2 ] && ok || bad "both clients reaped"
calls | grep 'setsid nohup' | grep -q "/tmp/client_run.sh /tmp/pool.jsonl.gz s54-[0-9]* $USERS_PER_CLIENT $RAMP_S $((RAMP_S + 300)) $URL $MODEL" && ok || bad "launch arguments: $(calls | grep 'setsid nohup' | head -1)"
[ "$(calls | grep 'setsid nohup' | grep -c 'exec aiperf-client-')" = 2 ] && ok || bad "both clients launched"
grep -qE '^s54-[0-9]+$' "$TMP/runs/current" && ok || bad "run name not written to RUNS_DIR/current"
echo "$OUT" | grep -q 'ramp completed' && ok || bad "ramp completed message"

# 2. The vLLM pod is replaced mid-run (spot reclaim): exit 1, vLLM log dumped.
UID_CHANGE_AT=3 run run_test.sh
[ "$RC" = 1 ] && ok || bad "pod replaced: exit $RC, expected 1"
echo "$OUT" | grep -q 'restarted or was replaced' && ok || bad "pod replaced: reason"

# 3. A client with the wrong pool and no staged pool on the toolbox: exit 6 before any launch.
POOL1_GOT=deadbeef run run_test.sh
[ "$RC" = 6 ] && ok || bad "wrong pool: exit $RC, expected 6"
calls | grep -q 'setsid nohup' && bad "wrong pool: launched anyway" || ok
# ... with the pool staged: copied into the pod (kubectl cp), then checked again.
mkdir -p "$TMP/load"; echo x > "$TMP/load/pool_1.jsonl.gz"
POOL1_GOT=deadbeef run run_test.sh
calls | grep -q "kubectl cp $TMP/load/pool_1.jsonl.gz $NS/aiperf-client-1:/tmp/pool.jsonl.gz" && ok || bad "staged pool not copied: $(calls | grep ' cp ' | head -2)"
[ "$RC" = 6 ] && ok || bad "pool still wrong after the copy: exit $RC, expected 6"
rm -rf "$TMP/load"

# 4. AIPerf is not 0.13.0 in the clients: exit 6.
AIPERF_VERSION=0.11.0 run run_test.sh
[ "$RC" = 6 ] && ok || bad "AIPerf 0.11.0: exit $RC, expected 6"

# 5. The watchdog: server TTFT p95 20 s, armed at once, held 0 s -> exit 0, the load left running.
PROM_TTFT=20000 run run_test.sh WD_ARM_DELAY_S=0 WD_HOLD_S=0
[ "$RC" = 0 ] && ok || bad "watchdog: exit $RC"
echo "$OUT" | grep -q 'Watchdog ended the test' && ok || bad "watchdog message"
calls | grep -q 'SIGINT' && bad "run_test.sh must leave the stop to stop_aiperf.sh" || ok
calls | grep 'prom ' | grep -q "vllm:time_to_first_token_seconds_bucket{namespace=\"$NS\"}\[150s\]" && ok || bad "watchdog query: $(calls | grep 'prom ' | head -1)"

# 6. A client exits non-zero before the watchdog: exit 1.
EXIT0="1 30" run run_test.sh
[ "$RC" = 1 ] && ok || bad "client failed: exit $RC, expected 1"
echo "$OUT" | grep -q 'client 0' && ok || bad "client failed: names the client"

# 7. stop_aiperf.sh: SIGINT both clients, saves both exports in RUNS_DIR/<run>.
mkdir -p "$TMP/runs"; echo s54-123 > "$TMP/runs/current"
EXIT0="0 1" EXIT1="0 1" run stop_aiperf.sh
[ "$RC" = 0 ] && ok || bad "stop: exit $RC"
[ "$(calls | grep -c SIGINT)" -ge 2 ] && ok || bad "stop: SIGINT to both clients"
[ "$(gzip -dc "$TMP/runs/s54-123/c0.jsonl.gz" 2>/dev/null)" = records ] && ok || bad "stop: c0 export not saved"
[ "$(gzip -dc "$TMP/runs/s54-123/c1.jsonl.gz" 2>/dev/null)" = records ] && ok || bad "stop: c1 export not saved"

# 8. aiperf_csv.sh: runs the converter on the saved exports with the workload's SLO and pools.
mkdir -p "$TMP/load" "$TMP/csv" "$TMP/csvfull" "$TMP/runs/s54-9"
echo s54-9 > "$TMP/runs/current"
cat > "$TMP/bin/fakeconv" <<'EOF'
#!/bin/bash
echo "conv $*" >> "$ST/calls"; echo "knee: none"
EOF
chmod +x "$TMP/bin/fakeconv"
printf 'r\n' | gzip -c > "$TMP/runs/s54-9/c0.jsonl.gz"; printf 'r\n' | gzip -c > "$TMP/runs/s54-9/c1.jsonl.gz"
mkdir -p "$TMP/st"; : > "$TMP/st/calls"
OUT=$(env ST="$TMP/st" RUNS_DIR="$TMP/runs" LOAD_DIR="$TMP/load" CSV_DIR="$TMP/csv" CSV_FULL_DIR="$TMP/csvfull" \
      CONVERTER="$TMP/bin/fakeconv" SCORER="$TMP/bin/fakeconv" bash "$HERE/aiperf_csv.sh" 2>&1); RC=$?
[ "$RC" = 0 ] && ok || bad "aiperf_csv: exit $RC"
C=$(grep '^conv --pair' "$TMP/st/calls" | head -1)
[[ "$C" == *"--pair $TMP/runs/s54-9/c0.jsonl $TMP/load/pool_0.jsonl.gz --pair $TMP/runs/s54-9/c1.jsonl $TMP/load/pool_1.jsonl.gz"* ]] && ok || bad "aiperf_csv: pairs: $C"
[[ "$C" == *"--out $TMP/csv/s54-9.csv --full-out $TMP/csvfull/s54-9.csv"* ]] && ok || bad "aiperf_csv: outputs: $C"
[[ "$C" == *"--slo-ttft-ms $SLO_TTFT_MS --slo-itl-ms $SLO_ITL_MS --min-window-reqs $MIN_WINDOW_REQS --breach-buckets $BREACH_BUCKETS"* ]] && ok || bad "aiperf_csv: SLO args: $C"
rm -f "$TMP/runs/s54-9/c1.jsonl.gz"
OUT=$(env ST="$TMP/st" RUNS_DIR="$TMP/runs" LOAD_DIR="$TMP/load" CSV_DIR="$TMP/csv" CSV_FULL_DIR="$TMP/csvfull" \
      CONVERTER="$TMP/bin/fakeconv" SCORER="$TMP/bin/fakeconv" bash "$HERE/aiperf_csv.sh" 2>&1); RC=$?
[ "$RC" = 2 ] && ok || bad "aiperf_csv without c1: exit $RC, expected 2"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
