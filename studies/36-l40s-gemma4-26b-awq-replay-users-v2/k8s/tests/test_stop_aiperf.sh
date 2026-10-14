#!/bin/bash
# stop_aiperf.sh: graceful stop, Job deleted, records saved once, gzipped, as <epoch>.jsonl.gz (stub kubectl).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
printf '%s\n' '{"metadata":{"was_cancelled":false},"metrics":{}}' \
  '{"metadata":{"was_cancelled":true},"metrics":{},"error":{"type":"cancelled"}}' \
  '{"metadata":{"was_cancelled":false},"metrics":{}}' > "$TMP/src.jsonl"
DEST=$TMP/dest
OUT=$DEST/1791815986.jsonl.gz
same() { cmp -s <(gzip -dc "$OUT" 2>/dev/null) "$TMP/src.jsonl"; }  # the saved gzip holds exactly the records
run() {  # env assignments for the stub, then stop_aiperf.sh
  : > "$TMP/kubectl.log"; rm -f "$TMP/sigint" "$TMP/copies"
  env PATH="$HERE/stub-stop:$PATH" STUB_LOG="$TMP/kubectl.log" STUB_DIR="$TMP" STUB_JSONL="$TMP/src.jsonl" \
    SA_DEST="$DEST" SA_POLL_S=0 "$@" bash "$K8S/stop_aiperf.sh" > "$TMP/out.txt" 2>&1
}
# A. RunTest left the Job running (watchdog): SIGINT, Job deleted, records saved.
run STUB_POD=aiperf-replay-x STUB_PHASE=Running STUB_RUN_DIR=/benchmarks/study36/replay-benchmark/data/results/run_202610122359_1791815986; rc=$?
[ $rc = 0 ] && ok "running: exit 0" || ko "running: exit $rc ($(tail -3 "$TMP/out.txt"))"
grep -q 'exec aiperf-replay-x -c aiperf -- python3 -c' "$TMP/kubectl.log" && ok "running: SIGINT sent to aiperf" || ko "running: SIGINT sent"
grep -q 'logs aiperf-replay-x -c aiperf --since=' "$TMP/out.txt" "$TMP/kubectl.log" && ok "running: logs since the SIGINT dumped" || ko "running: logs dumped"
grep -q 'delete job -l app=aiperf-replay --ignore-not-found --cascade=foreground --wait=true' "$TMP/kubectl.log" \
  && ok "Job deleted, foreground" || ko "Job deleted, foreground"
s=$(grep -n 'SIGINT\|python3 -c' "$TMP/kubectl.log" | head -1 | cut -d: -f1); d=$(grep -n 'delete job' "$TMP/kubectl.log" | head -1 | cut -d: -f1)
[ -n "$s" ] && [ -n "$d" ] && [ "$s" -lt "$d" ] && ok "SIGINT before the Job delete" || ko "SIGINT before the Job delete"
grep -q '^applied aiperf-export$' "$TMP/kubectl.log" && ok "export pod applied" || ko "export pod applied"
same && ok "saved as <epoch>.jsonl.gz, identical" || ko "saved as <epoch>.jsonl.gz ($(ls "$DEST" 2>&1))"
ls "$DEST"/*.tmp >/dev/null 2>&1 && ko "no .tmp left" || ok "no .tmp left"
grep -q 'records: 3, with an error: 1' "$TMP/out.txt" && ok "record summary printed" || ko "record summary printed"
grep -q 'delete pod aiperf-export --ignore-not-found --wait=false' "$TMP/kubectl.log" && ok "export pod deleted at exit" || ko "export pod deleted at exit"
# B. Run again (Apply config failed in the next trial: the volume holds the same run): no copy.
run STUB_RUN_DIR=/benchmarks/study36/replay-benchmark/data/results/run_202610122359_1791815986; rc=$?
[ $rc = 0 ] && ok "again: exit 0" || ko "again: exit $rc"
grep -q 'already saved' "$TMP/out.txt" && ok "again: already saved" || ko "again: already saved"
grep -q -- '-- gzip -c ' "$TMP/kubectl.log" && ko "again: not copied twice" || ok "again: not copied twice"
grep -q 'python3 -c' "$TMP/kubectl.log" && ko "no pod: no SIGINT" || ok "no pod: no SIGINT"
rm -rf "$DEST"
# C. The ramp completed (pod Succeeded): no SIGINT, records saved.
run STUB_POD=aiperf-replay-x STUB_PHASE=Succeeded STUB_RUN_DIR=/benchmarks/study36/replay-benchmark/data/results/run_202610122359_1791815986; rc=$?
[ $rc = 0 ] && ok "completed: exit 0" || ko "completed: exit $rc"
grep -q 'python3 -c' "$TMP/kubectl.log" && ko "completed: no SIGINT" || ok "completed: no SIGINT"
same && ok "completed: saved" || ko "completed: saved"
rm -rf "$DEST"
# D. Stopped before the measured run started: nothing to save, exit 0.
run STUB_RUN_DIR=; rc=$?
[ $rc = 0 ] && ok "no run: exit 0" || ko "no run: exit $rc"
grep -q 'nothing to save' "$TMP/out.txt" && ok "no run: nothing to save" || ko "no run: nothing to save"
# E. Not enough room on the toolbox volume: not saved, exit 1.
run STUB_RUN_DIR=/benchmarks/study36/replay-benchmark/data/results/run_202610122359_1791815986 SA_MIN_FREE_MB=999999999; rc=$?
[ $rc = 1 ] && ok "no room: exit 1" || ko "no room: exit $rc"
[ -e "$OUT" ] || ls "$DEST"/*.tmp >/dev/null 2>&1 && ko "no room: nothing written" || ok "no room: nothing written"
rm -rf "$DEST"
# F. The first copy arrives truncated: retried, the second is saved intact.
run STUB_RUN_DIR=/benchmarks/study36/replay-benchmark/data/results/run_202610122359_1791815986 STUB_BAD_COPIES=1; rc=$?
[ $rc = 0 ] && ok "truncated once: exit 0" || ko "truncated once: exit $rc ($(tail -3 "$TMP/out.txt"))"
grep -q 'copy 1 of 3: .* sha256 mismatch' "$TMP/out.txt" && ok "truncated once: mismatch reported" || ko "truncated once: mismatch reported"
same && ok "truncated once: saved intact" || ko "truncated once: saved intact"
rm -rf "$DEST"
# G. Every copy arrives truncated: not saved, exit 1.
run STUB_RUN_DIR=/benchmarks/study36/replay-benchmark/data/results/run_202610122359_1791815986 STUB_BAD_COPIES=9; rc=$?
[ $rc = 1 ] && ok "always truncated: exit 1" || ko "always truncated: exit $rc"
[ "$(cat "$TMP/copies")" = 3 ] && ok "always truncated: 3 tries" || ko "always truncated: $(cat "$TMP/copies") tries"
[ -e "$OUT" ] || ls "$DEST"/*.tmp >/dev/null 2>&1 && ko "always truncated: nothing written" || ok "always truncated: nothing written"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
