#!/bin/bash
# apply_config.sh must refuse an invalid params.env BEFORE any kubectl call (Review Focus 1).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); STUDY=$(dirname "$K8S")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
run_with() {  # $1 name, $2 params.env content
  printf '%s\n' "$2" > "$TMP/params.env"; : > "$TMP/kubectl.log"
  PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUDY_DIR="$STUDY" PARAMS="$TMP/params.env" \
    RENDERED="$TMP/sts.yaml" bash "$K8S/apply_config.sh" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -s "$TMP/kubectl.log" ]; } && ok "$1: exit 2, no kubectl call" \
    || ko "$1: rc=$rc, kubectl calls: $(wc -l < "$TMP/kubectl.log")"
}
GOOD='MIG_PROFILE=1g.16gb
CPU_LIMIT=2500
MEMORY_LIMIT=12000
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=4096
LINEAR_BACKEND=cutlass
ATTENTION_BACKEND=FLASHINFER'
with() { printf '%s\n' "$GOOD" | sed "$1"; }   # GOOD with one sed edit
run_with "empty value" "$(with 's|^LINEAR_BACKEND=.*|LINEAR_BACKEND=|')"
# shellcheck disable=SC2016
run_with "leftover token" "$(with 's|^CPU_LIMIT=.*|CPU_LIMIT=${container.cpu_limit}|')"
run_with "FLASH_ATTN with fp8" "$(with 's|^ATTENTION_BACKEND=.*|ATTENTION_BACKEND=FLASH_ATTN|')"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
