#!/bin/bash
# Stage study 54's load on the toolbox (or any host): download TraceLab v0.0.2 into OUT_DIR if missing
# (checked against its sha256), build the two client pools with prepare_pool.py (fixed arguments from
# ../k8s/workload.env), copy the shared system prompt, and fail unless the pools and the prompt are
# bit for bit the calibration's (sha256 of the decompressed pools, md5 of the prompt).
# Pools already in OUT_DIR with the right checksums are kept as they are (nothing downloaded or built):
# building them holds the whole trace in memory (~5 GB), more than the akamas-41 toolbox's container
# allows (killed, 2026-11-20), so there they are built elsewhere and copied in, and this script checks them.
# Usage: bash load/prepare_load.sh [OUT_DIR]   (default: workload.env's LOAD_DIR)
# Exit 0 staged and verified; 2 download or checksum failure.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$HERE/../k8s/workload.env"
OUT=${1:-$LOAD_DIR}
mkdir -p "$OUT"
sha() { if command -v sha256sum >/dev/null; then sha256sum | cut -c1-64; else shasum -a 256 | cut -c1-64; fi; }
md5() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -c1-32; else md5 -q "$1"; fi; }
staged() {
  [ "$(gzip -dc "$OUT/pool_0.jsonl.gz" 2>/dev/null | sha)" = "$POOL0_SHA" ] \
    && [ "$(gzip -dc "$OUT/pool_1.jsonl.gz" 2>/dev/null | sha)" = "$POOL1_SHA" ] \
    && [ -f "$OUT/sys8k.txt" ] && [ "$(md5 "$OUT/sys8k.txt")" = "$SYS_MD5" ]
}
if staged; then echo "pool_0, pool_1 and sys8k.txt already staged in $OUT, checksums ok"; exit 0; fi
T=$OUT/syfi_coding_trace.jsonl.gz
if [ ! -s "$T" ] || [ "$(sha < "$T")" != "$TRACELAB_SHA256" ]; then
  echo "downloading TraceLab v0.0.2 ($TRACELAB_URL)"
  curl -fsSL -o "$T.part" "$TRACELAB_URL" || { echo "error: download failed" >&2; exit 2; }
  mv "$T.part" "$T"
fi
[ "$(sha < "$T")" = "$TRACELAB_SHA256" ] || { echo "error: $T sha256 differs from $TRACELAB_SHA256" >&2; exit 2; }
python3 -I "$HERE/prepare_pool.py" "$T" "$OUT/pool" --sessions "$POOL_SESSIONS" --pools 2 \
  --seed "$POOL_SEED" --horizon-min "$HORIZON_MIN" --cap-s "$CAP_S" --sys-tokens "$SYS_TOKENS"
cp "$HERE/sys8k.txt" "$OUT/sys8k.txt"
fail=0
for i in 0 1; do
  want=$([ "$i" = 0 ] && echo "$POOL0_SHA" || echo "$POOL1_SHA")
  got=$(gzip -dc "$OUT/pool_$i.jsonl.gz" | sha)
  [ "$got" = "$want" ] && echo "pool_$i.jsonl.gz: sha256 ok" || { echo "error: pool_$i sha256 $got, expected $want" >&2; fail=1; }
done
[ "$(md5 "$OUT/sys8k.txt")" = "$SYS_MD5" ] && echo "sys8k.txt: md5 ok" || { echo "error: sys8k.txt md5 differs" >&2; fail=1; }
[ "$fail" = 0 ] || exit 2
echo "load staged in $OUT"
