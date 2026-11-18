#!/bin/bash
# Local tests of download_checkpoint.sh with a stub curl (no network): files land with the listed
# sizes, an interrupted transfer resumes (-C -), a complete checkpoint is skipped, a size mismatch
# fails without marking the checkpoint complete.
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"
# Stub curl: writes the remote file's bytes (size from STUB_SIZES "name=size ...") to -o, resuming with
# -C - from the .part size; the first call for a name in STUB_CUT stops halfway with exit 28.
cat > "$T/bin/curl" <<'EOF'
#!/bin/bash
out=""; resume=0; url=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out=$2; shift ;; -C) resume=1; shift ;; -H|--retry|--retry-delay|--speed-limit|--speed-time) shift ;; http*) url=$1 ;; esac
  shift
done
name=${url#*/resolve/}; name=${name#*/}   # drop the revision, keep subfolders
size=$(echo "$STUB_SIZES" | tr ' ' '\n' | awk -F= -v n="$name" '$1 == n { print $2 }')
have=0; [ "$resume" = 1 ] && [ -f "$out" ] && have=$(wc -c < "$out" | tr -d ' ')
echo "$name" >> "$STUB_LOG"
if [[ " $STUB_CUT " == *" $name "* ]] && [ ! -f "$STUB_DIR/cut-$name" ]; then
  touch "$STUB_DIR/cut-$name"; head -c $((size / 2 - have)) /dev/zero >> "$out"; exit 28
fi
head -c $((size - have)) /dev/zero >> "$out"
EOF
chmod +x "$T/bin/curl"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
export PATH="$T/bin:$PATH" STUB_LOG="$T/calls" STUB_DIR="$T"
export STUB_SIZES="config.json=100 model-00001-of-00002.safetensors=5000 sub/model-00002-of-00002.safetensors=3000"
export FILE_LIST=$'config.json 100\nmodel-00001-of-00002.safetensors 5000\nsub/model-00002-of-00002.safetensors 3000'
D="$T/ckpt"

STUB_CUT="model-00001-of-00002.safetensors" bash "$HERE/download_checkpoint.sh" org/repo abc123 "$D" > "$T/out" 2>&1; RC=$?
[ "$RC" = 0 ] && ok || { ko "download exit $RC"; cat "$T/out"; }
[ -f "$D/.complete" ] && ok || ko "no .complete marker"
[ "$(wc -c < "$D/model-00001-of-00002.safetensors" | tr -d ' ')" = 5000 ] && ok || ko "resumed file has the listed size"
[ "$(wc -c < "$D/sub/model-00002-of-00002.safetensors" | tr -d ' ')" = 3000 ] && ok || ko "file in a subfolder has the listed size"
[ "$(grep -c 'model-00001' "$T/calls")" = 2 ] && ok || ko "interrupted file fetched twice (resume), got $(grep -c 'model-00001' "$T/calls")"
ls "$D"/*.part "$D"/sub/*.part >/dev/null 2>&1 && ko "no .part file left" || ok

: > "$T/calls"
bash "$HERE/download_checkpoint.sh" org/repo abc123 "$D" > "$T/out" 2>&1; RC=$?
[ "$RC" = 0 ] && [ ! -s "$T/calls" ] && ok || ko "complete checkpoint skipped without any download (rc $RC)"

rm -rf "$D"
STUB_SIZES="config.json=100 model-00001-of-00002.safetensors=4000 sub/model-00002-of-00002.safetensors=3000" \
  bash "$HERE/download_checkpoint.sh" org/repo abc123 "$D" > "$T/out" 2>&1; RC=$?
[ "$RC" != 0 ] && [ ! -f "$D/.complete" ] && ok || ko "size mismatch fails without .complete (rc $RC)"

# INCLUDE (an extended regex on the file names): only the matching files, e.g. a tokenizer.
rm -rf "$D"; : > "$T/calls"
INCLUDE='^config\.json$' bash "$HERE/download_checkpoint.sh" org/repo abc123 "$D" > "$T/out" 2>&1; RC=$?
[ "$RC" = 0 ] && [ -f "$D/config.json" ] && [ ! -e "$D/model-00001-of-00002.safetensors" ] && [ "$(wc -l < "$T/calls" | tr -d ' ')" = 1 ] && ok || ko "INCLUDE fetches only the matching files (rc $RC)"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
