#!/bin/bash
# Download a Hugging Face checkpoint at a pinned revision into a plain directory, resumably.
# Usage: download_checkpoint.sh REPO REVISION DIR
# Used by the vLLM pod's init container and by 03-download-checkpoint.yaml (mounted from the
# ConfigMap download-checkpoint, which apply_config.sh refreshes).
#
# Why not `hf download` (2026-11-18): transfers stalled mid-file (every 19 GB file stopped at
# 11-16 GB, anonymous or authenticated), retries started over with new partial-file names, and a
# killed process left lock files the next attempt waited on forever. Here each file is fetched with
# curl into DIR/<file>.part with resume (-C -), retries on every error, and an abort when the rate
# stays under 1 MB/s for 60 s; the file is renamed only when its size matches the listing, and
# DIR/.complete is written only when every file is there.
# Env: HF_TOKEN (optional, read-only), PARALLEL (default 5), HF_ENDPOINT (default https://huggingface.co),
# FILE_LIST ("path size" lines, instead of asking the API: local tests), INCLUDE (an extended regex:
# only the matching file names, e.g. a tokenizer).
set -uo pipefail
REPO=$1 REV=$2 DIR=$3
EP=${HF_ENDPOINT:-https://huggingface.co}
[ -f "$DIR/.complete" ] && { echo "checkpoint $REPO@$REV already complete in $DIR"; exit 0; }
mkdir -p "$DIR"
LIST=${FILE_LIST:-$(python3 - "$EP" "$REPO" "$REV" <<'PY'
import json, os, sys, urllib.request
ep, repo, rev = sys.argv[1:4]
req = urllib.request.Request(f"{ep}/api/models/{repo}/revision/{rev}?blobs=true")
if os.environ.get("HF_TOKEN"):
    req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
for s in json.load(urllib.request.urlopen(req, timeout=60))["siblings"]:
    print(s["rfilename"], s.get("size") or 0)
PY
)}
[ -z "${INCLUDE:-}" ] || LIST=$(echo "$LIST" | awk -v re="$INCLUDE" '$1 ~ re')
[ -n "$LIST" ] || { echo "error: empty file list for $REPO@$REV" >&2; exit 2; }

fetch_one() { # NAME SIZE
  local name=$1 size=$2 dst="$DIR/$1" have attempt
  if [ -f "$dst" ] && [ "$(wc -c < "$dst" | tr -d ' ')" = "$size" ]; then return 0; fi
  mkdir -p "$(dirname "$dst")"
  for attempt in $(seq 1 30); do
    curl -fsSL -C - --retry 5 --retry-delay 5 --speed-limit 1048576 --speed-time 60 \
      ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o "$dst.part" "$EP/$REPO/resolve/$REV/$name"
    have=$(wc -c < "$dst.part" 2>/dev/null | tr -d ' ')
    [ "${have:-0}" = "$size" ] && { mv "$dst.part" "$dst"; echo "done $name ($size bytes)"; return 0; }
    [ "${have:-0}" -gt "$size" ] && { echo "error: $name is ${have} bytes, the listing says $size" >&2; return 1; }
    echo "$name: ${have:-0} of $size bytes after attempt $attempt, resuming"
  done
  echo "error: $name incomplete after 30 attempts" >&2; return 1
}
export -f fetch_one; export DIR EP REPO REV
echo "$LIST" | xargs -P "${PARALLEL:-5}" -n 2 bash -c 'fetch_one "$0" "$1"' || { echo "error: some files did not download" >&2; exit 1; }
# Every listed file present with its size, then the marker.
echo "$LIST" | while read -r name size; do
  [ "$(wc -c < "$DIR/$name" 2>/dev/null | tr -d ' ')" = "$size" ] || { echo "error: $name missing or wrong size" >&2; exit 1; }
done || exit 1
touch "$DIR/.complete"
echo "checkpoint $REPO@$REV complete in $DIR ($(du -sh "$DIR" | cut -f1))"
