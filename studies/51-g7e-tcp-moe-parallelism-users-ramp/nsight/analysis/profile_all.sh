#!/bin/bash
# profile_all.sh <traces dir> <work dir>: runs profile_trace.py on every zip of <traces dir>, one
# at a time, and writes analysis/profiles/<layout>-<users>users-node-<a|b>.json. Each zip is
# unpacked in <work dir> (38 to 148 MB per trace) and removed again. A trace that already has its
# JSON is skipped. Needs uv: the Perfetto trace processor runs in a throwaway environment.
set -u
T=$1; D=$2; A=$(cd "$(dirname "$0")" && pwd); O=$A/profiles
mkdir -p "$O"
for z in "$T"/*-node-?.perfetto-trace.zip; do
  n=$(basename "$z" .perfetto-trace.zip)
  [ -s "$O/$n.json" ] && continue
  rm -rf "$D/$n"; mkdir -p "$D/$n"; unzip -o -q "$z" -d "$D/$n"
  uv run --no-project --with perfetto --with pandas --with numpy python -I "$A/profile_trace.py" \
    "$n=$(ls "$D/$n"/*.perfetto-trace)" > "$O/$n.json" 2> "$D/$n.err" || { echo "FAILED $n"; tail -3 "$D/$n.err"; rm -f "$O/$n.json"; }
  rm -rf "$D/$n"; echo "$(date +%T) $n"
done
