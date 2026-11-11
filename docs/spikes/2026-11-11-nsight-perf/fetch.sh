#!/bin/bash
# [POD=aiperf] fetch.sh <remote path in the pod> <local path>   (pod: nsight unless POD is set)
# Downloads a file from the pod in 8 MiB chunks, each one retried, and checks the md5 at
# the end. kubectl cp dropped the stream on files of a few tens of MB.
set -euo pipefail
R=$1; L=$2; NS=nsight-probe; POD=${POD:-nsight}; CH=8
size=$(kubectl -n $NS exec $POD -- stat -c %s "$R"); sum=$(kubectl -n $NS exec $POD -- md5sum "$R" | cut -d' ' -f1)
n=$(( (size + CH * 1048576 - 1) / (CH * 1048576) )); : > "$L.part"
for i in $(seq 0 $((n - 1))); do
  want=$(( size - i * CH * 1048576 )); [ "$want" -gt $((CH * 1048576)) ] && want=$((CH * 1048576))
  for try in 1 2 3 4 5 6; do
    kubectl -n $NS exec $POD -- dd if="$R" bs=1M skip=$((i * CH)) count=$CH status=none > "$L.chunk" 2>/dev/null || true
    [ "$(wc -c < "$L.chunk" | tr -d ' ')" = "$want" ] && break
    [ "$try" = 6 ] && { echo "chunk $i of $R failed"; exit 1; }
  done
  cat "$L.chunk" >> "$L.part"
done
rm -f "$L.chunk"
got=$(md5 -q "$L.part" 2>/dev/null || md5sum "$L.part" | cut -d' ' -f1)
[ "$got" = "$sum" ] || { echo "md5 mismatch for $R"; exit 1; }
mv "$L.part" "$L"; echo "ok $L $((size / 1048576)) MiB"
