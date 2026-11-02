#!/bin/bash
# Start study 45 once study 44 tappa C has stopped and released the shared node groups
# (study 44's results/stop-C.sh writes "STOP-C DONE"), with Akamas 4.1's start workaround.
TB="kubectl -n akamas-41 exec deploy/toolbox -c toolbox --"
LOG=$(cd "$(dirname "$0")" && pwd)/../../44-l4-pd-multinode/results/stop-C.log
cd "$(dirname "$0")"
logged_in() { local o; o=$($TB akamas whoami 2>&1); echo "$o" | grep -q "Logged User" && ! echo "$o" | grep -qiE "not currently logged|unauthorized"; }
until grep -q "STOP-C DONE" "$LOG" 2>/dev/null; do sleep 60; done
until logged_in; do echo "$(date -u +%T) LOGIN NEEDED: akamas CLI logged out, study 45 not started yet"; sleep 600; done
echo "$(date -u +%T) study 44 C stopped; starting study 45"
$TB akamas start study 45-GPU-Choice-TPS-per-Dollar-v2 2>&1 | tail -2
sleep 120
n=$($TB akamas list experiment 45-GPU-Choice-TPS-per-Dollar-v2 2>/dev/null | grep -cE "^[0-9]+ ")
if [ "${n:-0}" -eq 0 ]; then
  echo "$(date -u +%T) no experiment after start: finish + start (late Airflow DAG)"
  $TB akamas finish study 45-GPU-Choice-TPS-per-Dollar-v2 --workspace default 2>&1 | tail -1
  sleep 30
  $TB sh -c 'echo y | akamas start study 45-GPU-Choice-TPS-per-Dollar-v2' 2>&1 | tail -2
  sleep 120
fi
$TB akamas describe study 45-GPU-Choice-TPS-per-Dollar-v2 2>&1 | grep -iE "^status|status message"
$TB akamas list experiment 45-GPU-Choice-TPS-per-Dollar-v2 2>&1 | grep -E "^[0-9]+ " | cut -c1-140
echo "$(date -u +%T) START-45 DONE"
