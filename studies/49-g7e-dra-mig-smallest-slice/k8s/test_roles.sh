#!/bin/bash
# Local tests of lib_roles.sh (slice-roles exporter content and GPU instance id parsing).
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$HERE/lib_roles.sh"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }

OUT=$(roles_metrics 1g.24gb 3 5 6 7); RC=$?
[ $RC = 0 ] && echo "$OUT" | grep -qx 'study49_slice_role{GPU_I_ID="3",role="customer"} 1' \
  && [ "$(echo "$OUT" | grep -c 'role="neighbour"')" = 3 ] && echo "$OUT" | grep -qx 'study49_slice_role{role="gpu"} 1' \
  && ok || ko "roles: 1g layout"

OUT=$(roles_metrics 2g.48gb 1 2); RC=$?
[ $RC = 0 ] && echo "$OUT" | grep -qx 'study49_slice_role{GPU_I_ID="1",role="customer"} 1' \
  && echo "$OUT" | grep -qx 'study49_slice_role{GPU_I_ID="2",role="neighbour"} 1' && ok || ko "roles: 2g layout"

OUT=$(roles_metrics none ""); RC=$?
[ $RC = 0 ] && echo "$OUT" | grep -qx 'study49_slice_role{role="customer"} 1' && ! echo "$OUT" | grep -q GPU_I_ID \
  && ok || ko "roles: none layout"

roles_metrics 1g.24gb 3 "" 6 7 >/dev/null 2>&1 && ko "roles: missing id -> fail" || ok
roles_metrics 1g.24gb "" 5 6 7 >/dev/null 2>&1 && ko "roles: missing customer id -> fail" || ok
roles_metrics 2g.48gb 1 2 3 >/dev/null 2>&1 && ko "roles: wrong neighbour count -> fail" || ok
roles_metrics 1g.24gb 3 3 6 7 >/dev/null 2>&1 && ko "roles: duplicate id -> fail" || ok

# nvidia-smi -q excerpts (MIG device and whole GPU).
MIG='    MIG Device
        Index                             : 0
        GPU Instance ID                   : 5
        Compute Instance ID               : 0'
[ "$(echo "$MIG" | parse_gpu_instance_id)" = 5 ] && ok || ko "parse: MIG id"
WHOLE='    MIG Mode
        Current                           : Disabled
    GPU Instance ID                       : N/A'
[ -z "$(echo "$WHOLE" | parse_gpu_instance_id)" ] && ok || ko "parse: whole GPU -> empty"

# A writer that keeps writing after the id (kubectl exec streaming nvidia-smi -q in frames) must
# not get SIGPIPE: under set -o pipefail that would abort apply_config.sh with rc 141 (review).
RC=$(bash -c 'set -o pipefail; source "$1"; { echo "        GPU Instance ID : 5"; sleep 0.3; seq 1 200000; } | parse_gpu_instance_id >/dev/null; echo $?' _ "$HERE/lib_roles.sh")
[ "$RC" = 0 ] && ok || ko "parse: no SIGPIPE on a late writer (rc $RC)"

# gpu_instance_id must not fail when kubectl exec fails: empty id, then roles_metrics -> exit 7.
OUT=$(bash -c 'set -euo pipefail; kubectl() { return 1; }; source "$1"; X=$(gpu_instance_id ns pod); echo "rc=0 id=[$X]"' _ "$HERE/lib_roles.sh" 2>&1)
[ "$OUT" = "rc=0 id=[]" ] && ok || ko "gpu_instance_id tolerates exec failure ($OUT)"

# Prometheus answer -> scalar (the pre-load join check in apply_config.sh).
J='{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1762430000.1,"24064"]}]}}'
[ "$(echo "$J" | parse_prom_value)" = 24064 ] && ok || ko "prom: value"
[ -z "$(echo '{"status":"success","data":{"resultType":"vector","result":[]}}' | parse_prom_value)" ] && ok || ko "prom: empty result -> empty"
[ -z "$(echo 'garbage' | parse_prom_value)" ] && ok || ko "prom: garbage -> empty"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
