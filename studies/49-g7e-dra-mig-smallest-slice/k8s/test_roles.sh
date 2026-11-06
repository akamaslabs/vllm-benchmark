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

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
