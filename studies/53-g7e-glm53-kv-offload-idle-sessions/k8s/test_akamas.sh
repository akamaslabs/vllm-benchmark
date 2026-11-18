#!/bin/bash
# Local checks of study 53's Akamas YAML against the scripts (no Akamas, no cluster), and the
# histogram helper copied from study 52.
HERE=$(cd "$(dirname "$0")" && pwd)
A="$HERE/../akamas"
Y="$A/53-G7E-GLM53-Offload-Idle-Sessions.yaml" W="$A/53-G7E-GLM53-Offload-Workflow.yaml" P="$A/telemetry/prometheus.yaml"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
O=$(python3 - "$Y" "$W" "$P" <<'PY'
import re, sys, yaml
s, w, p = (yaml.safe_load(open(f)) for f in sys.argv[1:4])
print("objective", s["goal"]["objective"], s["goal"]["function"]["formula"])
print("trim", s["windowing"]["trim"], s["windowing"]["task"], "task_exists", s["windowing"]["task"] in [t["name"] for t in w["tasks"]])
print("duration", p["config"]["duration"])
print("constraints", sorted(c["formula"] for c in s["goal"]["constraints"]["absolute"]))
print("types", sorted({x["type"] for x in s["steps"]}), "steps", len(s["steps"]))
print("offloads", [x["values"].get("vllm.kv_offloading_size") for x in s["steps"]])
print("mnbt", [x["values"].get("vllm.max_num_batched_tokens") for x in s["steps"]])
print("bad_names", [x["name"] for x in s["steps"] if not re.match(r"^[a-zA-Z\s][a-zA-Z0-9_\s]*$", x["name"])])
print("kpis", len(s["kpis"]))
print("run_on_failure", [x.get("runOnFailure", False) for x in s["steps"]])
print("system", s["system"] == p["system"], "workflow", s["workflow"] == w["name"])
print("commands", " ".join(t["arguments"].get("command", "") for t in w["tasks"]))
PY
)
echo "$O" | grep -qx 'objective minimize vllm.time_to_first_token_avg' && ok || ko "goal: minimize the server TTFT average"
echo "$O" | grep -qx "trim \['830s', '60s'\] Load task_exists True" && ok || ko "trim [830s, 60s] on the Load task, which exists"
echo "$O" | grep -qx 'duration 120' && ok || ko "telemetry duration 120 s (one turn every ~15 s: 30 s windows are often empty)"
echo "$O" | grep -qx "constraints \['vllm.request_success_rate >= 0.06', 'vllm.time_per_output_token_p95 <= 80'\]" && ok || ko "constraints: keeps up (>= 0.06 turns/s) and the TPOT safety net"
echo "$O" | grep -qx "types \['baseline', 'preset'\] steps 10" && ok || ko "10 steps, presets only (no optimize)"
echo "$O" | grep -qx 'offloads \[None, 32, 64, 128, 256, 512, 512, 512, None, None\]' && ok || ko "offload sweep 0-512 then the prefill presets"
echo "$O" | grep -qx 'mnbt \[None, None, None, None, None, None, 16384, 32768, 32768, None\]' && ok || ko "prefill presets 16k / 32k"
echo "$O" | grep -qx 'bad_names \[\]' && ok || ko "step names match ^[a-zA-Z\\s][a-zA-Z0-9_\\s]*$"
echo "$O" | grep -qx 'kpis 8' && ok || ko "8 KPIs (Akamas' limit)"
# runOnFailure false (the default) skips a step after a failed one: one bad start or a reclaimed spot node
# would stop the whole sweep overnight.
echo "$O" | grep -qx 'run_on_failure \[False, True, True, True, True, True, True, True, True, True\]' && ok || ko "every preset runs even after a failed step"
echo "$O" | grep -qx 'system True workflow True' && ok || ko "study, telemetry and workflow names agree"
for c in apply_config.sh run_load.sh; do
  echo "$O" | grep -q "bash /work/study53/53-g7e-glm53-kv-offload-idle-sessions/k8s/$c" && [ -f "$HERE/$c" ] && ok || ko "workflow runs k8s/$c under /work/study53"
done
# Resource names and paths only (comments may cite study 52 on purpose).
N=$(cat "$A"/*.yaml "$A"/components/*.yaml "$P" | grep -v '^[[:space:]]*#' | grep -c 'study52\|52-G7E\|_52_\|52-g7e')
[ "$N" = 0 ] && ok || ko "no study 52 names or paths in the Akamas resources ($N lines)"
# Smoke helper (from study 52): interpolated histogram quantile between two /metrics snapshots.
T=$(mktemp -d)
printf '%s\n' 'vllm:time_per_output_token_seconds_bucket{le="0.025"} 0' 'vllm:time_per_output_token_seconds_bucket{le="0.05"} 0' \
  'vllm:time_per_output_token_seconds_bucket{le="0.075"} 0' 'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 0' > "$T/a"
printf '%s\n' 'vllm:time_per_output_token_seconds_bucket{le="0.025"} 50' 'vllm:time_per_output_token_seconds_bucket{le="0.05"} 90' \
  'vllm:time_per_output_token_seconds_bucket{le="0.075"} 100' 'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 100' > "$T/b"
H=$(python3 "$HERE/hist_quantile.py" vllm:time_per_output_token_seconds "$T/a" "$T/b" 0.95 2>&1)
[ "$H" = "62.5 ms" ] && ok || ko "hist_quantile interpolates inside the 50-75 ms bucket ($H)"
rm -rf "$T"
[ "$FAIL" = 0 ] || echo "$O"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
