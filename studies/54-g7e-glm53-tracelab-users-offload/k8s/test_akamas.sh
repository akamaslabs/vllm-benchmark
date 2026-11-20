#!/bin/bash
# Local checks of study 54's Akamas YAML against the scripts and workload.env (no Akamas, no cluster).
# Usage: bash k8s/test_akamas.sh   → "N passed, M failed", exit 1 on any failure.
HERE=$(cd "$(dirname "$0")" && pwd)
A="$HERE/../akamas"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
# shellcheck disable=SC1091
source "$HERE/workload.env"
O=$(python3 - "$A" "$HERE" "$SLO_TTFT_MS" "$SLO_ITL_MS" "$RAMP_S" "$DEADLINE_EXTRA_S" <<'PY'
import glob, os, re, sys, yaml
a, k8s, slo_ttft, slo_itl, ramp, extra = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
def load(p): return yaml.safe_load(open(p))
studies = [load(p) for p in glob.glob(os.path.join(a, "*.yaml")) if load(p).get("kind") == "study"]
flows = [load(p) for p in glob.glob(os.path.join(a, "*.yaml")) if load(p).get("kind") == "workflow"]
s, w = studies[0], flows[0]
system = load(os.path.join(a, "system.yaml"))
comps = {d["name"]: d for d in map(load, glob.glob(os.path.join(a, "components", "*.yaml")))}
tels = [load(p) for p in glob.glob(os.path.join(a, "telemetry", "*.yaml"))]
print("one_study_one_workflow", len(studies) == 1 and len(flows) == 1)
print("names", s["name"] == "54-G7E-GLM53-TraceLab-Users-Offload", w["name"] == "54-G7E-GLM53-TraceLab-Workflow", system["name"] == "vLLM_Benchmark_54_G7E_GLM53_TraceLab")
print("refs", s["system"] == system["name"] and s["workflow"] == w["name"] and all(d["system"] == system["name"] for d in list(comps.values()) + tels))
print("goal", s["goal"]["objective"], s["goal"]["function"]["formula"])
cons = {c["formula"] for c in s["goal"]["constraints"]["absolute"]}
print("slo", cons == {"aiperf.aiperf_time_to_first_token_p95_150s <= %g" % slo_ttft, "aiperf.aiperf_inter_token_latency_p95_150s <= %g" % slo_itl})
wd = s["windowing"]
print("windowing", wd["type"], wd["stability"]["width"], wd["stability"]["metric"], wd["stability"]["when"]["metric"], wd["stability"]["when"]["is"])
print("bad_step_names", [x["name"] for x in s["steps"] if not re.match(r"^[a-zA-Z\s][a-zA-Z0-9_\s]*$", x["name"])])
print("steps", [x["name"] for x in s["steps"]])
print("run_on_failure", [x.get("runOnFailure", False) for x in s["steps"] if x["type"] == "preset"])
print("offloads", [x.get("values", {}).get("vllm.kv_offloading_size") for x in s["steps"]])
print("backends", [x.get("values", {}).get("vllm.kv_offloading_backend") for x in s["steps"]])
opt = [x for x in s["steps"] if x["type"] == "optimize"][0]
print("optimize", opt["numberOfExperiments"], opt.get("optimizer"), "vllm.kv_offloading_backend" in opt.get("renderParameters", []))
sel = {p["name"]: p.get("domain") for p in s["parametersSelection"]}
print("domains", sel == {"vllm.kv_offloading_size": [0, 512], "vllm.gpu_memory_utilization": [0.85, 0.97],
      "vllm.max_num_batched_tokens": [2048, 32768], "vllm.max_num_seqs": [16, 1024], "vllm.spec_tokens": [0, 5]})
print("pconstraint", [c["formula"] for c in s.get("parameterConstraints", [])])
tokens = set(re.findall(r"\$\{vllm\.([a-z_]+)\}", open(os.path.join(k8s, "params.env.template")).read()))
rendered = {n.split(".", 1)[1] for n in sel}
for x in s["steps"]:
    rendered |= {n.split(".", 1)[1] for n in x.get("values", {})} | {n.split(".", 1)[1] for n in x.get("renderParameters", [])}
print("unrendered_tokens", sorted(t for t in tokens if t not in rendered))
dnr = [x for x in s["steps"] if x["type"] == "baseline"][0].get("doNotRenderParameters", [])
print("baseline_dnr", sorted(dnr))
print("kpis", len(s["kpis"]))
declared = {(t["name"], m["metric"]) for t in tels for m in t["metrics"]}
names = {m for _, m in declared}
used = set(re.findall(r"(?:vllm|aiperf)\.([a-z0-9_]+)", " ".join(k["formula"] for k in s["kpis"]) + " " + s["goal"]["function"]["formula"] + " " + " ".join(cons)))
print("undeclared_metrics", sorted(used - names))
cmds = {t["name"]: t for t in w["tasks"]}
print("tasks", list(cmds))
missing = []
for t in w["tasks"]:
    c = t["arguments"].get("command", "")
    for path in re.findall(r"/work/study54/54-g7e-glm53-tracelab-users-offload/(\S+)", c):
        if not os.path.exists(os.path.join(k8s, "..", path)): missing.append(path)
    for key in ("source", "target"):
        p = t["arguments"].get(key, {}).get("path", "") if isinstance(t["arguments"].get(key), dict) else ""
        m = re.match(r"/work/study54/54-g7e-glm53-tracelab-users-offload/(\S+)", p)
        if m and key == "source" and not os.path.exists(os.path.join(k8s, "..", m.group(1))): missing.append(m.group(1))
print("missing_paths", missing)
def minutes(v): return int(v[:-1]) * (60 if v.endswith("h") else 1)
print("runtest_timeout_ok", minutes(cmds["RunTest"]["arguments"]["timeout"]) * 60 > ramp + extra)
print("stop_always", cmds["Stop AIPerf"].get("alwaysRun") is True and cmds["Stop AIPerf"].get("critical") is False)
print("csv_task_critical", cmds["AIPerf to CSV"].get("critical", True) is True)
csvt = [t for t in tels if t["provider"] == "CSV File"][0]
print("csv_pattern", csvt["config"]["remoteFilePattern"])
print("component_types", sorted({d["componentType"] for d in comps.values()}))
PY
)
want() { echo "$O" | grep -qxF -- "$1" && ok || ko "expected '$1'; got: $(echo "$O" | grep -F -- "${1%% *}" | head -1)"; }
want "one_study_one_workflow True"
want "names True True True"
want "refs True"
want "goal maximize aiperf.aiperf_active_sessions"
want "slo True"
want "windowing stability 6 aiperf.aiperf_active_sessions aiperf.aiperf_active_sessions max"
want "bad_step_names []"
want "steps ['baseline', 'Native 128', 'Native 256', 'Native 512', 'LMCache 512', 'optimize']"
want "run_on_failure [True, True, True, True]"
want "offloads [None, 128, 256, 512, 512, None]"
want "backends [None, 'native', 'native', 'native', 'lmcache', None]"
want "optimize 30 AKAMAS True"
want "domains True"
want "pconstraint ['vllm.max_num_batched_tokens >= vllm.max_num_seqs']"
want "unrendered_tokens []"
want "baseline_dnr ['vllm.kv_offloading_size', 'vllm.max_num_batched_tokens', 'vllm.max_num_seqs']"
want "kpis 8"
want "undeclared_metrics []"
want "tasks ['Write config', 'Apply config', 'RunTest', 'Stop AIPerf', 'AIPerf to CSV']"
want "missing_paths []"
want "runtest_timeout_ok True"
want "stop_always True"
want "csv_task_critical True"
want "csv_pattern $CSV_DIR/*.csv"
want "component_types ['AIPerf', 'GPU', 'Kubernetes Cluster', 'Kubernetes Container', 'Kubernetes Node', 'Kubernetes Workload', 'vLLM']"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
