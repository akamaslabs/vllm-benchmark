# akamas/: 34-l40s-gemma4-26b-awq-customer-replay

**Created:** 2026-10-13 as a copy of study 35's resources (`../../35-l40s-gemma4-26b-awq-replay-users/akamas/`,
themselves study 34's first design, accepted shapes on Akamas 4.1) with new names and the
vllm-bench cluster (namespace `llm-l40s`, node group `llm-serving-l40s-1xl`, the in-cluster
Prometheus, the toolbox's `default` kubectl context). The study's one difference from study 35 is in
the load (`../k8s/caps.py`), not in these files. **Not created on the server yet.**

**Last modified:** 2026-10-13: replaced study 34's first design (thinking on, Shakespeare texts, vLLM
completed req/s with e2e p95 <= 30 s, 13 parameters) with study 35's resources; Akamas names changed
(`..._Replay` -> `..._Replay_Users`): none of the old ones was ever created.

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_34_L40S_Gemma4_AWQ_Replay_Users` |
| `components/*.yaml` | 7 components | `vllm` (vLLM), `aiperf` (AIPerf), `gpu0` (GPU), `container`, `container_loadtest` (Kubernetes Container), `cluster`, `cluster_loadtest` (Kubernetes Cluster) |
| `telemetry/prometheus.yaml` | telemetry instance, Prometheus | `Prometheus_34_L40S_Gemma4_AWQ_Replay_Users`: the vllm-bench Prometheus in-cluster (`kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090`); vLLM, GPU, Kubernetes |
| `telemetry/aiperf-csv.yaml` | telemetry instance, CSV File | `AIPerf_CSV_34_L40S_Gemma4_AWQ_Replay_Users`: `/work/aiperf-csv/34-l40s-gemma4-26b-awq-customer-replay/*.csv` on the toolbox over SCP; the 32 AIPerf pack metrics, column = metric |
| `34-L40S-Gemma4-AWQ-Replay-Users-Workflow.yaml` | workflow | Write config, Apply config, RunTest, Stop AIPerf (alwaysRun, not critical), AIPerf to CSV |
| `34-L40S-Gemma4-AWQ-Replay-Users.yaml` | study | goal `aiperf.aiperf_active_sessions` max, constraint `aiperf.aiperf_e2e_request_latency_p50_150s <= 3000`, 17 parameters, 10 steps |

Every task runs on the toolbox host over SSH (host `toolbox`, user `akamas`, key
`/home/akamas/.ssh/id_rsa`); the scripts reach the vllm-bench cluster through the toolbox's
`default` kubectl context (in-cluster).

## Validation

`python3 akamas/check_offline.py --packs <dir>` with the packs' `develop` branches (vLLM 1.13.0,
AIPerf 1.0.0, GPU 1.4.0, Kubernetes 1.9.0-dev; the default `~/workspace/akamas/optimization-packs`
does not exist on every workstation): 0 failures on 2026-10-13 (names, component types, every metric
the study reads against its component type and a telemetry instance, every parameter domain inside
the pack's, every baseline / preset rendered through the real `render_statefulset.sh`). Installed on
the 4.1 server on 2026-10-13: vLLM 1.13.0, AIPerf 1.0.0. **Server:** not created yet; `akamas
create` is the real validation (repo rule).

## Setup & run

From the 4.1 toolbox (`toolbox-ssh vllm-bench akamas-41`), in `/work/vllm-benchmark`, with the
`llm-serving-l40s-1xl` node up and the template on the toolbox (`../load/README.md`). Study 33 (or
any other study on this node) finished and its AIPerf Job gone.

```bash
git pull --ff-only
A=studies/34-l40s-gemma4-26b-awq-customer-replay/akamas
akamas list optimization-pack | grep -iE 'vllm|aiperf'      # vLLM 1.13.0, AIPerf 1.0.0

# Once, before the study (~1.5-3 h, the GPU busy): the per-request output caps
# (/work/customer-data/34-l40s-gemma4-26b-awq-customer-replay/caps.json.gz; ../README.md "Load").
bash studies/34-l40s-gemma4-26b-awq-customer-replay/k8s/calibrate_caps.sh

mkdir -p /work/aiperf-csv/34-l40s-gemma4-26b-awq-customer-replay   # the CSV pattern's folder
akamas create -f $A/system.yaml
for c in vllm aiperf gpu0 container container_loadtest cluster cluster_loadtest; do
  akamas create -f $A/components/$c.yaml
done
akamas create -f $A/telemetry/prometheus.yaml
akamas create -f $A/telemetry/aiperf-csv.yaml
akamas create -f $A/34-L40S-Gemma4-AWQ-Replay-Users-Workflow.yaml
akamas create -f $A/34-L40S-Gemma4-AWQ-Replay-Users.yaml
sleep 120   # studies 24/27/28: a start right after create can leave the study RUNNING with no experiment
akamas start study "34-L40S-Gemma4-AWQ-Replay-Users"
```

Typed form, same order: `akamas create system $A/system.yaml`; `akamas create component
$A/components/<name>.yaml vLLM_Benchmark_34_L40S_Gemma4_AWQ_Replay_Users` once per component file;
`akamas create telemetry-instance $A/telemetry/prometheus.yaml
vLLM_Benchmark_34_L40S_Gemma4_AWQ_Replay_Users` and the same for `aiperf-csv.yaml`; `akamas create
workflow $A/34-L40S-Gemma4-AWQ-Replay-Users-Workflow.yaml`; `akamas create study
$A/34-L40S-Gemma4-AWQ-Replay-Users.yaml`.

**First trial checks:** Apply config's log (the compose's flags only, `Raising
max_num_batched_tokens ... to 2496`, Marlin); RunTest's log (corpus reused from the calibration,
`rendered 240 users`, **the `CAPS:` line**: about 48k requests at their calibrated length, the rest
at their task median; a `not the calibrated one` error means the render changed: recalibrate);
AIPerf starts, the watchdog lines; Stop AIPerf saved `<epoch>.jsonl`; AIPerf to CSV wrote
`<epoch>.csv` with rows covering the run; the `aiperf` component has samples **inside** the trial's
window (if they are shifted by hours, the CSV provider reads `TS` in another time zone: add the zone
to the CSV and to `timestampFormat`); the score is a number of users.

**Changes after creation:** Akamas 4.1 has no `update workflow`: delete and recreate the workflow
(and the study, if created). Only the goal can be updated on a study with experiments.

**Room on the toolbox:** `/work` is a 1 GiB volume; each trial adds a records file (~25-60 MB at
200 users) under `/work/aiperf-exports/` and a small CSV. Move the records off the toolbox every
few experiments (`kubectl -n akamas-41 exec deploy/toolbox -c toolbox -- tar -C
/work/aiperf-exports/34-l40s-gemma4-26b-awq-customer-replay -cf - . | tar -C <local dir> -xf -`, then
delete them there); keep the CSVs (the provider reads them all).
