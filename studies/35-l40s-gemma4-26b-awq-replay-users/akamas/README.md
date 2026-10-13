# akamas/: 35-l40s-gemma4-26b-awq-replay-users

**Created:** 2026-10-13 as a copy of study 34's resources (Quartuccio's, accepted shapes on Akamas
4.1), modified by hand: new names, the `aiperf` component, the CSV File telemetry instance, a fifth
workflow task, the goal / constraint / windowing on client-side metrics, four more parameters.
**Not created on the server yet** (needs the AIPerf pack 1.0.0 installed first).

## Resources

| File | Kind | Name |
|---|---|---|
| `system.yaml` | system | `vLLM_Benchmark_35_L40S_Gemma4_AWQ_Replay_Users` |
| `components/*.yaml` | 7 components | `vllm` (vLLM), `aiperf` (AIPerf), `gpu0` (GPU), `container`, `container_loadtest` (Kubernetes Container), `cluster`, `cluster_loadtest` (Kubernetes Cluster) |
| `telemetry/prometheus.yaml` | telemetry instance, Prometheus | `Prometheus_35_L40S_Gemma4_AWQ_Replay_Users`: the vllm-bench-gatling Prometheus through its NLB; vLLM, GPU, Kubernetes |
| `telemetry/aiperf-csv.yaml` | telemetry instance, CSV File | `AIPerf_CSV_35_L40S_Gemma4_AWQ_Replay_Users`: `/work/aiperf-csv/35-l40s-gemma4-26b-awq-replay-users/*.csv` on the toolbox over SCP; the 32 AIPerf pack metrics, column = metric |
| `35-L40S-Gemma4-AWQ-Replay-Users-Workflow.yaml` | workflow | Write config, Apply config, RunTest, Stop AIPerf (alwaysRun, not critical), AIPerf to CSV |
| `35-L40S-Gemma4-AWQ-Replay-Users.yaml` | study | goal `aiperf.aiperf_active_sessions` max, constraint `aiperf.aiperf_e2e_request_latency_p50_150s <= 3000`, 17 parameters, 10 steps |

Every task runs on the toolbox host over SSH (host `toolbox`, user `akamas`, key
`/home/akamas/.ssh/id_rsa`); the scripts reach the vllm-bench-gatling cluster through the
toolbox's `gatling` kubectl context.

## Validation

`python3 akamas/check_offline.py` (packs from `~/workspace/akamas/optimization-packs`, vLLM 1.13.0
and AIPerf 1.0.0): names, component types, every metric the study reads against its component
type and a telemetry instance, every parameter domain inside the pack's, every baseline / preset
rendered through the real `render_statefulset.sh`. **Server:** not created yet; `akamas create`
is the real validation (repo rule).

## Setup & run

From the 4.1 toolbox (`toolbox-ssh vllm-bench akamas-41`), in `/work/vllm-benchmark`, after the
AIPerf pack is installed, the cluster layer is provisioned (`../infra/README.md`) and the template
is on the toolbox (`../load/README.md`):

```bash
git pull --ff-only
A=studies/35-l40s-gemma4-26b-awq-replay-users/akamas
akamas list optimization-pack | grep -iE 'vllm|aiperf'      # vLLM 1.13.0, AIPerf 1.0.0
mkdir -p /work/aiperf-csv/35-l40s-gemma4-26b-awq-replay-users   # the CSV pattern's folder
akamas create -f $A/system.yaml
for c in vllm aiperf gpu0 container container_loadtest cluster cluster_loadtest; do
  akamas create -f $A/components/$c.yaml
done
akamas create -f $A/telemetry/prometheus.yaml
akamas create -f $A/telemetry/aiperf-csv.yaml
akamas create -f $A/35-L40S-Gemma4-AWQ-Replay-Users-Workflow.yaml
akamas create -f $A/35-L40S-Gemma4-AWQ-Replay-Users.yaml
sleep 120   # studies 24/27/28: a start right after create can leave the study RUNNING with no experiment
akamas start study "35-L40S-Gemma4-AWQ-Replay-Users"
```

**First trial checks:** Apply config's log (the compose's flags only, `Raising
max_num_batched_tokens ... to 2496`, Marlin); RunTest's log (corpus built or reused, `rendered 240
users`, AIPerf starts, the watchdog lines); Stop AIPerf saved `<epoch>.jsonl.gz`; AIPerf to CSV wrote
`<epoch>.csv` with rows covering the run; the `aiperf` component has samples **inside** the
trial's window (if they are shifted by hours, the CSV provider reads `TS` in another time zone:
add the zone to the CSV and to `timestampFormat`); the score is a number of users.

**Changes after creation:** Akamas 4.1 has no `update workflow`: delete and recreate the workflow
(and the study, if created). Only the goal can be updated on a study with experiments.

**Room on the toolbox:** `/work` is a 1 GiB volume; each trial adds a records file (~25-60 MB at
200 users) under `/work/aiperf-exports/` and a small CSV. Move the records off the toolbox every
few experiments (`kubectl -n akamas-41 exec deploy/toolbox -c toolbox -- tar -C
/work/aiperf-exports/35-l40s-gemma4-26b-awq-replay-users -cf - . | tar -C <local dir> -xf -`, then
delete them there); keep the CSVs (the provider reads them all).
