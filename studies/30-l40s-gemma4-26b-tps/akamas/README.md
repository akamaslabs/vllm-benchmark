# akamas/ — 30-L40S-Gemma4-TPS

**Created:** 2026-10-05, with the `akamas-study-manager` plugin (0.3.0) conventions, from
study 28's resources (same system shape, telemetry catalog, workflow pipeline), without the
MIG layout, the neighbour replica and the cost goal. **Not yet on the Akamas server**: the
files are local until the user confirms the sync (push, `git pull` on the toolbox).

## What it optimizes

**How many tokens per second can one L40S serve with Gemma 4 26B-A4B within a chat SLA?**
The study maximizes vLLM's total token throughput (prompt + generation tokens/s) of
`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` on one NVIDIA L40S, over 11 base vLLM settings
(studies 0-1's set, single GPU, no parallelism), subject to TTFT p95 <= 1500 ms and ITL
p95 <= 300 ms (150 s p95, `:max` over the scored window). The load is study 27's open-loop
AIPerf rate ramp on ShareGPT; a watchdog ends each trial past the SLA, and the score is the
best valid 3-minute window. Design: `../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | 3.7.x |
| Optimization pack **vLLM** | 1.12.0 (component type `vLLM`; `total_token_throughput`, `performance_mode`, `optimization_level`, ordinal `block_size`) |
| Optimization pack **GPU** | 1.4.0 checkout (component type `GPU`; only metrics used) |
| Kubernetes pack | 1.9.0-dev (installed on the server, checked 2026-10-02 for study 28; `Kubernetes Container`, `Kubernetes Cluster`, only metrics used) |
| Serving | `vllm/vllm-openai:v0.29.0`, `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` served as `gemma4-26b-l40s` |
| Load generator | AIPerf 0.11.0, ShareGPT replay, open-loop linear rate ramp (gamma arrivals, smoothness 4, seed 30) |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter on the L40S node |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_30_L40S_Gemma4_TPS` |
| `components/vllm.yaml` | `vllm` (vLLM, pod `vllm-0`, model `gemma4-26b-l40s`): every tuned parameter, goal, constraints, windowing |
| `components/gpu0.yaml` | `gpu0` (GPU, `gpu="0"`, model `.*L40S.*`) |
| `components/container.yaml` | `container` (Kubernetes Container, pod `vllm-0`) |
| `components/cluster.yaml`, `cluster_loadtest.yaml`, `container_loadtest.yaml` | node and load-generator views (Kubernetes pack) |
| `telemetry/prometheus.yaml` | `Prometheus_30_L40S_Gemma4_TPS`: study 28's 117 metrics, `active_gpus` on this namespace/node role, TTFT/ITL p95 over `[150s]` |
| `30-L40S-Gemma4-TPS-Workflow.yaml` | `Write config` -> `Apply config` (45 m) -> `RunTest` (110 m, `RT_RATE=30 RT_RAMP_S=4500`), scripts in `../k8s/` |
| `30-L40S-Gemma4-TPS-Smoke-Workflow.yaml` | the same, `RT_RATE=40 RT_RAMP_S=900` with wider first-trial guards (`RT_FIRST_OK_S=2400 RT_STALL_S=1500 RT_DEADLINE_S=3300`, 60 m): it also builds the ShareGPT cache |
| `30-L40S-Gemma4-TPS.yaml` | study `30-L40S-Gemma4-TPS` |
| `30-L40S-Gemma4-TPS-Smoke.yaml` | study `30-L40S-Gemma4-TPS-Smoke` (baseline only) |
| `check_offline.py` | offline checks against the repo rules and the local pack checkouts |

**Steps, main study:** `baseline` (vLLM 0.29.0 defaults on a < 70 GB GPU, every parameter
written out), `baseline repeat`, `kv fp8`, `kv fp8 large batch`, `optimize` (AKAMAS, 0 init,
60 experiments, `maxFailedExperiments` 20). **Smoke:** `baseline` only.
**Windowing shape:** `when:` is nested under `stability:`, as in studies 27/28, which the
3.7.x server accepted. The plugin's schema reference shows `when:` as a sibling of
`stability:`; do not "fix" it without checking on the server.
**parameterConstraints:** `max_num_batched_tokens >= max_num_seqs` (vLLM 0.29.0 raises
otherwise). **Placeholders left:** none (host `toolbox`, user `akamas`, key
`/home/akamas/.ssh/id_rsa`, as every study here; Prometheus address as studies 24-29).

**May change after the startup probe (before `akamas create`):** if the probe finds >= 2
attention or linear backends that start and serve within 15 %, add the parameter to both
studies' `parametersSelection` (and every preset's `values`), add its line to
`../k8s/params.env.template`, and add the FLASH_ATTN + fp8 constraint if FLASH_ATTN enters
(`vllm.attention_backend != "FLASH_ATTN" || vllm.kv_cache_dtype == "auto"`). Then re-run
`python3 check_offline.py`.

## Validation

Offline, 2026-10-05: `python3 check_offline.py` -> 0 failures. It checks component names,
every `params.env.template` token in `parametersSelection` and every selected parameter
rendered by the template, domains and categories inside the local packs (vLLM 1.12.0, GPU
1.4.0, Kubernetes 1.9.0-dev), step names, every baseline/preset rendering every parameter
and rendering through the real `render_statefulset.sh`, <= 8 KPIs, every goal / constraint /
windowing / KPI metric bound to its component's type and produced by the telemetry
instance, no underscore in telemetry placeholders. Mutation-tested (a bogus KPI metric, a
missing template line, an out-of-pack domain are all reported). **Not yet validated on the
Akamas server** (pending the sync).

## Setup & run

From the toolbox, in `/work/vllm-benchmark`, after the user's go-ahead and a `git pull`.
The smoke study first; the main study only after RT_RATE / RT_RAMP_S are set (`../README.md`,
"Morning runbook").

```bash
A=studies/30-l40s-gemma4-26b-tps/akamas

# Every file carries kind: (and system: where system-scoped), so the repo rule
# (.claude/rules/akamas-yaml.md) is the -f form, one file at a time in dependency order.
akamas create -f $A/system.yaml
for c in vllm gpu0 container cluster cluster_loadtest container_loadtest; do
  akamas create -f $A/components/$c.yaml
done
akamas create -f $A/telemetry/prometheus.yaml
akamas create -f $A/30-L40S-Gemma4-TPS-Smoke-Workflow.yaml
akamas create -f $A/30-L40S-Gemma4-TPS-Workflow.yaml
akamas create -f $A/30-L40S-Gemma4-TPS-Smoke.yaml
sleep 120   # studies 24/27/28: a start right after create can leave the study RUNNING with no experiment
akamas start study "30-L40S-Gemma4-TPS-Smoke"

# After the smoke run (export it, then delete it):
akamas export study "30-L40S-Gemma4-TPS-Smoke" /tmp/30-smoke-export.tar.gz
akamas delete --force study "30-L40S-Gemma4-TPS-Smoke"
akamas create -f $A/30-L40S-Gemma4-TPS.yaml
sleep 120
akamas start study "30-L40S-Gemma4-TPS"
```

Typed equivalent of the first block (one file per call; the typed form never takes a folder):
`akamas create system $A/system.yaml`, then `akamas create component $A/components/<c>.yaml
vLLM_Benchmark_30_L40S_Gemma4_TPS` for each of the six components, `akamas create
telemetry-instance $A/telemetry/prometheus.yaml vLLM_Benchmark_30_L40S_Gemma4_TPS`, `akamas
create workflow <file>` for both workflows, `akamas create study <file>`. Do not run `-f` on
the whole folder: it would create both studies at once.

If RT_RATE / RT_RAMP_S change after the smoke run, edit
`30-L40S-Gemma4-TPS-Workflow.yaml` (command and, if needed, `timeout` above RT_RAMP_S +
1500 s), push, pull, then `akamas delete workflow 30-L40S-Gemma4-TPS-Workflow` and create it
again before creating the study: the 3.7 CLI has no `update workflow` (checked for study 28).
If a start leaves a study RUNNING with no experiment after ~5 min, delete it, create it
again, wait a minute and start it again.

Monitoring: `akamas list experiment "30-L40S-Gemma4-TPS"`,
`akamas log --dump -s "30-L40S-Gemma4-TPS" -e <n> -l ERROR,WARN` (never `-d`: it prints the
token). After an `akamas finish study`, delete the load Job as well
(`kubectl -n llm-l40s delete job -l app=aiperf-l40s`, study 29). At the end:
`akamas export study "30-L40S-Gemma4-TPS"
studies/30-l40s-gemma4-26b-tps/results/export.tar.gz`, then the `study-recap` skill.
