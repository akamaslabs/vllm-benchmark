# akamas/ — 33-L40S-Gemma4-AWQ-NoThinking

**Created:** 2026-10-12, with the `akamas-study-manager` plugin (0.3.0, modify mode on a copy),
from study 32's resources (`../../32-l40s-gemma4-26b-awq-thinking/akamas/`, as accepted by the
Akamas 4.1 server on 2026-10-10, plus its local four-task workflow of 2026-10-12). **Target:
Akamas Studio 4.1** (`akamas41.lab.akamas.io`, namespace `akamas-41`). **Created on the
server and started on 2026-10-12** (commit 854d7f9, after study 32 finished).

## What it optimizes

**Study 32 with thinking off: how many requests per second does one L40S complete with a
customer's Docker Compose setup of Gemma 4 26B-A4B when the replies hold no reasoning, with the
e2e p95 within 30 s, and which vLLM configuration beats their compose?** vLLM's completed
requests/s (`vllm.request_success_rate`) of `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a`
(W4A16 int4) on one NVIDIA L40S, over study 32's 13 vLLM settings, subject to
`vllm.e2e_request_latency_p95 <= 30000` (150 s p95 in ms, **averaged** over the scored window:
no `:max`, Akamas's default aggregation; study 32 used `:max`), with study 27's open-loop AIPerf
rate ramp, a watchdog past the knee and the 3-minute window with the most completed requests/s
as the score. What differs from study 32 outside this folder: `enable_thinking` false
(`../k8s/01-statefulset_template.yaml`) and the ramp to 6 req/s (`../k8s/run_test.sh`). Design:
`../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | Studio 4.1.0 (chart 1.9.0-rc14) |
| Optimization pack **vLLM** | 1.12.0 (component type `vLLM`), installed on the 4.1 server (`akamas list optimization-pack`, 2026-10-12); local checkout `~/akamas/offline/optimization-packs/vllm` at 458c2bd |
| Optimization pack **GPU** | 1.4.0 (component type `GPU`; only metrics used) |
| Kubernetes pack | 1.9.0 on the 4.1 server (`Kubernetes Container`, `Kubernetes Cluster`, only metrics used) |
| Serving | `vllm/vllm-openai:v0.29.0`, `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a` served as `gemma4-26b-awq-nothink`, `--max-model-len 96000`, tool calling, reasoning parser `gemma4`, **thinking off** (`enable_thinking` false) |
| Load generator | AIPerf 0.11.0, study 32's synthetic multi-turn chat (shared 1000-token system prompt, 300 +- 150-token messages, 3 +- 1 turns, history kept), no `max_tokens`, open-loop linear rate ramp 0 -> 6 req/s over 6000 s (gamma arrivals, smoothness 4, seed 30) |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter on the L40S node |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_33_L40S_Gemma4_AWQ_NoThinking` |
| `components/vllm.yaml` | `vllm` (vLLM, pod `vllm-0`, model `gemma4-26b-awq-nothink`): every tuned parameter, goal, constraint, windowing |
| `components/gpu0.yaml` | `gpu0` (GPU, `gpu="0"`, model `.*L40S.*`) |
| `components/container.yaml` | `container` (Kubernetes Container, pod `vllm-0`) |
| `components/cluster.yaml`, `cluster_loadtest.yaml`, `container_loadtest.yaml` | node and load-generator views (Kubernetes pack) |
| `telemetry/prometheus.yaml` | `Prometheus_33_L40S_Gemma4_AWQ_NoThinking`: study 32's metrics unchanged (the model filter comes from the `vllm` component) |
| `33-L40S-Gemma4-AWQ-NoThinking-Workflow.yaml` | `Write config` -> `Apply config` (45 m) -> `RunTest` (130 m, `RT_RATE=6 RT_RAMP_S=6000`, provisional; `RT_STOP_AIPERF=0`) -> `Stop AIPerf` (20 m, `alwaysRun`, `critical: false`), scripts in `../k8s/` |
| `33-L40S-Gemma4-AWQ-NoThinking.yaml` | study `33-L40S-Gemma4-AWQ-NoThinking` |
| `check_offline.py` | offline checks against the repo rules and the local pack checkouts |

**Goal:** maximize `vllm.request_success_rate` under `vllm.e2e_request_latency_p95 <= 30000`
(ms; 150 s p95; averaged over the 6 samples of the scored window: "each metric is aggregated by
default by average", Akamas docs, Goal & Constraints, checked 2026-10-12). Windowing on
`vllm.request_success_rate` (stability, 6 samples, filter disabled, `when: max`). The threshold
can be changed on a running study with `akamas update study` (goal changes keep the history).
**As study 32:** `parametersSelection` (13 parameters, same domains), `parameterConstraints`,
the 8 `kpis` (the two latency KPIs keep `aggregation: max`: they describe, they do not
constrain), `numberOfTrials`.

**Steps:** study 32's nine plus one preset:
- `baseline`: the customer's compose (`gpu_memory_utilization` 0.90, `max_num_seqs` 64
  rendered; the other 11 in `doNotRenderParameters`, no value: Akamas 4.1 refuses a parameter
  both in `values` and there). Cost (study 27): this step never reaches the optimizer engine.
- `baseline repeat`: the same values, every one rendered, so it reaches the optimizer.
- Presets, every parameter rendered: `kv fp8`, `kv fp8 mtp2`, `vllm defaults`, `kv fp8 large
  batch`, `study 30 best`, `study 31 best` (as study 32), and **`study 32 best`** (study 32's
  experiment 10: gmu 0.938, 408 sequences, 2496 batched tokens, fp8 KV, balanced, O1, fcfs, async
  off, capture 512, block 16, `humming`, MTP K=3; 0.633 completed req/s, +280 %, the best valid
  experiment on 2026-10-12 while study 32 ran its experiment 62).
- `optimize`: AKAMAS, 0 init, 60 experiments, `maxFailedExperiments` 20.

**parameterConstraints:** `max_num_batched_tokens >= max_num_seqs`; `spec_method != "none" ||
spec_tokens == 0` and `spec_method == "none" || spec_tokens > 0`. **Placeholders left:** none
(host `toolbox`, user `akamas`, key `/home/akamas/.ssh/id_rsa`, Prometheus address, as study 32).
`RT_RATE` is provisional (`../README.md`, "Load").

## Validation

Offline, 2026-10-12: `python3 check_offline.py` -> 0 failures (study 32's checks: names,
template tokens <-> `parametersSelection`, domains inside the local packs, step names, every
baseline/preset rendered through the real `render_statefulset.sh`, <= 8 KPIs, every referenced
metric bound and produced, no underscore in placeholders, every name on this study's system,
workflow paths in this folder, the `vllm` component's `model` = the served model name, the
not-rendered baseline, `RT_STOP_AIPERF=0` followed by an `alwaysRun` Stop AIPerf task). One check
added: the study-manifest glob (`33-*.yaml`) must find exactly one study, since a glob left at
another study's number skips every check and still reports 0 failures (the copy from study 32
had exactly that). `alwaysRun` / `critical` are the workflow task fields of the live docs
("Workflows template", checked 2026-10-12); the four-task workflow has not been created on any
server yet (study 32's running study keeps its three-task one). **Server:** not created yet.

## Setup & run

From the 4.1 toolbox (`toolbox-ssh vllm-bench akamas-41`, or prefix each command with
`kubectl -n akamas-41 exec deploy/toolbox -c toolbox --`), in `/work/vllm-benchmark`, after the
user's go-ahead, a push and a `git pull`. **Study 32 must be finished first** (same GPU node,
namespace, pod `vllm-0` and Job name): `akamas finish study "32-L40S-Gemma4-AWQ-Thinking"` (if it
has not ended by itself), its export, and its load Job deleted (`kubectl -n llm-l40s delete job
-l app=aiperf-l40s`).

```bash
A=studies/33-l40s-gemma4-26b-awq-no-thinking/akamas

# Every file carries kind: (and system: where system-scoped), so the repo rule
# (.claude/rules/akamas-yaml.md) is the -f form, one file at a time in dependency order.
akamas create -f $A/system.yaml
for c in vllm gpu0 container cluster cluster_loadtest container_loadtest; do
  akamas create -f $A/components/$c.yaml
done
akamas create -f $A/telemetry/prometheus.yaml
akamas create -f $A/33-L40S-Gemma4-AWQ-NoThinking-Workflow.yaml
akamas create -f $A/33-L40S-Gemma4-AWQ-NoThinking.yaml
sleep 120   # studies 24/27/28: a start right after create can leave the study RUNNING with no experiment
akamas start study "33-L40S-Gemma4-AWQ-NoThinking"
```

Typed equivalent (one file per call; the typed form never takes a folder): `akamas create
system $A/system.yaml`, then `akamas create component $A/components/<c>.yaml
vLLM_Benchmark_33_L40S_Gemma4_AWQ_NoThinking` for each of the six components, `akamas create
telemetry-instance $A/telemetry/prometheus.yaml vLLM_Benchmark_33_L40S_Gemma4_AWQ_NoThinking`,
`akamas create workflow $A/33-L40S-Gemma4-AWQ-NoThinking-Workflow.yaml`, `akamas create study
$A/33-L40S-Gemma4-AWQ-NoThinking.yaml`. `akamas create -f $A/` on the whole folder would also
work (one study, one workflow here), but the per-file order above is the one the repo uses.

**First trial checks (experiment 1, the baseline):**
- `params.env` on the toolbox
  (`/work/vllm-benchmark/studies/33-l40s-gemma4-26b-awq-no-thinking/k8s/params.env`) shows the
  11 not-rendered keys empty, and the Apply config log the compose's flags plus
  `--default-chat-template-kwargs {"enable_thinking": false}`.
- **Thinking is really off:** `vllm:request_generation_tokens` per request near study 32's
  answers (~620 tokens, not ~1500), and the Stop AIPerf records show `reasoning_token_count` 0 or
  absent.
- **The ramp reaches the knee:** the watchdog stops the run before 6000 s (TTFT p95 > 3 s or ITL
  p95 > 600 ms). If the baseline's knee is past ~3 req/s, raise `RT_RATE` (workflow command,
  `../k8s/run_test.sh` default, `../k8s/tests/test_run_test.sh`): the best configurations must
  still reach theirs before the end of the ramp.
- **AIPerf's start:** the conversation pool is 9001 entries at R = 6 (3001 in study 32); the
  measured run must start well inside `run_test.sh`'s deadline (RT_RAMP_S + 1500 s).

If RT_RATE / RT_RAMP_S change after the workflow was created: edit
`33-L40S-Gemma4-AWQ-NoThinking-Workflow.yaml` (command and, if needed, the RunTest `timeout`
above RT_RAMP_S + 1500 s), push, pull, then `akamas delete workflow
33-L40S-Gemma4-AWQ-NoThinking-Workflow` and create it again (no `update workflow` verb on 4.1);
with the study already created, delete and recreate the study too. If a start leaves the study
RUNNING with no experiment after ~5 min, delete it, create it again, wait a minute and start it
again.

**`Stop AIPerf` task** (study 32's, 2026-10-12). After RunTest, `../k8s/stop_aiperf.sh` sends
AIPerf a SIGINT (it cancels the requests in flight and writes its files; a plain Job delete
SIGKILLs it and loses the last partial batch of records and the summary json), deletes the Job,
and saves the measured run's per-request records (`profile_export.jsonl`) on the toolbox as
`/work/aiperf-exports/33-l40s-gemma4-26b-awq-no-thinking/<epoch>.jsonl`, `<epoch>` being the
measured run's start (Executor commands have no experiment-id placeholder, so a file is matched
to its experiment by the trial start times). `alwaysRun`: it also runs after a failed RunTest,
which then leaves the Job to it (`RT_STOP_AIPERF=0`); `critical: false`: a record that could not
be saved never fails the trial.
- **Room:** the toolbox's `/work` is a 1 GiB volume (778 MB free on 2026-10-12) that the Write
  config task also writes to. Study 32's records were 10-26 MB per watchdog-shortened run at 2
  req/s; at 6 req/s and with shorter replies a run holds ~2-3x the requests. `stop_aiperf.sh`
  saves nothing below 150 MB free (`SA_MIN_FREE_MB`). Copy them to the workstation and delete
  them from the toolbox every few experiments: `kubectl -n akamas-41 exec deploy/toolbox -c
  toolbox -- tar -C /work/aiperf-exports/33-l40s-gemma4-26b-awq-no-thinking -cf - . | tar -C
  studies/33-l40s-gemma4-26b-awq-no-thinking/results/aiperf_runs -xf -`.

Monitoring: `akamas list experiment "33-L40S-Gemma4-AWQ-NoThinking"`,
`akamas log --dump -s "33-L40S-Gemma4-AWQ-NoThinking" -e <n> -l ERROR,WARN` (never `-d`: it
prints the token). After an `akamas finish study`, delete the load Job as well
(`kubectl -n llm-l40s delete job -l app=aiperf-l40s`, study 29). At the end:
`akamas export study "33-L40S-Gemma4-AWQ-NoThinking"
studies/33-l40s-gemma4-26b-awq-no-thinking/results/export.tar.gz`, then the `study-recap` skill.
