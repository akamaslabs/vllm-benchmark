# akamas/ — 32-L40S-Gemma4-AWQ-Thinking

**Created:** 2026-10-09, with the `akamas-study-manager` plugin (0.3.0, modify mode), from
study 31's resources (`../../31-l40s-gemma4-26b-tps-thinking/akamas/`) as they were accepted by
the Akamas 4.1 server on 2026-10-08. **Target: Akamas Studio 4.1** (`akamas41.lab.akamas.io`,
namespace `akamas-41`). Created and started on the server 2026-10-09 (15:35 UTC), without the
probe and the smoke run (the user's decision).

## What it optimizes

**How many requests per second does one L40S complete with a customer's Docker Compose setup
of Gemma 4 26B-A4B, thinking on, with the e2e p95 within 30 s, and which vLLM configuration beats
their compose?** vLLM's completed requests/s (`vllm.request_success_rate`) of
`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a` (W4A16 int4) on one NVIDIA L40S, over 13
vLLM settings, subject to e2e p95 <= 30 s (150 s p95, `:max` over the scored window; the metric
is in ms: `<= 30000`), with study 27's open-loop AIPerf rate ramp, a watchdog past the knee and
the 3-minute window with the most completed requests/s as the score (goal and constraint
decided with the user 2026-10-09, replacing studies 30/31's tokens/s under TTFT / ITL). The rest
of what differs from study 31 is outside this folder: the compose's serving flags
(`../k8s/01-statefulset_template.yaml`) and the synthetic multi-turn load with ~3000-token
prompts (`../k8s/05-job_template.yaml`). Inside it: the baselines, the presets and
`linear_backend`'s domain. Design: `../README.md`.

## Versions

| Item | Version |
|---|---|
| Akamas | Studio 4.1.0 (chart 1.9.0-rc14) |
| Optimization pack **vLLM** | 1.12.0 (component type `vLLM`; `total_token_throughput`, `performance_mode`, `optimization_level`, ordinal `block_size`, `spec_method` / `spec_tokens`, `linear_backend`, `spec_decode_*`), as on the 4.1 server for studies 30/31; local checkout `~/akamas/offline/optimization-packs/vllm` at 458c2bd |
| Optimization pack **GPU** | 1.4.0 (component type `GPU`; only metrics used) |
| Kubernetes pack | 1.9.0 on the 4.1 server (`Kubernetes Container`, `Kubernetes Cluster`, only metrics used) |
| Serving | `vllm/vllm-openai:v0.29.0`, `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rev `0ef577a` served as `gemma4-26b-awq-think`, `--max-model-len 96000`, tool calling, reasoning parser `gemma4`, thinking on |
| Load generator | AIPerf 0.11.0, synthetic multi-turn chat (shared 1000-token system prompt, 300 +- 150-token messages, 3 +- 1 turns, history kept), no `max_tokens`, open-loop linear rate ramp (gamma arrivals, smoothness 4, seed 30) |
| Telemetry | Prometheus (kube-prometheus-stack), dcgm-exporter on the L40S node |

## Resources

| File | Resource |
|---|---|
| `system.yaml` | system `vLLM_Benchmark_32_L40S_Gemma4_AWQ_Thinking` |
| `components/vllm.yaml` | `vllm` (vLLM, pod `vllm-0`, model `gemma4-26b-awq-think`): every tuned parameter, goal, constraints, windowing |
| `components/gpu0.yaml` | `gpu0` (GPU, `gpu="0"`, model `.*L40S.*`) |
| `components/container.yaml` | `container` (Kubernetes Container, pod `vllm-0`) |
| `components/cluster.yaml`, `cluster_loadtest.yaml`, `container_loadtest.yaml` | node and load-generator views (Kubernetes pack) |
| `telemetry/prometheus.yaml` | `Prometheus_32_L40S_Gemma4_AWQ_Thinking`: study 31's metrics unchanged (the model filter comes from the `vllm` component); `prefix_cache_hit_rate` now reads a live value |
| `32-L40S-Gemma4-AWQ-Thinking-Workflow.yaml` | `Write config` -> `Apply config` (45 m) -> `RunTest` (130 m, `RT_RATE=2 RT_RAMP_S=6000`, provisional until the smoke run; `RT_STOP_AIPERF=0`) -> `Stop AIPerf` (20 m, `alwaysRun`, `critical: false`; added 2026-10-12, not on the server: "Setup & run"), scripts in `../k8s/` |
| `32-L40S-Gemma4-AWQ-Thinking.yaml` | study `32-L40S-Gemma4-AWQ-Thinking` |
| `check_offline.py` | offline checks against the repo rules and the local pack checkouts |

**`max_num_batched_tokens`:** domain 2496-16384 (512-16384 in study 31): the vision tower is
loaded, and one image (2496 tokens) must fit a batch (experiment 1 of the first creation failed
on it). **Goal:** maximize `vllm.request_success_rate` under `vllm.e2e_request_latency_p95:max <= 30000`
(ms; its telemetry query now takes a 150 s window, as the TTFT / ITL p95s); windowing on
`vllm.request_success_rate`. The threshold can be changed on a running study with `akamas update
study` (goal changes keep the history). **As study 31:** `parameterConstraints`, `kpis` (English names),
`numberOfTrials`, and `parametersSelection` but `linear_backend`: `[auto, triton, humming]`,
the int4 (W4A16) kernels of vLLM 0.29.0 that can run on SM 8.9 with bf16 activations and the
image's packages (`auto` = Marlin; `torch`, `marlin`, `exllama`, `conch` out, see the
manifest).

**Steps:**
- `baseline`: the customer's compose. `gpu_memory_utilization` 0.90 and `max_num_seqs` 64 are
  rendered; the other 11 parameters (`max_num_batched_tokens` among them: unset, vLLM raises it
  to 2496 for Gemma 4's images, while an explicit 2048 stops the engine) are in
  `doNotRenderParameters` (decided with the user 2026-10-09), so Akamas writes them empty and
  `render_statefulset.sh` passes no flag: vLLM picks its own default, as with the compose. They
  have no value in the step: Akamas 4.1 refuses a parameter both in `values` and in
  `doNotRenderParameters` (first create, 2026-10-09). Cost (study 27): this step never reaches
  the optimizer engine.
- `baseline repeat`: the same values, every one rendered (the user's choice), so it reaches
  the optimizer.
- Presets, every parameter rendered: `kv fp8` (compose + fp8 KV), `kv fp8 mtp2` (+ MTP K=2),
  `vllm defaults` (studies 30/31's baseline: 0.92, 256 seqs), `kv fp8 large batch` (study 31's),
  `study 30 best` and `study 31 best` (each study's best configuration, `linear_backend` auto
  instead of torch).
- `optimize`: AKAMAS, 0 init, 60 experiments, `maxFailedExperiments` 20.

**parameterConstraints:** `max_num_batched_tokens >= max_num_seqs`; `spec_method != "none" ||
spec_tokens == 0` and `spec_method == "none" || spec_tokens > 0`. **No Akamas smoke study:**
the probe and the smoke run are manual (`../probe/`, `../smoke/`). **Placeholders left:** none
(host `toolbox`, user `akamas`, key `/home/akamas/.ssh/id_rsa`, Prometheus address, as study
31); `RT_RATE` / `RT_RAMP_S` are provisional until the smoke run, and `linear_backend`'s
categories until the probe.

## Validation

Offline, 2026-10-09: `python3 check_offline.py` -> 0 failures. Study 31's checks (component
names, template tokens <-> `parametersSelection`, domains inside the local packs, step names,
every baseline/preset rendering through the real `render_statefulset.sh`, <= 8 KPIs, every
referenced metric bound and produced, no underscore in placeholders, every name on this study's
system, workflow paths in this folder, the `vllm` component's `model` = the served model name),
plus two for relative constraints, if any (a step of type `baseline` exists; the formula ends
in a signed percentage), and three for the not-rendered baseline: a step's
`doNotRenderParameters` are rendered empty (as Akamas does) before `render_statefulset.sh` reads
them; every entry is a selected parameter and no step combines it with `from`; `baseline repeat`
has the baseline's values and leaves out no more than the baseline does. Mutation-tested:
`baseline repeat` with another value, `gpu_memory_utilization` left unrendered, an unsigned
relative percentage and a relative constraint without a `baseline` step are all reported. 2026-10-12: one more check, a
RunTest that leaves the AIPerf Job running (`RT_STOP_AIPERF=0`) needs a later `alwaysRun` task
that runs `stop_aiperf.sh` (mutation-tested: `alwaysRun: false` is reported). **Server:** every resource created on Akamas 4.1 on 2026-10-09 with the commands below; the
first study create was refused (a parameter in both `values` and `doNotRenderParameters`), fixed
in commit 1fb98da; `akamas describe study` lists the 9 steps.

## Setup & run

From the 4.1 toolbox (`toolbox-ssh vllm-bench akamas-41`, or prefix each command with
`kubectl -n akamas-41 exec deploy/toolbox -c toolbox --`), in `/work/vllm-benchmark`, after the
user's go-ahead and a `git pull`; study 31 finished first (same GPU node), and the probe and the
smoke run done (`../README.md`, "Runbook": they set `linear_backend`'s categories and the
workflow's `RT_RATE` / `RT_RAMP_S`).

```bash
A=studies/32-l40s-gemma4-26b-awq-thinking/akamas

# Every file carries kind: (and system: where system-scoped), so the repo rule
# (.claude/rules/akamas-yaml.md) is the -f form, one file at a time in dependency order.
akamas create -f $A/system.yaml
for c in vllm gpu0 container cluster cluster_loadtest container_loadtest; do
  akamas create -f $A/components/$c.yaml
done
akamas create -f $A/telemetry/prometheus.yaml
akamas create -f $A/32-L40S-Gemma4-AWQ-Thinking-Workflow.yaml
akamas create -f $A/32-L40S-Gemma4-AWQ-Thinking.yaml
sleep 120   # studies 24/27/28: a start right after create can leave the study RUNNING with no experiment
akamas start study "32-L40S-Gemma4-AWQ-Thinking"
```

Typed equivalent (one file per call; the typed form never takes a folder): `akamas create
system $A/system.yaml`, then `akamas create component $A/components/<c>.yaml
vLLM_Benchmark_32_L40S_Gemma4_AWQ_Thinking` for each of the six components, `akamas create
telemetry-instance $A/telemetry/prometheus.yaml vLLM_Benchmark_32_L40S_Gemma4_AWQ_Thinking`,
`akamas create workflow $A/32-L40S-Gemma4-AWQ-Thinking-Workflow.yaml`, `akamas create study
$A/32-L40S-Gemma4-AWQ-Thinking.yaml`. `akamas create -f $A/` on the whole folder would also
work (one study, one workflow here), but the per-file order above is the one the repo uses.

**First trial check (memory: doNotRenderParameters renders an empty string):** within the
first minute of experiment 1, `params.env` on the toolbox
(`/work/vllm-benchmark/studies/32-l40s-gemma4-26b-awq-thinking/k8s/params.env`) must show the
10 not-rendered keys empty, and the Apply config log the compose's flags only.

If RT_RATE / RT_RAMP_S change after the workflow was created: edit
`32-L40S-Gemma4-AWQ-Thinking-Workflow.yaml` (command and, if needed, the RunTest `timeout`
above RT_RAMP_S + 1500 s), push, pull, then `akamas delete workflow
32-L40S-Gemma4-AWQ-Thinking-Workflow` and create it again before creating the study (no
`update workflow` verb). If a start leaves the study RUNNING with no experiment after ~5 min,
delete it, create it again, wait a minute and start it again.

**`Stop AIPerf` task (added 2026-10-12, local only: not pushed, not on the server).** After
RunTest, `../k8s/stop_aiperf.sh` sends AIPerf a SIGINT (it cancels the requests in flight and
writes its files; a plain Job delete SIGKILLs it and loses the last partial batch of records and
the summary json), deletes the Job, and saves the measured run's per-request records
(`profile_export.jsonl`) on the toolbox as
`/work/aiperf-exports/32-l40s-gemma4-26b-awq-thinking/<epoch>.jsonl`, `<epoch>` being the
measured run's start (the `aiperf-<epoch>` folder; Executor commands have no experiment-id
placeholder, so a file is matched to its experiment by the trial start times). `alwaysRun`: it
also runs after a failed RunTest, which then leaves the Job to it (`RT_STOP_AIPERF=0`);
`critical: false`: a record that could not be saved never fails the trial.
- **It does not reach the study created on 2026-10-10**, which keeps the three-task workflow:
  Akamas 4.1 has no `update workflow` (`akamas update --help` on the 4.1 toolbox, 2026-10-12:
  experiment, password, study, trial, user, workspace only). A `git pull` on the toolbox does
  not change that study either: its RunTest command does not set `RT_STOP_AIPERF`, and
  `run_test.sh`'s default (1) deletes the Job as before. `../results/grab_aiperf.sh` keeps
  copying the records from the workstation meanwhile.
- **To use it:** a new workflow and a new study, after the user's go-ahead (push, `git pull` on
  the toolbox), once the current study is finished and exported (command below): `akamas delete
  study "32-L40S-Gemma4-AWQ-Thinking"` (the manifest keeps its name, so the create is refused
  while it exists; the 2026-10-10 recreation did the same), `akamas delete workflow
  32-L40S-Gemma4-AWQ-Thinking-Workflow`, `akamas create -f
  $A/32-L40S-Gemma4-AWQ-Thinking-Workflow.yaml`, `akamas create -f
  $A/32-L40S-Gemma4-AWQ-Thinking.yaml`, `sleep 120`, `akamas start study
  "32-L40S-Gemma4-AWQ-Thinking"`. To keep the old study on the server instead, give the
  workflow and the study new names (and the study's `workflow:` field).
- **Trial timeline:** past the knee the load now runs a little longer (RunTest's log dump and the
  task handoff come before the SIGINT, tens of seconds); the scored window is before the knee,
  so the score does not move. First real run: the export pod mounts the results volume seconds
  after the Job's pod released it (300 s budget; if it times out, the step prints the pod's
  `describe` tail). The toolbox's kubeconfig can create pods, exec and delete Jobs in
  `llm-l40s` (`kubectl auth can-i`, 2026-10-12).
- **Room:** the toolbox's `/work` is a 1 GiB volume (778 MB free on 2026-10-12) that the Write
  config task also writes to; the records of one run are 10-26 MB (watchdog-shortened runs of
  2026-10-12), more for a full ramp. `stop_aiperf.sh` saves nothing below 150 MB free
  (`SA_MIN_FREE_MB`). Copy them to the workstation and delete them from the toolbox as they
  accumulate: `kubectl -n akamas-41 exec deploy/toolbox -c toolbox -- tar -C
  /work/aiperf-exports/32-l40s-gemma4-26b-awq-thinking -cf - . | tar -C
  studies/32-l40s-gemma4-26b-awq-thinking/results/aiperf_runs -xf -`.

Monitoring: `akamas list experiment "32-L40S-Gemma4-AWQ-Thinking"`,
`akamas log --dump -s "32-L40S-Gemma4-AWQ-Thinking" -e <n> -l ERROR,WARN` (never `-d`: it
prints the token). After an `akamas finish study`, delete the load Job as well
(`kubectl -n llm-l40s delete job -l app=aiperf-l40s`, study 29). At the end:
`akamas export study "32-L40S-Gemma4-AWQ-Thinking"
studies/32-l40s-gemma4-26b-awq-thinking/results/export.tar.gz`, then the `study-recap` skill.
