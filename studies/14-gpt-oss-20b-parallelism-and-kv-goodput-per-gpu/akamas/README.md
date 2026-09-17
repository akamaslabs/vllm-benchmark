# akamas/ — Akamas resources of 14-gpt-oss-20b-parallelism-and-kv-goodput-per-gpu

**This study creates exactly one resource on the instance: its study manifest.** The
system, the 9 components, the telemetry instance and the workflow it runs against already
exist and were created from study 13's folder on 2026-09-15 — this study reuses them by
name. Their YAML is nevertheless **kept here as well** (byte-identical copies added on
2026-09-16, so that this folder is self-contained to read, like every other study's):

| Resource | Name | File here (copy of study 13's) | Created from |
|---|---|---|---|
| System | `vLLM_Benchmark_13_GPT_OSS_20B_TP` | `system.yaml` | `../../13-gpt-oss-20b-tp-goodput-per-gpu/akamas/system.yaml` |
| Components (9) | vLLM, container, cluster, gpu0-3, container_loadtest, cluster_loadtest | `components/*.yaml` | `../../13-gpt-oss-20b-tp-goodput-per-gpu/akamas/components/` |
| Telemetry instance | `Prometheus_13_GPT_OSS_20B_TP` (108 metrics) | `telemetry/prometheus.yaml` | `../../13-gpt-oss-20b-tp-goodput-per-gpu/akamas/telemetry/prometheus.yaml` |
| Workflow | `13-GPT-OSS-20B-TP-Goodput-Per-GPU-Workflow` | `13-GPT-OSS-20B-TP-Goodput-Per-GPU-Workflow.yaml` | `../../13-gpt-oss-20b-tp-goodput-per-gpu/akamas/13-GPT-OSS-20B-TP-Goodput-Per-GPU-Workflow.yaml` |
| Study | `14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU` | `14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU.yaml` | this folder — the only resource this study owns |

The copies keep study 13's resource names on purpose: they are the names the study manifest
references (`system:` / `workflow:`) and the names that exist on the instance. If study 13's
originals are ever edited, re-copy them here (`diff -rq` between the two folders should
show only the two study manifests and the READMEs).

This is deliberate, and it is the condition for the import to work: the `baseline` and
`bootstrap` steps pull study 13's experiments into this study, and their parameters and
metrics only map one-to-one if both studies run against the same system. It is also a
documented deviation from the repo's usual "one folder, one set of resources" convention
(`.claude/rules/akamas-yaml.md`) — the same deviation study 9's v2/v3 lineage makes.

Do **not** run `akamas create -f` on this folder: the system, components, telemetry
instance and workflow already exist on the instance, so the bulk form would fail on every
file but the study manifest. Create the study alone, with the typed command in
`../README.md`. Only if study 13's resources were ever deleted would the copies here be
created first, in dependency order (system, components, telemetry instance, workflow) —
and the imported experiments would then be gone with study 13.

`14-GPT-OSS-20B-Parallelism-And-KV-Goodput-Per-GPU.yaml` is the file that was created on
the live instance on 2026-09-16 (copied here verbatim, from `~/Downloads/study.yaml`) —
it is the record of what is running, not a draft.

## What the manifest says, in one table

| Section | Content |
|---|---|
| `goal` / `windowing` | byte-identical to study 13 (comparability of imported experiments) |
| `parametersSelection` | 14 parameters; only `tensor_parallel_size` [1, 4] and `data_parallel_size` [1, 4] differ from study 13 |
| `parameterConstraints` | the same 5 as study 13 |
| `kpis` | 8 (study 13 has 5) — two of them are not collected, see `../README.md` |
| `steps` | `baseline` (import exp 1) -> `bootstrap` (import exp 2..16) -> `S1..S10` presets -> `optimize` 100/20 with `numberOfInitExperiments: 0` |

There is no `akamas update` verb for `parametersSelection`, `parameterConstraints`,
`kpis`, `windowing` or `steps` — only `goal` can be edited on a running study. Any change
to the rest means a new study that bootstraps this one.

Setup and run commands: see `../README.md`, "Setup & run".
