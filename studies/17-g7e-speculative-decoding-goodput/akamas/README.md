# akamas/ — study 17's Akamas resources

Generated 2026-09-21 with the `akamas-study-manager` plugin (`/akamas-study-manager:build`),
against the pack checkouts under `~/akamas/offline/optimization-packs/`, and rewritten with
the same plugin on 2026-09-22 when the hardware changed. Nothing here was hand-written from
memory.

**These resources exist on the live instance and are CURRENT.** They were created
2026-09-21 against the RTX PRO 6000, and the 2026-09-22 rewrite was applied the same day:
the study and the telemetry instance were deleted and recreated, and the `cluster`
component was deleted and recreated with its new `node_role`. Verified afterwards — the
study is `CREATED` with 12 steps and S8/S9/S10 read "draft_model 2/3/4 tokens", so the
re-cut grid is what the instance actually holds. The delete/recreate sequence below is
kept as the record of what was run and as the recipe if any of it changes again.

## What this study optimizes

Maximize `vLLM.prefill_token_throughput + vLLM.decode_token_throughput` on one NVIDIA L4
(g6.4xlarge, 23034 MiB, Ada/SM89) serving `Qwen/Qwen3-8B-FP8`, subject to TTFT p95 ≤ 1500 ms
and ITL p95 ≤ 300 ms. Single GPU, so the goal is also the per-GPU figure. Both the card and
the model changed on 2026-09-22: AWS had no capacity for any GPU above 24 GB in `us-east-2`,
and the 30.5 GiB `Qwen/Qwen3-32B-FP8` this study had settled on does not fit 22.03 GiB usable. It
duplicates
`2-larger-model-g7e`'s "2-Larger-goodput" and adds speculative decoding
(`vLLM.spec_method` / `vLLM.spec_tokens`) as the one new tuned dimension. Full rationale,
including why the goal and the new parameter pull in opposite directions, is in
[`../README.md`](../README.md).

## Versions this was built against

| Thing | Version | How it was established |
|---|---|---|
| Akamas | 3.7.x | repo-wide target (`CLAUDE.md`) |
| vLLM optimization pack | **1.10.1 INSTALLED** | confirmed 2026-09-22: the telemetry instance exists on the instance, and it maps the five `spec_decode_*` metrics that only >= 1.10.0 declares, so it could not have been created otherwise. Source branch `feature/speculative-decoding-metrics`, still committed-not-pushed |
| GPU optimization pack | **1.2.0** | same, `~/akamas/offline/optimization-packs/nvidia-gpu` |
| Kubernetes optimization pack | **1.9.0-dev** | same, `~/akamas/offline/optimization-packs/kubernetes` — study 16's README recorded 1.8.0-dev as installed, so **re-verify which is actually on the instance** |
| vLLM server | `vllm/vllm-openai:v0.29.0` | `../k8s/01-deployment_template.yaml` |
| GPU | **1x NVIDIA L4**, 23034 MiB, Ada/SM89, ~300 GB/s, 72 W cap, on `g6.4xlarge`, node label `llm-serving-l4-single` | `nvidia-smi` on the live node, 2026-09-22. CHANGED from the RTX PRO 6000 Blackwell this study was built for: AWS had no capacity for any GPU class above 24 GB in any `us-east-2` zone |
| Model | `Qwen/Qwen3-8B-FP8` as `qwen3-8b` (8.79 GiB), drafter `Qwen/Qwen3-0.6B` (1.11 GiB, BF16) | `../k8s/01-deployment_template.yaml`; sizes read from each repo's safetensors headers 2026-09-22 |
| Load generator | NVIDIA AIPerf 0.11.0, ShareGPT replay | `../k8s/05-job.yaml` |
| Telemetry provider | Prometheus (`kube-prometheus-stack`) | `telemetry/prometheus.yaml` |

**`akamas list optimization-pack` still cannot be used to confirm these** — it returns
`Access forbidden … requires the 'Administrator' role` for this account. The GPU and
Kubernetes rows above therefore come from the local pack checkouts, which are the pack's
own source repos at their released tags: authoritative for *what the pack declares*, not
proof of *what is installed*. The vLLM row is different and is now settled by indirect
evidence rather than by the checkout: a telemetry instance that maps a metric the
component type does not declare cannot be created at all, and
`Prometheus_17_G7e_Speculative_Decoding` exists.

## Files

| File | Kind | Notes |
|---|---|---|
| `system.yaml` | `system` | `vLLM_Benchmark_17_G7e_Speculative_Decoding` |
| `components/vllm.yaml` | `component` | componentType `vLLM`; carries all 16 tuned parameters |
| `components/gpu0.yaml` | `component` | componentType `GPU`; the single physical GPU, one component per GPU per repo convention |
| `components/container.yaml` | `component` | componentType `Kubernetes Container`; the vLLM pod |
| `components/container_loadtest.yaml` | `component` | componentType `Kubernetes Container`; the AIPerf pod |
| `components/cluster.yaml` | `component` | componentType `Kubernetes Cluster`; the GPU node (`node_role: llm-serving-l4-single`). The only component whose edit is a LIVE binding rather than a description |
| `components/cluster_loadtest.yaml` | `component` | componentType `Kubernetes Cluster`; the CPU node (`node_role: system-m8a`) |
| `telemetry/prometheus.yaml` | `telemetry-instance` | **131 metrics** — study 16's 110-metric catalog verbatim, plus 7 vLLM-pack and 9 Kubernetes-pack metrics no study had ever wired, plus the 5 speculative-decoding metrics (no longer commented out: pack 1.10.1 ships them). The catalog is byte-identical to study 16's for its first 393 lines, and study 16 ran on L4s — so the measurement layer is already proven on this card |
| `17-G7e-Speculative-Decoding-Goodput-Workflow.yaml` | `workflow` | 3 tasks: FileConfigurator → Apply config → RunTest, all on `toolbox` over SSH |
| `17-G7e-Speculative-Decoding-Goodput.yaml` | `study` | goal, 16 parameters, **9** `parameterConstraints`, 8 KPIs (an Akamas hard limit), baseline + **10 presets** + optimize |

## The metrics question, answered

Two different gaps were found, and only one of them is fixable from this repo.

**Fixable, and fixed here — 7 metrics the pack declares that nothing was collecting.**
The vLLM pack declares 45 metrics on its component type; study 16's telemetry instance,
the richest in the repo, wired 38 of them. The 7 left over are now wired in
`telemetry/prometheus.yaml`: `time_to_first_token_p90`, `inter_token_latency_p90`,
`time_per_output_token_p95`, `time_per_output_token_avg`, `request_prefill_time_p95`,
`request_decode_time_p95`, `request_queue_time_avg`. They are directly relevant here:
speculative decoding changes the shape of a decode step, so TPOT and the prefill/decode
split are where it shows up even when total throughput does not move.

**Also fixed — 9 `Kubernetes Cluster` metrics.** The right comparison is per component
type, not against the whole Kubernetes pack. Of the two types this study actually uses,
`Kubernetes Container` was already complete (25 of 25 wired) and `Kubernetes Cluster` had
**5 of 14** — study 16 wired the PSI ones and left node capacity, allocatable, aggregate
requests and utilisation. Those 9 are now wired, and unlike most of this catalog **their
PromQL was executed against this cluster's live Prometheus** before being committed: each
returned a plausible value rather than an empty result. The Kubernetes pack's other ~100
metrics belong to component types this study does not model (`Kubernetes Node`, `Pod`,
`Namespace`, `Workload`, the HPA types) and would need their own components first. The GPU
pack's 42 metrics were already fully wired.

**Shipped, and the block is now live — the speculative-decoding metrics.** An Akamas
telemetry instance may only map metrics the component type already declares, and pack 1.9.1
declared none for speculative decoding. That change was made on the pack's own repo rather
than here: branch **`feature/speculative-decoding-metrics`** in
`~/akamas/offline/optimization-packs/vllm`, off `origin/develop`, adding
`spec_decode_drafts_rate`, `spec_decode_draft_tokens_rate`,
`spec_decode_accepted_tokens_rate`, `spec_decode_acceptance_rate` and
`spec_decode_accepted_tokens_per_draft`, with the pack's own offline test suite passing.
Version **1.10.1** was built and installed on the instance on 2026-09-22, and the five
entries at the end of `telemetry/prometheus.yaml` are uncommented accordingly.

**The branch is still committed-locally and unpushed.** Nothing in this study depends on
that any more, but the pack's own repo has no record of a version its instance is running,
which is a debt worth clearing: `git push -u origin feature/speculative-decoding-metrics`
and a merge request against `develop`.

So this study **can** now measure acceptance rate, not merely tune speculative decoding —
which matters more on this card than it did on the previous one, because the draft_model
arm's economics are marginal here and "it did not help" and "the drafter was rarely
accepted" are different findings leading to different follow-ups.

The four real series are `vllm:spec_decode_num_drafts_total`,
`vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total`
and `vllm:spec_decode_num_accepted_tokens_per_pos_total` (Counters, labels `model_name`
and `engine`). The five names in vLLM's own documentation are V0-engine legacy and are
**not** emitted by the V1 engine in 0.29.0 — verified 2026-09-21 by reading
`vllm/v1/spec_decode/metrics.py` at the `v0.29.0` tag. Wiring those would give
permanently empty series rather than an error. The full note, including the PromQL and
the fact that these counters are *absent* rather than zero when speculation is off, is in
the commented block itself.

## Offline validation performed

No live instance was touched. These checks were run mechanically against the pack
checkouts and all passed:

- Every file parses as YAML and carries `kind:` (and `system:` where needed), so
  `akamas create -f` can dispatch on it.
- Every `componentType` resolves to a real component type (`vLLM`, `GPU`,
  `Kubernetes Container`, `Kubernetes Cluster`).
- All 16 `parametersSelection` entries resolve to parameters the vLLM component type
  declares, and every domain/category list is a **subset** of the pack's own.
- Every metric named in the goal, the two constraints, the windowing and the 10 KPIs
  resolves both to a pack-declared metric on the referencing component's type **and** to
  an entry in this study's telemetry instance.
- Every `parameterConstraints` formula token resolves to a real parameter.
- **All 10 presets satisfy all 7 constraints** (evaluated, not eyeballed), including the three `draft_model` cells.
- The **baseline** satisfies all 7 constraints under the pack's default values, which is
  what the unrendered parameters resolve to (`spec_method` default `none`, `spec_tokens`
  default `0`, `attention_backend` default `FLASH_ATTN` with `kv_cache_dtype` `auto`).
- Every `doNotRenderParameters` entry is a real parameter, and the baseline step does not
  combine `doNotRenderParameters` with `from` (Akamas forbids that pairing).
- All 28 `${vLLM.*}` tokens in `../k8s/01-deployment_template.yaml` resolve to real
  parameters: 16 tuned, 12 pinned.
- The speculative-flag mechanics in `../k8s/apply_config.sh` were simulated against the
  rendered template for five cases: baseline (tokens empty) → 0 spec flags;
  `spec_method=none` → all three flags removed; `ngram` → method and tokens kept,
  `--spec-model` removed; `draft_model` → all three kept including
  `--spec-model=Qwen/Qwen3-0.6B`; the forbidden real-method-plus-0 pairing → script exits
  2 with a message instead of starting a doomed 45-minute trial.

Still **unvalidated** and only resolvable against the live instance or a real run: the
installed pack versions, whether `akamas create` accepts each file, and whether the 7
newly-wired PromQL queries actually return data (their underlying vLLM series names were
read from pack metric descriptions, not scraped — ROADMAP Q7 applies).

## Placeholders / preconditions before this can be created

1. **The workflow key is `/home/akamas/.ssh/id_rsa` on the toolbox** — the toolbox's
   own key, mounted from the `toolbox-keys` Kubernetes secret. Changed 2026-09-23: the
   per-study copies under `studies/*/akamas/id_rsa` were a compromised key tracked in git
   and are gone; the key was rotated in the secret. `akamas create workflow` still refuses
   a `key:` path that does not exist on the toolbox, so run it from the toolbox. The
   workflows already stored in Akamas before that date carry the old key and are not
   reused — create a fresh one from this file.
2. **The toolbox needs this study's folder** pulled at
   `/work/vllm-benchmark/studies/17-g7e-speculative-decoding-goodput/`, since all three
   workflow tasks read scripts from it.
3. **The GPU node.** SUPERSEDED 2026-09-22 and worth reading as a caution rather than as
   an instruction. This step used to read: scale `llm-serving-g7e` to 1, no provisioning
   needed. That node group then failed to launch 56+ times in one day, a second on
   `g7e.8xlarge` rolled back, and a third pinned to `us-east-2a` reached `CREATE_FAILED`
   after 34 attempts — `InsufficientInstanceCapacity` every time. A direct probe found
   every GPU class above 24 GB empty in all three zones. The study now uses
   **`llm-serving-l4-single`** (`g6.4xlarge`, one NVIDIA L4, 1.32 USD/h), created and
   brought up first try in `us-east-2a`:
   `aws eks update-nodegroup-config --cluster-name vllm-bench --region us-east-2 --nodegroup-name llm-serving-l4-single --scaling-config minSize=0,maxSize=1,desiredSize=1`
   Before assuming any node group can simply be scaled up, run
   `infra/eks/gpu-capacity-fallback.sh probe` — it answers in seconds instead of after an
   Auto Scaling Group's four-minute retry cycle. Remember to scale it back to 0 when the
   study is not running.
   Note the live cluster does **not** match this study's inherited
   `infra/eks/cluster.yaml` in every detail — that file came from study 2 and declares a
   `system` node group, while the live cluster runs `system-m8a`; the load-generator Job
   and the `cluster_loadtest` component were set to `system-m8a` to match reality.
4. **Re-verify the installed pack versions** (see the table above).

## Re-applying the 2026-09-22 hardware change (do this first)

The system, its six components, the telemetry instance, the workflow and the study were
**already created** on the instance on 2026-09-21, against the RTX PRO 6000. The
2026-09-22 rewrite for one L4 touched three of them, and Akamas 3.7.x has **no update
verb** for any of the three — only a study's `goal` is editable in place. So they must be
deleted and recreated, in this order:

```bash
S="vLLM_Benchmark_17_G7e_Speculative_Decoding"
D=studies/17-g7e-speculative-decoding-goodput/akamas

# 1. The study first: it references the system, so it has to go before the component does.
#    Safe — it is status CREATED with ZERO experiments, so no history is lost.
akamas delete study "17-G7e-Speculative-Decoding-Goodput" -f

# 2. The telemetry instance: it maps metrics onto the component being replaced.
akamas delete telemetry-instance "Prometheus_17_G7e_Speculative_Decoding" "$S" -f

# 3. The cluster component: node_role changed from llm-serving-g7e to
#    llm-serving-l4-single. This is the one edit that is a LIVE binding rather than a
#    description — it is substituted as $NODE_ROLE$ into 14 PromQL queries.
akamas delete component cluster "$S" -f
akamas create component "$D/components/cluster.yaml" "$S"

# 4. Recreate, in dependency order.
akamas create telemetry-instance "$D/telemetry/prometheus.yaml" "$S"
akamas create study "$D/17-G7e-Speculative-Decoding-Goodput.yaml"
```

The other five components and the workflow are **unchanged** and must NOT be deleted.
`system.yaml`, `gpu0.yaml` and `vllm.yaml` had only their `description` edited; those are
cosmetic, carry no live binding, and are not worth a delete/recreate cycle on their own —
the files are correct for the next time the system is built from scratch.

Outside Akamas, two cluster-side steps belong to the same change and are easy to forget
because both fail silently:

```bash
# The DCGM exporter is pinned by nodeSelector and will not follow the node. Stale, ~30
# metrics including two of the eight KPIs simply have no series and nothing errors.
helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring --version 4.8.3 --reuse-values=false \
  -f studies/17-g7e-speculative-decoding-goodput/k8s/monitoring/dcgm-exporter-values.yaml
# equivalently: studies/17-.../infra/eks/gpu-capacity-fallback.sh dcgm-cover

# Push the edited k8s/ files to the toolbox checkout the workflow actually reads.
```

## Setup & run

Dependency order matters: Akamas resolves `system:` / `workflow:` references by name at
creation time. Use this section for a from-scratch build; for the 2026-09-22 change on an
instance that already has these resources, use the delete/recreate block above instead.

```bash
# Typed, one resource at a time — the component form takes exactly ONE file, never a folder
akamas create system            studies/17-g7e-speculative-decoding-goodput/akamas/system.yaml
akamas create component         studies/17-g7e-speculative-decoding-goodput/akamas/components/vllm.yaml               "vLLM_Benchmark_17_G7e_Speculative_Decoding"
akamas create component         studies/17-g7e-speculative-decoding-goodput/akamas/components/gpu0.yaml               "vLLM_Benchmark_17_G7e_Speculative_Decoding"
akamas create component         studies/17-g7e-speculative-decoding-goodput/akamas/components/container.yaml          "vLLM_Benchmark_17_G7e_Speculative_Decoding"
akamas create component         studies/17-g7e-speculative-decoding-goodput/akamas/components/container_loadtest.yaml "vLLM_Benchmark_17_G7e_Speculative_Decoding"
akamas create component         studies/17-g7e-speculative-decoding-goodput/akamas/components/cluster.yaml            "vLLM_Benchmark_17_G7e_Speculative_Decoding"
akamas create component         studies/17-g7e-speculative-decoding-goodput/akamas/components/cluster_loadtest.yaml   "vLLM_Benchmark_17_G7e_Speculative_Decoding"
akamas create telemetry-instance studies/17-g7e-speculative-decoding-goodput/akamas/telemetry/prometheus.yaml         "vLLM_Benchmark_17_G7e_Speculative_Decoding"
akamas create workflow          studies/17-g7e-speculative-decoding-goodput/akamas/17-G7e-Speculative-Decoding-Goodput-Workflow.yaml
akamas create study             studies/17-g7e-speculative-decoding-goodput/akamas/17-G7e-Speculative-Decoding-Goodput.yaml

akamas start study "17-G7e-Speculative-Decoding-Goodput"
```

Bulk alternative — every file here self-describes its `kind:` and `system:`, so one
command covers the whole folder. The same dependency order still applies internally:

```bash
akamas create -f studies/17-g7e-speculative-decoding-goodput/akamas/
akamas start study "17-G7e-Speculative-Decoding-Goodput"
```

Run these from the toolbox, where the CLI is configured:

```bash
~/bin/toolbox-ssh            # interactive session with agent forwarding
# or, non-interactively:
kubectl -n akamas exec deploy/toolbox -c toolbox -- akamas <verb> ...
```

Monitoring and teardown:

```bash
akamas describe study "17-G7e-Speculative-Decoding-Goodput"
akamas list experiment "17-G7e-Speculative-Decoding-Goodput"
akamas finish study "17-G7e-Speculative-Decoding-Goodput"     # the presets already run are kept
# NOTE: there is no `akamas stop`. `finish` is overloaded to mean both "end for good" and
# "pause, resumable later" via `akamas resume study`.
akamas export study "17-G7e-Speculative-Decoding-Goodput" studies/17-g7e-speculative-decoding-goodput/results/export.tar.gz
```

There is **no generic "apply changes" verb**: on Akamas 3.7.x only a study's `goal` can be
edited in place (`akamas update study`). Changing `parametersSelection`, `windowing` or
`steps` after the study has run experiments requires deleting and recreating it, which
loses history — so get the manifest right before starting, or plan a successor study the
way study 16 succeeded study 15.
