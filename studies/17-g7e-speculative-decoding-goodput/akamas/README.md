# akamas/ — study 17's Akamas resources

Generated 2026-09-21 with the `akamas-study-manager` plugin (`/akamas-study-manager:build`),
against the pack checkouts under `~/akamas/offline/optimization-packs/`. Nothing here was
hand-written from memory, and nothing here has been created on a live Akamas instance yet.

## What this study optimizes

Maximize `vLLM.prefill_token_throughput + vLLM.decode_token_throughput` on one NVIDIA
RTX PRO 6000 Blackwell (g7e.4xlarge, 96 GB GDDR7, SM120) serving
`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, subject to TTFT p95 ≤ 1500 ms and ITL p95 ≤ 300 ms. Single
GPU, so the goal is also the per-GPU figure. It duplicates
`2-larger-model-g7e`'s "2-Larger-goodput" and adds speculative decoding
(`vLLM.spec_method` / `vLLM.spec_tokens`) as the one new tuned dimension. Full rationale,
including why the goal and the new parameter pull in opposite directions, is in
[`../README.md`](../README.md).

## Versions this was built against

| Thing | Version | How it was established |
|---|---|---|
| Akamas | 3.7.x | repo-wide target (`CLAUDE.md`) |
| vLLM optimization pack | **1.10.0 REQUIRED** | branch `feature/speculative-decoding-metrics`, committed not pushed — 1.9.1 lacks `spec_method`'s `draft_model` |
| GPU optimization pack | **1.2.0** | same, `~/akamas/offline/optimization-packs/nvidia-gpu` |
| Kubernetes optimization pack | **1.9.0-dev** | same, `~/akamas/offline/optimization-packs/kubernetes` — study 16's README recorded 1.8.0-dev as installed, so **re-verify which is actually on the instance** |
| vLLM server | `vllm/vllm-openai:v0.29.0` | `../k8s/01-deployment_template.yaml` |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` as `qwen3-30b-a3b`, drafter `Qwen/Qwen3-0.6B` | same |
| Load generator | NVIDIA AIPerf 0.11.0, ShareGPT replay | `../k8s/05-job.yaml` |
| Telemetry provider | Prometheus (`kube-prometheus-stack`) | `telemetry/prometheus.yaml` |

**The installed pack versions were NOT confirmed with `akamas list optimization-pack`**
(the repo's normal precondition). That command returns `Access forbidden … requires the
'Administrator' role` for this account, and the toolbox CLI session was logged out at
build time. The versions above come from the local pack checkouts, which are the pack's
own source repos at their released tags — authoritative for *what the pack declares*, but
not proof of *what is installed*. Confirm before creating anything.

## Files

| File | Kind | Notes |
|---|---|---|
| `system.yaml` | `system` | `vLLM_Benchmark_17_G7e_Speculative_Decoding` |
| `components/vllm.yaml` | `component` | componentType `vLLM`; carries all 16 tuned parameters |
| `components/gpu0.yaml` | `component` | componentType `GPU`; the single physical GPU, one component per GPU per repo convention |
| `components/container.yaml` | `component` | componentType `Kubernetes Container`; the vLLM pod |
| `components/container_loadtest.yaml` | `component` | componentType `Kubernetes Container`; the AIPerf pod |
| `components/cluster.yaml` | `component` | componentType `Kubernetes Cluster`; the GPU node (`node_role: llm-serving-g7e`) |
| `components/cluster_loadtest.yaml` | `component` | componentType `Kubernetes Cluster`; the CPU node (`node_role: system-m8a`) |
| `telemetry/prometheus.yaml` | `telemetry-instance` | **126 metrics** — study 16's 110-metric catalog verbatim, plus 7 vLLM-pack and 9 Kubernetes-pack metrics no study had ever wired, plus a commented-out speculative-decoding block blocked on a pack release |
| `17-G7e-Speculative-Decoding-Goodput-Workflow.yaml` | `workflow` | 3 tasks: FileConfigurator → Apply config → RunTest, all on `toolbox` over SSH |
| `17-G7e-Speculative-Decoding-Goodput.yaml` | `study` | goal, 16 parameters, 7 `parameterConstraints`, 8 KPIs (an Akamas hard limit), baseline + **10 presets** + optimize |

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

**Written, not yet shipped — the speculative-decoding metrics.** An Akamas telemetry
instance may only map metrics the component type already declares, and pack 1.9.1 declares
none for speculative decoding. That change was made, on the pack's own repo rather than
here: branch **`feature/speculative-decoding-metrics`** in
`~/akamas/offline/optimization-packs/vllm`, off `origin/develop`, commit *"Add
speculative-decoding acceptance metrics (v1.10.0)"*, adding `spec_decode_drafts_rate`,
`spec_decode_draft_tokens_rate`, `spec_decode_accepted_tokens_rate`,
`spec_decode_acceptance_rate` and `spec_decode_accepted_tokens_per_draft`, with the pack's
own offline test suite passing (14 tests). **Committed locally, nothing pushed.** It still
needs a push, a merge request, a build and an install; only then does the commented block
at the end of `telemetry/prometheus.yaml` become uncommentable — its five entries already
use that branch's exact metric names, so enabling them is mechanical.

Until the pack ships, **this study can tune speculative decoding but cannot measure its
acceptance rate** — it will see the effect on latency and throughput, not the reason for
it, and those two readings lead to different follow-up studies.

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

1. **`akamas/id_rsa` must exist on the toolbox** at
   `/work/vllm-benchmark/studies/17-g7e-speculative-decoding-goodput/akamas/id_rsa`.
   `akamas create workflow` refuses a `key:` path that does not exist there. The file is
   never committed (`.gitignore`); copy it from another study's folder on the toolbox.
2. **The toolbox needs this study's folder** pulled at
   `/work/vllm-benchmark/studies/17-g7e-speculative-decoding-goodput/`, since all three
   workflow tasks read scripts from it.
3. **Scale the GPU node group to 1.** Confirmed 2026-09-21 from the EKS console: the node
   group **`llm-serving-g7e` exists in cluster `vllm-bench` and is Active**, instance type
   `g7e.4xlarge` — with **0 nodes**. No provisioning needed, only a scale-up:
   `eksctl scale nodegroup --cluster vllm-bench --name llm-serving-g7e --nodes 1`.
   `kubectl get nodes` confirmed only the `system-m8a` and `akamas` nodes were running.
   Note the live cluster does **not** match this study's inherited
   `infra/eks/cluster.yaml` in every detail — that file came from study 2 and declares a
   `system` node group, while the live cluster runs `system-m8a`; the load-generator Job
   and the `cluster_loadtest` component were set to `system-m8a` to match reality.
4. **Re-verify the installed pack versions** (see the table above).

## Setup & run

Dependency order matters: Akamas resolves `system:` / `workflow:` references by name at
creation time.

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
akamas stop study "17-G7e-Speculative-Decoding-Goodput"       # the presets already run are kept
akamas export study "17-G7e-Speculative-Decoding-Goodput" studies/17-g7e-speculative-decoding-goodput/results/export.tar.gz
```

There is **no generic "apply changes" verb**: on Akamas 3.7.x only a study's `goal` can be
edited in place (`akamas update study`). Changing `parametersSelection`, `windowing` or
`steps` after the study has run experiments requires deleting and recreating it, which
loses history — so get the manifest right before starting, or plan a successor study the
way study 16 succeeded study 15.
