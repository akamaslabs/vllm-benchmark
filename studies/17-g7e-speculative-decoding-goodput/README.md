# 17-g7e-speculative-decoding-goodput

**Status:** TODO — manifests written and validated offline, nothing created on the Akamas
instance yet
**Dates:** Scaffolded 2026-09-21

> Duplicate of `2-larger-model-g7e`'s **"2-Larger-goodput"** study — same GPU, same model,
> same goal, same SLA, same windowing, same 14 tuned parameters and same 3
> `parameterConstraints` — with **speculative decoding** added as the one new tuned
> dimension, and with the telemetry catalog upgraded to 126 metrics: study 16's 110 plus
> 16 that the installed packs declare but no study had ever collected. Read that study's
> README for the history of this hardware; this file is self-contained for everything
> study 17 changes.

## Objective

Find out whether **speculative decoding earns its place** on a single large-VRAM GPU
serving a 7B dense model under an interactive-chat SLA, and if so, with which method and
how many speculative tokens.

```
maximize  vLLM.prefill_token_throughput + vLLM.decode_token_throughput
subject to  vLLM.time_to_first_token_p95 <= 1500 ms
            vLLM.inter_token_latency_p95 <=  300 ms
```

One GPU, so the goal is also the per-GPU figure — no `active_gpus` divisor is needed,
unlike studies 9-16.

### Read this before interpreting the result: the goal fights the new parameter

This is deliberate, and it is the main thing to understand about study 17.

Speculative decoding pays off when the decode batch is **small enough that the verifier
has idle forward-pass capacity** for the drafter to exploit. Its gain is documented to
shrink and even invert once the batch saturates
(`ROADMAP.md` H3, and
`knowledge/notes/2026-07-distributed-inference-advanced-deployment-patterns.md`). This
study's goal maximizes throughput at whatever concurrency still meets the SLA — i.e.
precisely the saturated end of the ramp. **The optimizer is therefore expected to drive
`spec_method` toward `none`.**

That would be a legitimate finding, but only if every method was actually measured rather
than abandoned after one unlucky trial. Three design choices protect against the second
outcome:

1. **Seven preset steps run the whole grid head-to-head** (`none`, `ngram` × 3/5/8,
   `ngram_gpu` × 3/5/8) at otherwise-identical settings, before the optimizer gets to
   choose anything. S1 is speculation off *at those same settings*, so the six
   speculative cells have a proper reference rather than being compared against a
   baseline that differs in a dozen other ways.
2. **The concurrency sweep was re-floored from 150 → 8.** Study 2's ramp started at 150
   concurrent requests, which observes only the regime where speculation is expected to
   lose; a study using it would answer its own question by construction of the load
   pattern. The sweep is now 8 levels, 8 → 1024, doubling.
3. **Ten KPIs carry the latency metrics the goal ignores** — TTFT, ITL and especially
   TPOT p95, plus GPU SM/DRAM activity. "Same throughput, better latency" and "no effect"
   look identical in the goal and completely different in the KPIs.

If the honest answer turns out to be "not on chatbot traffic," that is worth knowing
cheaply: `ROADMAP.md` Section F already argues n-gram speculation shows its largest gains
on high-repetition workloads (RAG, code completion) that ShareGPT chat replay is not
shaped like. Eight preset/baseline experiments is a cheap way to settle it before
committing to a RAG-shaped study.

### Why this study exists now

Study 16's analysis (2026-09-21) found its best configurations pinned against the GPU's
power cap with the **memory bus only ~48% active and tensor cores ~11%** — spare compute
sitting idle on a memory-bound decode. Speculative decoding is the classic lever for
exactly that shape: it spends idle compute to reduce memory traffic per emitted token.
This study tests the lever on the hardware where it is easiest to isolate — one GPU, no
parallelism, a model that fits with room to spare.

## Stack & versions

- **Akamas version:** 3.7.x
- **Optimization packs:** vLLM **1.9.1**, GPU **1.2.0**, Kubernetes **1.9.0-dev** —
  **read from the local pack checkouts, NOT confirmed on the instance.**
  `akamas list optimization-pack` returns `Access forbidden … requires the
  'Administrator' role` for this account. Verify before creating the system. Study 16
  recorded Kubernetes 1.8.0-dev as installed, so that one in particular may differ.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  `Qwen/Qwen2.5-7B-Instruct`, served as `qwen2.5-7b`, namespace `llm-serving`. Pinned
  flags: `--enable-mfu-metrics`, `--no-enable-prefix-caching`.
  **Image changed from study 2's v0.22.0** — required, not cosmetic: the pack's reference
  version is 0.29.0, and the speculative-decoding counters this study will eventually
  read are V1-engine metrics 0.22.0 does not emit. Cost of the change: results are not
  directly comparable with 2-Larger-goodput's own numbers, so this study runs its own
  baseline instead of importing that study's experiments via `from`.
- **Cluster / hardware:** AWS `us-east-2`, EKS cluster `vllm-bench`, node group
  `llm-serving-g7e` = 1× `g7e.4xlarge` (1× NVIDIA RTX PRO 6000 Blackwell Server Edition,
  96 GB GDDR7, SM120, no MIG). Provisioning in `infra/`, duplicated from study 2 per this
  repo's atomic-per-study convention. **The node group already exists in the live cluster
  and is Active, with 0 nodes** (confirmed 2026-09-21) — so this is a scale-up, not a
  provisioning run. The CPU node the load generator runs on is `system-m8a` in the live
  cluster, not the `system` group study 2's `infra/eks/cluster.yaml` declares; the Job and
  the `cluster_loadtest` component follow the live cluster.
- **Load generator:** NVIDIA AIPerf 0.11.0, ShareGPT replay via a cached `inputs.json`,
  closed-loop concurrency ramp of **8 levels, 8 → 1024** (doubling), 300 s each,
  `--goodput time_to_first_token:1500 inter_token_latency:300`. Re-floored from study 2's
  150 → 1024 in 12 levels; see "the goal fights the new parameter" above. 8 × 300 s =
  40 min of load per trial, down from 60.
- **Telemetry:** Prometheus (`kube-prometheus-stack`), `duration: 30`, `stability`
  windowing on `prefill_token_throughput` (width 6). **126 metrics** — study 16's full
  110-metric catalog, plus the 7 vLLM-pack metrics and the 9 `Kubernetes Cluster` metrics
  that no telemetry instance in this repo had ever wired. The 7 vLLM ones are exactly the
  latency-breakdown metrics the pack gained in its 1.9.0/1.9.1 release that no study had
  picked up: TPOT p95/avg, TTFT and ITL p90, prefill and decode time p95, queue time avg.
  The 9 Kubernetes ones had their PromQL executed against the live Prometheus before being
  committed, which most of this catalog has never had.

## Parameters tuned

16 in total: study 2's 14, unchanged, plus the two new speculative-decoding ones.

| Parameter | Domain / categories | Baseline |
|---|---|---|
| `vLLM.gpu_memory_utilization` | [0.85, 0.95] | **0.90** (the only explicitly rendered one) |
| `vLLM.max_num_seqs` | [16, 1024] | *(vLLM default)* |
| `vLLM.max_num_batched_tokens` | [256, 8192] | *(vLLM default)* |
| `vLLM.kv_cache_dtype` | auto / fp8 / fp8_e4m3 / fp8_e5m2 | *(vLLM default)* |
| `vLLM.performance_mode` | balanced / interactivity / throughput | *(vLLM default)* |
| `vLLM.optimization_level` | [0, 3] | *(vLLM default)* |
| `vLLM.enforce_eager` | true / false | *(vLLM default)* |
| `vLLM.scheduling_policy` | fcfs / priority | *(vLLM default)* |
| `vLLM.disable_cascade_attn` | true / false | *(vLLM default)* |
| `vLLM.tokenizer_mode` | auto / hf / slow | *(vLLM default)* |
| `vLLM.async_scheduling` | true / false | *(vLLM default)* |
| `vLLM.max_cudagraph_capture_size` | [1, 1024] | *(vLLM default)* |
| `vLLM.block_size` | 16 … 128 (8 ordinals) | *(vLLM default)* |
| `vLLM.attention_backend` | FLASH_ATTN / FLASHINFER / TRITON_ATTN | *(vLLM default)* |
| **`vLLM.spec_method`** | **none / ngram / ngram_gpu** | **none (by absence)** |
| **`vLLM.spec_tokens`** | **[0, 16]**, 0 = off sentinel | **0** |

The baseline pins only `gpu_memory_utilization` and lists every other parameter in
`doNotRenderParameters`, so the reference point is vLLM's own stock startup path. The
tokens render empty and `k8s/apply_config.sh` strips the whole flag.

### `spec_method`: why two of the pack's five categories are missing

The pack declares `[none, ngram, ngram_gpu, suffix, mtp]`. **`mtp` and `suffix` are
excluded deliberately** — both failed 100% of the time in `1-goodput-realistic-load`, on
this same model:

- `mtp` → `NotImplementedError: Unsupported speculative method: 'mtp'`, raised
  unconditionally. Qwen2.5-7B has no MTP head; no other parameter value avoids it.
- `suffix` → `ImportError: Arctic Inference is required for suffix decoding`. The package
  is not in the `vllm-openai` image, for any configuration.

A categorical value that fails for every combination cannot be fenced with a
`parameterConstraint`; the only fix is not to offer it. **Both are worth re-testing on
v0.29.0** before a future study re-adds them — the image may now ship `arctic-inference`,
which would make `suffix` interesting in its own right.

### Constraints (7)

Three carried over from study 2, four new, every one of them traceable to a real crash or
to vLLM's own source.

1. `attention_backend != "FLASH_ATTN" || kv_cache_dtype == "auto"` — FLASH_ATTN supports
   only the auto KV dtype.
2. `attention_backend != "TRITON_ATTN" || kv_cache_dtype != "fp8_e5m2"` — query-quant bug.
3. `attention_backend != "FLASHINFER" || kv_cache_dtype != "fp8_e5m2"` — same.
4. **`spec_method != "none" || spec_tokens == 0`** — the off-sentinel gate, from the vLLM
   pack's own README. Stops the optimizer varying `spec_tokens` while speculation is off.
5. **`spec_method == "none" || spec_tokens > 0`** — the other direction. vLLM's
   `num_speculative_tokens` field declares `gt=0`, so a real method paired with the 0
   sentinel is a startup crash.
6. **`spec_method != "ngram_gpu" || optimization_level != 0`** — study 1, incident 3:
   `ValueError: No compilation mode is set`. `NgramProposerGPU`'s kernel is built on
   vLLM's `@support_torch_compile` machinery and needs an active compilation backend, but
   `optimization_level` 0 sets `CompilationMode.NONE`.
7. **`spec_method != "ngram" || async_scheduling == "false"`** — study 1, incident 5:
   Pydantic rejects the pair with *"async scheduling is only supported with
   EAGLE/MTP/Draft Model/NGram GPU kind of speculative decoding"*. Note the asymmetry: the
   **GPU** n-gram variant is on vLLM's supported list, plain CPU `ngram` is not. Do not
   widen this constraint to `ngram_gpu`.

Study 1 hit 5 distinct speculative-decoding crashes and responded by dropping the
parameters entirely. All 5 are addressed here: two by domain exclusion (#2 and #4 of that
list), two by constraints 6 and 7, and the fifth — a KV-budget exhaustion from speculation
plus a large `max_num_seqs` on a 24 GB A10G at `max_model_len` 32768 — is not expected to
recur on a 96 GB card with a 7B model, but is the first thing to check if a trial fails at
startup.

### Why "off" is expressed by deleting flags, not by a sentinel value

vLLM 0.29.0 treats speculative decoding as off only when **none** of
`--speculative-config` / `--spec-method` / `--spec-model` / `--spec-tokens` is passed at
all (`create_speculative_config` in `vllm/engine/arg_utils.py`). Passing a placeholder
makes `speculative_config` a non-`None` dict, which then fails validation two different
ways: argparse rejects `none` as a `--spec-method` choice, and Pydantic rejects
`num_speculative_tokens=0`.

So Akamas expresses "off" with the pack's sentinel pair (`spec_method: none` +
`spec_tokens: 0`, kept consistent by constraints 4 and 5), and `k8s/apply_config.sh`
**removes both flag lines** from the rendered Deployment when it sees
`--spec-method=none`. It also fails fast with exit 2 if it ever sees a real method
alongside `--spec-tokens=0`, which would mean the study manifest and the script had
drifted apart. Both paths were simulated against the rendered template before this study
was declared ready.

## Initial design: 8 preset/baseline experiments, then optimize

| Step | `spec_method` | `spec_tokens` | Purpose |
|---|---|---|---|
| baseline | *(unrendered → off)* | — | vLLM's own stock defaults, `gmu` 0.90 |
| S1 | none | 0 | speculation off **at the grid's settings** — the reference cell |
| S2 / S3 / S4 | ngram | 3 / 5 / 8 | CPU n-gram drafter, token-depth sweep |
| S5 / S6 / S7 | ngram_gpu | 3 / 5 / 8 | GPU n-gram drafter, same depths |
| optimize | free | free | 60 experiments, `numberOfInitExperiments: 0` |

Everything outside the two speculative parameters is held fixed across S1-S7
(`gmu` 0.90, `max_num_seqs` 256, `max_num_batched_tokens` 8192, `kv_cache_dtype` auto,
`balanced`, `optimization_level` 2, `block_size` 16, `TRITON_ATTN`, `enforce_eager` false,
`fcfs`, `disable_cascade_attn` false, `tokenizer_mode` auto, `async_scheduling` false,
`max_cudagraph_capture_size` 512). `optimization_level` 2 and `async_scheduling` false are
forced by constraints 6 and 7; holding `async_scheduling` false in **all seven** cells,
including the `ngram_gpu` ones that could legally set it true, keeps it from becoming a
hidden second variable between the two drafter families.

`numberOfInitExperiments: 0` is mandatory: any value above 0 makes the campaign service
run its own Sobol bootstrap on top of the presets, landing back on the deterministic head
of the Sobol sequence that aliased two categorical parameters perfectly in study 13 (see
`ROADMAP.md` section C).

**Budget.** Roughly 55-65 minutes per experiment. The 8 baseline/preset experiments are
about 8 hours and answer the study's core question on their own; the 60-experiment
optimize step is another ~60 hours on top. Study 16 was stopped early on cost at 27
optimizer experiments — decide the budget before starting, and note the optimize step can
be stopped at any point without losing what already ran.

## Known gap: this study cannot yet measure *why* speculation wins or loses

It tunes `spec_method` and `spec_tokens` and will see their effect on throughput and
latency, but **not the acceptance rate** — the one number that explains the effect.

Akamas telemetry may only map metrics the component type declares, and vLLM pack 1.9.1
declares 45 vLLM metrics, none of them speculative-decoding. The four real V1 counters
(`vllm:spec_decode_num_drafts_total`, `…_num_draft_tokens_total`,
`…_num_accepted_tokens_total`, `…_num_accepted_tokens_per_pos_total`) are written up, with
PromQL, in a commented-out block at the end of `akamas/telemetry/prometheus.yaml`.

**The pack change has been written** — outside this repo, as the rules require: branch
`feature/speculative-decoding-metrics` in `~/akamas/offline/optimization-packs/vllm`, off
`origin/develop`, bumping the pack to **1.10.0** and adding five metrics
(`spec_decode_drafts_rate`, `spec_decode_draft_tokens_rate`,
`spec_decode_accepted_tokens_rate`, `spec_decode_acceptance_rate`,
`spec_decode_accepted_tokens_per_draft`), with the pack's own offline test suite passing.
Committed locally, **not pushed**: it still needs a merge request, a build and an install
before the commented block can be enabled. That block's entries already carry the branch's
exact metric names, so enabling them is mechanical.

Worth doing before this study runs, not after: the acceptance rate is what distinguishes
"n-gram speculation does not fit chatbot traffic" from "speculation works but this
batch size is too large for it to pay," and those two conclusions lead to very different
next studies.

Note also that the five speculative-decoding metrics listed in vLLM's own documentation
(`spec_decode_draft_acceptance_rate`, `spec_decode_efficiency`,
`spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens`,
`spec_decode_num_emitted_tokens`) are **V0-engine legacy and are not emitted by the V1
engine in 0.29.0** — verified 2026-09-21 against `vllm/v1/spec_decode/metrics.py` at the
`v0.29.0` tag. Wiring those names would produce permanently empty series rather than an
error. Do not copy them from the docs page.

## Prerequisites still open

1. **Nothing has been created on the Akamas instance**, and no YAML here has been
   validated with `akamas create`. Everything passed a mechanical offline check against
   the pack checkouts — see `akamas/README.md` for exactly what was and was not verified.
2. **Confirm the installed pack versions** with `akamas describe optimization-pack`. The
   account used for scaffolding lacks the Administrator role that command needs.
3. **Place the SSH key** at
   `/work/vllm-benchmark/studies/17-g7e-speculative-decoding-goodput/akamas/id_rsa` on the
   toolbox — never committed. `akamas create workflow` refuses a `key:` path that does not
   exist there.
4. **BLOCKED — AWS has no `g7e.4xlarge` capacity in `us-east-2` (as of 2026-09-21 14:30 UTC).**
   This is the study's hard blocker, not a configuration mistake. State of play, checked
   with the `lab` AWS profile (account `916205288457`; the workstation's default profile
   points at a different account and cannot see this cluster):
   - The node group `llm-serving-g7e` is **already asking for a node**: `desiredSize: 1`,
     `minSize: 0`, `maxSize: 1`. Its status is `DEGRADED`, not `ACTIVE`.
   - Its health issue is `AsgInstanceLaunchFailures` /
     `InsufficientInstanceCapacity`, and the Auto Scaling group has logged **42 failed
     launches**, retrying roughly every 11 seconds.
   - **Both** availability zones are out of capacity. The error text suggests the other
     AZ each time, which reads like a fix but is circular: 14:22 failed in `us-east-2a`
     suggesting `2b`, 14:24 failed in `2b` suggesting `2a`, 14:26 failed in `2a` again.
     Restricting the ASG to one AZ was tried and reverted; both subnets are back in place
     so the group can take whichever frees up first.
   - `us-east-2c` is not a way out: the VPC does have a subnet there, but the `g7e`
     family **is not offered in `2c` at all** (only `2a` and `2b`).
   - **It is not a quota problem.** The "Running On-Demand G and VT instances" quota is
     64 vCPUs, nothing in the G family is running, and `g7e.4xlarge` needs 16.
     `InsufficientInstanceCapacity` is AWS-side scarcity; a quota block would raise
     `VcpuLimitExceeded` instead.

   Three ways forward, in order of least disruption:
   **(a) Wait.** The ASG retries on its own and will take the first node that frees up.
   Nothing to change.
   **(b) Change instance size, same GPU.** `g7e.2xlarge`, `g7e.4xlarge` and `g7e.8xlarge`
   all carry **exactly one RTX PRO Server 6000 with 96 GiB** — only vCPU and host RAM
   differ (8/64 GiB, 16/128 GiB, 32/256 GiB). A different size may have capacity when
   `4xlarge` does not, and the GPU under test would be unchanged. Caveats: the instance
   type of a managed node group is immutable, so this means creating a *new* node group;
   and on `g7e.2xlarge` the vLLM Deployment's 8-CPU request would consume the whole node,
   so lower it (`k8s/01-deployment_template.yaml`) before trying that size.
   **(c) On-Demand Capacity Reservation.** Guarantees the instance once granted, but bills
   from creation whether or not a node is running — a spending decision, deliberately not
   taken here.

   Related: the live cluster runs a `system-m8a` CPU node, not the `system` one this
   study's inherited `infra/eks/cluster.yaml` declares — the AIPerf Job's `nodeSelector`
   and the `cluster_loadtest` component were set to `system-m8a` to match. If the cluster
   is ever rebuilt from this study's own `infra/`, revisit that.
5. **Smoke-test v0.29.0 on this GPU's SM120 compute capability** before trusting a run.
   Study 2 flagged SM120 kernel maturity as an open question and never resolved it, on an
   older image. At minimum, start the server once by hand with `TRITON_ATTN` and once with
   `FLASHINFER`, and once with `spec_method=ngram_gpu`, before spending experiment budget.
6. **Delete the stale ShareGPT cache** if `/benchmarks/sharegpt-cache/inputs.json` on the
   benchmark PVC still holds study 2's file — it embeds the served model name, and a
   mismatch produces 100% 404s at every concurrency level (study 2 hit this in reverse).
   The model name is unchanged from study 2, so this is a low risk here, but confirm.
7. **Re-calibrate the ramp after the baseline.** The 8 → 1024 list is re-floored but its
   *ceiling* is inherited from an A10G-era calibration. If the last level is still
   SLA-compliant with headroom, raise it — that is exactly the failure that left 10 of
   study 16's experiments scored at its ramp's last level with their real capacity
   unmeasured.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
