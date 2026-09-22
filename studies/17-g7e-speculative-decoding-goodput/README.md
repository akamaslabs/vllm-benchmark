# 17-g7e-speculative-decoding-goodput

**Status:** TODO — **blocked on vLLM optimization pack 1.10.0 being installed.** The
Akamas system, components, telemetry instance and workflow are created; the study itself
was created once in its 7B form and must be **deleted and recreated** after the pack ships
(`parametersSelection` cannot be edited on an existing study).
**Dates:** Scaffolded 2026-09-21, reconfigured the same day (model swap + `draft_model`)

> Started as a duplicate of `2-larger-model-g7e`'s **"2-Larger-goodput"** — same single
> RTX PRO 6000 Blackwell, same goal, SLA, windowing and 14 tuned parameters — with
> **speculative decoding** as the new dimension. It has since diverged on four points
> (model, image, ramp, timeouts), each recorded below, so it imports nothing from that
> study and runs its own baseline. Telemetry carries 126 metrics: study 16's 110 plus 16
> the installed packs declare that no study had ever collected.

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
   pattern. The sweep is now 9 levels, 8 → 2048, doubling — the ceiling raised too,
   because study 16 served this same model and ran out of ramp before its best
   configurations ran out of SLA.
3. **Ten KPIs carry the latency metrics the goal ignores** — TTFT, ITL and especially
   TPOT p95, plus GPU SM/DRAM activity. "Same throughput, better latency" and "no effect"
   look identical in the goal and completely different in the KPIs.

If the honest answer turns out to be "not on chatbot traffic," that is worth knowing
cheaply: `ROADMAP.md` Section F already argues n-gram speculation shows its largest gains
on high-repetition workloads (RAG, code completion) that ShareGPT chat replay is not
shaped like. Eight preset/baseline experiments is a cheap way to settle it before
committing to a RAG-shaped study.

### Three drafter families, compared head-to-head

The word "speculative decoding" covers two mechanisms that fail for opposite reasons, and
this study measures both rather than assuming one stands for the other.

- **`ngram` / `ngram_gpu`** propose tokens by finding a repeated n-gram in the context and
  replaying what followed it. No second model, no extra VRAM. They pay off only on
  **repetitive** text, which ShareGPT chat replay is not (`ROADMAP.md` section F).
- **`draft_model`** is the classic form: a real, much smaller model proposes and the
  target verifies. Acceptance does not depend on repetition at all, but the drafter's own
  forward passes consume the GPU the verifier would otherwise use. The drafter here is
  **`Qwen/Qwen3-0.6B`**, which shares the Qwen3 tokenizer and its 151936-token vocabulary
  with the target and costs about 1.2 GB of the 96 GB card.

Without both families in the grid, a poor result would be unattributable: "n-gram did not
help on chat traffic" and "speculation does not help at this batch size" are different
findings with different follow-ups.

`draft_model` is why this study needs **vLLM pack >= 1.10.0** — 1.9.1's `spec_method`
domain does not contain it. The drafter reference itself is deliberately **not** an Akamas
parameter: like the target model, it is a study constant rendered by
`k8s/01-deployment_template.yaml`, and `apply_config.sh` keeps or drops the
`--spec-model` flag depending on the method.

### OPEN DECISION: the target model is probably still wrong

Recorded 2026-09-22 after an adversarial review, and **not yet acted on** — the choice is
the team's.

The 7B was swapped out partly on an argument that is simply backwards. Weight bytes read
per batch-1 decode step, computed from each `config.json` and cross-checked against
HuggingFace's own reported tensor totals:

| Model | Weights on card | Read per decode step |
|---|---|---|
| Qwen2.5-7B-Instruct bf16 *(replaced)* | ~15 GB | **~15.2 GB** |
| **Qwen3-30B-A3B-Instruct-2507-FP8** *(current)* | 29.03 GiB | **~3.35 GB** |
| Qwen3-32B-FP8 *(dense candidate)* | 34.32 GB | **~32.8 GB** |
| Qwen3-14B-FP8 *(dense candidate)* | 16.33 GB | **~14.8 GB** |

Only 8 of 128 experts are active per token, so the current MoE reads about a **quarter**
of what the 7B read. The swap made the verifier step ~4.5× **cheaper**, leaving *less* for
speculation to amortise, not more.

It is worse than neutral. When the target verifies K+1 drafted tokens in one pass, each
token routes independently, so the pass reads the **union** of the experts they touch — up
to `min(128, 8·(K+1))` per layer. At `spec_tokens` 8 that is up to 72 of 128 experts,
roughly 9× the expert traffic of a single-token step, a cost a dense target does not pay
at all. **A sparse MoE is close to the worst possible target for showing speculative
decoding work.**

**`Qwen3-32B-FP8` is the better vehicle** and costs little to switch to: dense, 34.32 GB of
weights on a 96 GB card leaving ~55 GB of KV, ~32.8 GB read per decode step (about 10× this
model), `vocab_size` 151936 identical to `Qwen3-0.6B` so the **same drafter works
unchanged**. Switching means editing the model name, the served-model-name, the AIPerf
model/tokenizer and generating a new ShareGPT cache file; the Akamas study itself does not
change, since the model is not a tuned parameter.

**The counter-argument for keeping the MoE**, which is why this is a decision and not a
fix: if the model the team actually intends to serve in production is a sparse MoE, then
"speculative decoding does not pay on our model" is the true and useful answer, and
measuring it on a dense model would be measuring someone else's question. Study 15/16
comparability is a second, weaker reason to keep it.

What is NOT in doubt either way: the two reasons that survive are that the 7B used 15 of
96 GB, and that this model needed four L4s in studies 15/16 and fits one card here.

### Why this study exists now

Study 16's analysis (2026-09-21) found its best configurations pinned against the GPU's
power cap with the **memory bus only ~48% active and tensor cores ~11%** — spare compute
sitting idle on a memory-bound decode. Speculative decoding is the classic lever for
exactly that shape: it spends idle compute to reduce memory traffic per emitted token.
This study tests the lever on the hardware where it is easiest to isolate — one GPU, no
parallelism, a model that fits with room to spare.

## Stack & versions

- **Akamas version:** 3.7.x
- **Optimization packs:** vLLM **1.10.0 REQUIRED** (1.9.1 lacks `spec_method`'s
  `draft_model` category; the change is on branch `feature/speculative-decoding-metrics`
  in the pack's own repo, committed and not pushed), GPU **1.2.0**, Kubernetes
  **1.9.0-dev** —
  **read from the local pack checkouts, NOT confirmed on the instance.**
  `akamas list optimization-pack` returns `Access forbidden … requires the
  'Administrator' role` for this account. Verify before creating the system. Study 16
  recorded Kubernetes 1.8.0-dev as installed, so that one in particular may differ.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  **`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`**, served as `qwen3-30b-a3b`, namespace
  `llm-serving`, with **`Qwen/Qwen3-0.6B`** as the drafter for the `draft_model` cells.
  Pinned flags: `--enable-mfu-metrics`, `--no-enable-prefix-caching`.
  **Model changed from study 2's Qwen2.5-7B-Instruct** (2026-09-21). The 7B was
  inherited, not chosen: 15 GB of a 96 GB card, and a cheap decode, which is the worst
  case for showing speculation work. The 30B MoE is also the exact model of studies
  15/16 — there it did not fit one 24 GB L4 and forced a whole topology study; here it
  fits one card with ~60 GB to spare, so topology is constant at 1 and only speculation
  varies. **Image changed from v0.22.0** for the pack's reference version and because
  the V1 speculative counters do not exist on 0.22.0. Both changes mean this study is
  not comparable with 2-Larger-goodput's numbers and imports nothing from it.
- **Weights and memory:** 29.03 GiB of FP8 weights (measured in study 15) on a 97 887 MiB
  card, leaving roughly 50-60 GiB of KV cache depending on `gpu_memory_utilization`. At
  96 KiB/token (study 15's measured figure for this architecture) that is over 500 k
  tokens of cache, so unlike studies 15/16 this study is **not** KV-bound — which is why
  KV usage and preemption were dropped from the KPIs.
- **Cluster / hardware:** AWS `us-east-2`, EKS cluster `vllm-bench`, node group
  `llm-serving-g7e` = 1× `g7e.4xlarge` (1× NVIDIA RTX PRO 6000 Blackwell Server Edition,
  96 GB GDDR7, SM120, no MIG). Provisioning in `infra/`, duplicated from study 2 per this
  repo's atomic-per-study convention. **The node group already exists in the live cluster
  and is Active, with 0 nodes** (confirmed 2026-09-21) — so this is a scale-up, not a
  provisioning run. The CPU node the load generator runs on is `system-m8a` in the live
  cluster, not the `system` group study 2's `infra/eks/cluster.yaml` declares; the Job and
  the `cluster_loadtest` component follow the live cluster.
- **Load generator:** NVIDIA AIPerf 0.11.0, ShareGPT replay, closed-loop concurrency ramp
  of **9 levels, 8 → 2048** (doubling), 300 s each,
  `--goodput time_to_first_token:1500 inter_token_latency:300`. 45 min of load per trial.
  The **floor** moved down from study 2's 150 because a sweep starting there observes only
  the saturated regime where speculation is expected to lose — it would answer this
  study's question by construction of the load pattern. The **ceiling** moved up from 1024
  because study 16 served this same model and ended with 10 experiments scored at its
  ramp's last level, ITL p95 at 284-291 ms against a 300 ms SLA: their real capacity was
  never measured. The dataset cache is now **per model** (`inputs-qwen3-30b-a3b.json`,
  study 16's convention); that file already exists on the shared volume from 2026-09-18,
  so nothing needs regenerating.
- **Timeouts:** raised to study 16's proven values for this checkpoint — 1800 s rollout
  deadline and startup probe (not the 7B's 1200 s), 1740 s rollout wait, Akamas tasks at
  60 m / 95 m. 29.03 GiB of FP8 weights do not load cold in a 7B's budget.
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
| **`vLLM.spec_method`** | **none / ngram / ngram_gpu / draft_model** | **none (by absence)** |
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

## Initial design: 11 preset/baseline experiments, then optimize

| Step | `spec_method` | `spec_tokens` | Purpose |
|---|---|---|---|
| baseline | *(unrendered → off)* | — | vLLM's own stock defaults, `gmu` 0.90 |
| S1 | none | 0 | speculation off **at the grid's settings** — the reference cell |
| S2 / S3 / S4 | ngram | 3 / 5 / 8 | CPU n-gram lookup, token-depth sweep |
| S5 / S6 / S7 | ngram_gpu | 3 / 5 / 8 | GPU n-gram lookup, same depths |
| S8 / S9 / S10 | draft_model | 3 / 5 / 8 | **Qwen3-0.6B drafter**, same depths |
| optimize | free | free | 60 experiments, `numberOfInitExperiments: 0` |

Everything outside the two speculative parameters is held fixed across S1-S10
(`gmu` 0.90, `max_num_seqs` 256, `max_num_batched_tokens` 8192, `kv_cache_dtype` auto,
`balanced`, `optimization_level` 2, `block_size` 16, `TRITON_ATTN`, `enforce_eager` false,
`fcfs`, `disable_cascade_attn` false, `tokenizer_mode` auto, `async_scheduling` false,
`max_cudagraph_capture_size` 512), so the only difference between "ngram at 5 tokens" and
"draft_model at 5 tokens" is which drafter produced the proposal.

`optimization_level` 2 and `async_scheduling` false are forced by constraints 6 and 7.
`async_scheduling` is held false in **all ten** cells, including the `ngram_gpu` and
`draft_model` ones that could legally set it true, so it never becomes a hidden second
variable between the families.

`numberOfInitExperiments: 0` is mandatory: any value above 0 makes the campaign service
run its own Sobol bootstrap on top of the presets, landing back on the deterministic head
of the Sobol sequence that aliased two categorical parameters perfectly in study 13 (see
`ROADMAP.md` section C).

**Budget.** Roughly 60-75 minutes per experiment now (45 min of load plus rollout and a
29 GiB weight load). The **11 baseline/preset experiments are about 12 hours** and answer
the study's core question on their own; the 60-experiment optimize step is another ~65
hours on top. Study 16 was stopped early on cost at 27 optimizer experiments — decide the
budget before starting, and note the optimize step can be stopped at any point without
losing what already ran.

## Known gap: acceptance rate is not measurable until the pack ships

The study tunes `spec_method` and `spec_tokens` and will see their effect on throughput
and latency, but **not the acceptance rate** — the one number that explains the effect,
and the one that separates "wrong drafter for this traffic" from "right drafter, wrong
operating point". Those two readings lead to completely different follow-up studies.

Akamas telemetry may only map metrics the component type declares, and vLLM pack 1.9.1
declares none for speculative decoding. **The pack change is written**: branch
`feature/speculative-decoding-metrics` adds five metrics (`spec_decode_drafts_rate`,
`spec_decode_draft_tokens_rate`, `spec_decode_accepted_tokens_rate`,
`spec_decode_acceptance_rate`, `spec_decode_accepted_tokens_per_draft`) built on the four
real V1 counters, and `draft_model` alongside them. It is committed locally and not
pushed; see prerequisite 1.

The five entries sit commented out at the end of `akamas/telemetry/prometheus.yaml` with
their PromQL and that branch's exact metric names, so enabling them is mechanical. Two
traps are documented there: they are **absent rather than zero** when speculation is off,
so they need `defaultValue: 0` and must never appear in a goal formula or constraint; and
`spec_decode_accepted_tokens_per_draft` is vLLM's logged "mean acceptance length"
**minus one**, because that figure includes the always-emitted bonus token.

Note also that the five speculative metrics named in vLLM's own documentation
(`spec_decode_draft_acceptance_rate`, `spec_decode_efficiency`,
`spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens`,
`spec_decode_num_emitted_tokens`) are **V0-engine legacy and are not emitted by the V1
engine in 0.29.0** — verified 2026-09-21 against `vllm/v1/spec_decode/metrics.py` at the
`v0.29.0` tag. Wiring those names produces permanently empty series rather than an error.
Do not copy them from the docs page.

## Prerequisites still open

1. **BLOCKED — install vLLM optimization pack 1.10.0.** `spec_method`'s `draft_model`
   category does not exist in the installed 1.9.1, so `akamas create study` fails against
   it. The change is committed on branch `feature/speculative-decoding-metrics` in
   `~/akamas/offline/optimization-packs/vllm` (two commits: the five acceptance metrics,
   then `draft_model`), the pack's own test suite passes, and **nothing has been pushed**.
   It needs a push, a merge request, a build, and an install — and **the install requires
   the Administrator role, which the account used here does not have**
   (`akamas list optimization-pack` → "Access forbidden"). This is the one hard blocker.
2. **Then delete and recreate the study.** A study named
   `17-G7e-Speculative-Decoding-Goodput` was created on 2026-09-21 in its earlier 7B,
   seven-preset form (id `29fa60c3-8a96-4a00-9f66-ea05875c9f68`, 0 experiments run).
   `parametersSelection` and `steps` cannot be edited on an existing study on Akamas 3.7,
   so it must go:
   `akamas delete study "17-G7e-Speculative-Decoding-Goodput"` then
   `akamas create study .../17-G7e-Speculative-Decoding-Goodput.yaml`.
   The system, the six components and the workflow are already created and unaffected.
3. **Recreate the telemetry instance** if the five speculative metrics should be
   collected — they are commented out at the end of `akamas/telemetry/prometheus.yaml`
   and become valid only once 1.10.0 is installed. There is no update verb for a
   telemetry instance on 3.7, so it is a delete and recreate, and it must happen **before**
   the study starts, not halfway through.
4. **Smoke-test the drafter pairing by hand, once.** vLLM checks drafter/target vocabulary
   compatibility at engine init. `Qwen/Qwen3-0.6B` and
   `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` are both Qwen3 with a 151936-token vocabulary,
   but that is reasoned, not observed. Start the pair manually before spending three
   experiments on S8-S10. While there, also confirm `TRITON_ATTN` and `FLASHINFER` start
   on this GPU's SM120 compute capability — study 2 flagged SM120 kernel maturity as an
   open question and never resolved it.
5. **Bring the GPU node back up.** The node group `llm-serving-g7e` exists and is Active;
   it ran a node on 2026-09-21 (`ip-192-168-9-238`, us-east-2b) which was then scaled back
   to 0 to stop the spend. Scale it to 1 the morning the study runs:
   `eksctl scale nodegroup --cluster vllm-bench --name llm-serving-g7e --nodes 1`, or the
   EKS console. Note AWS had **no g7e.4xlarge capacity in either us-east-2 AZ** for a
   while that day, with 42 failed launches; if it recurs, the sibling sizes
   `g7e.2xlarge`/`g7e.8xlarge` carry the identical single RTX PRO 6000 and may have
   capacity when `4xlarge` does not (they need a new node group, and `2xlarge`'s 8 vCPUs
   need the Deployment's CPU request lowered).
6. **The DCGM exporter follows the node.** It is Helm release `dcgm-exporter` in
   `monitoring`, re-pointed at whichever GPU node group is in use — revision 19 targets
   `llm-serving-g7e` and was verified reporting `gpu="0"`, model name
   "NVIDIA RTX PRO 6000 Blackwell Server Edition", 97 887 MiB. Do **not** try to install a
   second release beside it; the chart hardcodes a ConfigMap name and refuses. Nothing to
   redo when the node returns — the DaemonSet schedules itself.
7. **`akamas/id_rsa` is in place on the toolbox** (copied 2026-09-21, mode 600, the same
   key every study shares). Not committed, per `.gitignore`.
8. **The ShareGPT cache is ready**: `inputs-qwen3-30b-a3b.json` has been on the
   `aiperf-results` volume since 2026-09-18, generated by study 16 for this exact
   served-model-name. No regeneration, and no risk of the stale-cache 404 storm that cost
   studies 2 and 13 a run each.
9. **Re-calibrate the ramp after the baseline.** 8 → 2048 is a reasoned guess for this
   card, not a measurement. If the last level is still SLA-compliant with headroom, raise
   the ceiling again rather than repeating study 16's mistake.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
