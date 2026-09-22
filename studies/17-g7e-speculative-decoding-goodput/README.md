# 17-g7e-speculative-decoding-goodput

**Status:** TODO — configuration complete, not yet started.
**Dates:** Scaffolded 2026-09-21, model swapped 2026-09-21, **hardware and model changed
again 2026-09-22** (see the box below).

> ### The card changed on 2026-09-22, and most of this file predates that
>
> This study was built for a single **RTX PRO 6000 Blackwell** (96 GB, SM120) on a
> `g7e.4xlarge`. AWS ran out of them: 56+ consecutive `InsufficientInstanceCapacity`
> failures in one day, a second node group on `g7e.8xlarge` rolled back with the same
> error, a third pinned to `us-east-2a` reached `CREATE_FAILED` after 34 attempts. A
> direct capacity probe then found **every GPU class above 24 GB per GPU empty in all
> three `us-east-2` zones** — g7e, g6e/L40S and `g6.12xlarge` alike.
>
> The study now runs on **one NVIDIA L4** (`g6.4xlarge`, 23034 MiB, Ada/SM89, ~300 GB/s,
> 72 W cap, 1.32 USD/h) serving **`Qwen/Qwen3-8B-FP8`**, because `Qwen/Qwen3-32B-FP8`
> needs 30.5 GiB of weights and does not fit. The **name is now a misnomer** and is kept
> on purpose: renaming would mean rewriting nine hardcoded workflow paths, the toolbox
> checkout and every already-created Akamas resource.
>
> **The authoritative files are `akamas/` and `k8s/`, which are current.** Sections of
> this README below still argue from the 96 GB card; they are kept because the reasoning
> is what justified each choice, but where they conflict with `akamas/README.md`, that
> file wins. `study-recap` will rewrite this one when the study closes.

> Started as a duplicate of `2-larger-model-g7e`'s **"2-Larger-goodput"** — same goal,
> SLA, windowing and 14 tuned parameters — with **speculative decoding** as the new
> dimension. It has since diverged on five points (model, image, ramp, timeouts, and the
> domains re-derived for 22.03 GiB usable), each recorded below, so it imports nothing from that
> study and runs its own baseline. Telemetry carries 131 metrics: study 16's 110 plus 16
> the installed packs declare that no study had ever collected, plus the 5
> speculative-decoding acceptance metrics that pack 1.10.1 now ships.

## Objective

Find out whether **speculative decoding earns its place** on a single GPU serving a dense
model under an interactive-chat SLA, and if so, with which method and how many speculative
tokens. The card is now an L4 rather than the 96 GB Blackwell, which sharpens the question
rather than blunting it: at ~300 GB/s the decode step is deeper into the
memory-bandwidth-bound regime speculation exists to exploit.

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

1. **Ten preset steps run the whole grid head-to-head** (`none`, `ngram` × 3/5/8,
   `ngram_gpu` × 3/5/8, `draft_model` × 2/3/4 — the last family uses lower draft lengths
   because its drafter is 1.11 GiB against an 8.79 GiB target and carries its own KV
   cache) at otherwise-identical settings, before the optimizer gets to
   choose anything. S1 is speculation off *at those same settings*, so the six
   speculative cells have a proper reference rather than being compared against a
   baseline that differs in a dozen other ways.
2. **The concurrency sweep was re-floored from 150 → 8.** Study 2's ramp started at 150
   concurrent requests, which observes only the regime where speculation is expected to
   lose; a study using it would answer its own question by construction of the load
   pattern. The sweep is now 8 levels, 1 → 128, doubling — the ceiling raised too,
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

### The target model, and why it changed twice

Settled 2026-09-22 on **`Qwen/Qwen3-32B-FP8`**, dense. The route there is worth recording,
because the middle step was taken on an argument that was simply backwards.

| Model | Weights on card | Read per decode step |
|---|---|---|
| Qwen2.5-7B-Instruct bf16 *(inherited from study 2)* | ~15 GB | ~15.2 GB |
| Qwen3-30B-A3B-Instruct-2507-FP8 *(2026-09-21, retracted)* | 29.03 GiB | **~3.35 GB** |
| **Qwen3-32B-FP8 *(final)*** | **34.32 GB** | **~32.8 GB** |

The MoE was chosen partly because "a 7B decode is cheap, the worst case for showing
speculation work". True premise, wrong conclusion: only 8 of 128 experts are active per
token, so that model read **a quarter** of what the 7B read. The swap made the verify step
~4.5× cheaper and left *less* for speculation to amortise.

It was worse than neutral. Verifying K+1 drafted tokens routes each token independently,
so the pass reads the **union** of the experts they touch — up to `min(128, 8·(K+1))` per
layer, roughly 9× single-token expert traffic at `spec_tokens` 8. A dense target pays none
of that: its verify cost is flat in draft length. **A sparse MoE is close to the worst
possible target for showing speculative decoding work.**

`Qwen3-32B-FP8` fixes both halves: dense, so verify cost is flat; ~32.8 GB read per step,
about 10× the MoE, which is the quantity speculation amortises; 34.32 GB of weights on a
95.59 GiB card leaving ~50 GiB of KV; and `vocab_size` 151936 identical to the drafter, so
**`Qwen/Qwen3-0.6B` carries over unchanged**.

Two costs, stated plainly. Comparability with studies 15/16 is gone — they served the MoE.
And this is a **hybrid-thinking** checkpoint, which the previous one was not.

### The thinking-mode trap, and the one flag that closes it

Qwen3-32B emits chain-of-thought by default. Its chat template only suppresses reasoning
when the caller passes `enable_thinking=false`, and the guard is

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
```

so **undefined is not false**. AIPerf sends plain OpenAI chat-completions with no
`chat_template_kwargs`, so every request would open its own `<think>` block; the model card
recommends a 32 768-token output budget in thinking mode. The benchmark would have measured
reasoning length, not ShareGPT replay, and the numbers would have looked like a catastrophic
regression against every prior study.

The deployment therefore passes:

```
--default-chat-template-kwargs '{"enable_thinking": false}'
```

That is the only server-side option in vLLM 0.29 that stops the tokens being **generated**.
`--reasoning-parser qwen3` and a request's `include_reasoning: false` merely move the text
out of `content` while the GPU still pays for it — and if the load generator counted output
tokens from `content` rather than from `usage`, that would under-report output length while
throughput stayed depressed. Request-level kwargs would override the server default, and
AIPerf sends none, so it holds for every request.

**Pre-flight gate**: after the server starts, fire one request by hand and confirm both that
`content` has no `<think>` and that `usage.completion_tokens` is plausibly short. That single
request is what verifies the deployed build matches the documentation.

### Why this study exists now### Why this study exists now

Study 16's analysis (2026-09-21) found its best configurations pinned against the GPU's
power cap with the **memory bus only ~48% active and tensor cores ~11%** — spare compute
sitting idle on a memory-bound decode. Speculative decoding is the classic lever for
exactly that shape: it spends idle compute to reduce memory traffic per emitted token.
This study tests the lever on the hardware where it is easiest to isolate — one GPU, no
parallelism, a model that fits with room to spare.

## Stack & versions

- **Akamas version:** 3.7.x
- **Optimization packs:** vLLM **1.10.1, INSTALLED** — confirmed indirectly on 2026-09-22,
  since the telemetry instance exists on the instance and maps the five `spec_decode_*`
  metrics only >= 1.10.0 declares, which it could not otherwise have been created with.
  GPU **1.2.0**, Kubernetes **1.9.0-dev**, both still **read from the local pack
  checkouts and NOT confirmed on the instance**: `akamas list optimization-pack` returns
  `Access forbidden … requires the 'Administrator' role` for this account. Study 16
  recorded Kubernetes 1.8.0-dev as installed, so that one in particular may differ.
  The vLLM pack's source branch `feature/speculative-decoding-metrics` is still committed
  locally and unpushed — a debt, since the instance runs a version its repo has no record
  of.
- **Workload under test:** `vllm/vllm-openai:v0.29.0` serving
  **`Qwen/Qwen3-8B-FP8`** (dense), served as `qwen3-8b`, namespace `llm-serving`, with
  **`Qwen/Qwen3-0.6B`** as the drafter for the `draft_model` cells (same 151936 vocabulary).
  Pinned flags: `--enable-mfu-metrics`, `--no-enable-prefix-caching`,
  `--max-model-len=4096` (a literal, not a rendered token — see `k8s/` for why that
  distinction cost a startup failure), and the mandatory
  `--default-chat-template-kwargs '{"enable_thinking": false}'`.
  **Model changed from study 2's Qwen2.5-7B-Instruct** (2026-09-21). The 7B was
  inherited, not chosen: 15 GB of a 96 GB card, and a cheap decode, which is the worst
  case for showing speculation work. The 30B MoE is also the exact model of studies
  15/16 — there it did not fit one 24 GB L4 and forced a whole topology study; here it
  fits one card with ~60 GB to spare, so topology is constant at 1 and only speculation
  varies. **Image changed from v0.22.0** for the pack's reference version and because
  the V1 speculative counters do not exist on 0.22.0. Both changes mean this study is
  not comparable with 2-Larger-goodput's numbers and imports nothing from it.
- **Weights and memory (re-derived 2026-09-22 from each HF repo's own config.json and
  safetensors headers):** 8.79 GiB of FP8 weights plus 1.11 GiB of drafter on a 22.03 GiB usable
  card. KV costs 144 KiB/token for the target alone and **256 KiB/token once the drafter
  is loaded**, because vLLM 0.29.0 gives the draft model its own KV cache. At
  `gpu_memory_utilization` 0.88 that leaves roughly 8 GiB of pool, on the order of 32 k
  tokens, i.e. around 80 concurrent ShareGPT requests before the scheduler starts
  preempting. So this study **is** KV-bound after all, the opposite of what the 96 GB
  version assumed — which is why `vLLM.kv_cache_usage_avg` came back into the KPIs in
  place of GPU memory bandwidth.
  Rejected alternatives, for the record: **Qwen3-14B-FP8** leaves ~1.7 GiB with the
  drafter and fails vLLM's one-request KV-fit check outright; **Qwen3-4B-FP8** has
  identical layers and KV heads to the 8B so KV costs the same while the verify pass reads
  half as much, making the drafter ~27% of a target step and any null result
  unattributable between "speculation does not help" and "this drafter is too big".
- **Cluster / hardware:** AWS `us-east-2`, EKS cluster `vllm-bench`, node group
  **`llm-serving-l4-single` = 1× `g6.4xlarge`** (1× NVIDIA L4, 23034 MiB, Ada/SM89,
  ~300 GB/s, 72 W power cap, no MIG), created 2026-09-22 and up first try in
  `us-east-2a`. Provisioning in `infra/`; the capacity workarounds and the full
  alternatives comparison are in `infra/README.md` and
  `infra/eks/gpu-capacity-fallback.sh`. The `llm-serving-g7e` group still exists at
  desiredSize 1 and still fails; if its capacity ever returns, note that **only one GPU
  node may be scraped at a time** — every DCGM query filters on `pod` and `gpu`, never on
  a node label, so two exporters would be averaged together. The CPU node the load generator runs on is `system-m8a` in the live
  cluster, not the `system` group study 2's `infra/eks/cluster.yaml` declares; the Job and
  the `cluster_loadtest` component follow the live cluster.
- **Load generator:** NVIDIA AIPerf 0.11.0, ShareGPT replay, closed-loop concurrency ramp
  of **8 levels, 1 → 128** (doubling), 300 s each,
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
| `vLLM.gpu_memory_utilization` | **[0.80, 0.90]** | **0.85** (the only explicitly rendered one) |
| `vLLM.max_num_seqs` | **[16, 128]** | *(vLLM default)* |
| `vLLM.max_num_batched_tokens` | **[256, 4096]** | *(vLLM default)* |
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
| **`vLLM.spec_tokens`** | **[0, 8]**, 0 = off sentinel | **0** |

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

### Constraints (9)

> All four numeric domains above were **re-derived on 2026-09-22** for a 22.03 GiB usable card;
> the values they replace were sized for 96 GB. The constraint count went from 7 to 9 in
> the same change: the sampler-warmup guard came back (the manifest had removed it with an
> explicit "re-add it if the card or the vocabulary changes"), joining the
> `max_num_batched_tokens >= max_num_seqs` rule added just before it.

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
`balanced`, `optimization_level` 2, `block_size` 16, `FLASH_ATTN`, `enforce_eager` false,
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

## ~~Known gap: acceptance rate is not measurable until the pack ships~~ — CLOSED 2026-09-22

> Resolved. Pack 1.10.1 is installed and the four V1 counters were read live off the
> server during the smoke test below, with a 58% acceptance rate on the S8 cell. The
> section is kept because its analysis of which metric names are real and which are V0
> legacy is still the reason the right ones were wired.

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

## Smoke test, 2026-09-22 — run, and what it settled

Run on the live L4 with the **S8 cell** (`draft_model`, 2 speculative tokens) — chosen
because it is the riskiest of the twelve steps: it is the only one that loads a second
model, and the one whose memory budget the arithmetic said was tightest. `apply_config.sh`
exited 0 and vLLM reached `Application startup complete`.

**Four things that were reasoned before and are now observed.**

1. **Every flag rendered correctly.** vLLM's own `non-default args` line reports
   `model='Qwen/Qwen3-8B-FP8'`, `served_model_name=['qwen3-8b']`, `max_model_len=4096`,
   `attention_backend='FLASH_ATTN'`, `gpu_memory_utilization=0.88`, `max_num_seqs=128`,
   `spec_method='draft_model'`, `spec_model='Qwen/Qwen3-0.6B'`, `spec_tokens=2`, and
   `default_chat_template_kwargs={'enable_thinking': False}`. The boolean-flag rewriting
   and the speculative-flag branching in `apply_config.sh` both work on this config.
2. **The thinking-mode trap is closed.** A real chat request ("what is the capital of
   France, answer in one sentence") returned 11 completion tokens, no `<think>` in the
   content and an empty `reasoning_content`. Without the flag this benchmark would have
   measured chain-of-thought length.
3. **The drafter pairing holds, and the acceptance metrics work.** After five generation
   requests, `/metrics` reports `vllm:spec_decode_num_drafts_total` 584,
   `..._num_draft_tokens_total` 1168 (exactly 2 per draft, as configured),
   `..._num_accepted_tokens_total` 682 and the per-position counter at positions 0 and 1.
   That is a **58% acceptance rate**, and it is the whole chain the pack change existed
   for, working end to end.
4. **The drafter inherits the context limit.** `Overriding draft model max model len from
   40960 to 4096` — so pinning `--max-model-len` as a literal fixes it for both models.

**The memory numbers, measured rather than estimated.**

| | |
|---|---|
| Card total, as vLLM sees it | **22.03 GiB** |
| Requested at `gpu_memory_utilization` 0.88 | 19.39 GiB |
| Weights + non-torch | 11.36 GiB |
| Peak activation | 1.91 GiB |
| CUDA graph | 0.77 GiB |
| **KV cache** | **6.12 GiB = 25,072 tokens** |

Two consequences. First, **study 16's constant of 22.03 is exactly right for this card** —
vLLM reports the same figure — so the re-added sampler-warmup guard is calibrated on a
measurement, not a guess. Second, the KV pool is 6.12 GiB against the ~8 GiB the
arithmetic predicted; the reserve was underestimated. The 1→128 ramp still stands but the
saturation knee will be lower than expected.

**One thing left on the table.** vLLM reports that up to 7.67 GiB of KV would "fully
utilize gpu memory", i.e. roughly a quarter more of the resource that binds this study,
and separately that CUDA-graph profiling alone costs the equivalent of 0.033 of
`gpu_memory_utilization`. The preset grid captures CUDA graphs for batch sizes up to 512
while `max_num_seqs` is 128, so about half the captures can never be used. Whether
lowering `max_cudagraph_capture_size` recovers that is being measured.

**Not reproduced, and worth knowing:** an `async_llm.py` traceback appeared at the exact
moment the rollout replaced the pod, consistent with the engine being terminated
mid-request rather than with a serving fault. The pod's logs were gone by the time it was
investigated, so it is recorded rather than explained.

## Prerequisites still open

1. **Recreate the Akamas resources.** Not a blocker, just an ordering requirement: the
   system, six components, telemetry instance, workflow and study were all created on
   2026-09-21 against the RTX PRO 6000, and Akamas 3.7 has no update verb for a component,
   a telemetry instance, or anything in a study but its `goal`. The exact delete/recreate
   sequence is in [`akamas/README.md`](akamas/README.md). The study is status `CREATED`
   with zero experiments, so nothing is at risk.
2. **~~Install vLLM pack 1.10.0~~ — DONE.** 1.10.1 is installed; confirmed because the
   telemetry instance exists on the instance and maps the five `spec_decode_*` metrics that
   only >= 1.10.0 declares. The pack's source branch
   `feature/speculative-decoding-metrics` is still committed-locally and unpushed, which is
   a debt against the pack's own repo rather than against this study.
3. **~~Smoke-test the drafter pairing~~ — DONE**, see the section above.
4. **Scale the node group back to 0 when not running.** `llm-serving-l4-single` costs
   1.32 USD/h and the study does not need it between runs.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
