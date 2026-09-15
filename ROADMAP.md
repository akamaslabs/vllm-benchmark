# ROADMAP — living plan across studies

This file holds what cuts **across** studies — what to try next. For a factual recap of
what's already been built and observed, see `studies/README.md` instead. What's specific
to one study — its exact stack, versions, parameters, and results — lives in that
study's own `studies/<name>/README.md`, not here (see `README.md` on why studies are
self-contained). Update this file:
- whenever a cross-study hypothesis is born, confirmed, or retired (section A);
- whenever a study is decided, started, or completed (section B);
- at the end of a study, via the `study-recap` skill (section C);
- when a backlog study's real-world setup (GPU/instance, load generator, dataset) is
  decided or changes (section D);
- whenever a pack gap, tooling evaluation, or other non-study action is identified or
  resolved (section E);
- when a study idea is deprioritized (not abandoned) in favor of a sharper near-term
  plan, or picked back up from that list (section F).

---

## A. Hypotheses / open questions

> All hypotheses below are **TO BE CONFIRMED**: they come from general vLLM-tuning
> literature and reasoning, not from measured results in this repo yet. They're
> intentionally stated without hardware/model specifics — those belong to whichever
> study actually tests them, in that study's own README.

- **H1 — `max_num_batched_tokens` drives a TTFT/throughput trade-off** [TO BE CONFIRMED —
  not cleanly testable by study #1] `studies/0-explorative` searched this jointly with 15
  other parameters via Akamas' optimizer (not a controlled single-variable sweep), and its
  goal metric is throughput-only (no TTFT/latency term) — across its 80 finished trials,
  `max_num_batched_tokens` showed essentially **zero raw correlation with the throughput
  goal** (Pearson r ≈ 0.02). That's not evidence against the trade-off itself (the goal
  formula can't see TTFT), just a note that this study's design can't confirm or refute
  H1 as stated — a future study would need either a TTFT term in its goal/constraints or
  a dedicated single-parameter sweep to test this properly. Range cross-check unchanged:
  independent sources give different ranges depending on model/hardware —
  `knowledge/notes/2026-07-vllm-recipe-llama3-3-70b.md` baselines 8192 (up to 16384) for
  Llama-3.3-70B on B200/H100/H200, while `knowledge/notes/2026-07-auto-tune-vllm.md`'s
  own example configs search 1024–8192 or 2048–16384 — confirms a study's domain should
  be derived from its own model/hardware, not copied from either source.

- **H2 — `gpu_memory_utilization` shows diminishing returns past some threshold**
  [PARTIALLY CONFIRMED, within a pre-narrowed range — see `studies/0-explorative/README.md`]
  That study narrowed this parameter's domain to `[0.85, 0.95]` for OOM-avoidance reasons
  (not to test H2), so the full `[0,1]` range was never explored — but *within* that
  narrow, already-high band, `gpu_memory_utilization` showed only a weak correlation with
  the throughput goal (Pearson r ≈ 0.15) across 80 finished trials, consistent with H2's
  "diminishing returns" shape near the top of the range, though not a strong enough
  signal to call this conclusively settled. The alternative study-design pattern below
  (calibrate once, hold fixed) looks more attractive given this weak effect — a future
  study could reasonably spend less optimizer budget on this dimension. Range
  cross-check unchanged: two independent sources
  (`knowledge/notes/2026-07-practical-vllm-performance-tuning.md`'s manual "push toward
  0.95" guidance and `knowledge/notes/2026-07-auto-tune-vllm.md`'s own 0.85–0.95 search
  range) both center on a narrow band above vLLM's 0.9 default rather than the full [0,1]
  domain. Alternative study-design pattern worth considering, per
  `knowledge/notes/2026-07-vllm-official-auto-tune-script.md` (vLLM's own upstream
  `auto_tune.sh`): rather than searching `gpu_memory_utilization` jointly with
  `max_num_seqs`/`max_num_batched_tokens`, calibrate its safe ceiling once (highest
  value that avoids OOM for the model+hardware) and hold it fixed, scoping
  `parametersSelection` to just the other two parameters.

- **H3 — `max_num_seqs` saturates before its upper bound** [LEANS REJECTED for
  Qwen2.5-7B-Instruct / single A10G — see `studies/0-explorative/README.md`] Across that
  study's 80 finished trials, `max_num_seqs` showed a *positive* correlation with the
  throughput goal (Pearson r ≈ 0.36, no goal-metric plateau observed toward the domain's
  upper end: trials with `max_num_seqs` in the top half of `[16, 1024]` averaged ~13%
  higher throughput than the bottom half) — no clear saturation within the range tested,
  contrary to H3's premise. **But** the two experiments that failed with an OOM
  (`max_num_seqs` 917 and 1016, both near the domain ceiling, both paired with
  near-maximum `gpu_memory_utilization`) show this parameter does hit a *reliability*
  ceiling before it hits a *throughput* one — worth carrying into any future test of this
  hypothesis: check for a reliability cliff (crash rate), not just a throughput plateau,
  when raising `max_num_seqs`. This is one model/GPU pair's result, not a general claim —
  a future study on different hardware (more/less VRAM, different vocab size) should
  re-test rather than assume this transfers.
  Additional supporting evidence (see
  `knowledge/notes/2026-07-distributed-inference-advanced-deployment-patterns.md`):
  speculative decoding's throughput gain shrinks or inverts at large batch sizes, because
  an already-saturated decode fleet has little idle forward-pass capacity left for a
  draft model to exploit — another case of "more concurrency isn't purely additive"
  alongside this hypothesis and H2, even though this specific study's data leans against
  a throughput-side saturation for this stack.

- **H4 — co-tuning Kubernetes resource limits alongside vLLM parameters changes the
  optimum** [TO BE CONFIRMED] **Numbering note (2026-07-15)**: every "backlog #4" cross-
  reference below this hypothesis (and in Q6 and the autoscaling operating-practice
  note) predates Section D's 2026-07-15 rewrite around Goal A/B — backlog #4 now names
  the **MIG right-sizing** study (Section D, study #4), a related but distinct question
  from what H4 describes here. The generic "replica-count/HPA/autoscaling co-tuning"
  question H4 itself is about now lives in **Section F's "Kubernetes resource
  co-tuning"** futuribile item, not an active backlog slot — read every "see backlog #4"
  below as pointing there, not at the current study #4. Akamas can tune Kubernetes-level
  parameters (container CPU/memory requests/limits, HPA settings, ...) in the same
  study as vLLM parameters.
  Whether jointly optimizing both finds a materially better configuration than tuning
  vLLM alone (with "reasonable" fixed k8s resources) is untested — worth a dedicated
  study once a vLLM-only baseline exists to compare against. A specific mechanism worth
  testing (see `knowledge/notes/2026-07-gpu-memory-bound-large-batch-inference.md`):
  deliberately under-allocating `gpu_memory_utilization`/`max_num_seqs` below the
  single-replica optimum frees GPU memory that a second (or third) vLLM replica on the
  same GPU can use, potentially beating a single maximally-tuned replica on aggregate
  throughput — this needs a Kubernetes-pack replica-count/HPA parameter tuned jointly
  with the vLLM ones, see backlog #4. For what that parameter should ideally represent,
  see `knowledge/notes/2026-07-nvidia-dynamo-planner.md`: NVIDIA Dynamo's Planner is a
  concrete example of an SLA-aware autoscaler that scales prefill and decode replica
  counts *independently* (compute-bound vs. memory/KV-bound) rather than one aggregate
  replica count — a single Kubernetes replica-count parameter may not be expressive
  enough for a disaggregated topology, though it's sufficient for this repo's current
  non-disaggregated studies. Strongest quantitative evidence yet for this hypothesis's
  underlying pattern, per `knowledge/notes/2026-07-gpu-fractioning-nvidia-runai.md`:
  NVIDIA Run:ai's GPU-fractioning benchmarks show two 0.5-GPU Llama-3.1-8B replicas
  reaching ~305K tokens/s combined vs. ~199K tokens/s for one full-GPU replica on the
  same hardware — the same "smaller allocation × more replicas beats one big
  allocation" shape as "Mind the Memory Gap," but via a **different mechanism**:
  scheduler-level GPU fractioning (Run:ai-specific, hard memory isolation across
  independent Pods) rather than vLLM-internal `gpu_memory_utilization` tuning within one
  process. If the target cluster has a GPU-sharing scheduler, the missing Kubernetes
  parameter needs a **GPU-fraction-per-replica** dimension (Request/Limit), not just a
  replica count — but this entire mechanism is conditional on that scheduler (Run:ai,
  MIG, time-slicing, or MPS) actually being installed on whatever cluster a study runs
  against; confirm before assuming it's available.

- **H5 — `tensor_parallel_size`'s optimum is a "minimum that fits," not a value worth
  sweeping freely** [TO BE CONFIRMED] Per
  `knowledge/notes/2026-07-distributed-inference-scaling-dimensions.md`, general
  guidance for dense models is: quantize first, then set the minimum TP degree that fits
  the model plus KV cache headroom on the target GPU, then scale further capacity via
  data parallelism (replicas) rather than higher TP — higher TP only pays off under a
  strict single-request TTFT SLO. If confirmed, a study tuning `tensor_parallel_size`
  should narrow its domain around that computed minimum instead of the pack's full
  range, and treat DP (replica count) as the throughput lever instead — ties into H4/
  backlog #4's replica-count question. A related but distinct angle, per
  `knowledge/notes/2026-07-practical-vllm-performance-tuning.md`: for a *fixed* GPU
  budget, the split between replica count and TP degree per replica (e.g. 16 GPUs as 4
  replicas × TP=2, 2 replicas × TP=4, or 1 replica × TP=8) is itself an empirical
  trade-off — not just "minimum TP, then add replicas," but "given N total GPUs, which
  TP/replica split wins," which also interacts with HA/scheduling-flexibility needs, not
  just throughput. Worth testing explicitly if a study ever has a fixed multi-GPU budget
  to allocate, distinct from backlog #4's single-GPU replica-packing question. Note per
  `knowledge/notes/2026-07-nvidia-dynamo-aiconfigurator.md`: if a future multi-GPU
  study's topology is decided externally (e.g. via a topology-planning tool) before the
  study starts, `tensor_parallel_size` may not need to be an Akamas-searched parameter
  at all for that study — it would already be pinned by the topology decision, and only
  per-instance parameters within that fixed topology would go into
  `parametersSelection`. Concrete numbers to sanity-check a future TP>1 study against,
  per `knowledge/notes/2026-07-generative-ai-on-kubernetes-gpu-production-patterns.md`:
  tensor-parallel communication overhead can consume **50-70% of inference time** if the
  interconnect is poorly partitioned (the reason it's recommended single-node-only, with
  NVLink/NVSwitch — cross-node bandwidth is roughly two orders of magnitude slower);
  within one node, ~4 GPUs deliver "approximately three and a half times" one GPU's
  throughput for a well-optimized model (near-linear but not perfectly so) — a future
  study measuring TP scaling efficiency below that ballpark should suspect interconnect
  or partitioning issues before concluding TP itself doesn't pay off. **Node-provisioning
  prerequisite to check before that study, per
  `knowledge/notes/2026-07-generative-ai-on-kubernetes-training-job-scheduling.md`**: the
  Kubernetes Topology Manager (Kubelet component, policies `none`/`best-effort`/
  `restricted`/`single-numa-node`) coordinates CPU/GPU NUMA locality — cross-NUMA-socket
  memory access costs roughly 3x the latency of local access, which can eat into TP's
  own communication budget on a multi-socket node if the pod's GPUs and pinned CPUs land
  on different sockets. Worth confirming the target node's Topology Manager policy (and
  that the GPUs assigned to a TP>1 pod are NUMA-local) as part of that study's setup, not
  something Akamas tunes — a provisioning check, like the NVLink/NVSwitch interconnect
  check already noted above. **Concrete GPU-count sizing formula to compute that
  "minimum that fits" from first principles**, per
  `knowledge/notes/2026-07-inference-engineering-techniques-quantization-speculation-parallelism.md`:
  `vram_required = (bits_precision / 8) × params_billions × kv_cache_allocation_factor`
  — round up to the nearest available instance size to get the minimum GPU count, then
  derive the minimum TP degree from that. The same source states Tensor Parallelism
  "should be your default strategy" for multi-GPU inference (supports both dense and MoE
  models), reinforcing this hypothesis's premise, and gives a firm general rule worth
  citing directly: "unless your model and KV cache are so large as to require multi-node
  inference, it probably isn't the best use of extra hardware — better off scaling
  replicas horizontally, or disaggregating."

- **H6 — `tensor_parallel_size` and `max_num_seqs` should not be analyzed as independent
  effects within a single replica** [TO BE CONFIRMED] Per
  `knowledge/notes/2026-07-vllm-recipe-llama3-3-70b.md` (vLLM-project's own Llama-3.3-70B
  recipe): raising TP shards the KV cache across more GPUs, freeing per-GPU headroom that
  can let `max_num_seqs` run higher than it could at lower TP — partially offsetting TP's
  own per-request throughput cost. If a study's `parametersSelection` searches both
  parameters together, its results analysis should look for this coupling (e.g. does the
  optimizer converge to a higher `max_num_seqs` at higher TP than at lower TP) rather than
  reporting each parameter's marginal effect as if the other were held constant. Distinct
  from H5 (TP vs. replica *count* across GPUs) — this is TP vs. batch size *within* one
  replica.

- **Q1 — Windowing**: for a given load-test duration, does `trim` windowing reliably
  capture steady state, or does a given study need `stability` windowing instead? Decide
  per study based on how variable that study's load generator's ramp-up is.
- **Q2 — Load generator choice**: studies so far have used GuideLLM; NVIDIA's
  GenAI-Perf/AIPerf is being evaluated as an alternative. This is a **per-study**
  decision, not a repo-wide default — document the actual tool + version in each study's
  README. Now backed by `knowledge/notes/2026-07-llm-inference-load-testing-tools.md`'s
  methodology comparison — concretely: GuideLLM and vLLM's own `benchmark_serving.py`
  are open-loop (arrivals scheduled independent of responses; `benchmark_serving.py`
  additionally supports a `--burstiness` gamma-distribution knob GuideLLM's poisson/
  constant split doesn't have), while AIPerf/GenAI-Perf keeps a fixed number of
  concurrent requests active continuously (closed-loop-style) and defines
  ITL/TPOT as `(e2e_latency − TTFT) / (total_output_tokens − 1)` — a different
  concurrency model *and* a different formula than GuideLLM's. If both tools ever get
  used across studies, don't compare raw numbers directly: record which tool, which
  rate-type/concurrency mode, and note that fixed-concurrency (AIPerf-style) runs can
  look artificially stable under saturation compared to true open-loop (GuideLLM
  `constant`/`poisson`, or `benchmark_serving.py`'s Poisson/gamma) runs at "the same"
  load. **A third candidate evaluated 2026-07-14: `kubernetes-sigs/inference-perf`**
  (https://github.com/kubernetes-sigs/inference-perf, wg-serving-affiliated, active —
  weekly-ish releases, v0.6.0 as of 2026-06). Unlike GuideLLM (open-loop only,
  synthetic fixed-shape prompts only) and AIPerf (closed-loop only), it supports
  **both** open-loop and closed-loop load, **plus real dataset/trace replay**: ShareGPT,
  CNN/DailyMail, BillSum, Infinity Instruct, VisionArena, Azure production-trace replay,
  full OpenTelemetry trace/DAG replay, and an explicit `shared_prefix` generator for
  prefix-cache-hit-rate testing. This directly closes the gap this repo's studies have
  had so far — every study to date (`0-explorative` included) used GuideLLM's synthetic
  fixed-shape prompts (`prompt_tokens=512,output_tokens=128`), which cannot produce
  realistic prefix-cache reuse or content-dependent behavior. It also reports a
  **Goodput** (SLO-constrained throughput) metric neither GuideLLM nor AIPerf has. It
  deploys the same way GuideLLM does today (a plain Kubernetes Job + ConfigMap, no
  Gateway API Inference Extension integration despite the `kubernetes-sigs` name) — a
  drop-in swap for the existing workflow task, not a re-architecture. One doc
  inconsistency flagged, not yet resolved: its own `docs/config.md` says `server.type`
  is "currently only vLLM supported" while the README claims broader OpenAI-compatible
  support — confirm directly against the CLI before relying on non-vLLM claims (not a
  concern for this repo, which only ever targets vLLM). **Decision (2026-07-15):
  committed directly to `inference-perf` with ShareGPT replay repo-wide**, superseding
  the earlier plan to first run a dedicated side-by-side validation study against
  GuideLLM — see Section D's intro for the current studies (#2-#4), all of which use
  `inference-perf` from the start. Full findings in
  `knowledge/notes/2026-07-kubernetes-sigs-inference-perf.md`.
- **Q3 — Optimization pack versions**: pack lifecycle (which parameters/metrics exist)
  is managed outside this repo. When a study starts, record in its README exactly which
  pack versions were installed at the time — packs can change independently of this repo.
  - **2026-09-14 — vLLM 0.22.0 is no longer enough; next study should pin
    `vllm/vllm-openai:v0.29.0`** (released 2026-09-09, amd64 image verified on Docker Hub).
    Checked against vLLM `v0.22.0` and `v0.29.0` sources while extending the vLLM pack after
    the vLLM x AgentX post (`knowledge/notes/2026-09-vllm-agentx-agentic-serving.md`):
    - `--prefill-schedule-interval` and `--watermark` exist only from **0.24.0**;
      `--max-num-queued-reqs`/`--max-num-queued-tokens` only from **0.29.0**.
    - **0.28.0 removed `--max-num-partial-prefills` and `--max-long-partial-prefills`**: the
      pack keeps both parameters (instance-wide definitions), so a 0.28+ study must drop the
      two flags in its deploy-rendering script (`apply_config.sh`-style) or vLLM fails at
      startup on an unknown argument. `--long-prefill-token-threshold` survives with the
      per-step-cap semantics the post describes, and 0.29 raises if it exceeds `max_model_len`.
    - Every categorical value the pack already declares (`attention_backend`, `kv_cache_dtype`,
      `tokenizer_mode`, `prefix_caching_hash_algo`, `performance_mode`, `scheduling_policy`) is
      still accepted by 0.29.0; the 37 `vllm:*` Prometheus series are unchanged.
    - Do **not** rewrite the historical studies' `v0.22.0` image pins (studies #0-#9 are
      factual records); the upgrade applies to whatever study is scaffolded next, whose README
      must record the vLLM tag and the pack versions actually installed.
    - Pack MRs opened the same day, both targeting `develop`, both validated only offline
      (`akamas build` + the packs' unit tests; the Akamas instance was unreachable from the
      CLI): vLLM pack **1.9.0**
      (<https://gitlab.com/akamas/optimization-packs/vllm/-/merge_requests/5>: 8 scheduling/KV
      admission knobs, 9 latency-breakdown metrics incl. `request_queue_time_p95` and
      `total_token_throughput`) and Kubernetes pack **1.9.0-dev**
      (<https://gitlab.com/akamas/optimization-packs/kubernetes/-/merge_requests/10>: new
      metrics-only `Kubernetes Node` component type, 71 `k8s_node_*` metrics = kube-state-metrics
      capacity/allocatable/requests/GPU + node-exporter OS set + per-node PSI; ships without the
      signed `crpCredits` licensing field, to be added by the pack maintainers). Both are
      numbered above the still-unmerged `feature/gpu-efficiency-metrics` (vLLM 1.8.0, needed by
      study #9) and `feature/psi-ratio-convention` (Kubernetes 1.8.1-dev) branches. A study
      using `Kubernetes Node` needs its own Prometheus telemetry queries for `k8s_node_*` (the
      pack ships names only; the Linux-pack query mapping can be reused for the OS metrics).
- **Q4 — Prefix-cache hit rate as a hard constraint, not just an observed metric**: per
  `knowledge/notes/2026-07-vllm-official-auto-tune-script.md`, vLLM's own `auto_tune.sh`
  supports `MIN_CACHE_HIT_PCT` as an admission gate — a configuration that tanks cache
  hit rate shouldn't win purely on throughput. Worth adopting as a `goal.yaml` constraint
  for any future RAG-shaped or prompt-reuse-heavy study. First confirm with
  `akamas describe optimization-pack vLLM` whether the installed pack exposes an actual
  cache-*hit-rate* (reuse) metric, distinct from the cache-*usage*/occupancy metric
  already referenced by H2 — this repo's own tracking at the time only confirmed the
  latter.
- **Q5 — What granularity does the installed GPU pack's "utilization" metric actually
  report?**: per `knowledge/notes/2026-07-modal-gpu-glossary.md`, `nvidia-smi`-style GPU
  utilization can report 100% while a kernel occupies under 1% of an H100's streaming
  multiprocessors — "GPU utilization" is not a reliable proxy for compute saturation
  unless it's specifically an SM-level (or finer, e.g. per-pipe/tensor-core) metric.
  Before trusting a GPU-utilization reading in any study's results (including H2's
  verification metrics and backlog #3's energy-efficiency study), confirm via
  `akamas describe optimization-pack GPU` which granularity the installed pack's metric
  actually measures.
- **Q6 — Is llm-d worth adopting for backlog #4's multi-replica work?** Per
  `knowledge/notes/2026-07-llm-d-distributed-inference-platform.md`, llm-d is a
  Kubernetes-native orchestration layer above vLLM (KV-cache-aware routing, prefill/
  decode disaggregation, SLO-aware autoscaling via `InferencePool`/`InferenceModel`
  CRDs) — same category as Dynamo's Planner/aiconfigurator already tracked here: a
  topology/deployment choice made *before* a study starts, not a vLLM parameter change.
  It's N/A for this repo's current single-GPU/single-replica scope; revisit only once
  backlog #4/H4-H6 actually reach multi-replica territory. If adopted then, llm-d's own
  routing/scaling policy (EPP scoring weights, prefill:decode pool ratio, autoscaling
  SLO targets) would need a **new routing-component pack type** — not an extension of
  the existing vLLM component — flag as a fresh pack request at that time rather than
  now. **Concrete, falsifiable thresholds for when disaggregation (llm-d, Dynamo, or
  otherwise) is worth it at all**, per
  `knowledge/notes/2026-07-inference-engineering-techniques-quantization-speculation-parallelism.md`:
  reach for it only when **all three** hold — serving 100M-1B+ tokens/day (scale-
  dependent on model size), serving a model of at least ~100B parameters, and traffic
  that is prefill-heavy with long input sequences. This repo's studies (single 7B-class
  model, single replica, no production traffic) fail the first two cleanly — a sharper,
  more falsifiable statement than the previous qualitative "N/A until multi-replica
  scope," worth citing here instead of just asserting non-applicability.
- **Q7 — Does this repo's telemetry config actually point at vLLM's real Prometheus
  metric names?** Per
  `knowledge/notes/2026-07-generative-ai-on-kubernetes-model-observability.md`, vLLM
  exposes metrics under names like `vllm:time_to_first_token_seconds`,
  `vllm:num_requests_waiting`, `vllm:prompt_tokens_total`/`vllm:generation_tokens_total`
  (distinct from — and not fully mirrored by — OpenTelemetry's competing semantic-
  convention names, e.g. no OTel equivalent exists for the throughput metric at all).
  This repo's `telemetry-instance` configs (e.g.
  `studies/0-explorative/akamas/telemetry/prometheus.yaml`) were inherited from the
  pre-restructure setup and, per that study's own README, were never independently
  re-verified against a live vLLM `/metrics` endpoint. Worth a one-time check (e.g.
  `curl <vllm-pod>:8000/metrics` against a running study) that every `vLLM.*` Akamas
  metric identity actually maps to the real underlying vLLM metric name, before trusting
  results from a study whose telemetry config was copied forward rather than verified.
- **Operating practice — diagnosing anomalous study runs**: per
  `knowledge/notes/2026-07-distributed-inference-blueprints-troubleshooting.md`'s
  troubleshooting playbook, when a study's results look off, check TTFT and TPOT
  together (not just the goal metric) before concluding a parameter caused a regression,
  and avoid changing more than one scheduler-adjacent parameter between experiments where
  possible so a symptom can be attributed. Concretely: a TPOT rise with flat TTFT and
  climbing KV-cache utilization points to preemption/fragmentation, not the parameter
  under test — worth ruling out before trusting an experiment's result. This is a
  practice to apply when interpreting results, not a hypothesis to confirm/deny itself.
  Two more named symptom→fix mappings, per
  `knowledge/notes/2026-07-ai-systems-performance-serving-tuning-checklist.md`'s Table
  16-1: a KV cache preemption warning in the vLLM scheduler log points to insufficient
  KV cache space — try raising `gpu_memory_utilization` or lowering
  `max_num_batched_tokens`, not the parameter under test; a cache-hit rate below ~60%
  under load points to unbalanced shard placement or a missing/misconfigured prefix
  cache — check the prefix-caching config before attributing a throughput drop to the
  parameter under test.
- **Operating practice — don't run autoscaling during a controlled Akamas experiment**:
  per `knowledge/notes/2026-07-nvidia-dynamo-planner.md`, LLM-aware autoscalers
  (Dynamo's Planner and likely others) terminate workers on scale-down **without
  draining in-flight requests** — if any future study (e.g. backlog #4's replica-count
  work) runs with autoscaling active during a load test, a scale-down event could drop
  in-flight requests and look like a vLLM-parameter effect. Freeze/disable autoscaling
  (or pin replica count) during a controlled experiment, or explicitly account for
  autoscaling events in that study's methodology if it's testing the autoscaler itself.
  **This generalizes beyond Dynamo's Planner** — per
  `knowledge/notes/2026-07-kubernetes-cluster-config-autoscaling-multitenancy.md`, an
  open vLLM RFC (vllm-project/vllm#24885) confirms vLLM's own SIGTERM handling has been
  **inconsistent historically**: it doesn't reliably wait for in-flight requests to
  finish today, regardless of which autoscaler (KEDA/HPA included) triggers the
  scale-down. Until that's hardened upstream, this practice applies to *any*
  autoscaler-driven scale-down of vLLM pods, not just Dynamo-specific tooling. If
  backlog #4 does add autoscaling, the same note names concrete tools to evaluate: KEDA
  (pod-level, scales on `vLLM.num_requests_waiting` — a metric this repo's telemetry
  already tracks), Karpenter (node-level provisioning, complementary to KEDA not an
  alternative), and Kueue (multi-tenant GPU quota fairness, if the cluster is ever
  shared across studies/teams). **Concrete knob names any such autoscaling parameter
  set would need to expose**, per
  `knowledge/notes/2026-07-inference-engineering-production-autoscaling-deployment.md`:
  min replicas, max replicas, autoscaling window (sliding timeframe for decisions),
  scale-down delay (grace period before removing a replica), and concurrency target
  (requests per replica before scaling up — must match the replica's actual batch-size
  configuration or the autoscaler's capacity model is wrong). Worth using as a checklist
  once backlog #4 is actually scoped, not a new pack-request by itself.

## B. Prioritized study backlog

**Reframed 2026-07-15 around two explicit validation goals** (see Section D for full
detail): **(Goal A)** extract more throughput at parity of hardware; **(Goal B)** given
a target throughput, use the right amount of hardware (right-sizing, whole-GPU via
tensor parallelism and sub-GPU via MIG). Studies #2-#4 below are the near-term,
concretely-scoped plan for these two goals, framed around a **chatbot** use case (the
simplest real use case to start with, per this repo's own decision — RAG/agent use
cases are deliberately deferred, see Section F). Everything from the previous version
of this backlog that isn't one of these two goals has moved to **Section F
("Futuribile")** rather than being deleted — it's deprioritized, not abandoned.

| # | Study | Target component(s) | Objective (sketch) | Status |
|---|-------|----------------------|---------------------|--------|
| 1 | [0-explorative](studies/0-explorative/README.md) | vLLM | Maximize token throughput (no latency constraint, matching the pre-restructure S3.1 study it replaces) — the first study to establish a baseline, against the vLLM pack 1.3.1 (16 of 26 parameters tuned). **Result: +12.5% over baseline** (`FLASHINFER`+`fp8_e4m3`+`block_size=32`), with lower latency and higher success rate too — not a throughput/latency trade-off at this optimum. | DONE |
| 1b | [1-goodput-realistic-load](studies/1-goodput-realistic-load/README.md) | vLLM | **Preliminary step before Goal A/B's H100 studies (2026-07-15)** — same A10G/model/vLLM version as `0-explorative` (deliberately unchanged, to isolate the variables below), but: switched load generator to NVIDIA AIPerf (via `inference-perf` initially, replaced 2026-07-17) with real ShareGPT-dataset replay and a 7-level concurrency sweep instead of GuideLLM's synthetic fixed-shape saturating benchmark; tuned 14 pack v1.5.1 parameters (`spec_method`/`spec_tokens` dropped after 5 crashes); goal is **goodput** (throughput subject to a P95 TTFT/ITL SLA) instead of throughput-only. **Result: +45.5% goodput over baseline, staying SLA-compliant** — stopped by the user at 31 experiments (1 baseline + 30 optimize, 1 no-score); best trial (exp 29) reached 3131 tokens/s vs. baseline's 2152 tokens/s, both within the 1500ms TTFT/300ms ITL SLA (Akamas' `stability` windowing selects the best compliant sub-window from each trial's concurrency sweep, not a whole-trial average) — `FLASHINFER`+`fp8`+`gpu_memory_utilization≈0.92`+`max_num_seqs≈770`, confirmed by 5 independent top trials converging on the same region. See the study's own README "Conclusions" — also flags a real export-bundle-analysis pitfall (array-index-to-experiment-number offset differs from what the `akamas-study-analyzer` plugin's reference doc assumed). | DONE |
| 2 | [2-larger-model-g7e](studies/2-larger-model-g7e/README.md) | vLLM | **Redirected 2026-08-20** from the original Goal-A plan below (kept as a superseded historical record in Section D) — instead of re-baselining the *same* Qwen2.5-7B-Instruct model on 8-GPU `p5.48xlarge`, this asks whether a **~4x larger model** (`Qwen/Qwen3.8-27B`, dense) on a single, higher-VRAM GPU improves goodput over `1-goodput-realistic-load`'s A10G result — same goodput objective/methodology, bigger model instead of more GPUs. **Hardware swapped again 2026-08-21**: planned single-GPU H100 (`p5.4xlarge`) had no capacity in-region, now runs on `g7e.4xlarge` (1x NVIDIA RTX PRO 6000 Blackwell 96GB) — a third, distinct GPU architecture from both A10G and H100 (lower memory bandwidth than H100, less mature vLLM kernel support on this compute capability) — see the study's own README for what this changes. **Does not answer Goal A/B as originally framed** — see Section D's explicit flag on studies #3/#4, which still depend on the superseded plan's baseline. **Inherited-SLA note (2026-08-25)**: this study carried forward `1-goodput-realistic-load`'s same 1500ms TTFT/300ms ITL thresholds and goal formula — that study's own now-completed result found those thresholds *achievable and beatable* above its own baseline on A10G (+45.5%, see row 1b), so there's no reason to revisit them on that account. Whether the same SLA pair is achievable on this study's different model/hardware is still open — worth checking early rather than assuming it transfers. | TODO |
| 3 | `<tbd>` | vLLM + Kubernetes (DRA, multi-GPU) | **Goal B, whole-GPU granularity** — given a target throughput, find the minimum `tensor_parallel_size` (GPU count) that satisfies it, requested dynamically via DRA. Tests H5/H6. **Setup detail: Section D, study #3.** | IDEA |
| 4 | `<tbd>` | vLLM + GPU/Kubernetes (DRA→MIG, classic fallback) | **Goal B, sub-GPU granularity** — given a (smaller) target throughput, find the minimum MIG slice that satisfies it. Primary path via DRA; explicit classic device-plugin fallback if DRA's MIG support proves unworkable (confirmed still not officially supported upstream as of 2026-07-15). **Setup detail: Section D, study #4.** | IDEA |
| 8 | [8-parallelism-tuning](studies/8-parallelism-tuning/README.md) | vLLM (4-GPU topology + per-instance) | **Tests H5's second angle and H6 directly (scaffolded 2026-09-09)** — same goodput goal/SLA/windowing as `2-larger-model-g7e` and the same 4x L4 `g6.12xlarge` node, per-GPU components and 102-metric telemetry as `7-tensor-parallelism`, but `tensor_parallel_size` and `data_parallel_size` (both [1,4]) join the 14 tuned vLLM parameters so the optimizer searches the TP/replica split itself instead of running it as presets. Two topology `parameterConstraints` (`TP x DP <= 4`, `TP != 3` — 28 attention heads) leave 7 valid pairs. Not Goal B/#3 (no right-sizing, no DRA): fixed 4-GPU budget, maximize goodput. | TODO |
| 9 | [9-goodput-per-gpu](studies/9-goodput-per-gpu/README.md) | vLLM (4-GPU topology + per-instance) | **Same search as #8 (16 parameters, 7 valid TP/DP topologies), different objective (scaffolded 2026-09-10)**: maximize `(prefill_token_throughput + decode_token_throughput) / active_gpus` — goodput per GPU actually holding weights — instead of raw goodput, so a configuration is not rewarded merely for occupying all four L4s. `active_gpus` (DCGM-sourced) and `active_dp_engines` (vLLM `engine` label) are new vLLM-pack 1.8.0 metrics; the DCGM counter set was also aligned with NVIDIA's Run:ai GPU-profiling reference (+3 NVLink fields, GPU pack 1.2.0 — empty on L4). Complements #8: #8 answers "most goodput from this node", #9 answers "most goodput per GPU" — the pair measures TP/DP scaling efficiency directly (H5's 3.5x-for-4-GPUs ballpark). | TODO |
| 10 | [10-gpt-oss-20b-goodput-per-gpu](studies/10-gpt-oss-20b-goodput-per-gpu/README.md) | vLLM (MoE model, 4-GPU topology + expert parallelism) | **Same goodput-per-GPU objective as #9, new model (scaffolded 2026-09-14)**: `openai/gpt-oss-20b` (21B MoE, 32 experts / top-4, MXFP4 expert weights = 12.8 GiB, attention sinks + alternating sliding window) on vLLM **0.29.0** — the version the vLLM pack 1.9.0 now references. Source-verified fit on one L4: Marlin MXFP4 MoE kernel (capability >= 7.5) keeps weights 4-bit, `TRITON_ATTN` is the only sink-capable attention backend on SM 8.9, so TP1 is legal and the optimizer compares TP1/TP2/TP4 x DP layouts plus `enable_expert_parallel` (constraints: `TP x DP <= 4`, `TP != 3` for 64/8 heads, EP only if TP x DP > 1, per-GPU memory fit). Carries the study-9 lessons: `gpu_memory_utilization` <= 0.90 and a workflow restart guard that fails crashed trials. Blocked on `g6.12xlarge` capacity (node group DEGRADED since 2026-09-14); `g6e.12xlarge` (L40S, same SM 8.9 path) or a region move are the alternatives — see its `infra/README.md`. | TODO |
| 11 | [11-gpt-oss-20b-tps-peak](studies/11-gpt-oss-20b-tps-peak/README.md) | vLLM (MoE model, 4-GPU topology + expert parallelism) | **Short-ramp twin of #10, scaffolded 2026-09-15 to make the search affordable**: identical model (`openai/gpt-oss-20b`), node, goal, SLA, 14 parameters, domains and 5 `parameterConstraints` — only the AIPerf load profile changes, from 12 levels x 300s (~67 min of load per trial, ~10 days for 200 experiments) to 4 log-spaced levels 128/256/512/1024 x 90s (~6 min per trial, ~11 h for 60). The ramp still ends at 1024 because the knee sits **at** `max_num_seqs`: #10's baseline measured it directly on this node (concurrency 253 -> `Waiting=0`, engine-side TTFT p95 < 500 ms; 302 -> 46 queued, TTFT p95 5-7.5 s), so a ceiling below `max_num_seqs`'s [16, 1024] domain would leave configurations tuned toward ~770 (study 1b's winning region) unsaturated and their TPS peak invisible. `stability` windowing kept unchanged on purpose — it picks the best compliant sub-window, so weak configs come out infeasible and strong ones feasible — with only `resolution: 15s` added so the 6-sample window equals one 90s level. | TODO |

Before activating a study: run `/new-study` (reads this roadmap and `knowledge/README.md`
for you), pick a real name, and let it scaffold `studies/<name>/` from the template.
Once scaffolded, replace the `<tbd>` row above with the real study name and a link.
**Recommended execution order (original plan, now partly stale — see below): #1b → #2 →
#3 → #4** — #1b runs first, on the already-proven A10G hardware, to de-risk the
tooling/pack changes (inference-perf, pack v1.5.1) before #2 also changes hardware to
H100; #2 re-establishes the H100 baseline and validates the new load-testing tool that
#3/#4 both reuse; #3 (DRA for generic multi-GPU, more mature per Section D's research)
should be de-risked before #4 (DRA for MIG, explicitly less mature) so any DRA/Akamas
integration problems surface on the easier case first.

**Ordering caveat, added 2026-08-20, hardware detail updated 2026-08-21**: #2 has been
redirected to `2-larger-model-g7e` (single-GPU, a different model) — it no longer
produces the 8-GPU `p5.48xlarge`/Qwen2.5-7B baseline #3 and #4 were designed to build
on. Its single GPU is also no longer H100: originally `p5.4xlarge`, swapped
2026-08-21 to `g7e.4xlarge` (RTX PRO 6000 Blackwell) for regional capacity reasons — a
second, independent divergence from #3/#4's `p5.48xlarge`/H100 hardware, on top of the
already-open model/topology divergence. Their own setup detail (Section D) still
references "study #2's winning region" — **that dependency is now open/unreconciled**,
not resolved by this note. Before scoping #3 or #4, either run the original
(superseded) Study #2 plan under a new slot to produce the baseline they actually
need, or revise #3/#4's design to not depend on it.

## C. Consolidated learnings

- **`studies/0-explorative`** (Qwen2.5-7B-Instruct, single A10G, vLLM 0.22.0): a
  `parameterConstraint` written purely to stop a crash can also mark the actual optimum,
  not just a guardrail — `FLASHINFER` was the *worst*-performing attention backend when
  restricted to `kv_cache_dtype=auto` (the only value the other two backends could use
  safely), but became the best overall once paired with fp8-family quantization
  (`fp8_e4m3` specifically), which is exactly the pairing the crash investigation had
  already restricted to `FLASHINFER` for correctness reasons. Worth checking for this
  pattern in future studies: a constraint born from a crash may be pointing at the
  interesting part of the search space, not just fencing off a bad one.
- **`studies/0-explorative`**: vLLM's own memory accounting (the `gpu_memory_utilization`
  budget / "Available KV cache memory" it reports) does **not** cover every memory
  consumer — a dummy sampler-warmup step allocates a buffer sized roughly
  `max_num_seqs × vocab_size`, *after* weights+KV cache are already reserved, and this
  caused an OOM at `max_num_seqs≈900-1000` combined with `gpu_memory_utilization` near
  the top of its range, even though vLLM's own reported KV-cache budget looked fine.
  Narrowing `gpu_memory_utilization`'s domain alone doesn't fully guard against this —
  future studies with a similarly high `max_num_seqs` ceiling should watch for this
  specific failure mode too, not just the KV-cache-deficit OOM the domain narrowing
  targets.
- **`studies/1-goodput-realistic-load`** (Qwen2.5-7B-Instruct, single A10G, vLLM
  1.5.1, stopped 2026-07-21 at 31 experiments): **a concurrency-sweep load pattern
  combined with `stability` windowing lets Akamas find the best SLA-compliant
  operating point within each trial, not just average across a trial that includes a
  saturating tail** — the winning trial (+45.5% goodput vs. baseline) stayed within
  both the 1500ms TTFT and 300ms ITL budgets. Worth carrying this sweep-then-select
  load-pattern design into future goodput-constrained studies rather than a
  fixed-rate benchmark, when the goal itself has a hard latency constraint.
  `attention_backend=FLASHINFER` + `kv_cache_dtype=fp8` + `gpu_memory_utilization`
  ~0.90-0.92 + `max_num_seqs` ~750-800 reconfirms `0-explorative`'s own
  `FLASHINFER`+fp8-family finding under a completely different methodology (real
  ShareGPT/goodput-constrained optimization vs. synthetic fixed-shape/throughput-only)
  — 5 independent top trials converged on the same region, and two independent
  studies now agree on this attention-backend/quantization pairing for
  Qwen2.5-7B/A10G, worth treating as a reasonably solid default for future studies on
  similar hardware.
- **Export-bundle analysis pitfall, found analyzing `studies/1-goodput-realistic-load`
  (2026-08-25)**: the `akamas-study-analyzer` plugin's bundled script assumes
  `last-optimization.json` array index `i` = experiment `i+1`; this bundle's true
  mapping was `i+2` (baseline excluded from the arrays entirely), and its
  `metricConstraintsViolations[i]` array does not describe whether the experiment's
  *scored* window violates the goal's constraints (with `stability` windowing, the
  scored window is deliberately the compliant one — the violations array reflects a
  different, worse window in the same trial). Using the script's assumed mapping
  uncritically produced a completely fabricated first-draft finding ("no trial
  satisfied the SLA") that only got caught by cross-checking against `logs.json`'s
  directly-tagged per-experiment log lines. **Any future study's export analysis
  should verify the array-to-experiment offset against `logs.json` first** — don't
  trust the `+1` mapping (or a `metricConstraintsViolations` entry) without that
  check. Flagged in the plugin's own `reference/export-schema.md` "Known gaps".

## D. Incremental Study Plan — Setup Detail (Infra / Study / Load config)

**Rewritten 2026-07-15** around two explicit validation goals, replacing the previous
8-study incremental plan (moved to **Section F, "Futuribile"** rather than deleted).

> **Goal A**: extract more throughput at parity of hardware.
> **Goal B**: given a target throughput, use the right amount of hardware —
> right-sized at whole-GPU granularity (tensor parallelism) and sub-GPU granularity
> (MIG).

**Use case: chatbot** (this repo's own explicit choice, 2026-07-15 — the simplest real
use case to start with; RAG/agent use cases are deliberately deferred to Section F).
This choice concretely determines the dataset and load pattern used below:

- **Dataset**: ShareGPT (real multi-turn chat conversations) via `inference-perf`'s
  `data.type: shareGPT` — this *is* chatbot traffic, not a proxy for it, and gives
  realistic prompt/response length distributions and genuine prefix-cache reuse across
  conversation turns, unlike `0-explorative`'s synthetic fixed-shape prompts.
- **Load pattern differs by goal**: Goal A (study #2) wants the *ceiling* — push load
  via a saturation sweep to find maximum sustainable throughput. Goal B (studies #3/#4)
  wants a *fixed target* — hold load at a specific rate representing an assumed peak
  chatbot traffic level, then let Akamas search for the minimum hardware that satisfies
  it. Don't reuse one load config for both goals; they're asking different questions.
- **SLA-awareness throughout**: a chatbot is an interactive, human-facing product —
  every study below includes a latency guardrail (P95 TTFT/ITL via
  `goal.constraints`), not just a throughput number. Maximizing throughput at the cost
  of unacceptable per-response latency isn't a real win for this use case.

**Load-generator decision (2026-07-15, supersedes the earlier "validate first" plan):
`kubernetes-sigs/inference-perf`**, committed directly rather than run as a side-by-side
validation study first (the earlier plan's study #6 is dropped, not deferred — the
decision is made). Rationale unchanged from
`knowledge/notes/2026-07-kubernetes-sigs-inference-perf.md`: it's the only one of
GuideLLM/AIPerf/inference-perf with native real-dataset replay (ShareGPT) *and* both
open- and closed-loop load, plus a saturation-sweep mode Goal A needs directly.

**Hardware: NVIDIA H100 via AWS `p5.48xlarge`** (8× H100 80 GB, NVLink 4.0 — unchanged
from the previous plan; verify current pricing/availability at study-design time).
**Superseded for study #2 specifically (2026-08-20, hardware updated again 2026-08-21)**:
the original single-fixed-GPU plan below now runs as `2-larger-model-g7e` instead, on
a single GPU — planned as `p5.4xlarge` (1x H100 80GB), swapped 2026-08-21 to
`g7e.4xlarge` (1x NVIDIA RTX PRO 6000 Blackwell 96GB) for regional capacity reasons —
not 1 of 8 GPUs on a shared `p5.48xlarge` — see that study's own README and the "Study
#2" subsection below for the historical plan and why it changed. Studies
#3/#4 are unaffected in their own hardware choice (`p5.48xlarge`, multi-GPU via **DRA**)
but their designs still reference study #2's original baseline — see the flag on each
below.

**Hardware-baseline caveat (unchanged)**: every number `0-explorative` produced (A10G,
Ampere) is **not** directly comparable to any study below (H100, Hopper) — different
compute capability, different FP8 Tensor Core support (H100 has native FP8 Tensor
Cores; `0-explorative`'s `TRITON_ATTN`+fp8 Triton-compilation incident was
Ampere-specific and should not recur on H100 — re-verify empirically, don't just
assume). Treat study #2's own `baseline` experiment as the new hardware reference
point for every study after it.

### DRA prerequisites and risk — read before scoping studies #3/#4

Researched directly 2026-07-15 (not carried over from the earlier, less precise
"technical preview" note):

- **Core DRA API** (`resource.k8s.io`: `ResourceClaim`, `ResourceClaimTemplate`,
  `DeviceClass`) is genuinely GA since Kubernetes 1.34. **EKS supports DRA from 1.33**,
  **recommended 1.34+** (AWS's own docs cite an upstream bug, k8s issue #133920, fixed
  at 1.34). **Not compatible with Karpenter or EKS Auto Mode** — the node group running
  these studies must be a managed or self-managed node group. The NVIDIA DRA driver
  **does not support Bottlerocket** — use an AL2023-based AMI. DRA drivers and device
  plugins must not run on the same node for the same device type — don't leave the
  classic NVIDIA device plugin active on this node once DRA is installed.
- **NVIDIA's DRA driver** (now `kubernetes-sigs/dra-driver-nvidia-gpu`, donated to
  CNCF/kubernetes-sigs; latest release `v0.4.1`, requires Kubernetes 1.32+): generic
  multi-GPU allocation (requesting N devices from a `DeviceClass`) uses the current
  GA `v1` API and is the more solid path — **this is what study #3 relies on**.
  **GPU-allocation features specifically (which is where MIG-profile selection
  lives) are explicitly stated by the driver's own README as "not yet officially
  supported," and the GPU kubelet plugin ships disabled by default.** The only
  existing MIG example in the driver's repo uses the old alpha API
  (`resource.k8s.io/v1alpha2`, `MigDeviceClaimParameters`), not the current GA `v1`
  API — there is no working GA-API MIG example today. Only `ComputeDomain` (aimed at
  GB200-class multi-node NVLink) is called "officially supported," which doesn't apply
  to single-node H100 work here.
- **Akamas + DRA integration is a genuine open question** — no public evidence either
  way that Akamas can template `ResourceClaim`/`ResourceClaimTemplate` as a component
  parameter (as opposed to classic `resources.requests`/`resources.limits`). **Confirm
  this with Akamas support before scoping either study #3 or #4** — if unsupported
  natively, a workflow-level pre-render step (rendering the `ResourceClaimTemplate`
  YAML from the tuned parameter's value before `kubectl apply`, the same pattern
  `0-explorative`'s `apply_config.sh` already uses for other templating) is the likely
  workaround, but this needs confirming, not assuming.
- **Decision (2026-07-15, per this repo's own explicit choice)**: proceed with DRA for
  both studies #3 and #4, **with an explicit documented fallback for study #4 only**
  (classic NVIDIA device plugin + static MIG resource names, e.g.
  `nvidia.com/mig-3g.40gb`, which is production-proven) if DRA-based MIG allocation
  proves unworkable — so Goal B's sub-GPU right-sizing question isn't blocked by the
  driver's own stated immaturity for that specific feature. Study #3 (generic
  multi-GPU) doesn't need this fallback — its DRA path is the more mature one.

---

### Study #2 — Maximize Throughput at Fixed Hardware (Goal A, chatbot) — SUPERSEDED 2026-08-20

**This plan was never built and is now superseded** — kept below as a historical
record (this repo's convention: never silently overwrite a documented plan, mark it
superseded instead), not deleted, because studies #3/#4 below still reference it as
their own baseline/setup precedent. The actual study #2 that now exists is
`2-larger-model-g7e` (own `README.md`), which asks a different question (bigger model
on a single GPU — RTX PRO 6000 Blackwell as of 2026-08-21, not H100 — rather than more
throughput on the same model at 8-GPU scale) and does
**not** produce the `p5.48xlarge`/Qwen2.5-7B baseline the text below assumes. **Anyone
scoping study #3 or #4 needs to either produce that baseline separately first, or
revise #3/#4 to not depend on it** — not resolved by this note.

Re-baselines `0-explorative` on H100 with a realistic dataset/load pattern and an
explicit latency guardrail — this repo's first study under the new validation framing.

- **Infra config**: `p5.48xlarge`, 1 of its 8 H100s, classic `nodeSelector`/resource
  request (no DRA — single fixed GPU). Same `llm-serving`/`llm-benchmark`/`monitoring`
  namespace pattern and PVC-based model cache as `0-explorative` (verify PVC/storage
  class details against the new node rather than assuming identical). Qwen2.5-7B-
  Instruct, same as `0-explorative`.
- **Study config**: adapt `0-explorative`'s 16-parameter `parametersSelection` and
  `parameterConstraints` to H100 — **re-verify each constraint against vLLM source for
  Hopper specifically**, don't assume they transfer as-is (the `TRITON_ATTN`+fp8
  Triton-compilation exclusion was Ampere-specific; H100 supports `fp8e4nv` natively,
  so that exclusion may no longer be needed — confirm empirically, don't remove it on
  the strength of this note alone). **Add the two now-available pack parameters**:
  `vLLM.spec_method`/`vLLM.spec_tokens` (categorical gate `none`/`ngram`/`ngram_gpu`/
  `suffix`/`mtp`, default `none`, plus the integer dependent) — copy in both
  `parameterConstraints` from
  `knowledge/notes/2026-07-vllm-pack-v1.5.0-speculative-decoding-gate-pattern.md`, and
  implement the deploy-script drop-flag step it documents (dropping `--spec-method`/
  `--spec-tokens` from the rendered command whenever the gate is `"none"` — passing
  the sentinels through would crash vLLM at startup). N-gram speculation is a plausible
  (if modest — chatbot conversation isn't as repetitive as code/RAG) throughput lever
  now worth letting the optimizer try, given `"none"` is still the safe default if it
  doesn't help. **Goal**: maximize `vLLM.prefill_token_throughput +
  vLLM.decode_token_throughput`, subject to a `goal.constraints` latency SLA (e.g.
  `vLLM.time_to_first_token_p95 <= <X ms>` and/or an ITL bound) — `X` chosen empirically
  from this study's own baseline experiment, not invented or carried over from
  `0-explorative`'s A10G-era numbers. Consider narrowing `gpu_memory_utilization`
  further per H2's weak-correlation finding to spend more optimizer budget on
  parameters that actually move the throughput/latency trade-off.
- **Load config**: `inference-perf`, `data.type: shareGPT`, **saturation-sweep mode**
  (linear/geometric rate progression until the server saturates) to find the ceiling
  throughput under realistic chatbot content — this is Goal A's defining load pattern,
  distinct from studies #3/#4's fixed-target load below.

### Study #3 — Multi-GPU Right-Sizing via Tensor Parallelism (Goal B, whole-GPU granularity, tests H5/H6)

**Open dependency, flagged 2026-08-20**: this study's design below carries forward
"study #2's winning region" and treats study #2's single-GPU ceiling as its throughput
target's reference point — but study #2 (as originally planned, `p5.48xlarge`/
Qwen2.5-7B) was never run; that slot now redirects to `2-larger-model-g7e`, a
different model on different hardware that doesn't produce this baseline. Before
scoping this study for real, either run the superseded Study #2 plan (above) under a
new slot, or rework this section to reference a different/no baseline.

Given a target throughput (a larger chatbot deployment's expected peak), find the
minimum GPU count that satisfies it — the inverse question from study #2's "maximize
given fixed hardware."

- **Infra config**: `p5.48xlarge`, via **DRA**: a `ResourceClaimTemplate` requesting
  `count: N` devices from a GPU `DeviceClass` (`resource.k8s.io/v1`, the current GA
  API — the more mature DRA path per the research above), where `N` is driven by
  `vLLM.tensor_parallel_size`'s tuned value. **Prerequisites to confirm before this
  study can run** (see the DRA section above): an EKS managed/self-managed node group
  (not Karpenter/Auto Mode) on Kubernetes 1.34+, AL2023 AMI, NVIDIA DRA driver v0.4.1+
  installed via Helm with GPU allocation enabled (disabled by default), and confirmed
  with Akamas support whether `ResourceClaimTemplate` can be templated as a component
  parameter or needs a workflow pre-render step. Confirm the Kubernetes Topology
  Manager policy and that the requested GPUs are NUMA-local to the pod's pinned CPUs
  (per H5's provisioning-prerequisite note) — real NVLink 4.0 between all 8 H100s on
  this instance, unlike the A10G `g5` family which had none.
- **Study config**: `vLLM.tensor_parallel_size` ∈ `{1, 2, 4, 8}` (not the pack's full
  `[1,16]` range — per H5's "minimum that fits" framing and this instance's 8-GPU cap),
  `pipeline_parallel_size`/`data_parallel_size` pinned to 1 (single node, isolates TP),
  carrying forward `attention_backend`/`kv_cache_dtype`/`block_size`/`spec_method`/
  `spec_tokens` from study #2's winning region rather than re-searching them. **Goal
  (inverted from study #2)**: minimize `vLLM.tensor_parallel_size` (the GPU-count
  proxy) subject to `goal.constraints` requiring the throughput formula to meet or
  exceed a target `Y` tokens/s representing the assumed peak chatbot traffic tier for
  this study (choose `Y` deliberately higher than study #2's single-GPU ceiling, so
  the study can actually demonstrate needing more than one GPU) — this is the concrete
  Akamas expression of Goal B. Cross-check the result against H5's "~3.5× for 4 GPUs"
  reference figure and the 50-70%-communication-overhead warning for poorly-partitioned
  interconnect (not expected to bite here, given real NVLink 4.0, but verify — a
  measured efficiency well below that figure is a stronger signal of a real issue than
  it would have been on the earlier PCIe-only A10G plan).
- **Load config**: `inference-perf`, `data.type: shareGPT`, but **fixed at the target
  rate `Y`** (open-loop `poisson`, not a sweep) — the question here is "what hardware
  satisfies this specific load," not "what's the ceiling."

### Study #4 — MIG Right-Sizing (Goal B, sub-GPU granularity)

**Same open dependency as Study #3, flagged 2026-08-20** — this study's design also
carries forward vLLM software parameters "from study #2's winning region," which no
longer exists (see Study #3's flag above and `2-larger-model-g7e`'s own README).

Given a *smaller* target throughput (a lower-traffic chatbot deployment), find the
minimum MIG slice that satisfies it — the sub-GPU-granularity complement to study #3.

- **Infra config**: `p5.48xlarge`, 1 of its 8 H100s MIG-partitioned. **Primary path:
  DRA**, per this repo's own decision — but per the DRA-risk section above, expect to
  need the driver's older alpha API (`resource.k8s.io/v1alpha2`,
  `MigDeviceClaimParameters`) since no working GA-API MIG example exists yet, and
  expect the GPU kubelet plugin to need explicit enabling (disabled by default,
  "not yet officially supported" per the driver's own README). **Explicit fallback**:
  if DRA-based MIG allocation proves unworkable within reasonable effort, switch to
  the classic NVIDIA device plugin + static MIG resource names (`nvidia.com/mig-
  <profile>`, e.g. `nvidia.com/mig-3g.40gb`) — production-proven, just not DRA-based —
  and note the fallback explicitly in this study's own README rather than silently
  abandoning the DRA attempt without a record.
- **Study config**: a new categorical/ordinal parameter representing MIG slice size
  (smallest→largest, e.g. `1g.10gb` < `2g.20gb` < `3g.40gb` < `4g.40gb` < `7g.80gb` on
  an H100 80GB — confirm exact available profiles at study-design time) — **this
  parameter doesn't exist in any installed pack yet and needs new pack-engineering
  work** (most naturally a new Kubernetes-pack parameter or a new component type
  representing "which GPU resource to request," analogous to the already-flagged
  GPU-fraction-per-replica gap in H4/backlog, not something vLLM itself controls). Use
  `ordinal`, not `categorical`, for this parameter from the start — the pack audit
  already established Akamas' `ordinal` type exists specifically for this shape of
  ordered-value problem (see the `block_size` conversion). Carry forward vLLM software
  parameters from study #2's winning region, rescaled where needed to the smaller
  slice's actual VRAM (`max_num_seqs`/`max_num_batched_tokens` in particular — a MIG
  slice's memory is fixed and smaller than a full GPU's). **Goal (same inverted shape
  as study #3, one granularity finer)**: minimize the MIG-slice-size parameter subject
  to `goal.constraints` requiring the throughput formula to meet or exceed a target
  `Z` tokens/s representing a smaller chatbot traffic tier (`Z` deliberately lower than
  study #3's `Y`, so the narrative across studies #2→#3→#4 is "single GPU → multiple
  GPUs for large traffic → GPU fraction for small traffic," a full right-sizing
  spectrum).
- **Load config**: `inference-perf`, `data.type: shareGPT`, fixed at target rate `Z`
  (same reasoning as study #3 — a specific target, not a sweep).

## E. Debt / non-study actions
- [ ] **Evaluate Run:ai Model Streamer to cut per-experiment model-load time**, per
      `knowledge/notes/2026-07-generative-ai-on-kubernetes-production-tuning-routing.md`.
      Every Akamas experiment in this repo's studies pays a 5-15 min model-load cost
      before the actual load test starts (see `studies/0-explorative/README.md`'s own
      timing notes) — model loading is called out as the single biggest lever in vLLM's
      startup-time breakdown. `--load-format runai_streamer` (with
      `--model-loader-extra-config '{"concurrency": 16}'`) needs no model repackaging,
      unlike CoreWeave Tensorizer/fastsafetensor which require pre-serialization — worth
      trying against this repo's actual model/PVC setup before committing to any
      pre-serialization approach. Not an Akamas parameter — a workflow/Deployment-level
      flag change to test manually, potentially cutting wall-clock time across every
      future study's experiment count. **Second, independent motivation for this same
      evaluation**, per
      `knowledge/notes/2026-07-inference-engineering-production-autoscaling-deployment.md`:
      quantizing model weights also speeds up *cold-start* loading time itself, not just
      steady-state inference throughput — this repo's per-experiment model-load cost is
      exactly a cold-start cost, so this is a second lever (alongside Model Streamer)
      worth testing, not just an inference-throughput side effect.
- **Percent-scale bug in studies 5/7/8 telemetry (found 2026-09-10, fixed only in `9-goodput-per-gpu`)**: Akamas' `percent` unit is a 0-1 ratio (its own Prometheus metrics mapping returns plain ratios; the UI renders ×100), but the Kubernetes-pack percent queries written in `5-pack-changes` and carried into 7 and 8 prefix `100 *` — PSI at 8.5% displays as 850%, `container_cpu_util` likewise ×100 — while 5 DCGM `DEV_*` utilization queries (0-100 natively, e.g. `gpu_util`) are not divided by 100 and display ×100 too. `container_cpu_throttled_millicores` also used the wrong counter (`throttled_periods` instead of `throttled_seconds`). Any comparison of these metrics across studies must rescale; recreating those studies' telemetry instances with study 9's queries is the fix. Pack descriptions now spell out the convention (Kubernetes 1.8.1-dev `feature/psi-ratio-convention`, GPU 1.2.0).
- [ ] **If vLLM's multi-node Ray executor backend is ever adopted** (for TP/PP spanning
      multiple nodes, per H5's "minimum TP that fits" discussion), check
      `knowledge/notes/2026-07-generative-ai-on-kubernetes-training-job-scheduling.md`'s
      security caveat first: Ray ships with **no built-in authentication/encryption by
      default** (explicitly documented as "not built for use in untrusted environments") —
      any process with network access to the Ray cluster can execute arbitrary code.
      Mitigation is a Kubernetes `NetworkPolicy` deny-all-by-default + explicit allow
      rules scoped to the Ray cluster's pods, not something vLLM configures itself. N/A
      for every study so far (all single-node) — flagged for if/when this changes.
- [ ] **SECURITY**: this repo's git history contains a real, unencrypted private SSH key
      committed in an earlier revision (path `akamas/workflows/id_rsa` at the time).
      Revoke/rotate it wherever it grants access, and treat it as compromised regardless
      of whether the file is still present on disk — removing a file from the working
      tree does not remove it from git history; that requires a history rewrite.
- [x] **Cluster provisioning is now atomic per study, not shared** (resolved
      2026-07-15, supersedes this item's original framing). `0-explorative` gained a
      complete `infra/` layer (`eksctl` cluster config, `provision.sh`, Kubernetes
      bootstrap namespaces/StorageClasses) ported from `_old/infra/` and organized
      under `studies/0-explorative/infra/` — see that folder's own `README.md`.
      `studies/_TEMPLATE/infra/` scaffolds the same shape for future studies (own
      `CLAUDE.md` line updated to match). Studies #2-#4 in Section D each still need
      their *own* `infra/eks/cluster.yaml` scaffolded for `p5.48xlarge`/H100 before
      they can run — this item resolves the general pattern, not every study's
      specific cluster yet. **Update 2026-08-20**: study #2's slot is done —
      `2-larger-model-g7e/infra/` is scaffolded, but for a single GPU (not
      `p5.48xlarge`), since that slot was redirected (see Section D). #3/#4 still
      need their own `p5.48xlarge` scaffolding, unaffected by this redirect. **Update
      2026-08-21**: study #2's single-GPU node group swapped from `p5.4xlarge`
      (H100) to `g7e.4xlarge` (RTX PRO 6000 Blackwell) — no regional capacity for
      `p5.4xlarge`; see the study's own README/`infra/README.md` for the full detail.
- [ ] Scaffold and run the first real study under `studies/` via `/new-study`.
- [x] **Pack request — implemented directly on the open branch, 2026-07-14** (pack
      version now `1.5.1`, commits `2121136`, `ba048e4`, and `9e177fd` on
      `feature/attention-backend-and-block-size-categorical`, not pushed/merged — see
      the urgent branch/version item below). **Added, source-verified against vLLM
      `v0.22.0`**: `decode_context_parallel_size` (integer `[1,16]`, default 1) and
      `prefill_context_parallel_size` (integer `[1,8]`, default 1) — the decode/prefill
      context parallelism gap first flagged in
      `knowledge/notes/2026-07-distributed-inference-scaling-dimensions.md` is closed.
      Both need the `parameterConstraints` documented in
      `knowledge/notes/2026-07-vllm-pack-v1.4.0-dcp-pcp-parameterconstraints.md` (also
      now in the pack's own README) copied into any study that tunes them — in
      particular `decode_context_parallel_size` must evenly divide
      `tensor_parallel_size`, and `decode_context_parallel_size > 1` is only safe for
      MLA-architecture models (not Qwen2.5-7B), a model-config-dependent constraint that
      can't be expressed as a pack-level formula. **Confirmed definitively absent from
      vLLM `v0.22.0`** (exhaustive source grep, not a guess — no longer open questions
      for the pack owner): `swap_space` (flag removed upstream), `scheduler_delay_factor`
      (zero occurrences in source), `max_seq_len_to_capture` (superseded by
      `max_cudagraph_capture_size`, already a pack parameter), and a KV-cache
      preemption-threshold parameter (preemption is purely automatic, not
      CLI-configurable). **Speculative decoding — also now implemented**, via a
      reusable "gate parameter + forced-sentinel dependent" pattern: added
      `spec_method` (categorical `none`/`ngram`/`ngram_gpu`/`suffix`/`mtp`, default
      `none`) and `spec_tokens` (integer `[0,16]`, default 0) — named to mirror vLLM's
      own `--spec-method`/`--spec-tokens` CLI flags directly, matching this pack's own
      naming convention (an initial `speculative_decoding_method`/
      `num_speculative_tokens` naming was caught and corrected the same day, commit
      `9e177fd`, before the branch was ever pushed). Draft-model-dependent methods
      (`eagle`/`medusa`/`mlp_speculator`/`draft_model`/etc.) were deliberately **not**
      added — they'd need a `--spec-model` reference this pack doesn't provide
      (workload/target-model-specific, not a fixed pack-level domain) and would crash
      vLLM at startup without one. Full pattern, source citations, and the two
      required `parameterConstraints` (forcing `spec_tokens=0` whenever the gate is
      `"none"`, and forbidding `0` whenever it isn't) are in
      `knowledge/notes/2026-07-vllm-pack-v1.5.0-speculative-decoding-gate-pattern.md`.
      **Any study using these two parameters must also implement a deploy-script
      drop-flag step** (documented in both that note and the pack's own README) —
      passing the sentinel values through to vLLM (`--spec-method none` /
      `--spec-tokens 0`) would crash it at startup, confirmed from source (argparse
      rejects `none` as a choice; Pydantic rejects `num_speculative_tokens=0`, the
      internal field `spec_tokens` maps to) — they must be *omitted from the rendered
      command entirely* when the gate is `"none"`, the same shape of fix
      `0-explorative`'s own `apply_config.sh` already applies to boolean CLI flags.
      **Deliberately not attempted** (unchanged conclusion): MoE expert-count/EPLB
      toggles and disaggregation/prefill-decode topology remain deployment-architecture
      choices, not per-instance parameters.
- [ ] **URGENT — confirm which pack version is actually installed on the Akamas
      instance**: as of 2026-07-14 the open branch
      `feature/attention-backend-and-block-size-categorical` is now at **`1.5.1`**
      (block_size ordinal conversion + DCP/PCP parameters + speculative-decoding gate
      pattern with corrected `spec_method`/`spec_tokens` naming, commits `2121136`,
      `ba048e4`, and `9e177fd`, still unpushed/unmerged/untagged locally). Both
      `origin/develop` and `origin/master` remain pinned at the **tagged** `1.2.0`,
      which has the old, broken `block_size: integer [1,128]` domain (no
      multiple-of-16 guard) — no `1.3.1`/`1.4.0`/`1.5.0`/`1.5.1` tag exists anywhere
      in the repo. Run `akamas describe optimization-pack vLLM` before starting any new
      study: if it reports `1.2.0` (or anything other than this feature-branch build),
      the crash this repo already root-caused and fixed is live again for that
      installation, and none of the new DCP/PCP/speculative-decoding parameters or the
      ordinal `block_size` conversion are available either. Ask the pack owner
      to merge and tag the branch (opening a PR/MR is the user's own separate decision,
      not yet made) rather than relying on an unreleased build persisting on the Akamas
      instance indefinitely.
- [x] **Pack improvement — `block_size` converted from `categorical` to `ordinal`,
      implemented 2026-07-14** (same commit `2121136` as above). Confirmed directly
      against Akamas' own docs (`docs.akamas.io/akamas-docs/reference/glossary/
      parameter`) that Akamas has a real, currently-shipping `ordinal` domain type
      distinct from `categorical` — categorical values are one-hot encoded with **no
      notion of order or adjacency** to the optimizer, while `ordinal` values are
      converted to an ordered real value the optimizer *can* exploit. Same 8 values/
      default as before, only the domain `type` changed. **Not yet validated against a
      live Akamas instance** — no `akamas` CLI was available where this work was done;
      the exact YAML syntax (`type: ordinal`, reusing `categories:`) was inferred by
      analogy with two other real Akamas packs that use `ordinal` for the same shape of
      problem (AWS pack's `aws_ec2_instance_size`, Node.js pack's
      `v8_max_semi_space_size_ordinal`), not confirmed from a fetchable raw-YAML
      example. **Run `akamas build optimization-pack`/`akamas describe optimization-pack
      vLLM` against a real 3.7.x instance before relying on this in a live study**, per
      `.claude/rules/akamas-yaml.md` — if rejected, the documented fallback is reverting
      to `categorical` with categories left in ascending order (still correct, just
      less search-efficient).
- [ ] **Two smaller pack-audit findings, informational for future study design, not pack
      bugs**: `prefix_caching_hash_algo`'s `xxhash`/`xxhash_cbor` categories require the
      optional `xxhash` Python package per the pack's own description — the same *shape*
      of risk as the original `block_size` problem (a categorical value not universally
      valid) but gated by an environment precondition, not a domain-type fix; confirm
      the package is installed in a study's target image before including these values,
      or add a `parameterConstraints` exclusion the way `0-explorative` already excludes
      them. Separately, `max_long_partial_prefills`'s own description states it "must be
      ≤ `max_num_partial_prefills`" — an unenforced cross-parameter constraint Akamas
      packs have no mechanism to encode themselves; any future study tuning both
      parameters together needs its own `parameterConstraints` entry for this, the same
      pattern `0-explorative` already used for the FlashInfer/`block_size` and
      `FLASH_ATTN`/`kv_cache_dtype` cases.
- [ ] **Evaluate NVIDIA Dynamo's `aiconfigurator`** for deployment-topology planning
      (aggregated vs. disaggregated, TP/PP degree, prefill/decode worker count, GPU
      count/type) ahead of any future multi-GPU or MoE study, per
      `knowledge/notes/2026-07-nvidia-dynamo-aiconfigurator.md`. This doesn't require
      pack support — it's a way to fix topology *before* scaffolding a study (via
      `/new-study`) rather than asking Akamas or the pack owner to model topology search
      directly. Coverage is limited to its pre-profiled model/GPU database (GPT/LLAMA/
      Qwen/Mixtral/DeepSeek-V3 on H100/H200/A100/B200/GB200) — check whether a study's
      actual model/GPU is covered before relying on it, and still validate its output
      with real load testing (`inference-perf`, per Q2) rather than trusting its
      simulated estimate alone.
- [ ] **Confirm whether the target cluster has a GPU-sharing scheduler** (NVIDIA
      Run:ai, MIG, time-slicing, MPS, or a Dynamic Resource Allocation (DRA) driver)
      before scoping backlog #4's MIG right-sizing study. **Narrowed by this repo's own
      2026-07-14/2026-07-15 decisions**: Section D's study #4 commits to **MIG only**
      as the partitioning mechanism (no time-slicing/MPS/Run:ai), requested via **DRA**
      with a classic device-plugin fallback (see Section D's "DRA prerequisites and
      risk") — so for this repo's own near-term plan, the open question is specifically
      "does the `p5.48xlarge` node's H100s support the MIG profile sizes study #4
      needs" (yes, H100 supports up to 7 MIG instances same as A100) and "does DRA or
      the classic device plugin end up being the actual working path" (per the
      fallback plan), not a broader "which of five mechanisms is available" survey.
      The general background below stays useful if this decision is ever revisited
      (e.g. a future study wants soft/oversubscribed sharing instead of MIG's hard
      isolation). Per
      `knowledge/notes/2026-07-gpu-fractioning-nvidia-runai.md` and
      `knowledge/notes/2026-07-kubernetes-gpu-scheduling-patterns.md`. If one is
      available, backlog #4 should test GPU-fraction-per-replica (not just replica
      count) as an additional dimension — this is a cluster/infra capability, not
      something the installed vLLM/Kubernetes packs currently model (confirm with
      `akamas describe optimization-pack vLLM`/`Kubernetes`, or the pack's own repo,
      https://gitlab.com/akamas/optimization-packs/vllm), and ties into the
      existing "Provision/document access to the shared cluster" debt item above: record
      which GPU-sharing mechanism (if any) that cluster has as part of that provisioning
      work. This isn't a binary yes/no — per
      `knowledge/notes/2026-07-ai-systems-performance-serving-tuning-checklist.md`, MIG
      (hard-partitioned GPU instances, up to 7 slices, guaranteed resources but
      unsuitable for tightly-coupled parallel jobs like TP-sharded inference — confirmed
      by `2026-07-kubernetes-gpu-scheduling-patterns.md`: MIG instances don't expose
      NVLink between them, so TP≥2 can't span MIG slices) and MPS (soft concurrent
      kernel sharing across processes, no hard isolation) are two distinct mechanisms
      with different trade-offs from each other and from Run:ai's fractional scheduler —
      confirm which one (if any) specifically, not just whether "a scheduler" exists.
      Also check the cluster's **Kubernetes version**: DRA (the newer, more expressive
      API for partial/topology-aware device requests) graduated to GA in **v1.34**
      (Sept 2025) — if the cluster predates that, DRA-based fractional GPU requests
      aren't available regardless of which GPU-sharing mechanism is otherwise installed.
      **Correction, per `2026-07-generative-ai-on-kubernetes-gpu-production-patterns.md`**:
      even on a v1.34+ cluster, **NVIDIA's own DRA driver is still a technical preview
      as of early 2026, not supported for production** — don't treat DRA as a viable
      path yet regardless of Kubernetes version; the device-plugin + label-based
      scheduling model (nodeSelector/affinity/taints, per the same source) remains the
      thing to actually check for and use today. Same source also notes MIG's
      isolation/granularity trade-off is explicitly framed as suited to *many small
      models sharing a GPU*, not this repo's single-large-model-per-GPU case — worth
      keeping in mind if backlog #4 is ever scoped around MIG specifically.
      **Update 2026-07-15 — decision made to proceed with DRA anyway**: per a more
      precise, direct re-check (see Section D's "DRA prerequisites and risk"), the core
      DRA API is genuinely GA and EKS-supported since 1.33/1.34; NVIDIA's driver's
      generic multi-GPU allocation is workable, but its MIG-specific GPU-allocation
      feature is still explicitly "not yet officially supported" upstream (confirmed
      directly from the driver's own README, not just inferred). This repo's own
      decision (2026-07-15) is to use DRA for both studies #3 (multi-GPU, the more
      mature path) and #4 (MIG, explicitly the less mature path), with a documented
      classic-device-plugin fallback for study #4 specifically if DRA-based MIG
      allocation proves unworkable — see Section D for the full detail. This
      supersedes the blanket "don't treat DRA as viable yet" framing above for this
      repo's own near-term plan, though the underlying caution (driver immaturity for
      MIG specifically) is still accurate and the reason the fallback exists.

## F. Futuribile — deprioritized, not abandoned

**Added 2026-07-15**, when Section D was rewritten around the two explicit validation
goals (Goal A: throughput at fixed hardware; Goal B: right-sized hardware for a target
throughput). Everything below doesn't map onto either goal directly — deprioritized in
favor of studies #2-#4, not deleted. Revisit any of these once the current plan
completes or a concrete need arises.

- **Kubernetes resource co-tuning (generic)** — co-tuning container CPU/memory
  requests/limits alongside vLLM parameters (tests H4, the original framing before this
  session's Goal A/B reframing). Orthogonal to both current goals; worth doing once
  studies #2-#4 establish GPU baselines to compare against (H100 for #3/#4;
  `2-larger-model-g7e`'s own baseline is RTX PRO 6000 Blackwell as of 2026-08-21, not
  H100 — see Section D).
- **Energy efficiency** — maximize tokens/s per watt (a GPU power-draw metric).
  Candidate prior, per `knowledge/notes/2026-07-ai-systems-performance-serving-tuning-checklist.md`:
  "for some models, going from a 100% to 80% power limit yields nearly the same speed
  at 20% less power usage." Confirm via `akamas describe optimization-pack GPU`
  whether a power-limit parameter exists; if not, power-capping needs a workflow task
  (`nvidia-smi -pl`) rather than an Akamas parameter.
- **MIG multi-tenant exploration** (distinct from study #4's single-model right-sizing
  question) — 2-3 *smaller, different* models (e.g. 1-3B parameter class) sharing MIG
  slices of one H100, aggregate multi-tenant throughput vs. a shared-full-GPU baseline.
  MIG's actual intended use case per the knowledge base, but not this repo's current
  single-model-per-GPU question — revisit if a genuinely multi-tenant workload need
  arises.
- **RAG and agent use cases** — deliberately deferred per this repo's own 2026-07-15
  decision to start with chatbot as the simplest real use case. Each needs a distinct
  dataset/load-pattern choice, not a reuse of the chatbot ShareGPT setup: RAG needs
  long-context/retrieval-heavy prompts and a strong focus on prefix-cache-hit-rate (per
  Q4's `MIN_CACHE_HIT_PCT`-style constraint idea); agent workloads need multi-step
  tool-calling traces, most naturally captured via `inference-perf`'s OpenTelemetry
  trace/DAG replay rather than its simpler ShareGPT/synthetic generators. Revisit once
  the chatbot use case's three studies are done and validated.
- **Speculative decoding as its own dedicated study** (beyond being one tunable
  dimension folded into the original study #2 plan, now superseded — `2-larger-model-g7e`
  pins it off instead, see that study's own README) — n-gram-style speculation shows its largest gains
  on high-repetition workloads (per `knowledge/notes/2026-07-speculative-decoding-survey.md`),
  which chatbot conversation isn't especially shaped for. A RAG or code-completion use
  case (see above) would be a much stronger test of `spec_method`/`spec_tokens`'s real
  value than the chatbot use case alone.
- **Hopper-specific FP8/NVFP4 deep dive** — H100's native FP8 Tensor Cores and NVFP4
  support go beyond what a general parameter search will naturally explore in depth;
  a dedicated quantization-focused study could push further once the chatbot baseline
  exists. **Note (2026-08-21)**: this is no longer study #2's territory — that study
  moved off Hopper entirely to RTX PRO 6000 Blackwell (SM120, a different FP8/FP4
  kernel-support story, see its own README) — relevant to studies #3/#4 instead, which
  remain on `p5.48xlarge`/H100.
- **llm-d adoption** — unchanged from Q6: N/A until real production-scale traffic
  exists (100M+ tokens/day, 100B+ parameter model, per the concrete thresholds already
  cited there). Nothing in Section D blocks adopting it later.
- **Disaggregation topology, MoE expert-count/EPLB** — unchanged conclusion from the
  pack-engineering decisions already made: deployment-architecture choices, not
  per-instance parameters: not attempted in the vLLM pack, not scoped as a study here.
- **Load-generator validation (GuideLLM vs. AIPerf vs. inference-perf)** — **resolved,
  not deferred**: this repo committed directly to `inference-perf` (2026-07-15, see
  Section D's intro) rather than running a dedicated validation study first. Listed
  here only for the historical record, not as a pending item.
- **Multi-node (beyond one 8-GPU `p5.48xlarge` node)** — not yet needed at any scale
  studies #2-#4 test; revisit only if a target throughput in a future study genuinely
  exceeds what 8 H100s on one node can deliver.
