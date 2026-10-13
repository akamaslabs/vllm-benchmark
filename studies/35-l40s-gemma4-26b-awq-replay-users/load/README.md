# load/: the customer's dataset template (not in git)

The customer's replay dataset arrives as a **template**: their AIPerf mooncake_trace file with every
text replaced by a placeholder of the same length (`{{instructions: 3113 chars}}`, `{{user: 30
chars}}`, `{{assistant: 139 chars}}`, `{{input: 186 chars}}`, `{{text: omitted}}`). The template is
customer data: `.gitignore` here keeps every `*.jsonl` / `*.jsonl.gz` out of git. Keep the file
here (for the local dry run) and on the toolbox (for the study); never commit it.

## Received

- 2026-10-12, by email from the customer, with their `run_benchmark.sh` (study 34's `load/`; the same
  file again on 2026-10-13):
  `dataset_u000.template.jsonl` (864 KB, 2848 lines). Their full dataset is 100 users over one
  hour (> 300 MB, not shared: its synthetic texts may not be fully anonymized); this template is
  its first 10 users (`user_0000`-`user_0009`): 10 users rendered here come to ~26 MB, so 100 to
  ~300 MB, as their figure.

## What it holds (computed 2026-10-12)

- 10 sessions, 212-416 requests each (mean 285), ~3400-3900 s of think time per session
  (3.3-7.2 requests/min per user at zero latency). Delays: median 0.5 s, 12 % at the 60 s cap;
  requests come in chains (delay < 2 s) of 2.3 on average, up to 16: several calls per user
  action.
- Per request, characters (instructions + input): median ~6,200, mean ~9,300, p90 ~23,000, p99
  ~51,000, max ~108,000 (~1,500 / 2,300 / 5,800 / 13,000 / 27,000 tokens at ~4 characters per
  token, an estimate). 1544 requests carry `instructions` (892 distinct lengths, 83 of them in
  more than one session) and an `input` list with the history (1, 3, 5, 7 or 9 messages); 1304
  carry a single string `input`.
- Replies in the history (real answers, thinking off in their data): median 325 characters,
  p90 ~1,600, p99 ~5,100. No output length anywhere else: no `max_output_tokens`.
- 71 requests carry `text` (an output format, omitted): dropped by the renderer.
- Every payload: model `aiservices_gemma`, temperature 0.1 (9 at 0), top_p 1, store false,
  stream true; no `reasoning`, no tools.

## On the toolbox

```bash
D=/work/customer-data/35-l40s-gemma4-26b-awq-replay-users
kubectl -n akamas-41 exec -i deploy/toolbox -c toolbox -- \
  sh -c "mkdir -p $D && cat > $D/dataset_u000.template.jsonl" \
  < studies/35-l40s-gemma4-26b-awq-replay-users/load/dataset_u000.template.jsonl
```

`run_test.sh` reads it from there (`RT_TEMPLATE`), gzips it (28 KB) into the ConfigMap
`aiperf-replay` with `../k8s/render_dataset_it.py`, and the Job renders it with Italian texts.

The same file (study 34's copy, identical: received 2026-10-12, sent again 2026-10-13 with
`usage_stats.png`, their production distributions, which the calibration in
`../../../probes/gemma4-customer-dataset-it/` uses).

## Open (to ask the customer)

- The other 90 users' templates: they hold no text either, and with them no rendered user
  repeats a template session.
- An id in each placeholder for identical texts (their "many repeated prompts"): the renderer
  would give the same text to the same id instead of guessing from the length.
- The output tokens per request (`usage.output_tokens`), the `text` schemas, how `delay` was
  measured, their `benchmark.env` (USERS, DURATION, RAMP, CAP, GRACE) and AIPerf version.
