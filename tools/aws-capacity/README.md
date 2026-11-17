# istanceAws

A local page with live AWS GPU capacity for the lab account (`AWS_PROFILE=lab`). Type
`istanceAws` in a terminal: it starts a small server on `127.0.0.1:8765` and opens the page; if
the server is already running it only opens the page. Ctrl+C in that terminal stops it.

Design: `docs/superpowers/specs/2026-11-17-aws-gpu-capacity-app-design.md`.

## What the page shows

- Every GPU instance type offered in the shown regions (default `us-east-2`, `us-east-1`,
  `eu-central-1`; tick more at the top), grouped by GPU and GPU memory, with vCPU, RAM and the
  on-demand price.
- For each availability zone, two lines:
  - **on-demand**: `YES` / `no` / `quota` from the last probe and how long ago; blank if never
    probed; a hatched cell means the type is not offered in that zone;
  - **spot**: the spot placement score (1 to 10, one request per type, target capacity 1) and the
    current spot price. Refreshed every 10 minutes and with "Refresh spot". A score is a
    likelihood, not a guarantee.
- Per region, the free on-demand vCPU quota for the G (G and VT) and P buckets. A `quota` cell
  means the type does not fit in what is free.

## On-demand probes cost money

On-demand capacity can only be checked by trying: the probe creates a 1-instance capacity
reservation and cancels it at once. A reservation is billed per second with a **60-second
minimum** at the on-demand price, so each `YES` costs one minute of the hourly price (about $0.06
for a g7e.4xlarge, about $1 for a p5en.48xlarge). A `no` costs nothing. The button shows the
worst case before you press it, and the page keeps a running total.

Probes run only when you press the button, only for the types you ticked and only in the regions
you chose next to it (default `us-east-2`). Inside a region they run one at a time.

Every reservation the tool creates carries the tag `created-by=istanceAws`. After each probe run,
at start-up and at Ctrl+C the tool cancels any active reservation with that tag, and never
touches others. The page shows how many tagged reservations are open (it should say 0). Each
reservation also has an end date one hour out, so it expires on its own if every cancel fails.

## Files

`awscli.py` (CLI wrapper), `catalog.py` (types, zones, prices cached for a day in
`~/.cache/istanceAws/`), `spot.py`, `quota.py`, `probe.py`, `app.py` (state and jobs),
`server.py` (HTTP and `main`), `static/index.html`, `istanceAws` (launcher, symlinked into
`~/.local/bin`). Python standard library and the `aws` CLI only.

## Tests

```bash
cd tools/aws-capacity && python3 -m unittest discover -s tests -v
```

The tests use fake AWS readers and never call AWS.

## Known gaps

- Capacity Blocks (24 h H100/H200 slots) are not shown.
- Spot placement scores: AWS accepts about 50 new request configurations per 24 hours
  (`MaxConfigLimitExceeded` after that, seen on 2026-11-17 with 65 GPU types) and lets
  already-used ones repeat. A configuration includes the region list, so scores are always asked
  for `us-east-2`, `us-east-1`, `eu-central-1` together, whatever the page shows, biggest GPUs
  first. Types past the limit say "score limited" and still show the spot price; other shown
  regions get the spot price only.
