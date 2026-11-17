# istanceAws: live AWS GPU capacity page (design, 2026-11-17)

## Intent

Typing `istanceAws` in a terminal opens a browser page that lists every AWS GPU instance type,
labelled with its GPU, and shows where capacity is right now: spot (automatic) and on-demand
(only for the types the user picks). It replaces the hand-run capacity-reservation probes used
before each study (see `studies/17-.../infra/eks/gpu-capacity-fallback.sh probe`) and answers
"where can I get these GPUs today" without opening the console.

## Goals

- One command, `istanceAws`, starts a local server in the foreground (Ctrl+C stops it) and opens
  the page in the default browser. If the server is already running, it only opens the page.
- Every GPU instance type offered in the selected regions (default `us-east-2`, `us-east-1`,
  `eu-central-1`; more can be ticked, e.g. `us-west-2`), grouped by GPU model (RTX PRO 6000,
  RTX PRO 4500, L40S, L4, A10G, T4, A100 40/80, H100, H200, B200, B300...). Each row: type, vCPU,
  RAM, `N x GPU model, memory per GPU`, total GPU memory, on-demand price per region.
- **Spot, automatic:** per AZ, the spot placement score (1-10, one request per type, target
  capacity 1, single-AZ) and the current spot price. Refreshed every 10 minutes and on a button.
  Free.
- **On-demand, on request only:** the user ticks types (a tick on a GPU group ticks all its
  sizes), picks the region(s) to probe in a selector of its own, separate from the regions
  shown (default `us-east-2` alone), and presses "Probe on-demand". Before running, the page
  shows how many reservations it will try and the worst-case cost. For each ticked type and
  each AZ of the picked regions that offers it: create a
  1-instance targeted capacity reservation, cancel it immediately. Cell shows YES / no / quota /
  error with the time of the probe. Results update live while the probe runs.
- Quota strip per region: G+VT and P on-demand vCPU quota, vCPU in use (running instances plus
  unused reservations), free. Explains why a cell says "quota".
- Session cost counter: probes that found capacity x (on-demand price / 60), since a
  reservation is billed per second with a 60-second minimum.

## Non-goals (v1)

- Launching instances or node groups, scaling the cluster.
- Capacity Block offerings (H200/H100 24 h slots). Read-only and free, a candidate for v2.
- History or charts over time. Results live in the server's memory for the session.

## Anti-goals

- A reservation left alive. Every reservation the tool creates carries the tag
  `created-by=istanceAws`; after each probe, at start-up and at Ctrl+C the server cancels any
  active reservation with that tag, and the page shows how many are active (must be 0).
- Touching reservations the tool did not create (e.g. the open one behind the g7e.12xlarge node).
- Any on-demand probe the user did not start.

## Constraints

- AWS account via `AWS_PROFILE=lab` only (the default profile points at another account).
- No package installs: Python 3 standard library plus the `aws` CLI v2 already on the machine
  (`boto3` is not installed).
- Probes inside one region run one at a time (two reservations at once can trip the vCPU quota
  and report a false "quota"); regions run in parallel.
- AWS may throttle new spot placement score configurations within 24 h; repeating the same
  ones is allowed. A throttled row shows "score limited" and keeps the spot price.
- Repo files in English, so the page text is in English.

## Approaches

- **A (picked by the user, 2026-11-17):** spot and prices automatic for every GPU type;
  on-demand probed only for the types the user ticks, in the region(s) the user picks for the
  probe (e.g. `us-east-2` only), on a button, with the worst-case cost shown first.
- B: as A, plus an automatic on-demand re-probe of the ticked types every N minutes with a cost
  counter. Dropped: spends money without a click.
- C: on-demand probe of every GPU type on each refresh. Dropped: about 500 reservations per
  round, minutes long, several dollars each time, and most large types fail on quota anyway.

## Shape

```
istanceAws                         regions: [x] us-east-2 [x] us-east-1 [x] eu-central-1 [ ] us-west-2
quota  us-east-2  G 48/64 (16 free)  P 0/384     us-east-1  G 0/768  P 0/768   ...
spot refreshed 09:41 UTC [refresh]
probe in: (x) us-east-2 ( ) us-east-1 ( ) eu-central-1   [Probe on-demand: 4 types, 12 AZs, max 0.60 USD]

[x] RTX PRO Server 6000 (96 GB)          on-demand              spot score / price
    type          vCPU  GPUs     $/h     2a  2b  2c | 1b  1c  1d  ...   2a     2b     2c   ...
[x] g7e.2xlarge     8   1x96GB   3.36    no  no  no | no  YES YES        3 1.12 1 1.30 6 1.05
[x] g7e.12xlarge   48   2x96GB   8.29    quota...   | no  no  YES        ...
[ ] H200 (141 GB)
[ ] p5en.48xlarge 192   8x141GB  63.30   -- not probed --               1 ...
```

## Left to the builder

- Where the code lives: `tools/aws-capacity/` (server, page, launcher); the launcher is
  symlinked as `~/.local/bin/istanceAws`, already on `PATH`.
- Port (default 8765), polling interval of the page, layout details, colours.
- How prices are fetched (Pricing API per type, cached on disk for a day).
