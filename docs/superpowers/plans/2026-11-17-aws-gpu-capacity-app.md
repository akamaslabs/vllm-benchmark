# istanceAws Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `istanceAws` opens a local browser page listing every AWS GPU instance type by GPU, with
live spot score/price per AZ and on-demand capacity probes the user starts per type and region.

**Architecture:** A Python standard-library HTTP server (`server.py`) owns an `App` object that
reads AWS through the `aws` CLI (`awscli.py`), keeps everything in memory and runs background
threads (region loading, spot refresh every 600 s, one probe job per region). A single static page
polls `/api/state` and posts actions. A bash launcher starts the server or just opens the page.

**Tech Stack:** Python 3.12 stdlib (`http.server`, `threading`, `subprocess`, `unittest`), AWS CLI
v2, vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-11-17-aws-gpu-capacity-app-design.md`

## Global Constraints

- Every `aws` call runs with `AWS_PROFILE=lab` in its environment and `--output json`.
- No package installs; `boto3` is not available. Tests use `unittest`, not pytest.
- Code, comments and page text in English.
- Tag on every reservation the tool creates: `created-by=istanceAws`. Only reservations with
  that tag are ever cancelled.
- Default shown regions: `us-east-2`, `us-east-1`, `eu-central-1`. Selectable extra regions:
  `us-west-2`, `eu-north-1`, `eu-west-1`, `ap-northeast-1`. Default probe region: `us-east-2` only.
- Spot auto-refresh every 600 s. Port 8765 (env `ISTANCEAWS_PORT` overrides).
- Probes inside one region strictly sequential; regions in parallel.
- Reservation: 1 instance, `Linux/UNIX`, `targeted`, `--end-date-type limited`, end date now + 1 h
  (safety net if every cancel fails).
- Probe cost model: a probe that finds capacity costs on-demand price / 60 (60-second minimum).
- Pricing cache: `~/.cache/istanceAws/prices-<region>.json`, valid 24 h.
- Commits: none during execution; this repo commits only when the user asks.

## Review Focus

1. Ctrl+C (SIGINT) while a probe job is mid-reservation: every reservation tagged by the tool must
   end cancelled before the process exits (Task 4, `test_shutdown_sweeps_every_region`).
2. Expired or missing AWS credentials: the page shows the AWS message in an error banner instead of
   an empty table (Task 1 `test_parse_error_without_code`, Task 4 `test_load_error_reaches_snapshot`).
3. A second "Probe" click on a region that is already probing: rejected with 409, never two
   reservations alive in one region (Task 4, `test_second_probe_same_region_rejected`).
4. A type offered in the region but not in some AZ: that cell is "not offered", never probed and
   never shown as "no" (Task 4, `test_probe_skips_azs_not_offering_type`).
5. Spot placement score throttled for some types: those rows say "score limited" and still show the
   spot price (Task 2, `test_scores_error_is_returned_not_raised`).

---

## File Structure

```
tools/aws-capacity/
  istanceAws          bash launcher (symlinked as ~/.local/bin/istanceAws)
  awscli.py           run the aws CLI, parse JSON and errors
  catalog.py          GPU instance types, AZ offerings, on-demand prices (disk cache)
  spot.py             spot prices, AZ id -> name map, spot placement scores
  quota.py            G/P vCPU quota and usage
  probe.py            one capacity probe (create + cancel), tagged sweep
  app.py              App: in-memory state, background jobs, snapshot()
  server.py           HTTP routes, main(), signal handling, browser open
  static/index.html   the page
  README.md           what it does, how it spends money, how to run tests
  tests/test_*.py     unittest, fake `call` functions, no network
```

All AWS-reading functions take `call: Call = awscli.aws` so tests inject a fake.

---

### Task 1: AWS CLI wrapper and GPU catalog

**Files:**
- Create: `tools/aws-capacity/awscli.py`, `tools/aws-capacity/catalog.py`
- Test: `tools/aws-capacity/tests/test_awscli.py`, `tools/aws-capacity/tests/test_catalog.py`

**Interfaces:**
- Produces:
  - `class AwsError(Exception)` with attributes `code: str`, `message: str`
  - `Call = Callable[[list[str], str | None], Any]`
  - `aws(args: list[str], region: str | None = None) -> Any` (runs `aws <args> --output json [--region r]`, env `AWS_PROFILE=lab`, returns parsed JSON or `{}` on empty stdout, raises `AwsError` on non-zero exit)
  - `parse_error(stderr: str) -> AwsError` (code from `An error occurred (Code)`, else code `"CliError"` and the last non-empty stderr line as message)
  - `@dataclass(frozen=True) class GpuType: name: str; vcpu: int; mem_gib: int; gpu_count: int; gpu_model: str; gpu_mem_gib: int; group: str`
  - `parse_instance_types(doc: dict) -> list[GpuType]` (skips types without `GpuInfo`)
  - `load_gpu_types(region: str, call: Call = aws) -> list[GpuType]` (`ec2 describe-instance-types --filters Name=instance-type,Values=g*,p*`)
  - `load_offerings(region: str, call: Call = aws) -> dict[str, list[str]]` (type -> sorted AZ names, `ec2 describe-instance-type-offerings --location-type availability-zone`, same filter)
  - `parse_prices(price_list: list[str]) -> dict[str, float]` (each item is a JSON string from the Pricing API)
  - `load_prices(region: str, call: Call = aws, cache_dir: Path = DEFAULT_CACHE, now: Callable[[], float] = time.time) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

```python
# test_awscli.py
def test_parse_error_with_code(self):
    e = awscli.parse_error("\nAn error occurred (InsufficientInstanceCapacity) when calling the CreateCapacityReservation operation: There is no Spot capacity...\n")
    self.assertEqual(e.code, "InsufficientInstanceCapacity")
def test_parse_error_without_code(self):
    e = awscli.parse_error("\nUnable to locate credentials. You can configure credentials by running \"aws configure\".\n")
    self.assertEqual(e.code, "CliError"); self.assertIn("Unable to locate credentials", e.message)

# test_catalog.py  (fixture dicts inline, shaped like the CLI output)
def test_parse_g7e_12xlarge(self):
    t = catalog.parse_instance_types(DOC)[0]   # g7e.12xlarge: 48 vCPU, 524288 MiB, 2x "RTX PRO Server 6000" 98304 MiB
    self.assertEqual((t.vcpu, t.mem_gib, t.gpu_count, t.gpu_mem_gib), (48, 512, 2, 96))
    self.assertEqual(t.group, "RTX PRO Server 6000 96 GB")
def test_parse_skips_types_without_gpu(self): ...   # an entry without GpuInfo -> not returned
def test_parse_fractional_gpu(self):              # g6f.large: Count 0, L4, 2048 MiB
    self.assertEqual(t.group, "L4 fractional 2 GB")
def test_offerings_grouped_and_sorted(self):     # fake call returns 2c,2a for g7e.2xlarge
    self.assertEqual(catalog.load_offerings("us-east-2", call=fake)["g7e.2xlarge"], ["us-east-2a", "us-east-2c"])
def test_parse_prices(self):                     # one product JSON string, OnDemand USD "8.2860800000"
    self.assertAlmostEqual(catalog.parse_prices([P])["g7e.12xlarge"], 8.28608)
def test_prices_cache_hit_skips_call(self):      # cache file younger than 86400 s -> fake call never invoked
def test_prices_cache_expired_calls_again(self): # now() = mtime + 86401 -> fake call invoked once
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tools/aws-capacity && python3 -m unittest discover -s tests -v`
Expected: FAIL / ImportError on `awscli`, `catalog`.

- [ ] **Step 3: Implement `awscli.py` and `catalog.py`**

Pricing call: `pricing get-products --service-code AmazonEC2` on region `us-east-1`, filters
`TERM_MATCH` on `regionCode=<region>`, `operatingSystem=Linux`, `tenancy=Shared`,
`preInstalledSw=NA`, `capacitystatus=Used`, `licenseModel=No License required`,
`instanceFamily=GPU instance`. Group label: `f"{model} {mem} GB"`, or `f"{model} fractional {mem} GB"`
when `Count == 0`; `gpu_mem_gib = round(SizeInMiB / 1024)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tools/aws-capacity && python3 -m unittest discover -s tests -v`
Expected: all PASS.

- [ ] **Step 5: Check against real AWS (read-only, free)**

Run: `cd tools/aws-capacity && python3 -c "import catalog as c; t=c.load_gpu_types('us-east-2'); print(len(t)); print(len(c.load_prices('us-east-2')))"`
Expected: `56` GPU types (2026-11-17 count) and a price count close to it; list the types missing a
price, if any, in the README as a known gap.

---

### Task 2: Spot and quota readers

**Files:**
- Create: `tools/aws-capacity/spot.py`, `tools/aws-capacity/quota.py`
- Test: `tools/aws-capacity/tests/test_spot.py`, `tools/aws-capacity/tests/test_quota.py`

**Interfaces:**
- Consumes: `awscli.aws`, `awscli.AwsError`, `awscli.Call`
- Produces:
  - `parse_spot_prices(doc: dict) -> dict[str, dict[str, float]]` (type -> AZ -> latest price by `Timestamp`)
  - `load_spot_prices(region: str, call: Call = aws) -> dict[str, dict[str, float]]` (`ec2 describe-spot-price-history --product-descriptions Linux/UNIX --start-time <now UTC> --filters Name=instance-type,Values=g*,p*`)
  - `az_id_map(region: str, call: Call = aws) -> dict[str, str]` (zone id -> zone name)
  - `parse_scores(doc: dict, id_to_name: dict[str, str]) -> dict[str, int]` (AZ name -> score)
  - `load_scores(type_name: str, regions: list[str], id_to_name: dict[str, str], call: Call = aws) -> dict[str, int] | AwsError` (`ec2 get-spot-placement-scores --instance-types <t> --target-capacity 1 --single-availability-zone --region-names <regions>` on region `us-east-1`; returns the error instead of raising)
  - `QUOTA_CODES = {"G": "L-DB2E81BA", "P": "L-417A185B"}`
  - `bucket(type_name: str) -> str | None` (`g*`/`vt*` -> `"G"`, `p*` -> `"P"`, else `None`)
  - `used_vcpus(running: list[str], unused_reserved: list[tuple[str, int]], vcpu_of: dict[str, int]) -> dict[str, int]`
  - `load_quota(region: str, vcpu_of: dict[str, int], call: Call = aws) -> dict[str, dict[str, int]]` (`{"G": {"quota", "used", "free"}, "P": {...}}`; running = `describe-instances` in `pending`/`running`; reserved = active reservations' `AvailableInstanceCount`)

- [ ] **Step 1: Write the failing tests**

```python
def test_spot_price_keeps_latest(self):   # same type/AZ at 07:00 (2.0) and 09:00 (1.8) -> 1.8
def test_scores_map_zone_ids(self):       # {"AvailabilityZoneId": "use2-az3", "Score": 9} + map -> {"us-east-2c": 9}
def test_scores_error_is_returned_not_raised(self):
    r = spot.load_scores("g7e.2xlarge", ["us-east-2"], {}, call=raising(AwsError("MaxConfigLimitExceeded", "...")))
    self.assertIsInstance(r, AwsError)
def test_bucket(self):
    self.assertEqual([quota.bucket(t) for t in ("g7e.12xlarge", "vt1.3xlarge", "p5en.48xlarge", "m5a.xlarge")], ["G", "G", "P", None])
def test_used_vcpus_counts_unused_reservations(self):
    u = quota.used_vcpus(["g7e.12xlarge", "m5a.xlarge"], [("g7e.4xlarge", 1)], {"g7e.12xlarge": 48, "g7e.4xlarge": 16})
    self.assertEqual(u, {"G": 64, "P": 0})
def test_load_quota_free(self):           # quota 64.0, one running g7e.12xlarge -> {"quota": 64, "used": 48, "free": 16}
```

- [ ] **Step 2: Run tests to verify they fail** — same command, FAIL on import.
- [ ] **Step 3: Implement `spot.py` and `quota.py`.**
- [ ] **Step 4: Run tests to verify they pass** — all PASS.

---

### Task 3: Probe and tagged sweep

**Files:**
- Create: `tools/aws-capacity/probe.py`
- Test: `tools/aws-capacity/tests/test_probe.py`

**Interfaces:**
- Consumes: `awscli.aws`, `awscli.AwsError`, `awscli.Call`
- Produces:
  - `TAG_KEY = "created-by"`, `TAG_VALUE = "istanceAws"`
  - `classify(code: str) -> str` (`InsufficientInstanceCapacity` -> `"no"`; `InstanceLimitExceeded`, `ReservationCapacityExceeded`, `VcpuLimitExceeded` -> `"quota"`; else `"error"`)
  - `@dataclass class ProbeResult: region: str; type: str; az: str; result: str; at: float; detail: str = ""` (`result` in `"YES" | "no" | "quota" | "error" | "LEFT_ACTIVE"`)
  - `probe_one(region: str, type_name: str, az: str, call: Call = aws, now: Callable[[], float] = time.time) -> ProbeResult`
  - `list_tagged_active(region: str, call: Call = aws) -> list[str]` (filters `tag:created-by=istanceAws`, `state=active,pending`)
  - `sweep(region: str, call: Call = aws) -> list[str]` (cancels each, returns the ids it cancelled)

- [ ] **Step 1: Write the failing tests**

```python
def test_yes_creates_tagged_then_cancels(self):
    r = probe.probe_one("us-east-2", "g4dn.xlarge", "us-east-2a", call=rec)
    self.assertEqual(r.result, "YES")
    create, cancel = rec.calls
    self.assertIn("ResourceType=capacity-reservation,Tags=[{Key=created-by,Value=istanceAws}]", create)
    self.assertEqual(cancel[:2], ["ec2", "cancel-capacity-reservation"]); self.assertIn("cr-123", cancel)
def test_no_capacity(self):      # create raises AwsError("InsufficientInstanceCapacity") -> "no", no cancel call
def test_quota(self):            # AwsError("InstanceLimitExceeded") -> "quota"
def test_unknown_error(self):    # AwsError("UnauthorizedOperation", "msg") -> "error", detail "UnauthorizedOperation: msg"
def test_cancel_retried_then_left_active(self):
    # create ok, cancel raises 3 times -> result "LEFT_ACTIVE", detail contains "cr-123", exactly 3 cancel attempts
def test_sweep_cancels_only_listed(self):   # list returns ["cr-a"]; cancel called once with cr-a
```

- [ ] **Step 2: Run tests to verify they fail** — FAIL on import.
- [ ] **Step 3: Implement `probe.py`** (cancel retried 3 times, 1 s apart; end date `now() + 3600` as `%Y-%m-%dT%H:%M:%SZ`).
- [ ] **Step 4: Run tests to verify they pass** — all PASS.

---

### Task 4: App state, background jobs, HTTP API

**Files:**
- Create: `tools/aws-capacity/app.py`, `tools/aws-capacity/server.py`
- Test: `tools/aws-capacity/tests/test_app.py`, `tools/aws-capacity/tests/test_server.py`

**Interfaces:**
- Consumes: everything above. `App` receives its readers as constructor arguments so tests pass fakes.
- Produces:
  - `class Busy(Exception)` (message names the region)
  - `class App:`
    - `__init__(self, regions: list[str], probe_regions: list[str], readers: Readers, clock=time.time)` where `Readers` is a dataclass of the callables `load_gpu_types, load_offerings, load_prices, az_id_map, load_spot_prices, load_scores, load_quota, probe_one, list_tagged_active, sweep` (defaults: the real module functions)
    - `load_region(region: str) -> None` (synchronous; errors go to `errors`)
    - `set_regions(regions: list[str]) -> None` (loads new ones in threads)
    - `refresh_spot() -> None` (synchronous)
    - `plan_probe(types: list[str], regions: list[str]) -> dict` -> `{"planned": int, "max_cost": float}` (counts only AZs that offer each type)
    - `start_probe(types: list[str], regions: list[str]) -> dict` (raises `Busy` if any requested region is probing, `ValueError` if nothing to probe; starts one thread per region; after each region job: `sweep`, refresh `active_tagged` and `quota`)
    - `snapshot() -> dict` (JSON-safe; keys below)
    - `shutdown() -> None` (sweep every region ever shown or probed)
  - `make_server(app: App, port: int) -> ThreadingHTTPServer`, `main(argv: list[str] | None = None) -> int` (`--port`, `--no-browser`)
  - Routes: `GET /` (static/index.html), `GET /api/ping` -> `{"ok": true, "name": "istanceAws"}`, `GET /api/state`, `POST /api/spot/refresh` -> 202, `POST /api/regions {"regions": [...]}` -> 202, `POST /api/probe {"types": [...], "regions": [...]}` -> 202 with `plan_probe` result, 409 `{"error"}` on `Busy`, 400 on `ValueError`.

Snapshot keys (the page depends on these exact names):
`regions, probe_regions, all_regions, azs{region: [az]}, groups[{group, types[]}]` (sorted by
`gpu_mem_gib` desc, then group name), `types{name: GpuType fields}`, `offered{region: {type: [az]}}`,
`price{region: {type: usd}}`, `spot_price{region: {type: {az: usd}}}`, `spot_score{type: {az: int}}`,
`spot_score_error{type: str}`, `spot_at`, `probe{region: {type: {az: {result, at, detail}}}}`,
`probing{region: {done, total, current}}`, `quota{region: {...}}`, `active_tagged{region: int}`,
`cost_usd`, `errors[str]` (last 20), `loading[region]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_plan_counts_offered_azs_and_cost(self):
    # g7e.2xlarge offered in us-east-2a/2b/2c at 3.36 -> {"planned": 3, "max_cost": round(3*3.36/60, 2)}
def test_probe_skips_azs_not_offering_type(self):
    # type offered only in us-east-2b -> fake probe_one called once with az "us-east-2b"
def test_probe_records_results_and_cost(self):
    # fake probe_one returns YES for 2a, no for 2b -> snapshot probe[...]["us-east-2a"]["result"] == "YES", cost_usd == round(3.36/60, 4)
def test_second_probe_same_region_rejected(self):
    # first job blocked on an Event inside fake probe_one; second start_probe(["x"], ["us-east-2"]) raises Busy
def test_region_job_ends_with_sweep(self):    # fake sweep called once with "us-east-2" after the job
def test_shutdown_sweeps_every_region(self):  # regions shown + probed -> sweep called for each
def test_scores_error_marks_type(self):       # load_scores returns AwsError -> spot_score_error[type] == "score limited", spot_price kept
def test_load_error_reaches_snapshot(self):   # load_gpu_types raises AwsError("ExpiredToken", "...") -> "ExpiredToken" in snapshot errors[0]
# test_server.py: real ThreadingHTTPServer on port 0 with an App built from fakes
def test_ping(self): ...
def test_state_is_json(self): ...
def test_probe_conflict_is_409(self): ...
def test_probe_empty_is_400(self): ...
```

- [ ] **Step 2: Run tests to verify they fail** — FAIL on import.
- [ ] **Step 3: Implement `app.py` and `server.py`.** One `threading.Lock` guards all state;
  `snapshot()` copies under the lock. `main()`: sweep the default regions, start the server, load
  regions in background threads, start a daemon thread that calls `refresh_spot()` once all regions
  have loaded and then every 600 s, open `http://127.0.0.1:<port>/` with `webbrowser.open` unless
  `--no-browser`, and on SIGINT/SIGTERM call `app.shutdown()` before exiting (print the ids it
  cancelled).
- [ ] **Step 4: Run tests to verify they pass** — all PASS.

---

### Task 5: The page

**Files:**
- Create: `tools/aws-capacity/static/index.html` (inline CSS and JS, no external assets)

**Interfaces:**
- Consumes: the routes and snapshot keys of Task 4.

Required content (layout and colours are the builder's call; light and dark via
`prefers-color-scheme`):
- Header: title `istanceAws`, shown-region checkboxes (all `all_regions`), text filter (matches
  type, GPU model or group).
- Quota strip: one block per shown region, `G used/quota (free)` and `P used/quota (free)`.
- Spot line: `spot refreshed HH:MM UTC` and a Refresh button.
- Probe bar: probe-region checkboxes (default from `probe_regions`), button text
  `Probe on-demand: N types, M AZs, max X USD` computed client-side the same way as `plan_probe`,
  disabled when M is 0; progress `done/total, current` while `probing`; session cost `cost_usd`;
  `active_tagged` total, highlighted if not 0.
- Error banner with `errors`.
- Table: one header row per group (checkbox ticks all its types, label = group), one row per type:
  checkbox, type, vCPU, RAM GiB, `N x model, M GB`, on-demand $/h (first shown region that has a
  price). Then one column per AZ of every shown region, grouped under the region name. Each AZ cell:
  - top: on-demand result badge (`YES` green, `no` red, `quota` amber, `error`/`LEFT_ACTIVE` with
    `detail` as tooltip), with age (`3m`); blank if never probed; `—` if not offered there;
  - bottom: spot score (1-10, coloured scale) and spot price; `score limited` when
    `spot_score_error` has the type.
- Polling: `/api/state` every 2 s while any region is probing or loading, else every 15 s.

- [ ] **Step 1: Write the page.**
- [ ] **Step 2: Verified in Task 6 Step 3** against real read-only data (no demo mode).

---

### Task 6: Launcher, README, real run

**Files:**
- Create: `tools/aws-capacity/istanceAws` (bash, executable), `tools/aws-capacity/README.md`
- Symlink: `~/.local/bin/istanceAws -> <repo>/tools/aws-capacity/istanceAws`

Launcher behaviour: resolve its own directory through the symlink (`readlink -f`); if
`curl -fsS http://127.0.0.1:$PORT/api/ping` answers, `open` the page and exit; else
`exec python3 server.py --port "$PORT" "$@"`.

README: what it shows, that on-demand probes cost one minute of on-demand price per YES, the tag and
sweep, how to run the tests, known gaps (types without price).

- [ ] **Step 1: Write launcher and README, `chmod +x`, create the symlink.**
- [ ] **Step 2: Run all tests.** `cd tools/aws-capacity && python3 -m unittest discover -s tests -v` — all PASS.
- [ ] **Step 3: Real run.** `istanceAws --no-browser` in the background; `curl -s localhost:8765/api/state`
  until `loading` is empty and `spot_at` is set. Expected: three regions, `groups` includes
  `RTX PRO Server 6000 96 GB` and `H200 141 GB`, `quota.us-east-2.G.quota == 64`. Open the page
  in Chrome and take one screenshot: groups, quota strip, spot cells and the probe button render,
  no JS errors in the console.
- [ ] **Step 4: One real probe, cheapest type.** `POST /api/probe {"types": ["g4dn.xlarge"], "regions": ["us-east-2"]}`;
  wait for the job; expected results in all three AZs, `active_tagged.us-east-2 == 0`, and
  `aws ec2 describe-capacity-reservations` shows only `cr-0aeb8916b9312c7a2` active.
- [ ] **Step 5: Second launch while running.** `istanceAws` again prints nothing and opens the page
  (ping path), no second server.
- [ ] **Step 6: Ctrl+C.** Stop the server; the log lists the sweep; no tagged reservation remains.
