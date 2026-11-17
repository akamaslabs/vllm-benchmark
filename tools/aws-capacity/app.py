"""In-memory state of istanceAws and the background jobs that fill it."""
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable

import catalog
import probe
import quota
import spot
from awscli import AwsError

ALL_REGIONS = ["us-east-2", "us-east-1", "eu-central-1", "us-west-2", "eu-north-1", "eu-west-1",
               "ap-northeast-1"]
DEFAULT_REGIONS = ["us-east-2", "us-east-1", "eu-central-1"]
DEFAULT_PROBE_REGIONS = ["us-east-2"]
SPOT_INTERVAL_S = 600
SCORE_WORKERS = 8
# AWS accepts about 50 new spot placement score configurations per 24 h (MaxConfigLimitExceeded
# after that) and lets already-used ones repeat. A configuration includes the region list, so it
# stays fixed whatever regions the page shows, and the biggest GPUs are asked for first.
SCORE_REGIONS = ["us-east-2", "us-east-1", "eu-central-1"]


CLEAN_RESULTS = ("YES", "no", "quota")  # results after which no reservation of ours can be alive


class Busy(Exception):
    pass


@dataclass
class Readers:
    load_gpu_types: Callable = catalog.load_gpu_types
    load_offerings: Callable = catalog.load_offerings
    load_prices: Callable = catalog.load_prices
    az_id_map: Callable = spot.az_id_map
    load_spot_prices: Callable = spot.load_spot_prices
    load_scores: Callable = spot.load_scores
    load_quota: Callable = quota.load_quota
    probe_one: Callable = probe.probe_one
    list_tagged_active: Callable = probe.list_tagged_active
    sweep: Callable = probe.sweep


class App:
    def __init__(self, regions: list[str], probe_regions: list[str], readers: Readers | None = None,
                 clock: Callable[[], float] = time.time):
        self.r = readers or Readers()
        self.clock = clock
        self.lock = threading.Lock()
        self.regions = list(regions)
        self.probe_regions = list(probe_regions)
        self.touched = set(regions) | set(probe_regions)
        self.types: dict[str, catalog.GpuType] = {}
        self.region_types: dict[str, set[str]] = {}
        self.azs: dict[str, list[str]] = {}
        self.id_to_name: dict[str, str] = {}
        self.offered: dict[str, dict[str, list[str]]] = {}
        self.price: dict[str, dict[str, float]] = {}
        self.spot_price: dict[str, dict] = {}
        self.spot_score: dict[str, dict[str, int]] = {}
        self.spot_score_error: dict[str, str] = {}
        self.spot_at: float | None = None
        self.spot_running = False
        self.probe: dict[str, dict] = {}
        self.probing: dict[str, dict] = {}
        self.quota: dict[str, dict] = {}
        self.active_tagged: dict[str, int] = {}
        self.cost_usd = 0.0
        self.errors: deque[str] = deque(maxlen=20)
        self.loading: set[str] = set()
        self.stopping = threading.Event()
        self.threads: list[threading.Thread] = []
        self.probe_threads: list[threading.Thread] = []

    # --- helpers -------------------------------------------------------------------------
    def _error(self, where: str, exc: BaseException) -> None:
        stamp = time.strftime("%H:%M:%S", time.gmtime(self.clock()))
        with self.lock:
            self.errors.appendleft(f"{stamp} UTC {where}: {exc}")

    def _vcpu_of(self) -> dict[str, int]:
        with self.lock:
            return {n: t.vcpu for n, t in self.types.items()}

    def _refresh_quota_and_tags(self, region: str) -> None:
        try:
            q = self.r.load_quota(region, self._vcpu_of())
            with self.lock:
                self.quota[region] = q
        except Exception as e:  # noqa: BLE001 - shown on the page
            self._error(f"{region} quota", e)
        try:
            n = len(self.r.list_tagged_active(region))
            with self.lock:
                self.active_tagged[region] = n
        except Exception as e:  # noqa: BLE001
            self._error(f"{region} reservations", e)

    # --- loading ---------------------------------------------------------------------------
    def load_region(self, region: str) -> None:
        with self.lock:
            self.loading.add(region)
        try:
            types = self.r.load_gpu_types(region)
            offerings = self.r.load_offerings(region)
            names = {t.name for t in types}
            with self.lock:
                for t in types:
                    self.types[t.name] = t
                self.region_types[region] = names
                self.offered[region] = {k: v for k, v in offerings.items() if k in names}
            try:
                prices = self.r.load_prices(region)
                with self.lock:
                    self.price[region] = {k: v for k, v in prices.items() if k in names}
            except Exception as e:  # noqa: BLE001
                self._error(f"{region} prices", e)
            m = self.r.az_id_map(region)
            with self.lock:
                self.id_to_name.update(m)
                self.azs[region] = sorted(m.values())
            self._refresh_quota_and_tags(region)
        except Exception as e:  # noqa: BLE001
            self._error(region, e)
        finally:
            with self.lock:
                self.loading.discard(region)

    def set_regions(self, regions: list[str]) -> None:
        unknown = [r for r in regions if r not in ALL_REGIONS]
        if unknown or not regions:
            raise ValueError(f"unknown or empty regions: {unknown or regions}")
        with self.lock:
            self.regions = [r for r in ALL_REGIONS if r in regions]
            self.touched |= set(regions)
            new = [r for r in regions if r not in self.region_types and r not in self.loading]
        if new:
            def load_then_spot():
                self._load_many(new)
                self.refresh_spot()
            self._spawn(load_then_spot)

    def _load_many(self, regions: list[str]) -> None:
        ts = [threading.Thread(target=self.load_region, args=(r,), daemon=True) for r in regions]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    def _spawn(self, fn, *args, probe_job: bool = False) -> threading.Thread:
        t = threading.Thread(target=fn, args=args, daemon=True)
        with self.lock:
            self.threads.append(t)
            if probe_job:
                self.probe_threads.append(t)
        t.start()
        return t

    # --- spot ------------------------------------------------------------------------------
    def refresh_spot(self) -> None:
        with self.lock:
            if self.spot_running:
                return
            self.spot_running = True
            regions = [r for r in self.regions if r in self.region_types]
            names = sorted(set().union(*(self.region_types[r] for r in regions))) if regions else []
            id_map = dict(self.id_to_name)
        try:
            for region in regions:
                try:
                    sp = self.r.load_spot_prices(region)
                    with self.lock:
                        self.spot_price[region] = sp
                except Exception as e:  # noqa: BLE001
                    self._error(f"{region} spot prices", e)
            if names:
                with ThreadPoolExecutor(SCORE_WORKERS) as pool:
                    results = list(pool.map(lambda n: (n, self.r.load_scores(n, SCORE_REGIONS, id_map)),
                                            self._score_order(names)))
                with self.lock:
                    for n, res in results:
                        if isinstance(res, AwsError):
                            self.spot_score_error[n] = ("score limited" if "Limit" in res.code or "Throttl" in res.code
                                                        else f"score error: {res.code}")
                        else:
                            self.spot_score[n] = res
                            self.spot_score_error.pop(n, None)
            with self.lock:
                self.spot_at = self.clock()
        finally:
            with self.lock:
                self.spot_running = False

    def _score_order(self, names: list[str]) -> list[str]:
        with self.lock:
            ts = [self.types[n] for n in names]
        return [t.name for t in sorted(ts, key=lambda t: (-t.gpu_mem_gib, -t.gpu_count, -t.vcpu, t.name))]

    def request_spot_refresh(self) -> None:
        self._spawn(self.refresh_spot)

    def start_background(self, interval: float = SPOT_INTERVAL_S) -> None:
        def loop():
            self._load_many(self.regions)
            while not self.stopping.is_set():
                self.refresh_spot()
                self.stopping.wait(interval)
        threading.Thread(target=loop, daemon=True).start()

    # --- probes ----------------------------------------------------------------------------
    def _jobs(self, types: list[str], regions: list[str]) -> dict[str, list[tuple[str, str]]]:
        with self.lock:
            return {r: [(t, az) for t in types for az in self.offered.get(r, {}).get(t, [])]
                    for r in regions}

    def plan_probe(self, types: list[str], regions: list[str]) -> dict:
        jobs = self._jobs(types, regions)
        with self.lock:
            cost = sum(self.price.get(r, {}).get(t, 0.0) / 60 for r, js in jobs.items() for t, _ in js)
        return {"planned": sum(len(js) for js in jobs.values()), "max_cost": round(cost, 2)}

    def start_probe(self, types: list[str], regions: list[str]) -> dict:
        if self.stopping.is_set():
            raise Busy("shutting down")
        unknown = [r for r in regions if r not in ALL_REGIONS]
        if unknown:
            raise ValueError(f"unknown regions: {unknown}")
        plan = self.plan_probe(types, regions)
        jobs = {r: js for r, js in self._jobs(types, regions).items() if js}
        with self.lock:
            busy = [r for r in regions if r in self.probing]
            if busy:
                raise Busy(f"probe already running in {', '.join(busy)}")
            if not jobs:
                raise ValueError("nothing to probe: tick types offered in the chosen regions")
            for r, js in jobs.items():
                self.probing[r] = {"done": 0, "total": len(js), "current": ""}
                self.touched.add(r)
        for r, js in jobs.items():
            self._spawn(self._probe_region, r, js, probe_job=True)
        return plan

    def _probe_region(self, region: str, jobs: list[tuple[str, str]]) -> None:
        try:
            for t, az in jobs:
                if self.stopping.is_set():
                    break
                with self.lock:
                    self.probing[region]["current"] = f"{t} {az}"
                try:
                    res = self.r.probe_one(region, t, az)
                except Exception as e:  # noqa: BLE001
                    self._error(f"probe {t} {az}", e)
                    res = probe.ProbeResult(region, t, az, "error", self.clock(), str(e))
                with self.lock:
                    cell = {"result": res.result, "at": res.at, "detail": res.detail}
                    self.probe.setdefault(region, {}).setdefault(t, {})[az] = cell
                    if res.result in ("YES", "LEFT_ACTIVE"):
                        self.cost_usd += self.price.get(region, {}).get(t, 0.0) / 60
                    self.probing[region]["done"] += 1
                if res.result not in CLEAN_RESULTS and not self._region_clean(region):
                    break  # never two reservations alive in one region
        finally:
            try:
                self.r.sweep(region)
            except Exception as e:  # noqa: BLE001
                self._error(f"{region} sweep", e)
            if not self.stopping.is_set():
                self._refresh_quota_and_tags(region)
            with self.lock:
                self.probing.pop(region, None)

    def _region_clean(self, region: str) -> bool:
        """After an unclear probe result a reservation may be alive: sweep and confirm none
        tagged by this tool is left before creating the next one."""
        try:
            self.r.sweep(region)
            left = self.r.list_tagged_active(region)
        except Exception as e:  # noqa: BLE001
            self._error(f"{region} stopped probing, could not confirm no reservation is open", e)
            return False
        if left:
            self._error(f"{region} stopped probing", RuntimeError(f"reservations still open: {', '.join(left)}"))
            return False
        return True

    def wait_idle(self, timeout: float = 60, probes_only: bool = False) -> None:
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                self.threads = [t for t in self.threads if t.is_alive()]
                self.probe_threads = [t for t in self.probe_threads if t.is_alive()]
                alive = list(self.probe_threads if probes_only else self.threads)
            if not alive or time.monotonic() > deadline:
                return
            alive[0].join(max(0.0, deadline - time.monotonic()))

    def shutdown(self, timeout: float = 60) -> dict[str, dict[str, list[str]]]:
        """Stop starting probes, let in-flight probes return (not spot or region loads), then
        cancel every tagged reservation in every region this session touched and list what is
        still open."""
        self.stopping.set()
        self.wait_idle(timeout, probes_only=True)
        with self.lock:
            regions = sorted(self.touched)
        report = {}
        for r in regions:
            cancelled, left = [], []
            try:
                cancelled = self.r.sweep(r)
            except Exception:  # noqa: BLE001 - the list below says what is left
                pass
            try:
                left = self.r.list_tagged_active(r)
            except Exception as e:  # noqa: BLE001
                left = [f"unknown, listing failed: {e}"]
            report[r] = {"cancelled": cancelled, "still_active": left}
        return report

    # --- snapshot --------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            shown = set().union(*(self.region_types.get(r, set()) for r in self.regions)) if self.regions else set()
            groups: dict[str, list[catalog.GpuType]] = {}
            for n in shown:
                t = self.types[n]
                groups.setdefault(t.group, []).append(t)
            ordered = sorted(groups.items(), key=lambda kv: (-kv[1][0].gpu_mem_gib, kv[0]))
            return {
                "regions": list(self.regions),
                "probe_regions": list(self.probe_regions),
                "all_regions": list(ALL_REGIONS),
                "azs": {r: list(v) for r, v in self.azs.items()},
                "groups": [{"group": g, "types": [t.name for t in sorted(ts, key=lambda t: (t.gpu_count, t.vcpu, t.name))]}
                           for g, ts in ordered],
                "types": {n: asdict(self.types[n]) for n in shown},
                "offered": {r: {k: list(v) for k, v in o.items()} for r, o in self.offered.items()},
                "price": {r: dict(p) for r, p in self.price.items()},
                "spot_price": {r: {k: dict(v) for k, v in p.items()} for r, p in self.spot_price.items()},
                "spot_score": {k: dict(v) for k, v in self.spot_score.items()},
                "spot_score_error": dict(self.spot_score_error),
                "spot_at": self.spot_at,
                "spot_running": self.spot_running,
                "probe": {r: {t: dict(c) for t, c in p.items()} for r, p in self.probe.items()},
                "probing": {r: dict(p) for r, p in self.probing.items()},
                "quota": {r: dict(q) for r, q in self.quota.items()},
                "active_tagged": dict(self.active_tagged),
                "cost_usd": round(self.cost_usd, 4),
                "errors": list(self.errors),
                "loading": sorted(self.loading),
                "now": self.clock(),
            }
