"""An App wired to in-memory fake readers (no AWS)."""
import threading

import app as appmod
from catalog import GpuType
from probe import ProbeResult

TYPES = [
    GpuType("g7e.2xlarge", 8, 64, 1, "RTX PRO Server 6000", 96, "RTX PRO Server 6000 96 GB"),
    GpuType("g7e.12xlarge", 48, 512, 2, "RTX PRO Server 6000", 96, "RTX PRO Server 6000 96 GB"),
    GpuType("g6.xlarge", 4, 16, 1, "L4", 22, "L4 22 GB"),
]
OFFERED = {"g7e.2xlarge": ["us-east-2a", "us-east-2b", "us-east-2c"],
           "g7e.12xlarge": ["us-east-2b"], "g6.xlarge": ["us-east-2a"]}
PRICES = {"g7e.2xlarge": 3.36, "g7e.12xlarge": 8.29, "g6.xlarge": 0.8}


class Recorder:
    def __init__(self):
        self.sweeps = []
        self.probes = []
        self.events = []  # ordered log of ("probe", az) / ("sweep", region)


def make_app(regions=("us-east-2",), probe_answers=None, gate=None, **overrides):
    """probe_answers: az -> result string (default "no"). gate: threading.Event the fake
    probe waits on before answering, to hold a job open."""
    rec = Recorder()
    probe_answers = probe_answers or {}

    def probe_one(region, type_name, az):
        rec.probes.append((region, type_name, az))
        if gate is not None:
            gate.wait(5)
        rec.events.append(("probe", az))
        return ProbeResult(region, type_name, az, probe_answers.get(az, "no"), 1000.0)

    def sweep(region):
        rec.sweeps.append(region)
        rec.events.append(("sweep", region))
        return []

    readers = appmod.Readers(
        load_gpu_types=lambda r: list(TYPES),
        load_offerings=lambda r: {k: list(v) for k, v in OFFERED.items()},
        load_prices=lambda r: dict(PRICES),
        az_id_map=lambda r: {"use2-az1": "us-east-2a", "use2-az2": "us-east-2b", "use2-az3": "us-east-2c"},
        load_spot_prices=lambda r: {"g7e.2xlarge": {"us-east-2a": 1.96}},
        load_scores=lambda t, regions, m: {"us-east-2a": 3},
        load_quota=lambda r, v: {"G": {"quota": 64, "used": 48, "free": 16},
                                 "P": {"quota": 384, "used": 0, "free": 384}},
        probe_one=probe_one,
        list_tagged_active=lambda r: [],
        sweep=sweep,
    )
    for k, v in overrides.items():
        setattr(readers, k, v)
    a = appmod.App(list(regions), ["us-east-2"], readers, clock=lambda: 2000.0)
    return a, rec


def loaded_app(**kw):
    a, rec = make_app(**kw)
    for r in a.regions:
        a.load_region(r)
    return a, rec


def blocked_app():
    gate = threading.Event()
    a, rec = loaded_app(gate=gate)
    return a, rec, gate
