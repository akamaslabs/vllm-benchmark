"""GPU instance types, the AZs that offer them, and their on-demand prices."""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from awscli import Call, aws

DEFAULT_CACHE = Path.home() / ".cache" / "istanceAws"
PRICE_TTL_S = 86400
GPU_FILTER = "Name=instance-type,Values=g*,p*"


@dataclass(frozen=True)
class GpuType:
    name: str
    vcpu: int
    mem_gib: int
    gpu_count: int
    gpu_model: str
    gpu_mem_gib: int
    group: str


def parse_instance_types(doc: dict) -> list[GpuType]:
    out = []
    for it in doc.get("InstanceTypes", []):
        gpus = (it.get("GpuInfo") or {}).get("Gpus") or []
        if not gpus:
            continue
        g = gpus[0]
        model = g.get("Name", "?")
        mem = round(g.get("MemoryInfo", {}).get("SizeInMiB", 0) / 1024)
        count = int(g.get("Count", 0))
        group = f"{model} {mem} GB" if count else f"{model} fractional {mem} GB"
        out.append(GpuType(
            name=it["InstanceType"],
            vcpu=int(it["VCpuInfo"]["DefaultVCpus"]),
            mem_gib=round(it["MemoryInfo"]["SizeInMiB"] / 1024),
            gpu_count=count, gpu_model=model, gpu_mem_gib=mem, group=group))
    return out


def load_gpu_types(region: str, call: Call = aws) -> list[GpuType]:
    doc = call(["ec2", "describe-instance-types", "--filters", GPU_FILTER], region)
    return parse_instance_types(doc)


def load_offerings(region: str, call: Call = aws) -> dict[str, list[str]]:
    doc = call(["ec2", "describe-instance-type-offerings", "--location-type", "availability-zone",
                "--filters", GPU_FILTER], region)
    out: dict[str, list[str]] = {}
    for o in doc.get("InstanceTypeOfferings", []):
        out.setdefault(o["InstanceType"], []).append(o["Location"])
    return {t: sorted(azs) for t, azs in out.items()}


def parse_prices(price_list: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in price_list:
        p = json.loads(item) if isinstance(item, str) else item
        name = p["product"]["attributes"].get("instanceType")
        for term in p.get("terms", {}).get("OnDemand", {}).values():
            for dim in term.get("priceDimensions", {}).values():
                usd = float(dim["pricePerUnit"].get("USD", 0))
                if name and usd > 0 and name not in out:
                    out[name] = usd
    return out


def load_prices(region: str, call: Call = aws, cache_dir: Path = DEFAULT_CACHE,
                now: Callable[[], float] = time.time) -> dict[str, float]:
    cache = cache_dir / f"prices-{region}.json"
    if cache.exists() and now() - cache.stat().st_mtime < PRICE_TTL_S:
        return json.loads(cache.read_text())
    filters = [f"Type=TERM_MATCH,Field={k},Value={v}" for k, v in (
        ("regionCode", region), ("operatingSystem", "Linux"), ("tenancy", "Shared"),
        ("preInstalledSw", "NA"), ("capacitystatus", "Used"),
        ("licenseModel", "No License required"), ("instanceFamily", "GPU instance"))]
    doc = call(["pricing", "get-products", "--service-code", "AmazonEC2", "--filters", *filters],
               "us-east-1")
    prices = parse_prices(doc.get("PriceList", []))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(prices, sort_keys=True))
    return prices
