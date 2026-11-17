"""Spot prices per AZ and spot placement scores (1-10) per AZ."""
from datetime import datetime, timezone

from awscli import AwsError, Call, aws

GPU_FILTER = "Name=instance-type,Values=g*,p*"
SCORE_ENDPOINT_REGION = "us-east-1"


def parse_spot_prices(doc: dict) -> dict[str, dict[str, float]]:
    latest: dict[tuple[str, str], tuple[str, float]] = {}
    for h in doc.get("SpotPriceHistory", []):
        key = (h["InstanceType"], h["AvailabilityZone"])
        ts = h["Timestamp"]
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, float(h["SpotPrice"]))
    out: dict[str, dict[str, float]] = {}
    for (t, az), (_, price) in latest.items():
        out.setdefault(t, {})[az] = price
    return out


def load_spot_prices(region: str, call: Call = aws) -> dict[str, dict[str, float]]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = call(["ec2", "describe-spot-price-history", "--product-descriptions", "Linux/UNIX",
                "--start-time", now, "--filters", GPU_FILTER], region)
    return parse_spot_prices(doc)


def az_id_map(region: str, call: Call = aws) -> dict[str, str]:
    doc = call(["ec2", "describe-availability-zones"], region)
    return {z["ZoneId"]: z["ZoneName"] for z in doc.get("AvailabilityZones", [])}


def parse_scores(doc: dict, id_to_name: dict[str, str]) -> dict[str, int]:
    out = {}
    for s in doc.get("SpotPlacementScores", []):
        zone_id = s.get("AvailabilityZoneId")
        out[id_to_name.get(zone_id, zone_id)] = int(s["Score"])
    return out


def load_scores(type_name: str, regions: list[str], id_to_name: dict[str, str],
                call: Call = aws) -> "dict[str, int] | AwsError":
    try:
        doc = call(["ec2", "get-spot-placement-scores", "--instance-types", type_name,
                    "--target-capacity", "1", "--single-availability-zone",
                    "--region-names", *regions], SCORE_ENDPOINT_REGION)
    except AwsError as e:
        return e
    return parse_scores(doc, id_to_name)
