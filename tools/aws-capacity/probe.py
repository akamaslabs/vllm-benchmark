"""One on-demand capacity probe: create a 1-instance capacity reservation, cancel it at once.

A reservation is billed per second with a 60-second minimum, so a probe that finds capacity
costs one minute of the on-demand price. Every reservation carries TAG_KEY=TAG_VALUE, and
sweep() cancels only reservations with that tag."""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from awscli import AwsError, Call, aws

TAG_KEY = "created-by"
TAG_VALUE = "istanceAws"
CANCEL_ATTEMPTS = 3
SAFETY_END_S = 3600  # the reservation expires by itself after this if every cancel fails

NO_CAPACITY = {"InsufficientInstanceCapacity"}
QUOTA = {"InstanceLimitExceeded", "ReservationCapacityExceeded", "VcpuLimitExceeded"}


@dataclass
class ProbeResult:
    region: str
    type: str
    az: str
    result: str  # "YES" | "no" | "quota" | "error" | "LEFT_ACTIVE"
    at: float
    detail: str = ""


def classify(code: str) -> str:
    if code in NO_CAPACITY:
        return "no"
    if code in QUOTA:
        return "quota"
    return "error"


def probe_one(region: str, type_name: str, az: str, call: Call = aws,
              now: Callable[[], float] = time.time,
              sleep: Callable[[float], None] = time.sleep) -> ProbeResult:
    at = now()
    end = datetime.fromtimestamp(at + SAFETY_END_S, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        doc = call(["ec2", "create-capacity-reservation", "--instance-type", type_name,
                    "--instance-platform", "Linux/UNIX", "--availability-zone", az,
                    "--instance-count", "1", "--instance-match-criteria", "targeted",
                    "--end-date-type", "limited", "--end-date", end,
                    "--tag-specifications",
                    f"ResourceType=capacity-reservation,Tags=[{{Key={TAG_KEY},Value={TAG_VALUE}}}]"],
                   region)
    except AwsError as e:
        result = classify(e.code)
        return ProbeResult(region, type_name, az, result, at,
                           "" if result == "no" else f"{e.code}: {e.message}")
    try:
        cr = doc["CapacityReservation"]["CapacityReservationId"]
    except (KeyError, TypeError):
        # a reservation may exist without an id we can read: the caller sweeps by tag
        return ProbeResult(region, type_name, az, "error", at, f"unreadable create response: {doc!r:.200}")
    if _cancel(region, cr, call, sleep):
        return ProbeResult(region, type_name, az, "YES", at)
    return ProbeResult(region, type_name, az, "LEFT_ACTIVE", at, f"{cr} not cancelled; the sweep retries it")


def _cancel(region: str, cr: str, call: Call, sleep: Callable[[float], None]) -> bool:
    for attempt in range(CANCEL_ATTEMPTS):
        try:
            call(["ec2", "cancel-capacity-reservation", "--capacity-reservation-id", cr], region)
            return True
        except Exception:  # noqa: BLE001 - any failure: retry, then report it as left open
            if attempt < CANCEL_ATTEMPTS - 1:
                sleep(1)
    return False


def list_tagged_active(region: str, call: Call = aws) -> list[str]:
    doc = call(["ec2", "describe-capacity-reservations", "--filters",
                f"Name=tag:{TAG_KEY},Values={TAG_VALUE}", "Name=state,Values=active,pending"], region)
    return [c["CapacityReservationId"] for c in doc.get("CapacityReservations", [])]


def sweep(region: str, call: Call = aws, sleep: Callable[[float], None] = time.sleep) -> list[str]:
    """Cancel every active reservation tagged by this tool; return the ids cancelled. Ones that
    still fail stay listed by list_tagged_active, which callers check afterwards."""
    return [cr for cr in list_tagged_active(region, call=call) if _cancel(region, cr, call, sleep)]
