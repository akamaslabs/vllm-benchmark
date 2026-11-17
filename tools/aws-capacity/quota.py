"""On-demand vCPU quota and usage for the G (G and VT) and P instance buckets."""
from awscli import Call, aws

QUOTA_CODES = {"G": "L-DB2E81BA", "P": "L-417A185B"}


def bucket(type_name: str) -> str | None:
    family = type_name.split(".")[0]
    if family.startswith(("g", "vt")):
        return "G"
    if family.startswith("p"):
        return "P"
    return None


def used_vcpus(running: list[str], unused_reserved: list[tuple[str, int]],
               vcpu_of: dict[str, int]) -> dict[str, int]:
    used = {"G": 0, "P": 0}
    for t in running:
        b = bucket(t)
        if b:
            used[b] += vcpu_of.get(t, 0)
    for t, count in unused_reserved:
        b = bucket(t)
        if b:
            used[b] += vcpu_of.get(t, 0) * count
    return used


def load_quota(region: str, vcpu_of: dict[str, int], call: Call = aws) -> dict[str, dict[str, int]]:
    limits = {}
    for b, code in QUOTA_CODES.items():
        doc = call(["service-quotas", "get-service-quota", "--service-code", "ec2",
                    "--quota-code", code], region)
        limits[b] = int(doc["Quota"]["Value"])
    inst = call(["ec2", "describe-instances", "--filters",
                 "Name=instance-state-name,Values=pending,running"], region)
    running = [i["InstanceType"] for r in inst.get("Reservations", []) for i in r.get("Instances", [])]
    res = call(["ec2", "describe-capacity-reservations", "--filters",
                "Name=state,Values=active,pending"], region)
    reserved = [(c["InstanceType"], int(c.get("AvailableInstanceCount", 0)))
                for c in res.get("CapacityReservations", [])]
    used = used_vcpus(running, reserved, vcpu_of)
    return {b: {"quota": limits[b], "used": used[b], "free": max(limits[b] - used[b], 0)}
            for b in QUOTA_CODES}
