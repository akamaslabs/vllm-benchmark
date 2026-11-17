import unittest

import probe
from awscli import AwsError
from fakes import FakeCall

NOW = lambda: 1_800_000_000.0  # noqa: E731
NO_SLEEP = lambda s: None  # noqa: E731


def created(cr="cr-123"):
    return {"CapacityReservation": {"CapacityReservationId": cr}}


class ProbeOneTest(unittest.TestCase):
    def run_probe(self, fake):
        return probe.probe_one("us-east-2", "g4dn.xlarge", "us-east-2a", call=fake, now=NOW, sleep=NO_SLEEP)

    def test_yes_creates_tagged_then_cancels(self):
        fake = FakeCall([created(), {"Return": True}])
        r = self.run_probe(fake)
        self.assertEqual(r.result, "YES")
        self.assertEqual((r.region, r.type, r.az, r.at), ("us-east-2", "g4dn.xlarge", "us-east-2a", NOW()))
        (create, region1), (cancel, region2) = fake.calls
        self.assertEqual(create[:2], ["ec2", "create-capacity-reservation"])
        self.assertIn("ResourceType=capacity-reservation,Tags=[{Key=created-by,Value=istanceAws}]", create)
        self.assertEqual(create[create.index("--availability-zone") + 1], "us-east-2a")
        self.assertEqual(create[create.index("--instance-count") + 1], "1")
        self.assertEqual(create[create.index("--end-date") + 1], "2027-01-15T09:00:00Z")
        self.assertEqual(cancel[:2], ["ec2", "cancel-capacity-reservation"])
        self.assertIn("cr-123", cancel)
        self.assertEqual((region1, region2), ("us-east-2", "us-east-2"))

    def test_no_capacity(self):
        fake = FakeCall([AwsError("InsufficientInstanceCapacity", "none")])
        r = self.run_probe(fake)
        self.assertEqual(r.result, "no")
        self.assertEqual(len(fake.calls), 1)

    def test_quota(self):
        r = self.run_probe(FakeCall([AwsError("InstanceLimitExceeded", "vcpu")]))
        self.assertEqual(r.result, "quota")

    def test_unknown_error(self):
        r = self.run_probe(FakeCall([AwsError("UnauthorizedOperation", "msg")]))
        self.assertEqual(r.result, "error")
        self.assertEqual(r.detail, "UnauthorizedOperation: msg")

    def test_cancel_retried_then_left_active(self):
        boom = AwsError("RequestLimitExceeded", "slow down")
        fake = FakeCall([created(), boom, boom, boom])
        r = self.run_probe(fake)
        self.assertEqual(r.result, "LEFT_ACTIVE")
        self.assertIn("cr-123", r.detail)
        self.assertEqual(sum(1 for a, _ in fake.calls if a[1] == "cancel-capacity-reservation"), 3)

    def test_cancel_succeeds_on_retry(self):
        fake = FakeCall([created(), AwsError("RequestLimitExceeded", "x"), {"Return": True}])
        self.assertEqual(self.run_probe(fake).result, "YES")

    def test_cancel_unexpected_exception_is_left_active(self):
        boom = ValueError("bad json")
        r = self.run_probe(FakeCall([created(), boom, boom, boom]))
        self.assertEqual(r.result, "LEFT_ACTIVE")

    def test_unreadable_create_response_is_error_not_yes(self):
        r = self.run_probe(FakeCall([{"Unexpected": {}}]))
        self.assertEqual(r.result, "error")


class SweepTest(unittest.TestCase):
    def test_sweep_retries_failed_cancel(self):
        fake = FakeCall([{"CapacityReservations": [{"CapacityReservationId": "cr-a"}]},
                         AwsError("RequestLimitExceeded", "x"), {"Return": True}])
        self.assertEqual(probe.sweep("us-east-2", call=fake, sleep=NO_SLEEP), ["cr-a"])

    def test_list_tagged_active_filters(self):
        fake = FakeCall([{"CapacityReservations": [{"CapacityReservationId": "cr-a"}]}])
        self.assertEqual(probe.list_tagged_active("us-east-2", call=fake), ["cr-a"])
        args = fake.calls[0][0]
        self.assertIn("Name=tag:created-by,Values=istanceAws", args)
        self.assertIn("Name=state,Values=active,pending", args)

    def test_sweep_cancels_only_listed(self):
        fake = FakeCall([{"CapacityReservations": [{"CapacityReservationId": "cr-a"}]}, {"Return": True}])
        self.assertEqual(probe.sweep("us-east-2", call=fake), ["cr-a"])
        cancels = [a for a, _ in fake.calls if a[1] == "cancel-capacity-reservation"]
        self.assertEqual(len(cancels), 1)
        self.assertIn("cr-a", cancels[0])


if __name__ == "__main__":
    unittest.main()
