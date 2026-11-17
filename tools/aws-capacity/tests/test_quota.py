import unittest

import quota
from fakes import FakeCall


class QuotaTest(unittest.TestCase):
    def test_bucket(self):
        self.assertEqual([quota.bucket(t) for t in ("g7e.12xlarge", "vt1.3xlarge", "p5en.48xlarge", "m5a.xlarge")],
                         ["G", "G", "P", None])

    def test_used_vcpus_counts_unused_reservations(self):
        u = quota.used_vcpus(["g7e.12xlarge", "m5a.xlarge"], [("g7e.4xlarge", 1)],
                             {"g7e.12xlarge": 48, "g7e.4xlarge": 16})
        self.assertEqual(u, {"G": 64, "P": 0})

    def test_load_quota_free(self):
        fake = FakeCall(by_op={
            "get-service-quota": [{"Quota": {"Value": 64.0}}, {"Quota": {"Value": 384.0}}],
            "describe-instances": {"Reservations": [{"Instances": [{"InstanceType": "g7e.12xlarge"},
                                                                   {"InstanceType": "m5a.xlarge"}]}]},
            "describe-capacity-reservations": {"CapacityReservations": [
                {"InstanceType": "g7e.12xlarge", "AvailableInstanceCount": 0}]},
        })
        q = quota.load_quota("us-east-2", {"g7e.12xlarge": 48}, call=fake)
        self.assertEqual(q["G"], {"quota": 64, "used": 48, "free": 16})
        self.assertEqual(q["P"], {"quota": 384, "used": 0, "free": 384})


if __name__ == "__main__":
    unittest.main()
