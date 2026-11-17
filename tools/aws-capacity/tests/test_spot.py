import unittest

import spot
from awscli import AwsError
from fakes import FakeCall


class SpotPriceTest(unittest.TestCase):
    def test_spot_price_keeps_latest(self):
        doc = {"SpotPriceHistory": [
            {"InstanceType": "g7e.2xlarge", "AvailabilityZone": "us-east-2a", "SpotPrice": "2.000000",
             "Timestamp": "2026-11-17T07:00:00+00:00"},
            {"InstanceType": "g7e.2xlarge", "AvailabilityZone": "us-east-2a", "SpotPrice": "1.800000",
             "Timestamp": "2026-11-17T09:00:00+00:00"},
            {"InstanceType": "g7e.2xlarge", "AvailabilityZone": "us-east-2b", "SpotPrice": "1.500000",
             "Timestamp": "2026-11-17T08:00:00+00:00"},
        ]}
        self.assertEqual(spot.parse_spot_prices(doc),
                         {"g7e.2xlarge": {"us-east-2a": 1.8, "us-east-2b": 1.5}})


class ScoresTest(unittest.TestCase):
    def test_az_id_map(self):
        fake = FakeCall([{"AvailabilityZones": [{"ZoneName": "us-east-2c", "ZoneId": "use2-az3"}]}])
        self.assertEqual(spot.az_id_map("us-east-2", call=fake), {"use2-az3": "us-east-2c"})

    def test_scores_map_zone_ids(self):
        doc = {"SpotPlacementScores": [{"Region": "us-east-2", "AvailabilityZoneId": "use2-az3", "Score": 9}]}
        self.assertEqual(spot.parse_scores(doc, {"use2-az3": "us-east-2c"}), {"us-east-2c": 9})

    def test_scores_request_shape(self):
        fake = FakeCall([{"SpotPlacementScores": []}])
        spot.load_scores("g7e.2xlarge", ["us-east-2", "us-east-1"], {}, call=fake)
        args, region = fake.calls[0]
        self.assertEqual(region, "us-east-1")
        self.assertIn("--single-availability-zone", args)
        self.assertEqual(args[args.index("--region-names") + 1:args.index("--region-names") + 3],
                         ["us-east-2", "us-east-1"])

    def test_scores_error_is_returned_not_raised(self):
        fake = FakeCall([AwsError("MaxConfigLimitExceeded", "limit")])
        r = spot.load_scores("g7e.2xlarge", ["us-east-2"], {}, call=fake)
        self.assertIsInstance(r, AwsError)


if __name__ == "__main__":
    unittest.main()
