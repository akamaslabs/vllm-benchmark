import threading
import time
import unittest

import app as appmod
from appfakes import blocked_app, loaded_app, make_app
from awscli import AwsError


class PlanTest(unittest.TestCase):
    def test_plan_counts_offered_azs_and_cost(self):
        a, _ = loaded_app()
        self.assertEqual(a.plan_probe(["g7e.2xlarge"], ["us-east-2"]),
                         {"planned": 3, "max_cost": round(3 * 3.36 / 60, 2)})

    def test_empty_plan_is_value_error(self):
        a, _ = loaded_app()
        with self.assertRaises(ValueError):
            a.start_probe([], ["us-east-2"])


class ProbeJobTest(unittest.TestCase):
    def test_probe_skips_azs_not_offering_type(self):
        a, rec = loaded_app()
        a.start_probe(["g7e.12xlarge"], ["us-east-2"])
        a.wait_idle(5)
        self.assertEqual(rec.probes, [("us-east-2", "g7e.12xlarge", "us-east-2b")])

    def test_probe_records_results_and_cost(self):
        a, _ = loaded_app(probe_answers={"us-east-2a": "YES"})
        a.start_probe(["g7e.2xlarge"], ["us-east-2"])
        a.wait_idle(5)
        s = a.snapshot()
        cells = s["probe"]["us-east-2"]["g7e.2xlarge"]
        self.assertEqual(cells["us-east-2a"]["result"], "YES")
        self.assertEqual(cells["us-east-2b"]["result"], "no")
        self.assertEqual(s["cost_usd"], round(3.36 / 60, 4))
        self.assertEqual(s["probing"], {})

    def test_second_probe_same_region_rejected(self):
        a, rec, gate = blocked_app()
        a.start_probe(["g7e.2xlarge"], ["us-east-2"])
        try:
            with self.assertRaises(appmod.Busy):
                a.start_probe(["g6.xlarge"], ["us-east-2"])
        finally:
            gate.set()
            a.wait_idle(5)
        self.assertNotIn(("us-east-2", "g6.xlarge", "us-east-2a"), rec.probes)

    def test_region_job_ends_with_sweep(self):
        a, rec = loaded_app()
        a.start_probe(["g7e.2xlarge"], ["us-east-2"])
        a.wait_idle(5)
        self.assertEqual(rec.sweeps, ["us-east-2"])
        self.assertEqual(rec.events[-1], ("sweep", "us-east-2"))


class ShutdownTest(unittest.TestCase):
    def test_shutdown_sweeps_every_region(self):
        a, rec = loaded_app(regions=("us-east-2", "us-east-1"))
        a.shutdown()
        self.assertEqual(sorted(rec.sweeps), ["us-east-1", "us-east-2"])

    def test_shutdown_waits_inflight_probe_then_sweeps(self):
        a, rec, gate = blocked_app()
        a.start_probe(["g7e.2xlarge"], ["us-east-2"])  # 3 AZs, first probe held by the gate
        while not rec.probes:
            time.sleep(0.01)
        t = threading.Thread(target=a.shutdown)
        t.start()
        time.sleep(0.05)
        gate.set()
        t.join(5)
        self.assertEqual(len(rec.probes), 1, "no new probe may start after shutdown")
        self.assertEqual(rec.events[0], ("probe", "us-east-2a"))
        self.assertIn(("sweep", "us-east-2"), rec.events[1:])


class SpotTest(unittest.TestCase):
    def test_refresh_spot_fills_scores_and_prices(self):
        a, _ = loaded_app()
        a.refresh_spot()
        s = a.snapshot()
        self.assertEqual(s["spot_price"]["us-east-2"]["g7e.2xlarge"]["us-east-2a"], 1.96)
        self.assertEqual(s["spot_score"]["g6.xlarge"], {"us-east-2a": 3})
        self.assertEqual(s["spot_at"], 2000.0)

    def test_scores_error_marks_type(self):
        a, _ = loaded_app(load_scores=lambda t, r, m: AwsError("MaxConfigLimitExceeded", "limit"))
        a.refresh_spot()
        s = a.snapshot()
        self.assertEqual(s["spot_score_error"]["g7e.2xlarge"], "score limited")
        self.assertEqual(s["spot_price"]["us-east-2"]["g7e.2xlarge"]["us-east-2a"], 1.96)


class SnapshotTest(unittest.TestCase):
    def test_load_error_reaches_snapshot(self):
        def expired(region):
            raise AwsError("ExpiredToken", "The security token included in the request is expired")
        a, _ = loaded_app(load_gpu_types=expired)
        s = a.snapshot()
        self.assertIn("ExpiredToken", s["errors"][0])
        self.assertEqual(s["loading"], [])

    def test_groups_sorted_by_gpu_memory(self):
        a, _ = loaded_app()
        s = a.snapshot()
        self.assertEqual([g["group"] for g in s["groups"]], ["RTX PRO Server 6000 96 GB", "L4 22 GB"])
        self.assertEqual(s["groups"][0]["types"], ["g7e.2xlarge", "g7e.12xlarge"])
        self.assertEqual(s["quota"]["us-east-2"]["G"]["free"], 16)
        self.assertEqual(s["azs"]["us-east-2"], ["us-east-2a", "us-east-2b", "us-east-2c"])
        self.assertEqual(s["offered"]["us-east-2"]["g7e.12xlarge"], ["us-east-2b"])

    def test_unknown_region_rejected(self):
        a, _ = make_app()
        with self.assertRaises(ValueError):
            a.set_regions(["mars-1"])



class ScorePolicyTest(unittest.TestCase):
    def test_scores_use_fixed_regions_whatever_is_shown(self):
        seen = []
        a, _ = loaded_app(regions=("us-east-2",), load_scores=lambda t, regions, m: seen.append(list(regions)) or {})
        a.refresh_spot()
        self.assertTrue(seen)
        self.assertTrue(all(r == appmod.SCORE_REGIONS for r in seen))

    def test_score_order_biggest_gpu_first(self):
        a, _ = loaded_app()
        self.assertEqual(a._score_order(["g6.xlarge", "g7e.2xlarge", "g7e.12xlarge"]),
                         ["g7e.12xlarge", "g7e.2xlarge", "g6.xlarge"])


class FailureSafetyTest(unittest.TestCase):
    def test_job_stops_when_unclear_result_leaves_reservation(self):
        a, rec = loaded_app(probe_answers={"us-east-2a": "LEFT_ACTIVE"},
                            list_tagged_active=lambda r: ["cr-1"])
        a.start_probe(["g7e.2xlarge"], ["us-east-2"])
        a.wait_idle(5)
        self.assertEqual([p[2] for p in rec.probes], ["us-east-2a"])
        self.assertTrue(any("cr-1" in e for e in a.snapshot()["errors"]))

    def test_job_continues_after_unclear_result_once_clean(self):
        a, rec = loaded_app(probe_answers={"us-east-2a": "error"})
        a.start_probe(["g7e.2xlarge"], ["us-east-2"])
        a.wait_idle(5)
        self.assertEqual([p[2] for p in rec.probes], ["us-east-2a", "us-east-2b", "us-east-2c"])
        self.assertEqual(rec.events[:2], [("probe", "us-east-2a"), ("sweep", "us-east-2")])

    def test_shutdown_reports_still_active(self):
        a, _ = loaded_app(list_tagged_active=lambda r: ["cr-x"])
        report = a.shutdown()
        self.assertEqual(report["us-east-2"]["still_active"], ["cr-x"])

    def test_shutdown_does_not_wait_for_spot_refresh(self):
        gate = threading.Event()
        a, _ = loaded_app(load_scores=lambda t, r, m: gate.wait(5) and {})
        a.request_spot_refresh()
        t0 = time.monotonic()
        a.shutdown()
        took = time.monotonic() - t0
        gate.set()
        self.assertLess(took, 2)


if __name__ == "__main__":
    unittest.main()
