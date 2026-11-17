import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import catalog
from fakes import FakeCall

DOC = {"InstanceTypes": [
    {"InstanceType": "g7e.12xlarge", "VCpuInfo": {"DefaultVCpus": 48},
     "MemoryInfo": {"SizeInMiB": 524288},
     "GpuInfo": {"Gpus": [{"Name": "RTX PRO Server 6000", "Manufacturer": "NVIDIA", "Count": 2,
                           "MemoryInfo": {"SizeInMiB": 98304}}]}},
    {"InstanceType": "m5a.xlarge", "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 16384}},
    {"InstanceType": "g6f.large", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 8192},
     "GpuInfo": {"Gpus": [{"Name": "L4", "Manufacturer": "NVIDIA", "Count": 0,
                           "MemoryInfo": {"SizeInMiB": 2048}}]}},
]}


def price_product(type_name, usd):
    return json.dumps({
        "product": {"attributes": {"instanceType": type_name}},
        "terms": {"OnDemand": {"A.B": {"priceDimensions": {"A.B.C": {
            "unit": "Hrs", "pricePerUnit": {"USD": usd}}}}}},
    })


class ParseInstanceTypesTest(unittest.TestCase):
    def test_parse_g7e_12xlarge(self):
        t = catalog.parse_instance_types(DOC)[0]
        self.assertEqual(t.name, "g7e.12xlarge")
        self.assertEqual((t.vcpu, t.mem_gib, t.gpu_count, t.gpu_mem_gib), (48, 512, 2, 96))
        self.assertEqual(t.gpu_model, "RTX PRO Server 6000")
        self.assertEqual(t.group, "RTX PRO Server 6000 96 GB")

    def test_parse_skips_types_without_gpu(self):
        names = [t.name for t in catalog.parse_instance_types(DOC)]
        self.assertNotIn("m5a.xlarge", names)

    def test_parse_fractional_gpu(self):
        t = [t for t in catalog.parse_instance_types(DOC) if t.name == "g6f.large"][0]
        self.assertEqual(t.gpu_count, 0)
        self.assertEqual(t.group, "L4 fractional 2 GB")


class OfferingsTest(unittest.TestCase):
    def test_offerings_grouped_and_sorted(self):
        fake = FakeCall([{"InstanceTypeOfferings": [
            {"InstanceType": "g7e.2xlarge", "Location": "us-east-2c"},
            {"InstanceType": "g7e.2xlarge", "Location": "us-east-2a"},
            {"InstanceType": "p5.48xlarge", "Location": "us-east-2b"},
        ]}])
        off = catalog.load_offerings("us-east-2", call=fake)
        self.assertEqual(off["g7e.2xlarge"], ["us-east-2a", "us-east-2c"])
        self.assertEqual(off["p5.48xlarge"], ["us-east-2b"])
        self.assertEqual(fake.calls[0][1], "us-east-2")


class PricesTest(unittest.TestCase):
    def test_parse_prices(self):
        prices = catalog.parse_prices([price_product("g7e.12xlarge", "8.2860800000")])
        self.assertAlmostEqual(prices["g7e.12xlarge"], 8.28608)

    def test_prices_cache_hit_skips_call(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "prices-us-east-2.json").write_text(json.dumps({"g7e.2xlarge": 3.36}))
            fake = FakeCall([])
            prices = catalog.load_prices("us-east-2", call=fake, cache_dir=Path(d))
            self.assertEqual(prices, {"g7e.2xlarge": 3.36})
            self.assertEqual(fake.calls, [])

    def test_prices_cache_expired_calls_again(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, "prices-us-east-2.json")
            f.write_text(json.dumps({"g7e.2xlarge": 1.0}))
            mtime = time.time() - 10
            os.utime(f, (mtime, mtime))
            fake = FakeCall([{"PriceList": [price_product("g7e.2xlarge", "3.36")]}])
            prices = catalog.load_prices("us-east-2", call=fake, cache_dir=Path(d),
                                         now=lambda: mtime + 86401)
            self.assertEqual(prices, {"g7e.2xlarge": 3.36})
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(fake.calls[0][1], "us-east-1")
            self.assertIn("Type=TERM_MATCH,Field=regionCode,Value=us-east-2", fake.calls[0][0])
            self.assertEqual(json.loads(f.read_text()), {"g7e.2xlarge": 3.36})


if __name__ == "__main__":
    unittest.main()
