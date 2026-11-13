#!/usr/bin/env python3
"""Per-node split of the GPU's time in study 51's Nsight traces.

    python3 analysis/gpu_time_split.py analysis/profiles run-logs/logs analysis/gpu_time_split.csv

Reads the per-trace profiles (profile_trace.py) and the capture logs. Per traced node: the share
of the trace with a compute kernel running, with only NCCL kernels on the GPU (the GPU waits for
the other node) and with no kernel, the mean of the trace's GR Active, SM Active and Tensor
Active counters, and the nvidia-smi reading taken at the end of the capture.
"""
import csv
import json
import re
import sys

profiles, logs, out = sys.argv[1:4]
ORDER = [("replicas", 256, "Two replicas", "ab"), ("single", 64, "One GPU", "a"), ("pp2", 512, "PP two", "ab"),
         ("dp2ep", 64, "DP two + expert parallel", "ab"), ("dp2", 64, "DP two", "ab"),
         ("tp2", 32, "TP two", "ab"), ("tp2ep", 64, "TP two + expert parallel", "ab")]
FIELDS = ["layout", "node", "users", "compute_kernel_pct", "only_nccl_kernel_pct", "no_kernel_pct",
          "gr_active_pct", "sm_active_pct", "tensor_active_pct", "nvidia_smi_util_pct", "nvidia_smi_power_w"]
rows = []
for key, users, name, nodes in ORDER:
    for node in nodes:
        with open(f"{profiles}/{key}-{users}users-node-{node}.json") as f:
            trace = next(iter(json.load(f).values()))
        kernels = trace["kernels"]
        counters = {k.split(" (")[0]: v["mean"] for k, v in trace["counters"].items()}
        with open(f"{logs}/{key}-capture-{node}.log") as f:
            smi = re.search(r"gpu: (\d+) %, \d+ MiB, ([\d.]+) W", f.read())
        compute, wait = kernels["compute_busy_pct"], kernels["nccl_only_pct"]
        rows.append([name, node, users, compute, wait, round(100 - compute - wait, 1),
                     round(counters["GR Active"], 1), round(counters["SM Active"], 1),
                     round(counters["Tensor Active"], 1), int(smi.group(1)), round(float(smi.group(2)))])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(FIELDS)
    w.writerows(rows)
print(f"{len(rows)} traced nodes written to {out}")
