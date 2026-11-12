#!/usr/bin/env python3
"""Node and GPU telemetry of every load level of study 51, read back from Prometheus.

    python3 results/level_telemetry.py results/levels.csv results/level_telemetry.csv

For each row of levels.csv (see parse_levels.py) it queries the cluster's Prometheus, through the
Kubernetes API proxy (`kubectl get --raw`), over the last 300 s of the level: network bytes sent
by each GPU node (the NCCL traffic over TCP), GPU utilization and power, node CPU. The study's
telemetry has the same series, but Akamas keeps them for the Hold window only and its export did
not carry the network ones. Prometheus keeps 10 days: run it before 2026-11-21.
"""
import csv
import json
import subprocess
import sys
import urllib.parse

DAY = "2026-11-11"
NODES = {"a": "ip-192-168-174-92", "b": "ip-192-168-182-253"}   # study51/role of the two g7e.4xlarge
PROXY = "/api/v1/namespaces/monitoring/services/kube-prometheus-stack-prometheus:9090/proxy/api/v1/query"
WINDOW = 300
ANY = "|".join(n + ".*" for n in NODES.values())
JOIN = f'* on(instance) group_left(nodename) node_uname_info{{nodename=~"{ANY}"}}'
QUERIES = {
    "net_out_mb_s": ("nodename", 1e-6,
                     f'sum by(nodename)(rate(node_network_transmit_bytes_total{{device!~"lo|veth.*|eni.*"}}[{WINDOW}s]) {JOIN})'),
    "tcp_retrans_per_s": ("nodename", 1,
                          f'sum by(nodename)(rate(node_netstat_Tcp_RetransSegs[{WINDOW}s]) {JOIN})'),
    "node_cpu_util": ("nodename", 1,
                      f'1 - avg by(nodename)(rate(node_cpu_seconds_total{{mode="idle"}}[{WINDOW}s]) {JOIN})'),
    "gpu_util": ("hostname", 0.01,
                 f'avg by(hostname)(avg_over_time(DCGM_FI_DEV_GPU_UTIL{{hostname=~"{ANY}"}}[{WINDOW}s]))'),
    "gpu_power_w": ("hostname", 1,
                    f'avg by(hostname)(avg_over_time(DCGM_FI_DEV_POWER_USAGE{{hostname=~"{ANY}"}}[{WINDOW}s]))'),
    "gpu_mem_copy_util": ("hostname", 0.01,
                          f'avg by(hostname)(avg_over_time(DCGM_FI_DEV_MEM_COPY_UTIL{{hostname=~"{ANY}"}}[{WINDOW}s]))'),
}


def query(expr, at):
    url = f"{PROXY}?query={urllib.parse.quote(expr)}&time={at}"
    out = subprocess.run(["kubectl", "get", "--raw", url], check=True, capture_output=True, text=True).stdout
    return json.loads(out)["data"]["result"]


def end_of(row):
    h, m, s = (int(x) for x in row["start_utc"].split(":"))
    t = h * 3600 + m * 60 + s + int(row["duration_s"])
    return f"{DAY}T{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}Z"


def main(levels_path, out_path):
    with open(levels_path, newline="") as f:
        levels = list(csv.DictReader(f))
    fields = ["experiment", "layout", "task", "users", "window_end_utc"] + \
             [f"{name}_{role}" for name in QUERIES for role in NODES]
    table = []
    for row in levels:
        at = end_of(row)
        out = {k: row[k] for k in ("experiment", "layout", "task", "users")}
        out["window_end_utc"] = at
        for name, (label, scale, expr) in QUERIES.items():
            for r in query(expr, at):
                for role, node in NODES.items():
                    if r["metric"][label].startswith(node):
                        out[f"{name}_{role}"] = round(float(r["value"][1]) * scale, 4)
        table.append(out)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} levels written to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
