"""Re-score study 25's experiments with the window ranked on TOTAL throughput.

The study's windowing ranked 6 x 30 s windows on prefill throughput, while its goal is
prefill + decode. This script rebuilds 30 s series from Prometheus (via `kubectl get
--raw`, Prometheus retention ~10 days: run before 2026-10-09), slides a 6-sample window
over each trial, and reports for each experiment: the prefill-ranked window (what
Akamas ranked on) and the best SLA-compliant window ranked on total throughput
(TTFT p95 <= 1500 ms, ITL p95 <= 300 ms, window averages as Akamas evaluates them), with
GPU power and SM clock over that window. Output: rescore_total_throughput.csv.

Approximation: Prometheus step alignment differs from Akamas' own sampling, so values
differ from Akamas' scores by up to ~1 % on the exclusive experiments, where both rankings
pick the same load level.
"""
import csv
import json
import subprocess
import urllib.parse

TRIALS = "trials.csv"
PROM = "/api/v1/namespaces/monitoring/services/kube-prometheus-stack-prometheus:9090/proxy/api/v1/query_range"
M = 'model_name="qwen3-4b", pod=~"^vllm-[0-9]+$"'
G = 'modelName=~".*RTX PRO 4500.*"'
Q = {
    "pre": f"sum(rate(vllm:prompt_tokens_total{{{M}}}[30s]))",
    "dec": f"sum(rate(vllm:generation_tokens_total{{{M}}}[30s]))",
    "ttft": f"histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{{{M}}}[30s])))*1000",
    "itl": f"histogram_quantile(0.95, sum by(le)(rate(vllm:inter_token_latency_seconds_bucket{{{M}}}[30s])))*1000",
    "run": f"sum(vllm:num_requests_running{{{M}}})",
    "watt": f"max(DCGM_FI_DEV_POWER_USAGE{{{G}}})",
    "clock": f"max(DCGM_FI_DEV_SM_CLOCK{{{G}}})",
}
W = 6


def series(expr, start, end):
    url = f"{PROM}?query={urllib.parse.quote(expr)}&start={start}&end={end}&step=30s"
    r = json.loads(subprocess.check_output(["kubectl", "get", "--raw", url]))["data"]["result"]
    return {int(float(t)): float(v) for t, v in r[0]["values"]} if r else {}


def windows(s):
    ts = sorted(set(s["pre"]) & set(s["dec"]))
    for i in range(len(ts) - W + 1):
        w = ts[i:i + W]
        avg = lambda k: sum(s[k].get(x, float("nan")) for x in w) / W  # noqa: E731
        pre, dec = avg("pre"), avg("dec")
        yield dict(start=w[0], pre=pre, dec=dec, tot=pre + dec, ttft=avg("ttft"), itl=avg("itl"),
                   run=avg("run"), watt=avg("watt"), clock=avg("clock"))


out = []
for t in csv.DictReader(open(TRIALS)):
    if not t["akamas_score"]:
        continue  # aborted experiment
    s = {k: series(q, t["start"] + "Z", t["end"] + "Z") for k, q in Q.items()}
    rows = list(windows(s))
    by_pre = max(rows, key=lambda r: r["pre"])
    ok = [r for r in rows if r["ttft"] <= 1500 and r["itl"] <= 300]
    best = max(ok, key=lambda r: r["tot"]) if ok else None
    out.append({
        "experiment": t["experiment"], "sharing_mode": t["sharing_mode"],
        "max_num_seqs": t["max_num_seqs"], "max_num_batched_tokens": t["max_num_batched_tokens"],
        "akamas_score_prefill_window": round(float(t["akamas_score"]), 1),
        "prefill_ranked_total": round(by_pre["tot"], 1), "prefill_ranked_ttft_p95": round(by_pre["ttft"]),
        "total_ranked_sla_ok_total": round(best["tot"], 1) if best else "",
        "total_ranked_running": round(best["run"]) if best else "",
        "total_ranked_ttft_p95": round(best["ttft"]) if best else "",
        "total_ranked_itl_p95": round(best["itl"]) if best else "",
        "total_ranked_gpu_w": round(best["watt"], 1) if best else "",
        "total_ranked_sm_clock": round(best["clock"]) if best else "",
    })

with open("rescore_total_throughput.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r)
