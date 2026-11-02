"""Study 44: study 27's goal recomputed from Prometheus for one manual run.

Study 27 (Akamas): goal = pd_topology.total_token_throughput / pd_topology.active_gpus, with
30 s telemetry samples, stability windowing of width 6 (180 s) on the throughput maximum,
and constraints TTFT p95 (150 s) :max <= 10000 ms and ITL p95 (150 s) :max <= 75 ms, all read
at the router. Here: every 6-sample window in [start, end], valid if both constraints hold
on all 6 samples, score = the best valid window's mean throughput / GPUs. Akamas picks the
window by its own rule, so this is the same quantity, not necessarily the same window.

    python3 score.py 2026-10-29T11:00:00Z 2026-10-29T12:00:00Z

Reads Prometheus through the API server proxy (kubectl get --raw), so it runs wherever
kubectl does.
"""
import datetime as dt
import json
import subprocess
import sys
import urllib.parse

PROXY = "/api/v1/namespaces/monitoring/services/kube-prometheus-stack-prometheus:9090/proxy"
STEP = 30
WIDTH = 6
R = 'model_name="qwen3-8b-router"'
Q = {
    "tput": f"sum(rate(vllm:prompt_tokens_total{{{R}}}[30s])) + sum(rate(vllm:generation_tokens_total{{{R}}}[30s]))",
    "rps": f"sum(rate(vllm:request_success_total{{{R}}}[30s]))",
    "ttft": f"histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{{{R}}}[150s])))*1000",
    "itl": f"histogram_quantile(0.95, sum by(le)(rate(vllm:inter_token_latency_seconds_bucket{{{R}}}[150s])))*1000",
    # vllm:cache_config_info has no model_name label (study 27 telemetry notes): one series per
    # vLLM process, told apart by the scrape's Service.
    "gpus": 'count(vllm:cache_config_info{namespace="llm-serving", service=~"vllm-pd-mn-(prefill|decode)"})',
    "prefill_leg": f"histogram_quantile(0.5, sum by(le)(rate(pd_router_prefill_leg_seconds_bucket{{{R}}}[150s])))*1000",
}


def get(path):
    return json.loads(subprocess.check_output(["kubectl", "get", "--raw", PROXY + path]))


def series(q, start, end):
    p = urllib.parse.urlencode({"query": q, "start": start, "end": end, "step": STEP})
    res = get("/api/v1/query_range?" + p)["data"]["result"]
    out = {}
    for r in res:
        for t, v in r["values"]:
            if v not in ("NaN", "+Inf", "-Inf"):
                out[int(float(t))] = float(v)
    return out


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%H:%M:%S")


def main(start, end):
    s = {k: series(q, start, end) for k, q in Q.items()}
    ts = sorted(s["tput"])
    gpus = max(s["gpus"].values(), default=0) or 1
    best = None
    for i in range(len(ts) - WIDTH + 1):
        w = ts[i:i + WIDTH]
        if w[-1] - w[0] != STEP * (WIDTH - 1):
            continue
        ttft = max((s["ttft"].get(t, float("inf")) for t in w))
        itl = max((s["itl"].get(t, float("inf")) for t in w))
        mean = sum(s["tput"][t] for t in w) / WIDTH
        if ttft <= 10000 and itl <= 75 and (best is None or mean > best[0]):
            best = (mean, w, ttft, itl)
    print(f"GPUs holding the model: {gpus:.0f}")
    if not best:
        print("no valid 180 s window (TTFT p95 <= 10 s and ITL p95 <= 75 ms on all 6 samples)")
        return
    mean, w, ttft, itl = best
    rps = sum(s["rps"].get(t, 0) for t in w) / WIDTH
    leg = [s["prefill_leg"][t] for t in w if t in s["prefill_leg"]]
    print(f"best valid window {iso(w[0])}-{iso(w[-1])} UTC: {mean:.1f} tokens/s total, "
          f"{mean / gpus:.2f} tokens/s per GPU (study 27 score), {rps:.3f} req/s, "
          f"max TTFT p95 {ttft:.0f} ms, max ITL p95 {itl:.1f} ms"
          + (f", prefill leg p50 {sum(leg) / len(leg):.0f} ms" if leg else ""))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
