"""Per-load-level analysis of study 26, rebuilt from Prometheus.

AIPerf keeps only the last experiment's artifacts (each Job deletes the previous
`aiperf-*` dirs), so the per-level picture comes from vLLM's and DCGM's own series
(Prometheus retention ~10 days: run before 2026-10-09). For each experiment (one AIPerf
pod each) the load levels are the plateaus of running + waiting requests, separated by
the drain to 0 between levels; the 60 s warm-up is skipped. Over each level (first 20 s
cut) it reports aggregate and per-replica throughput, TTFT / ITL p95 from the summed
histograms, queueing, KV usage, preemptions and GPU power / SM clock / temperature.

It also emulates the study's windowing (6 x 30 s windows ranked on total throughput,
`when: max`, window averages as Akamas evaluates them) to compare with Akamas' scores.

Output: levels.csv, windows.csv (next to this script). Run from this folder.
"""
import csv
import datetime
import json
import subprocess
import urllib.parse

PROM = "/api/v1/namespaces/monitoring/services/kube-prometheus-stack-prometheus:9090/proxy/api/v1/"
M = 'model_name="qwen3-4b"'
G = 'modelName=~".*RTX PRO 4500.*"'
LEVELS = [16, 32, 64, 96, 128, 192, 256, 320, 384, 512, 640, 768]
# Steps in study order (akamas/26-G7-4500-GPU-Slice-Right-Sizing.yaml); checked below
# against the replica count and the running-requests cap.
STEPS = [
    ("baseline", "none", 256), ("MIG whole GPU seqs 256", "2g.32gb", 256),
    ("MIG whole GPU seqs 512", "2g.32gb", 512), ("MIG half GPU seqs 256", "1g.16gb", 256),
    ("MIG half GPU seqs 128", "1g.16gb", 128), ("MIG half GPU seqs 384", "1g.16gb", 384),
    ("no MIG repeat", "none", 256),
]
SLA_TTFT, SLA_ITL = 1500, 300


def get(path, **params):
    url = PROM + path + "?" + urllib.parse.urlencode(params)
    return json.loads(subprocess.check_output(["kubectl", "get", "--raw", url]))["data"]["result"]


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def instant(query, at):
    r = get("query", query=query, time=at)
    return {tuple(sorted(s["metric"].items())): float(s["value"][1]) for s in r}


def scalar(query, at):
    v = list(instant(query, at).values())
    return v[0] if v else float("nan")


def series(query, start, end, step="30s"):
    r = get("query_range", query=query, start=iso(start), end=iso(end), step=step)
    return {int(float(t)): float(v) for t, v in r[0]["values"]} if r else {}


# Experiment boundaries: one AIPerf pod per experiment.
pods = get("query_range", query='max by(pod)(kube_pod_start_time{namespace="gpu-sharing",pod=~"aiperf-gpu-sharing-.*"})',
           start="2026-09-30T08:55:00Z", end="2026-10-01T00:00:00Z", step="60s")
starts = sorted(float(p["values"][0][1]) for p in pods)
assert len(starts) == len(STEPS), starts
bounds = list(zip(starts, starts[1:] + [starts[-1] + 90 * 60]))

level_rows, window_rows = [], []
for exp, ((step, profile, seqs), (t0, t1)) in enumerate(zip(STEPS, bounds), start=1):
    conc = series(f"sum(vllm:num_requests_running{{{M}}})+sum(vllm:num_requests_waiting{{{M}}})", t0, t1, "15s")
    runs, cur = [], None
    for t in sorted(conc):
        if conc[t] > 0.5:
            cur = [t, t] if cur is None else [cur[0], t]
        elif cur:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    runs = [r for r in runs if r[1] - r[0] >= 200]  # drops the 60 s warm-up
    for a, b in runs:
        a += 20
        d = int(b - a)
        at = iso(b)
        mean_c = sum(v for t, v in conc.items() if a <= t <= b) / max(1, sum(1 for t in conc if a <= t <= b))
        level = min(LEVELS, key=lambda lv: abs(lv - mean_c))
        rng = f"[{d}s]"
        per_pod = instant(f"sum by(pod)(increase(vllm:prompt_tokens_total{{{M}}}{rng}) + increase(vllm:generation_tokens_total{{{M}}}{rng}))", at)
        per_pod = {dict(k)["pod"]: v / d for k, v in per_pod.items() if v > 0}
        row = dict(
            experiment=exp, step=step, mig_profile=profile, max_num_seqs=seqs, level=level,
            replicas=len(per_pod),
            total_tps=round(sum(per_pod.values()), 1),
            prefill_tps=round(scalar(f"sum(increase(vllm:prompt_tokens_total{{{M}}}{rng}))", at) / d, 1),
            decode_tps=round(scalar(f"sum(increase(vllm:generation_tokens_total{{{M}}}{rng}))", at) / d, 1),
            per_replica_tps=" / ".join(f"{v:.0f}" for _, v in sorted(per_pod.items())),
            ttft_p95_ms=round(1000 * scalar(f"histogram_quantile(0.95, sum by(le)(increase(vllm:time_to_first_token_seconds_bucket{{{M}}}{rng})))", at)),
            itl_p95_ms=round(1000 * scalar(f"histogram_quantile(0.95, sum by(le)(increase(vllm:inter_token_latency_seconds_bucket{{{M}}}{rng})))", at)),
            running_avg=round(scalar(f"avg_over_time((sum(vllm:num_requests_running{{{M}}}))[{d}s:15s])", at), 1),
            waiting_avg=round(scalar(f"avg_over_time((sum(vllm:num_requests_waiting{{{M}}}))[{d}s:15s])", at), 1),
            kv_usage_max=round(scalar(f"max_over_time((max(vllm:kv_cache_usage_perc{{{M}}}))[{d}s:15s])", at), 3),
            preemptions=round(scalar(f"sum(increase(vllm:num_preemptions_total{{{M}}}{rng}))", at)),
            gpu_w=round(scalar(f"avg_over_time((max(DCGM_FI_DEV_POWER_USAGE{{{G}}}))[{d}s:15s])", at), 1),
            sm_mhz=round(scalar(f"avg_over_time((max(DCGM_FI_DEV_SM_CLOCK{{{G}}}))[{d}s:15s])", at)),
            temp_c=round(scalar(f"avg_over_time((max(DCGM_FI_DEV_GPU_TEMP{{{G}}}))[{d}s:15s])", at), 1),
            start=iso(a), end=at,
        )
        row["sla_ok"] = row["ttft_p95_ms"] <= SLA_TTFT and row["itl_p95_ms"] <= SLA_ITL
        level_rows.append(row)

    # Akamas-style windowing: 6 x 30 s samples, ranked on total throughput.
    s = {k: series(q, t0, t1) for k, q in {
        "tot": f"sum(rate(vllm:prompt_tokens_total{{{M}}}[30s]))+sum(rate(vllm:generation_tokens_total{{{M}}}[30s]))",
        "ttft": f"histogram_quantile(0.95, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket{{{M}}}[30s])))*1000",
        "itl": f"histogram_quantile(0.95, sum by(le)(rate(vllm:inter_token_latency_seconds_bucket{{{M}}}[30s])))*1000",
        "run": f"sum(vllm:num_requests_running{{{M}}})",
    }.items()}
    ts = sorted(s["tot"])
    wins = []
    for i in range(len(ts) - 5):
        w = ts[i:i + 6]
        avg = lambda k: sum(s[k].get(x, float("nan")) for x in w) / 6  # noqa: E731
        wins.append(dict(start=w[0], tot=avg("tot"), ttft=avg("ttft"), itl=avg("itl"), run=avg("run")))
    wins = [w for w in wins if w["tot"] == w["tot"] and w["ttft"] == w["ttft"]]  # drop NaN windows
    best = max(wins, key=lambda w: w["tot"])
    ok = [w for w in wins if w["ttft"] <= SLA_TTFT and w["itl"] <= SLA_ITL]
    best_ok = max(ok, key=lambda w: w["tot"]) if ok else None
    window_rows.append(dict(
        experiment=exp, step=step, mig_profile=profile, max_num_seqs=seqs,
        max_window_total=round(best["tot"], 1), max_window_running=round(best["run"]),
        max_window_ttft_p95=round(best["ttft"]), max_window_itl_p95=round(best["itl"]),
        max_window_sla_ok=best["ttft"] <= SLA_TTFT and best["itl"] <= SLA_ITL,
        best_sla_ok_total=round(best_ok["tot"], 1) if best_ok else "",
        best_sla_ok_running=round(best_ok["run"]) if best_ok else "",
        max_window_start=iso(best["start"]),
    ))

for name, rows in (("levels.csv", level_rows), ("windows.csv", window_rows)):
    with open(name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print(f"{len(level_rows)} levels, {len(window_rows)} experiments")
