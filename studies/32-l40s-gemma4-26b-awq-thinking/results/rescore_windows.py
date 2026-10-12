"""Re-score study 32's experiments under other window rules, from an Akamas export.

Usage: python3 rescore_windows.py <unpacked export dir> [out.csv]

The study's score is the best 6-sample (30 s) window by vllm.request_success_rate with
vllm.e2e_request_latency_p95 :max <= 30 s; with thinking on, the p95 output length is
2000-4000 tokens, so that constraint mostly samples the output-length tail and the same
configuration scores very differently from run to run (2026-10-12: experiments 34 / 35,
identical, 0.573 / 0.088). This script slides the same 6-sample window over the exported
series of each experiment and reports the best window's completed requests/s under:
  e2e30   e2e p95 :max <= 30 s (the study's own rule: Akamas' score for 48 of 53 experiments
          on 2026-10-12; 43, 47, 50, 52 differ, all below 0.1 req/s, near samples missing a
          metric, which this script skips (how Akamas treats them is not checked); 49 violated
          the constraint)
  e2e60   e2e p95 :max <= 60 s
  e2e120  e2e p95 :max <= 120 s
  ttft3   TTFT p95 :max <= 3 s (capacity before queueing; identical configurations agree
          within ~5-8 %)
  noqueue num_requests_waiting :max == 0
plus, per rule, whether that best window is the warm-up (<rule>_warmup): the Job's 8 discarded
single-turn requests (~1300-token prompts) complete in the trial's first minutes, before the
measured ramp, and when no window of the measured run meets the rule, the best one is theirs
(2026-10-12: experiments 24, 35, 44, 47, 56, 58, 59 under e2e30, scored 0.08-0.10 req/s on it).
Flagged when the window's mean input_sequence_length_avg is below 1300 tokens: the measured
run's is ~3000-4000. Then the experiment's parameters and Akamas' score and goal state.
"""
import collections
import csv
import json
import sys

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "rescore_windows.csv"
study = json.load(open(f"{src}/study.json"))
steps = {s["id"]: s["name"] for s in study["study"]["steps"]}
METRICS = ["request_success_rate", "e2e_request_latency_p95", "time_to_first_token_p95",
           "num_requests_waiting"]
series = collections.defaultdict(dict)
for m in METRICS:
    for p in json.load(open(f"{src}/metrics-vllm-{m}.json")):
        for x in p["studyExperimentTrialIds"]:
            series[x["experiment"]].setdefault(p["timestamp"][:19], {})[m] = p["value"]
# Read apart from METRICS: a sample without it does not make a window invalid.
isl = collections.defaultdict(dict)
for p in json.load(open(f"{src}/metrics-vllm-input_sequence_length_avg.json")):
    for x in p["studyExperimentTrialIds"]:
        isl[x["experiment"]][p["timestamp"][:19]] = p["value"]

RULES = {
    "e2e30": lambda w: max(v["e2e_request_latency_p95"] for v in w) <= 30000,
    "e2e60": lambda w: max(v["e2e_request_latency_p95"] for v in w) <= 60000,
    "e2e120": lambda w: max(v["e2e_request_latency_p95"] for v in w) <= 120000,
    "ttft3": lambda w: max(v["time_to_first_token_p95"] for v in w) <= 3000,
    "noqueue": lambda w: max(v["num_requests_waiting"] for v in w) < 0.5,
}


def best(ex, ok):
    """The best window's mean completed requests/s and its mean ISL (None if no valid window)."""
    pts = sorted(series[ex].items())
    top = None
    for i in range(len(pts) - 5):
        w = [v for _, v in pts[i:i + 6]]
        if any(any(m not in v for m in METRICS) for v in w) or not ok(w):
            continue
        s = sum(v["request_success_rate"] for v in w) / 6
        if top is None or s > top[0]:
            vals = [isl[ex][t] for t, _ in pts[i:i + 6] if t in isl[ex]]
            top = (s, sum(vals) / len(vals) if vals else None)
    return top


PARAMS = ["gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens", "kv_cache_dtype",
          "performance_mode", "optimization_level", "scheduling_policy", "async_scheduling",
          "max_cudagraph_capture_size", "block_size", "linear_backend", "spec_method",
          "spec_tokens"]
rows = []
for e in sorted(study["experiments"], key=lambda x: x["id"]):
    conf = e["manualConfiguration"] or {}
    p = conf.get("parameters") or e["suggestedConfiguration"]["parameters"]
    row = {"experiment": e["id"], "step": steps[e["step"]], "akamas_score": e["score"],
           "goal_state": e["goalState"]}
    for r, ok in RULES.items():
        v = best(e["id"], ok)
        row[r] = None if v is None else round(v[0], 3)
        row[r + "_warmup"] = None if v is None or v[1] is None else v[1] < 1300
    for k in PARAMS:
        v = p.get(f"vllm.{k}")
        row[k] = v.get("value") if isinstance(v, dict) else v
    rows.append(row)

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} experiments -> {out}")
