#!/usr/bin/env python3
"""Per-level table of study 51 from the Akamas task logs.

    python3 results/parse_levels.py results/logs results/levels.csv

Reads logs/exp<N>.csv (one `akamas log --dump` per experiment) and writes one row per load level
of each experiment: the levels of the Ramp task and the one level of the Hold task. Every number
comes from the lines run_ramp.sh / run_hold.sh print: the combined RESULT line (requests started
after minute 10; its latencies are the worse of the two AIPerf clients, not a pooled percentile),
the two per-client RESULT lines, the vLLM counter deltas over the whole 16-minute run, and the
per-answer means.
"""
import csv
import re
import sys

LAYOUTS = {1: "One GPU", 2: "Two replicas", 3: "DP two", 4: "DP two + expert parallel",
           5: "TP two", 6: "TP two + expert parallel", 7: "PP two"}
START = re.compile(r"^(\d\d:\d\d:\d\d) run_(ramp|hold): \[(?:ramp|hold)-(\d+)\] users=(\d+): .*duration=(\d+)s")
RESULT = re.compile(r"^RESULT sent=(\d+) completed=(\d+) errors=(\d+) ttft_p50_ms=(\d+) ttft_p95_ms=(\d+) itl_p95_ms=([\d.]+)")
CLIENT = re.compile(r"^client (\d): RESULT sent=\d+ completed=\d+ errors=\d+ ttft_p50_ms=(\d+) ttft_p95_ms=(\d+)")
COUNTER = re.compile(r"^vllm:(\w+) (\d+)")
VERDICT = re.compile(r"RAMP level=(\d+) (pass|fail) (.*)")
PER_ANSWER = re.compile(r"^client \d: per answer (\w+): mean ([\d.]+) .*\(n=(\d+)\)")
CPU = re.compile(r"^client \d: cpu used \d+ s \(([\d.]+) cores on average\)")
FIELDS = ["experiment", "layout", "task", "users", "start_utc", "duration_s", "verdict", "verdict_detail",
          "sent", "completed", "errors", "ttft_p50_ms", "ttft_p95_ms", "itl_p95_ms",
          "ttft_p50_ms_client0", "ttft_p95_ms_client0", "ttft_p50_ms_client1", "ttft_p95_ms_client1",
          "ttft_mean_ms", "itl_mean_ms", "first_output_token_mean_ms", "input_tokens_mean",
          "generation_tokens", "prompt_tokens", "prefix_cache_hits", "prefix_cache_queries",
          "prefix_cache_hit_rate", "preemptions", "requests_served", "output_tokens_per_s",
          "client_cpu_cores_max"]


def task_text(rows, task):
    # Airflow cuts long output into fixed-size messages: joining them as they are rebuilds the lines.
    return "".join(r["message"] for r in rows if r["workflowTaskName"] == task)


def levels(text, experiment):
    out, cur = [], None
    for line in text.splitlines():
        m = START.match(line)
        if m:
            cur = {"experiment": experiment, "layout": LAYOUTS[experiment], "task": m.group(2),
                   "users": int(m.group(4)), "start_utc": m.group(1), "duration_s": int(m.group(5)),
                   "_means": {}, "_cpu": []}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = RESULT.match(line)
        if m:
            for k, v in zip(("sent", "completed", "errors", "ttft_p50_ms", "ttft_p95_ms"), m.groups()):
                cur[k] = int(v)
            cur["itl_p95_ms"] = float(m.group(6))
            continue
        m = CLIENT.match(line)
        if m:
            cur["ttft_p50_ms_client" + m.group(1)] = int(m.group(2))
            cur["ttft_p95_ms_client" + m.group(1)] = int(m.group(3))
            continue
        m = COUNTER.match(line)
        if m:
            cur["_c_" + m.group(1)] = int(m.group(2))
            continue
        m = PER_ANSWER.match(line)
        if m:
            cur["_means"].setdefault(m.group(1), []).append((float(m.group(2)), int(m.group(3))))
            continue
        m = CPU.match(line)
        if m:
            cur["_cpu"].append(float(m.group(1)))
            continue
        m = VERDICT.search(line)   # may follow the tail of a cut AIPerf log line
        if m and int(m.group(1)) == cur["users"]:
            cur["verdict"], cur["verdict_detail"] = m.group(2), m.group(3)
    return out


def finish(row):
    def mean(name):
        v = row["_means"].get(name)
        return round(sum(a * n for a, n in v) / sum(n for _, n in v), 1) if v else ""
    row["ttft_mean_ms"] = mean("time_to_first_token")
    row["itl_mean_ms"] = mean("inter_token_latency")
    row["first_output_token_mean_ms"] = mean("time_to_first_output_token")
    row["input_tokens_mean"] = mean("input_sequence_length")
    for src, dst in (("generation_tokens_total", "generation_tokens"), ("prompt_tokens_total", "prompt_tokens"),
                     ("prefix_cache_hits_total", "prefix_cache_hits"),
                     ("prefix_cache_queries_total", "prefix_cache_queries"),
                     ("num_preemptions_total", "preemptions"), ("request_success_total", "requests_served")):
        row[dst] = row.get("_c_" + src, "")
    if row.get("prefix_cache_queries"):
        row["prefix_cache_hit_rate"] = round(row["prefix_cache_hits"] / row["prefix_cache_queries"], 4)
    if row.get("generation_tokens") != "":
        row["output_tokens_per_s"] = round(row["generation_tokens"] / row["duration_s"], 1)
    row["client_cpu_cores_max"] = max(row["_cpu"]) if row["_cpu"] else ""
    return {k: row.get(k, "") for k in FIELDS}


def main(logs_dir, out_path):
    csv.field_size_limit(10 ** 9)
    table = []
    for experiment in sorted(LAYOUTS):
        with open(f"{logs_dir}/exp{experiment}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for task in ("Ramp", "Hold"):
            table += [finish(r) for r in levels(task_text(rows, task), experiment)]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} levels written to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
