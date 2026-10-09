#!/usr/bin/env python3
"""Paired accuracy comparison for study 30's accuracy check (eval/README.md, "Decision rule").

Usage: compare.py <results_dir>

Reads <results_dir>/<run>/lmeval/<protocol>/<task>/: lm-eval 0.4.13's results_*.json and
samples_<task>_*.jsonl[.gz] (--log_samples, one level down), and the vLLM finished-reason
snapshots finished_{before,after}.json. Writes comparison.md and comparison.json into
<results_dir>. Exit 2 on inconsistent inputs: different doc_id sets, different task
versions, a missing or duplicated file. Standard library only.
"""
import glob
import gzip
import json
import math
import os
import random
import sys

# task -> (filter, primary metric, margin in points, (filter, secondary metric))
TASKS = {
    "gsm8k_platinum_cot_llama": ("strict-match", "exact_match", 1.0, ("flexible-extract", "exact_match")),
    "ifeval": ("none", "prompt_level_strict_acc", 2.0, ("none", "inst_level_strict_acc")),
}
# RedHat's FP8 column (no thinking) and the tolerance of one run against a 3-seed mean.
ANCHOR = {"gsm8k_platinum_cot_llama": (95.37, 1.4), "ifeval": (89.34, 3.1)}
# (B, A): every comparison is B - A against the first baseline.
PAIRS = [("best", "baseline-a"), ("best-kv-auto", "baseline-a"),
         ("best-no-mtp", "baseline-a"), ("baseline-b", "baseline-a")]
BOOTSTRAP = 10000
SEED = 30
TRUNCATION_WARN = 1.0  # percent of a task's requests ending on "length"


class InputError(Exception):
    pass


def task_dir(root, run, protocol, task):
    return os.path.join(root, run, "lmeval", protocol, task)


def one_file(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if len(files) != 1:
        raise InputError(f"expected one file for {pattern}, found {len(files)}")
    return files[0]


def load_samples(root, run, protocol, task):
    """-> {(doc_id, filter): record}"""
    path = one_file(os.path.join(task_dir(root, run, protocol, task), "**", f"samples_{task}_*.jsonl*"))
    opener = gzip.open if path.endswith(".gz") else open
    out = {}
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["doc_id"], r["filter"])] = r
    return out


def task_version(root, run, protocol, task):
    path = one_file(os.path.join(task_dir(root, run, protocol, task), "**", "results_*.json"))
    with open(path) as f:
        return json.load(f)["versions"][task]


def scores(samples, flt, metric):
    """-> {doc_id: value}; an instruction-level value stays a list."""
    return {d: r[metric] for (d, f), r in samples.items() if f == flt}


def accuracy(values):
    flat = [float(x) for v in values for x in (v if isinstance(v, list) else [v])]
    return 100.0 * sum(flat) / len(flat)


def mcnemar_p(lost, gained):
    """Exact two-sided McNemar p-value on the discordant pairs."""
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b, seed=SEED, resamples=BOOTSTRAP):
    """a, b: {doc_id: 0/1}. Delta = b - a in points, 95 % CI by paired bootstrap."""
    if set(a) != set(b):
        raise InputError(f"different doc_id sets ({len(a)} vs {len(b)} docs)")
    ids = sorted(a)
    d = [float(b[i]) - float(a[i]) for i in ids]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=n)) / n for _ in range(resamples))
    lo, hi = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    lost = [i for i in ids if a[i] and not b[i]]
    gained = [i for i in ids if b[i] and not a[i]]
    return {"n": n, "acc_a": accuracy(a.values()), "acc_b": accuracy(b.values()),
            "delta": 100.0 * sum(d) / n, "ci": [100.0 * lo, 100.0 * hi],
            "lost": lost, "gained": gained, "mcnemar_p": mcnemar_p(len(lost), len(gained))}


def verdict(lo, hi, margin):
    if lo > -margin:
        return "no degradation" if hi >= 0 else "measurable drop within the margin"
    return "degradation" if hi < 0 else "inconclusive"


def truncations(root, run, protocol, task):
    d = task_dir(root, run, protocol, task)
    with open(os.path.join(d, "finished_before.json")) as f:
        before = json.load(f)
    with open(os.path.join(d, "finished_after.json")) as f:
        after = json.load(f)
    diff = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    total = sum(diff.values())
    length = diff.get("length", 0)
    return {"requests": total, "length": length, "share": 100.0 * length / total if total else 0.0}


def analyse(root):
    runs = sorted(r for r in os.listdir(root) if os.path.isdir(os.path.join(root, r, "lmeval")))
    report = {"anchor": {}, "accuracy": {}, "truncations": {}, "pairs": [], "warnings": []}
    for task, (flt, metric, margin, (flt2, metric2)) in TASKS.items():
        greedy, versions = {}, {}
        for run in runs:
            if not os.path.isdir(task_dir(root, run, "greedy", task)):
                # Every run of the plan has greedy results for both tasks.
                report["warnings"].append(f"{run} {task}: greedy results missing, its comparisons are left out")
                continue
            s = load_samples(root, run, "greedy", task)
            greedy[run] = s
            versions[run] = task_version(root, run, "greedy", task)
            report["accuracy"].setdefault(task, {})[run] = {
                metric: accuracy(scores(s, flt, metric).values()),
                f"{metric2},{flt2}": accuracy(scores(s, flt2, metric2).values())}
            tr = truncations(root, run, "greedy", task)
            report["truncations"].setdefault(task, {})[run] = tr
            if tr["share"] > TRUNCATION_WARN:
                report["warnings"].append(
                    f"{run} {task}: {tr['share']:.1f} % of requests truncated (> {TRUNCATION_WARN} %)")
        if len(set(versions.values())) > 1:
            raise InputError(f"{task}: different task versions {versions}")
        if os.path.isdir(task_dir(root, "baseline-a", "card", task)):
            acc = accuracy(scores(load_samples(root, "baseline-a", "card", task), flt, metric).values())
            ref, tol = ANCHOR[task]
            within = abs(acc - ref) <= tol
            tr = truncations(root, "baseline-a", "card", task)
            report["anchor"][task] = {"accuracy": acc, "reference": ref, "tolerance": tol, "within": within,
                                      "truncations": tr}
            if tr["share"] > TRUNCATION_WARN:
                report["warnings"].append(
                    f"baseline-a card {task}: {tr['share']:.1f} % of requests truncated (> {TRUNCATION_WARN} %)")
            if not within:
                report["warnings"].append(
                    f"anchor {task}: {acc:.2f} vs card {ref} +- {tol}: find out why before reading any delta")
        for b, a in PAIRS:
            # A run absent from the results is a skipped ablation; a missing reference is not.
            if b in greedy and a not in greedy:
                report["warnings"].append(f"{task}, {b} vs {a}: {a} missing, no comparison")
            if a in greedy and b in greedy:
                p = paired(scores(greedy[a], flt, metric), scores(greedy[b], flt, metric))
                p.update(task=task, a=a, b=b, metric=metric, margin=margin,
                         verdict=verdict(p["ci"][0], p["ci"][1], margin))
                report["pairs"].append(p)
    return report


def markdown(r):
    out = ["# Study 30 accuracy check: comparison", ""]
    if r["warnings"]:
        out += ["## Warnings", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    out += ["## Anchor (baseline-a, card protocol, vs RedHat's FP8 column)", "",
            "| Task | Accuracy | Card | Tolerance | Within | Truncated |", "|---|---|---|---|---|---|"]
    for task, x in r["anchor"].items():
        out.append(f"| {task} | {x['accuracy']:.2f} | {x['reference']} | +-{x['tolerance']} | "
                   f"{'yes' if x['within'] else 'NO'} | {x['truncations']['length']:.0f} / {x['truncations']['requests']:.0f} |")
    out += ["", "## Accuracy (greedy)", "", "| Task | Run | Primary | Secondary | Truncated |", "|---|---|---|---|---|"]
    for task, runs in r["accuracy"].items():
        for run, m in runs.items():
            (k1, v1), (k2, v2) = list(m.items())
            tr = r["truncations"][task][run]
            out.append(f"| {task} | {run} | {k1} {v1:.2f} | {k2} {v2:.2f} | "
                       f"{tr['length']:.0f} / {tr['requests']:.0f} |")
    out += ["", "## Paired comparisons (greedy, primary metric, B - A)", "",
            "| Task | B vs A | n | A | B | Delta | 95 % CI | Lost | Gained | McNemar p | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in r["pairs"]:
        out.append(f"| {p['task']} | {p['b']} vs {p['a']} | {p['n']} | {p['acc_a']:.2f} | {p['acc_b']:.2f} | "
                   f"{p['delta']:+.2f} | [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] | {len(p['lost'])} | "
                   f"{len(p['gained'])} | {p['mcnemar_p']:.3g} | {p['verdict']} |")
    out += ["", "## Flips (doc_id)", ""]
    for p in r["pairs"]:
        out.append(f"- {p['task']}, {p['b']} vs {p['a']}: lost {p['lost'][:30]}, gained {p['gained'][:30]}")
    return "\n".join(out) + "\n"


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = argv[1]
    try:
        report = analyse(root)
    except (InputError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    with open(os.path.join(root, "comparison.json"), "w") as f:
        json.dump(report, f, indent=1)
    md = markdown(report)
    with open(os.path.join(root, "comparison.md"), "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
