#!/usr/bin/env python3
"""GPU activity profile of study 51's Nsight traces: usage profile_trace.py <label>=<file.perfetto-trace> ...
Per trace: every counter (mean, and share of the 1 kHz samples below 5 % and above 95 %), the time
with at least one GPU kernel executing (union of the kernel slices), the idle gaps between
kernels, the kernel families and the CUDA driver calls with the most time."""
import json
import re
import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor


def family(name):
    name = name[5:] if name.startswith("void ") else name
    name = re.sub(r"^std::enable_if<[^>]*>::type ", "", name)
    return re.split(r"[<(]", name, maxsplit=1)[0] or name


out = {}
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    tp = TraceProcessor(trace=path)
    df = lambda sql: tp.query(sql).as_pandas_dataframe()
    b = df("select start_ts, end_ts from trace_bounds").iloc[0]
    secs = (b.end_ts - b.start_ts) / 1e9
    r = {"secs": round(secs, 2), "counters": {}, "categories": {}}
    for _, c in df("select category, count(*) n, sum(dur)/1e6 ms from slice group by 1").iterrows():
        r["categories"][c.category] = {"per_s": round(c.n / secs), "ms_per_s": round(c.ms / secs, 1)}
    cs = df("select ct.name name, c.value v from counter_track ct join counter c on c.track_id = ct.id")
    for name, g in cs.groupby("name"):
        v = g.v.to_numpy()
        r["counters"][name] = {"mean": round(float(v.mean()), 2), "p50": round(float(np.median(v)), 2),
                               "below5": round(float((v < 5).mean()) * 100, 1),
                               "above95": round(float((v > 95).mean()) * 100, 1), "n": int(len(v))}
    k = df("select ts, dur, name from slice where category = 'cuda_gpu_kernel_launch' order by ts")
    if len(k):
        s = k.ts.to_numpy(); e = s + k.dur.to_numpy()
        emax = np.maximum.accumulate(e)
        new = np.r_[True, s[1:] > emax[:-1]]              # a kernel that starts after everything before it ended
        starts = s[new]; ends = np.r_[emax[:-1][new[1:]], emax[-1]]
        busy = (ends - starts).sum()
        gaps = (starts[1:] - ends[:-1]) / 1e3             # microseconds
        span = (ends[-1] - starts[0]) / 1e9
        r["kernels"] = {"per_s": round(len(k) / secs), "busy_pct": round(100 * busy / (span * 1e9), 1),
                        "sum_ms_per_s": round(k.dur.sum() / 1e6 / secs, 1),
                        "gaps_per_s": round(len(gaps) / span),
                        "gap_p50_us": round(float(np.median(gaps)), 1) if len(gaps) else 0,
                        "gap_p99_us": round(float(np.percentile(gaps, 99)), 1) if len(gaps) else 0,
                        "gap_max_ms": round(float(gaps.max()) / 1e3, 2) if len(gaps) else 0,
                        "idle_pct_gaps_over_100us": round(100 * gaps[gaps > 100].sum() / (span * 1e6), 1),
                        "idle_pct_gaps_over_1ms": round(100 * gaps[gaps > 1000].sum() / (span * 1e6), 1),
                        "gaps_over_1ms_per_s": round(float((gaps > 1000).sum()) / span, 1)}
        k["fam"] = k.name.map(family)
        # NCCL's device kernels sit on the GPU while they wait for the peer: split the time with a
        # kernel on the GPU into "a compute kernel runs" and "only NCCL kernels run".
        def merged(x):
            s = x.ts.to_numpy(); e = s + x.dur.to_numpy(); m = np.maximum.accumulate(e)
            new = np.r_[True, s[1:] > m[:-1]]
            return s[new], np.r_[m[:-1][new[1:]], m[-1]]
        def cover(st, en, t):
            """Busy time of the merged intervals (st, en) before each time in t."""
            cum = np.r_[0, np.cumsum(en - st)]
            i = np.searchsorted(st, t, "right") - 1
            j = np.clip(i, 0, len(st) - 1)
            return np.where(i < 0, 0, cum[j] + np.clip(t - st[j], 0, en[j] - st[j]))
        is_nccl = k.fam.str.startswith("ncclDevKernel")
        comp = k[~is_nccl]; ncc = k[is_nccl]
        cs_, ce_ = merged(comp)
        r["kernels"]["compute_busy_pct"] = round(100 * (ce_ - cs_).sum() / (span * 1e9), 1)
        r["kernels"]["nccl_only_pct"] = round(r["kernels"]["busy_pct"] - r["kernels"]["compute_busy_pct"], 1)
        if len(ncc):
            ns_, ne_ = merged(ncc)
            r["kernels"]["nccl_on_gpu_pct"] = round(100 * (ne_ - ns_).sum() / (span * 1e9), 1)
            r["kernels"]["nccl_per_s"] = round(len(ncc) / secs)
            r["kernels"]["nccl_mean_ms"] = round(float(ncc.dur.mean()) / 1e6, 2)
            r["kernels"]["nccl_p95_ms"] = round(float(np.percentile(ncc.dur, 95)) / 1e6, 2)
        # Counter samples (1 kHz) split by how much of their millisecond had a compute kernel running.
        ct = df("select ct.name name, c.ts ts, c.value v from counter_track ct join counter c on c.track_id = ct.id "
                "where ct.name like '%(GPU Metric)' and ct.name not like '%Clock%'")
        r["by_compute"] = {}
        for name, g in ct.groupby("name"):
            ts = g.ts.to_numpy(); v = g.v.to_numpy(); ms = int(np.median(np.diff(ts)))
            best = None
            for side, lo, hi in (("before", ts - ms, ts), ("after", ts, ts + ms)):
                frac = (cover(cs_, ce_, hi) - cover(cs_, ce_, lo)) / ms
                idle = frac < 0.05; full = frac > 0.95
                d = {"window": side, "period_us": ms / 1e3,
                     "samples_no_compute_pct": round(100 * float(idle.mean()), 1),
                     "samples_all_compute_pct": round(100 * float(full.mean()), 1),
                     "mean_when_no_compute": round(float(v[idle].mean()), 2) if idle.any() else None,
                     "mean_when_all_compute": round(float(v[full].mean()), 2) if full.any() else None}
                gap = abs((d["mean_when_all_compute"] or 0) - (d["mean_when_no_compute"] or 0))
                if best is None or gap > best[0]: best = (gap, d)
            r["by_compute"][name.replace(" (Utilization) (GPU Metric)", "")] = best[1]
        f = k.groupby("fam").dur.agg(["count", "sum"]).sort_values("sum", ascending=False)
        tot = f["sum"].sum()
        r["families"] = [{"name": n, "pct": round(100 * x["sum"] / tot, 1), "ms_per_s": round(x["sum"] / 1e6 / secs, 1),
                          "per_s": round(x["count"] / secs)} for n, x in f.head(12).iterrows()]
    d = df("select name, count(*) n, sum(dur)/1e6 ms, max(dur)/1e6 mx from slice where category = 'cuda_driver' group by 1 order by ms desc limit 10")
    r["driver"] = [{"name": x["name"], "per_s": round(x.n / secs), "ms_per_s": round(x.ms / secs, 1), "max_ms": round(x.mx, 2)} for _, x in d.iterrows()]
    r["tracks"] = [{"name": x["name"], "n": int(x.n)} for _, x in
                   df("select coalesce(t.name,'?') name, count(*) n from slice s join track t on s.track_id = t.id group by 1 order by n desc limit 12").iterrows()]
    out[label] = r
    tp.close()
    print(label, "done", file=sys.stderr)
json.dump(out, sys.stdout, indent=1)
