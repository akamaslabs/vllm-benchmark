#!/usr/bin/env python3
"""One summary block per Perfetto trace: usage steps.py <label>=<file.perfetto-trace> ...
Per trace: duration, slices per second by category, the mean of every counter track, and the
GPU kernel families with the most total time. Needs the perfetto Python package (trace_processor)."""
import re
import sys
from perfetto.trace_processor import TraceProcessor

def rows(tp, sql):
    return [tuple(getattr(r, k) for k in r.__dict__ if not k.startswith("_")) for r in tp.query(sql)]

def family(name):
    """Kernel name without template and call arguments: one row per kernel family."""
    name = name[5:] if name.startswith("void ") else name
    # cuBLAS kernels are declared "std::enable_if<...>::type internal::gemvx::kernel<...>(...)".
    name = re.sub(r"^std::enable_if<[^>]*>::type ", "", name)
    return re.split(r"[<(]", name, maxsplit=1)[0] or name

for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    tp = TraceProcessor(trace=path)
    secs = rows(tp, "select (end_ts - start_ts) / 1e9 from trace_bounds")[0][0]
    print(f"== {label}: {secs:.2f} s")
    for cat, n, ms in rows(tp, "select category, count(*), sum(dur) / 1e6 from slice group by 1 order by 2 desc"):
        print(f"  slices {cat}: {n / secs:.0f}/s, {ms / secs:.0f} ms per second of trace")
    for name, n, avg, mx in rows(tp, "select ct.name, count(*), avg(c.value), max(c.value) from counter_track ct "
                                     "join counter c on c.track_id = ct.id group by 1 order by 1"):
        print(f"  counter {name}: mean {avg:.2f}, max {mx:.2f} ({n} samples)")
    fam = {}
    for name, n, ms in rows(tp, "select name, count(*), sum(dur) / 1e6 from slice where category = 'cuda_gpu_kernel_launch' group by 1"):
        f = family(name); a = fam.setdefault(f, [0, 0.0]); a[0] += n; a[1] += ms
    total = sum(a[1] for a in fam.values())
    for f, (n, ms) in sorted(fam.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"  kernel {100 * ms / total:5.1f} % of kernel time {ms / secs:7.1f} ms/s {n / secs:8.0f}/s  {f}")
    tp.close()
