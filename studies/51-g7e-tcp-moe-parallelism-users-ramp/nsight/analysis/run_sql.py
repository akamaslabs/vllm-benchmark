#!/usr/bin/env python3
"""Run a Perfetto SQL file on one or more traces and print each result as a table.

    uv run --no-project --with perfetto --with pandas --with numpy \
        python -I analysis/run_sql.py analysis/gpu_state.sql pp2-a=<file.perfetto-trace> [...]

The same SQL can be pasted in the Query (SQL) tab of https://ui.perfetto.dev with a trace open.
"""
import sys
import time

from perfetto.trace_processor import TraceProcessor

sql = open(sys.argv[1]).read()
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    tp = TraceProcessor(trace=path)
    start = time.time()
    result = tp.query(sql).as_pandas_dataframe()
    print("=====", label, f"{time.time() - start:.1f} s")
    print(result.to_string(index=False))
    tp.close()
