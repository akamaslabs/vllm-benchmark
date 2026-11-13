# Nsight traces of study 51's layouts

One NVIDIA Nsight Perf SDK for AI trace per layout of study 51, taken on the study's own pair of
g7e.4xlarge **after** the last experiment, from 22:43 UTC on 2026-11-11 to 00:43 UTC on
2026-11-12. Nsight did not run during the seven experiments: the recipe was only proven that
afternoon on a third node (`docs/spikes/2026-11-11-nsight-perf/README.md`), with the study
already running. So each layout was applied again with an Nsight variant of the vLLM pod, loaded
at the level the study found best for it, and traced for 10 seconds on every node that ran a
vLLM process.

## Captures

| Layout | Users | Capture start (UTC) | Nodes | TTFT p95 with Nsight (ms) | TTFT p95 in the study's Hold (ms) | ITL p95 (ms) | Sent / completed / errors | Zip size, node a / b (MB) |
|---|---|---|---|---|---|---|---|---|
| PP two | 512 | 22:58:20 | a, b | 260 | 254 | 50.0 | 1439 / 1438 / 1 | 9.3 / 10.0 |
| One GPU | 64 | 23:15:31 | a | 106 | 152 | 8.3 | 179 / 179 / 0 | 32.2 |
| Two replicas | 256 | 23:32:33 | a, b | 399 | 394 | 22.5 | 719 / 719 / 0 | 16.1 / 16.1 |
| TP two | 32 | 23:49:22 | a, b | 415 | 407 | 14.2 | 89 / 89 / 0 | 19.0 / 18.9 |
| DP two + expert parallel | 64 | 00:06:38 | a, b | 486 | 476 | 23.3 | 179 / 179 / 0 | 19.0 / 19.0 |
| TP two + expert parallel | 64 | 00:23:44 | a, b | 490 | 526 | 36.6 | 178 / 178 / 0 | 12.6 / 12.6 |
| DP two | 64 | 00:41:01 | a, b | 487 | 482 | 24.1 | 178 / 178 / 0 | 19.0 / 19.0 |

- Every level passed the study's rule (client TTFT p95 at most 500 ms) with the Nsight service
  attached, and TTFT p95 stayed within 10 ms of the study's Hold on five layouts. The service
  and the compatibility libcuda cost nothing visible at these levels.
- The two that differ, one GPU (106 against 152 ms) and TP two + expert parallel (490 against
  526 ms), are both *lower* here. The capture runs judge the requests started in a 3-minute
  window, the study's levels a 6-minute one, so these are fewer requests (179 against about
  twice as many) and not a like-for-like repeat. Do not read them as "Nsight makes it faster".
- PP two lost one request out of 1,439 (`ServerDisconnectedError` on client 0). The study's
  Hold at the same level lost none out of 2,878. One event does not say whether tracing caused
  it.
- Every vLLM pod had the compatibility `libcuda.so.620.70.06` mapped by its processes (12 or 18
  mappings per pod) before the load started: `run-logs/capture_all.log`.

## What the traces show

Read on 2026-11-13. A trace lists every kernel that ran on the GPU, with its start and duration.
That splits the GPU's time in three states: a compute kernel is running, only NCCL kernels are on
the GPU, or there is no kernel. NCCL kernels (`ncclDevKernel_*`) are the ones that exchange data
with the other node, so a GPU that holds only those is waiting for its peer.

| Layout | Node | Users | Compute kernel (%) | Only NCCL kernel (%) | No kernel (%) | GR Active (%) | SM Active (%) | Tensor Active (%) | nvidia-smi util (%) | Power (W) |
|---|---|---|---|---|---|---|---|---|---|---|
| Two replicas | a | 256 | 98.5 | 0.0 | 1.5 | 99.5 | 88.3 | 29.3 | 100 | 494 |
| Two replicas | b | 256 | 98.6 | 0.0 | 1.4 | 99.6 | 88.2 | 29.3 | 100 | 485 |
| One GPU | a | 64 | 82.9 | 0.0 | 17.1 | 85.2 | 63.0 | 6.8 | 100 | 337 |
| PP two | a | 512 | 76.9 | 23.0 | 0.1 | 100.0 | 64.1 | 15.8 | 100 | 401 |
| PP two | b | 512 | 70.3 | 29.2 | 0.5 | 99.9 | 63.6 | 16.2 | 100 | 393 |
| DP two + expert parallel | a | 64 | 37.5 | 60.2 | 2.3 | 99.2 | 27.7 | 3.9 | 100 | 238 |
| DP two + expert parallel | b | 64 | 38.1 | 59.6 | 2.3 | 99.1 | 28.2 | 4.0 | 100 | 256 |
| DP two | a | 64 | 36.4 | 61.4 | 2.2 | 99.2 | 27.7 | 3.9 | 100 | 237 |
| DP two | b | 64 | 36.7 | 60.8 | 2.5 | 99.0 | 27.9 | 4.0 | 100 | 252 |
| TP two | a | 32 | 24.2 | 43.4 | 32.4 | 68.7 | 16.0 | 1.8 | 98 | 201 |
| TP two | b | 32 | 24.0 | 43.6 | 32.4 | 68.7 | 15.9 | 1.8 | 100 | 200 |
| TP two + expert parallel | a | 64 | 22.4 | 76.3 | 1.3 | 99.4 | 17.9 | 2.6 | 99 | 236 |
| TP two + expert parallel | b | 64 | 22.8 | 75.9 | 1.3 | 99.4 | 18.3 | 2.7 | 99 | 236 |

The three time shares come from the kernel slices. GR Active, SM Active and Tensor Active are the
means of the trace's 1 kHz counters over the whole trace. The last two columns are one
`nvidia-smi` reading at the end of the capture (`run-logs/logs/<layout>-capture-<a|b>.log`).
Same table as a file: `analysis/gpu_time_split.csv`.

- **The DP and TP layouts wait for most of the time.** At 64 users they compute for 22 to 38 %
  of the trace and hold only an NCCL kernel for 60 to 76 %. TP two at 32 users waits for 43 to
  44 % and has no kernel at all for another 32 %: the GPU is empty for about 1 s out of every
  4 s.
- **PP two waits too, but much less:** 23 % on node a and 29 % on node b, with 70 to 77 % of the
  time in compute kernels.
- **One GPU and two replicas never wait:** they exchange nothing.
- **Utilization does not see the wait.** GR Active is the counter that `nvidia-smi` and DCGM
  report as GPU utilization. It reads 99 to 100 % on every layout at 64 users or more, whatever
  the share of waiting. SM Active and power follow the compute share instead: 88 % and 485 to
  494 W on the replicas, 64 % and 393 to 401 W on PP two, 16 to 28 % and 200 to 256 W on the DP
  and TP layouts.

### PP two, state by state

`analysis/gpu_state.sql` on the two PP two traces. Mean of each counter over the time spent in
each state, in percent:

| Node | State | Share of time (%) | GR Active | SM Active | Tensor Active | DRAM Read |
|---|---|---|---|---|---|---|
| a | compute kernel running | 76.9 | 100.0 | 81.9 | 20.2 | 67.5 |
| a | only NCCL kernel | 22.9 | 100.0 | 4.7 | 0.8 | 3.6 |
| a | whole trace | 100.0 | 100.0 | 64.1 | 15.8 | 52.8 |
| b | compute kernel running | 70.3 | 99.9 | 88.9 | 22.7 | 73.2 |
| b | only NCCL kernel | 29.2 | 100.0 | 2.7 | 0.4 | 1.7 |
| b | whole trace | 100.0 | 99.9 | 63.6 | 16.2 | 52.2 |

GR Active is 100 % in both states. SM Active drops from 82 to 89 % to under 5 %. Counted per
counter sample, in the milliseconds with no compute kernel at all (20.1 % of the samples on node
a, 26.3 % on node b), SM Active is 0.6 % and Tensor Active 0.0 % on both nodes, with GR Active at
100.0 % (`mean_when_no_compute` in `analysis/profiles/pp2-512users-node-<a|b>.json`). The
remaining 0.1 to 0.5 % of the time has no kernel: too few samples for its counters to mean
anything.

### How to reproduce

```bash
# One trace in the browser: unzip it, open the .perfetto-trace file on https://ui.perfetto.dev,
# paste analysis/gpu_state.sql in the "Query (SQL)" tab.

# The same query from a shell (uv builds a throwaway environment with the trace processor):
uv run --no-project --with perfetto --with pandas --with numpy \
  python -I analysis/run_sql.py analysis/gpu_state.sql pp2-a=<dir>/nsight-a.20261111_225820.perfetto-trace

# Every trace: one JSON profile per zip, then the table above.
bash analysis/profile_all.sh traces <work dir>
python3 analysis/gpu_time_split.py analysis/profiles run-logs/logs analysis/gpu_time_split.csv
```

### Limits of this reading

- **Ten seconds, after the study.** Each capture is a re-run of the layout at its best level,
  not the study's Hold.
- **Waiting includes the transfer.** All the time with only NCCL kernels on the GPU counts as
  waiting, the copy itself included. SM Active near zero in that state says the GPU does almost
  no work there.
- **Waiting for what is not in the trace.** It shows that the GPU waits for its peer, not whether
  the time goes to the TCP wire or to the peer's own compute. On PP two part of the wait is
  expected: each node holds half the layers and waits for the other half. The same captures on
  EFA (study 50) would separate the two.
- **Counters at 1 ms.** The time shares come from kernel start and end times. The counters are
  sampled every millisecond and the DP and TP kernels last 0.1 to 0.2 ms, so per-state counter
  means are approximate on those layouts.
- **Power is not in the trace.** It is the single `nvidia-smi` reading at the end of each
  capture.
- **Not checked:** DCGM read lower than 99 % on some 32-user levels of the study (50 % on one
  GPU). The TP two trace shows a GPU with no kernel for about 1 s in every 4 s at that level, and
  a 30 s scrape may catch that unevenly. This is a hypothesis.

## Where the traces are

13 Perfetto zips, 219 MB in total, in `traces/` of this folder on the machine that ran the
captures. **They are not in git** (`.gitignore`): the nodes are gone, so those local files are
the only copy. Names: `<layout>-<users>users-node-<a|b>.perfetto-trace.zip`.

Each zip holds one `<pod>.<timestamp>.perfetto-trace` file (38 MB for PP two, 148 MB for one GPU)
and `perfetto_manifest.json`. Unzip it and open the `.perfetto-trace` file on
<https://ui.perfetto.dev>. What the tracks mean is in the spike's README ("Reading it in
Perfetto").

## How it was done

1. `00-nsight-pods.yaml`: one privileged `hostPID` pod per GPU node (`nsight-a`, `nsight-b`),
   image `nvidia/cuda:13.5.0-devel-ubuntu24.04`. The node's driver (580.178.04) is not touched.
2. `build.sh`, run inside each pod: builds the SDK (commit
   `bd5d5901c6c9b72704245dd0f6e8e1a822bda527`) into `/work/sdk/build/bin` and copies the image's
   CUDA 13.5 compatibility libraries to `/work/compat` (`/var/lib/nsight/compat` on the node).
3. The service, in each pod: `LD_LIBRARY_PATH=/usr/lib64 ./nvidia-nsight-perf --global`. It has
   to be up before a vLLM process initialises CUDA.
4. `01-vllm_template.yaml`: the study's pod template with three additions marked `nsight:`. The
   compatibility libcuda (620.70.06) first in `LD_LIBRARY_PATH`, and the mounts of the service's
   socket directory (`/run/nvidia/nsight-perf`) and of the compatibility libraries.
5. `capture_all.sh <layout>:<users> ...`, run in the toolbox next to a copy of the study's `k8s/`
   folder with that template. Per layout: the study's `apply_config.sh` at the preset's flags
   (about 3.5 minutes), one load level of 780 s with the study's `run_ramp.sh`, and 690 s into
   it `nvtrace_streaming_consumer --duration-ms 10000` on each node. The exporter
   (`nvtrace_perfetto_trace_exporter --tsf-database-dir <tsf>`) then writes the zip.
6. `fetch.sh`: downloads a zip from a pod in 8 MiB chunks with an md5 check. `kubectl cp`
   dropped the stream on files of this size.

## What did not go as scripted

- **The export step of the run failed for every layout.** The copy of `capture_all.sh` that ran
  in the toolbox looked for the TSF directory as `nsight.*`, and the consumer names it
  `<pod hostname>.<timestamp>` (`nsight-a.20261111_225820`). The captures themselves were fine,
  and each one was exported by hand with the same exporter command right after its load level
  ended. `run-logs/capture_all.log` shows the failed `export` lines. The script in this folder
  has the glob fixed, and that fixed export step has not been run end to end.
- PP two's hand export ran at 22:59, about a minute before its load level ended, so it shared
  the node with the last requests of that level. The other six were exported after their level.
- The consumer prints `ConsumeNextChunk || V1 chunk overrun detected while chasing a chunk
  chain` at the start of most captures. The TSF database is still complete enough for the
  exporter: every export succeeded and every zip passes `unzip -t`.
- TSF databases were 131 to 231 MB per node for 10 seconds. A trace of a whole 16-minute level
  would be two orders of magnitude larger, which is why the captures are 10-second samples of
  the steady part of the level.

## Files

- `00-nsight-pods.yaml`, `01-vllm_template.yaml`, `build.sh`, `capture_all.sh`, `fetch.sh`: see
  above.
- `run-logs/capture_all.log`: the run's own log (apply, libcuda proof, vLLM gauges and GPU power
  at capture time, AIPerf result per layout).
- `run-logs/logs/<layout>-{apply,load,capture-<role>,export-<role>}.log`: the per-step logs.
- `traces/`: the zips, not in git.
- `analysis/gpu_state.sql`: the query that splits a trace by GPU state, for ui.perfetto.dev or
  `analysis/run_sql.py`.
- `analysis/profile_trace.py`, `analysis/profile_all.sh`: one JSON profile per trace (counters,
  kernel time, NCCL share, kernel families, CUDA driver calls), in `analysis/profiles/`.
- `analysis/gpu_time_split.py`, `analysis/gpu_time_split.csv`: the per-node table of "What the
  traces show".
