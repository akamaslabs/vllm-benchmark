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
