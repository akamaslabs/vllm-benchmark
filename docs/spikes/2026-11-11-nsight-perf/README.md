# Nsight Perf SDK for AI on a g7e node, 2026-11-11

Throwaway check that NVIDIA Nsight Perf SDK for AI
(<https://github.com/NVIDIA/nsight-perf-sdk-for-ai>) can trace vLLM on this cluster's GPU nodes
and give a file that opens in Perfetto: does it work on the installed driver, what does a trace
hold, how large is it, what does capturing cost. Run by hand on a third node next to study 51, no
Akamas study. Times are UTC.

**Verdict: it works, on the node's driver 580.178.04, with no driver update and no reboot.** The
traces name every CUDA kernel vLLM launches and carry 1 kHz GPU counters. A users ramp to
saturation was captured at every step. Capturing costs nothing measurable at 64 users; keeping the
setup on without capturing left throughput unchanged at saturation.

## Setup

- **Node:** one g7e.2xlarge on demand (RTX PRO 6000 Blackwell 96 GB, 8 vCPU, 64 GiB), us-east-2a,
  node group `nsight-probe-g7e-2xl` created with eksctl (`nodegroup-g7e-2xl-nsight.yaml`).
  AL2023, kernel 6.18.56, NVIDIA open driver 580.178.04. Up 13:59 to 15:49, 3.36 USD/h, about
  6.1 USD. Not tagged `AlwaysOn`: the lab's 17:00 stop was the cost guard.
- **Pods** (`00-base.yaml`, `vllm.yaml`, namespace `nsight-probe`): `nsight` (privileged,
  `hostPID`, `nvidia/cuda:13.5.0-devel-ubuntu24.04`), `vllm` (the AWS vLLM 0.29.0 image of study
  51, `openai/gpt-oss-120b`, the flags of study 51's one-GPU layout), `aiperf` (AIPerf 0.11.0).
- **SDK:** commit `bd5d5901c6c9b72704245dd0f6e8e1a822bda527` (2026-11-04), built with CMake in the
  `nsight` pod (`build.sh`). The proprietary service binary `nvidia-nsight-perf` is downloaded by
  the SDK's CMake from a public GitHub release: no NGC login.
- **Load:** the per-user load of study 51 (10 turns, 600 input and 200 output tokens per turn,
  one turn every 64 s, history re-sent).

## The recipe on driver 580

The SDK's documentation asks for a 620 driver. The kernel driver did not have to change:

1. **Service:** `./nvidia-nsight-perf --global` in the privileged `hostPID` pod
   (`NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=all`), started with
   `LD_LIBRARY_PATH=/usr/lib64` so that it loads the host's libcuda 580.178.04.
2. **Workload:** vLLM gets the CUDA 13.5 compatibility libraries (libcuda 620.70.06, from
   `/usr/local/cuda-13.5/compat` of the CUDA image) mounted read-only at `/nsight-compat` and
   first in `LD_LIBRARY_PATH` (`@COMPAT_DIR@` in `vllm.yaml`; empty = the host's libcuda, no
   tracing).
3. **Socket:** `/run/nvidia/nsight-perf` is a hostPath mounted at the same path in both pods.
   The service must be up before the workload initialises CUDA.
4. **Capture and export** (`cap.sh`): `nvtrace_streaming_consumer --duration-ms <ms>` writes a
   TSF directory, `nvtrace_perfetto_trace_exporter` turns it into `<name>.perfetto-trace.zip`.
   Unzip it and open the `.perfetto-trace` file on <https://ui.perfetto.dev>.

Unexplained: the first vLLM start with the compatibility libcuda failed (14:07 to 14:12, `Engine
core initialization failed ... Failed core proc(s): {}`). An identical second start worked, and so
did every later one.

## What a trace holds

- **Slices:** `cuda_gpu_kernel_launch` (every kernel, by name), `cuda_driver` (CUDA API calls),
  `cuda_gpu_command`, `system_info`, `gpu_grctx`.
- **Kernels by name:** Marlin MoE (`marlin_moe_wna16::Marlin`), cutlass GEMM, cuBLAS
  `internal::gemvx::kernel`, `kernel_mha`, flashinfer `BatchPrefillWithPagedKVCacheKernel`,
  `vllm::moe::*`, `triton_*`.
- **Counters at 1 kHz:** Tensor Active, SM Active, SM Issue, GR Active, DRAM Read and Write,
  Compute Warp Occupancy, PCIe TX and RX, clocks.
- **Windowed exports** (`--start-time-utc-ms`, `--end-time-utc-ms`) leave some kernel names
  unresolved (`??`). Export the whole capture when the names matter.

### Reading it in Perfetto

Opened zoomed out, a trace looks like a dozen rows. The data is underneath:

- **`GPU 0 (PAR 1)`** holds the kernels, about 100,000 slices per second at 64 users. Zoom to
  milliseconds; a click on a slice gives the kernel name and its launch dimensions.
- **`Thread 1`** under the process holds the CUDA API calls of vLLM.
- **The four `GPU 0 - ...` rows are groups:** expanded, they show the eleven counters (GR Active,
  SM Active, SM Issue, Tensor Active, Compute Warp Occupancy, DRAM Read, DRAM Write, PCIe RX,
  PCIe TX, GPC Clock, SYS Clock).
- **Query (SQL)** aggregates, for example the kernels by total time:
  `select name, count(*) n, sum(dur)/1e6 ms from slice group by 1 order by ms desc limit 20`.

### What else the service can collect (not tried)

These captures used the service's default configuration. Its counters are the default profile
`ReducedOverview.config`, which its authors keep "deliberately small" because it is always on.
Per the service's README (<https://github.com/NVIDIA/nsight-perf-daemon>), a YAML configuration
(`--export-default-config-only <file>`, edit, `--config <file>`, `--validate-config`) can change:

- **The counter set:** `GpuPmCounters.metricConfig` takes a MetricConfig v7 document with other
  PerfWorks counters of the chip. One set must fit one collection pass.
- **The sampling rate:** `pmSampleRateHz`, up to 20 kHz recommended (1 kHz here).
- **CUDA Runtime API calls:** `CudaApiTrace.traceRuntimeApiCalls` (driver API calls only here).
- **NVTX marks and push-pop ranges**, if the application emits them. vLLM was not checked.
- **GPU power:** only with a kernel driver R615 or newer and on Rubin GPUs, so not on this node.

Not in this tool: GPU memory in use, temperature, CPU profiling, and anything vLLM knows (tokens,
queue, KV cache). Those come from DCGM and from vLLM's `/metrics`, sampled here every 15 s in
`results/metrics.log`.

## Users ramp to saturation

`ramp.sh` adds one AIPerf cohort of 64 users every 240 s (14:48:30 to 15:17:08), and
`ramp_capture.sh` captures 10 s of trace 200 s into every step. Zero errors in every cohort.
Steps are 240 s: the numbers are not comparable with study 51's 960 s levels.

Client side, requests assigned to the step they started in (`analyze.py`):

| Users | Requests/s | TTFT p50 ms | TTFT p95 ms | Latency p50 ms | Latency p95 ms | Output tok/s |
|---|---|---|---|---|---|---|
| 64 | 0.98 | 91 | 97 | 1377 | 1408 | 187 |
| 128 | 1.98 | 107 | 115 | 2198 | 2250 | 378 |
| 192 | 2.98 | 281 | 581 | 5548 | 7316 | 569 |
| 256 | 3.98 | 464 | 827 | 11391 | 15848 | 760 |
| 320 | 4.98 | 918 | 7063 | 24766 | 31184 | 951 |
| 384 | 5.83 | 30433 | 61849 | 54482 | 85719 | 1114 |
| 448 | 3.97 | 67836 | 72808 | 91693 | 96647 | 758 |

The last cohort ended with 60 requests cancelled at the grace period.

Server side, mean per step (`server.py` on `results/metrics.log`):

| Users | Running | Waiting mean | Waiting max | KV cache % | GPU W | Prompt tok/s | Generated tok/s |
|---|---|---|---|---|---|---|---|
| 64 | 1.1 | 0.0 | 0 | 0.4 | 363 | 1546 | 200 |
| 128 | 4.6 | 0.0 | 0 | 2.7 | 400 | 4903 | 399 |
| 192 | 16.2 | 0.0 | 0 | 12.2 | 502 | 8999 | 595 |
| 256 | 45.8 | 0.0 | 0 | 36.3 | 548 | 13425 | 785 |
| 320 | 107.6 | 6.1 | 30 | 87.6 | 582 | 16995 | 951 |
| 384 | 119.5 | 156.8 | 277 | 98.6 | 580 | 17434 | 975 |
| 448 | 120.7 | 343.6 | 374 | 98.4 | 586 | 17383 | 979 |

**The server saturates at 320 users:** the queue forms, the KV cache fills, and generation tops
out at about 975 to 980 tok/s.

GPU profile from the 10 s trace of each step (`steps.py`, full in `results/steps-summary.txt`):

| Users | Kernel launches/s | CUDA calls/s | Tensor Active % | SM Active % | SM Issue % | DRAM Read % | Marlin MoE % | cutlass GEMM % | kernel_mha % |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 98144 | 6783 | 5.99 | 74.40 | 11.77 | 59.75 | 38.5 | 14.9 | 5.6 |
| 128 | 59653 | 4417 | 11.08 | 80.73 | 13.81 | 67.44 | 57.1 | 29.1 | 5.6 |
| 192 | 20355 | 3346 | 34.39 | 91.62 | 19.88 | 54.83 | 70.6 | 12.6 | 7.0 |
| 256 | 9426 | 2970 | 44.17 | 94.27 | 22.01 | 44.44 | 68.6 | 11.1 | 9.1 |
| 320 | 5424 | 2484 | 48.74 | 95.99 | 23.03 | 38.04 | 65.6 | 11.1 | 10.8 |
| 384 | 5211 | 2828 | 49.97 | 95.93 | 23.05 | 37.38 | 65.8 | 11.2 | 10.5 |
| 448 | 4594 | 2922 | 52.02 | 96.14 | 23.43 | 34.92 | 66.5 | 11.4 | 9.2 |

The kernel columns are shares of total kernel time.

- **GR Active is about 99.5 % at every step**, also at 64 users with one request running. The
  figure nvidia-smi calls GPU utilization says nothing about headroom here. Tensor Active (6 % to
  52 %) and kernel launches per second (98k to 5k) follow the load.
- **From 320 users the profile stops changing**, as the server-side numbers do.
- **At 64 users the dense layers run as cuBLAS batched GEMV** (`internal::gemvx::kernel`, 29.8 %
  of kernel time): the batch is often one request. From 128 users they run as cutlass GEMM.
- **Cross-check:** `kernel_mha` launches divided by the 36 layers give the decode steps per
  second: 155 at 64 users, 8 at 320 users with about 108 requests running, which matches the 950
  generated tok/s the server reports.

## Sizes and export cost

Load makes traces smaller: fewer, longer kernels.

| Capture | Zip | Export time | Export peak memory |
|---|---|---|---|
| 10 s at 64 users | 43 MB | 5.83 s | 1.37 GB |
| 10 s at 128 users | 28 MB | 3.73 s | 1.01 GB |
| 10 s at 192 users | 12 MB | 1.77 s | 0.61 GB |
| 10 s at 256 users | 7 MB | 1.26 s | 0.50 GB |
| 10 s at 320 to 448 users | 5 MB | 1.2 s | 0.45 GB |
| 30 s at 64 users | 126 MB | 16.33 s | 3.36 GB |

Unzipped, the 10 s trace is 206 MB at 64 users and 19 MB at 320 users. The 30 s trace at 64 users
is 607 MB.

## What tracing costs

**While capturing, at 64 users:** not measurable. TTFT mean 97.6 ms with the consumer attached
against 97.0 ms without, request latency 1430.3 ms against 1428.2 ms.

**Setup on without capturing, at saturation:** one run each of 448 fresh users for 300 s, same
seed (`ab.sh`). "Nsight" = service up and libcuda 620, no consumer attached (15:22). "Plain" =
service stopped, vLLM restarted on the host's libcuda 580 (15:33). Both completed 2069 requests
with no errors, at 7.00 requests/s and 1337 output tok/s in each of the first four minutes.

| Minute | TTFT p50 ms, Nsight | TTFT p50 ms, plain | Latency p50 ms, Nsight | Latency p50 ms, plain |
|---|---|---|---|---|
| 1 | 155 | 154 | 5588 | 5555 |
| 2 | 276 | 270 | 9380 | 9084 |
| 3 | 408 | 390 | 15600 | 14408 |
| 4 | 765 | 703 | 24824 | 24485 |
| 5 | 4860 | 3686 | 26499 | 25459 |

Throughput is the same. Latency is 1 % to 8 % higher with the Nsight setup. This is one run per
side, and the plain run had a freshly restarted server: it shows there is no large cost, it does
not size a small one.

## What it means for study 51

- Tracing the pair needs no driver update and no reboot: the service pod on each node, and the
  compatibility libcuda in the vLLM pods.
- Do not switch a running study to it. The experiments already done ran on libcuda 580; later ones
  would run on libcuda 620 with a service next to them.
- Capture each layout by hand after the last experiment and before the teardown, 10 s at a load
  near the layout's best level.

## Practical notes

- `kubectl cp` truncated files of a few tens of MB. `fetch.sh` downloads in 8 MiB chunks and
  checks the md5; `fetch_all.sh` brings everything.
- A long `kubectl exec` can drop with `i/o timeout` while the process in the pod finishes.
- AIPerf 0.11.0 takes about 1.16 GB per 64-user cohort. The `aiperf` pod was resized in place to
  14 GiB and 4 CPU before the seventh cohort.
- vLLM takes about 5.5 minutes to reload on this node (model on the local disk).

## Files

- `results/`: `steps-summary.txt`, `metrics.log`, `capture.log`, `post.log`, `ramp.log`, and
  AIPerf's summaries in `results/aiperf/` (ramp cohorts, the steady 64-user run, the two A/B
  runs). The per-request `profile_export.jsonl` files (55 MB) are kept locally, not in git.
- `traces/` (not in git, 348 MB): 18 Perfetto zips and `README.txt`, the index of which trace is
  which load step. `nsight-traces-2026-11-11.zip` bundles them.
- Scripts: `build.sh`, `cap.sh`, `ramp.sh`, `ramp_capture.sh`, `post.sh`, `ab.sh` run in the pods;
  `fetch.sh`, `fetch_all.sh`, `analyze.py`, `server.py`, `steps.py`, `q.py` read the results.

## Teardown

`kubectl delete namespace nsight-probe`, then `eksctl delete nodegroup -f
nodegroup-g7e-2xl-nsight.yaml --approve --wait`. Nothing shared on the cluster was changed.

Done on 2026-11-11: instance terminated from 15:48 UTC, CloudFormation stack and EKS node group
gone at 16:00 UTC, no node object left. `eksctl` itself exited with an error at 15:53 UTC, on a
DNS failure while it polled CloudFormation: the deletion went through anyway, so check the stack
by hand before running it again.
