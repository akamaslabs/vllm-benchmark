"""GPU-to-GPU interconnect check for the g7e.12xlarge (2x RTX PRO 6000, PCIe, no NVLink).

Adapted from studies/18-l4-pd-disaggregation-tps-per-gpu/k8s/p2p_nccl_diag.py. Answers whether
TP2 traffic goes peer-to-peer over PCIe and what an all-reduce costs at the sizes vLLM uses:
in decode a TP2 all-reduce is hidden_size (4096) x batch x 2 bytes, i.e. 8 KiB to ~4 MiB.

  1. CUDA peer access matrix.
  2. Copy bandwidth 0->1 and 1->0: direct device-to-device vs staged through pinned host memory.
  3. NCCL all-reduce sweep, 8 KiB to 256 MiB: latency (us) and bus bandwidth (GB/s), with the
     default transport and with NCCL_P2P_DISABLE=1. NCCL_DEBUG=INFO prints the transport
     ("via P2P/..." or "via SHM/...").

Only torch + NCCL from the vLLM image. No CUDA samples to build.
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

KB = 1 << 10
MB = 1 << 20
SIZES = [8 * KB, 32 * KB, 128 * KB, 512 * KB, 2 * MB, 8 * MB, 64 * MB, 256 * MB]


def copy_bandwidth(size_mb=256, iters=20):
    n = torch.cuda.device_count()
    print(f"\n== 2. copy bandwidth, {size_mb} MiB x {iters} (GB/s)")
    print(f"{'pair':>8} {'direct':>8} {'via host':>9} {'D2H':>6} {'H2D':>6}")
    host = torch.empty(size_mb * MB, dtype=torch.uint8, pin_memory=True)
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            a = torch.empty(size_mb * MB, dtype=torch.uint8, device=f"cuda:{s}")
            b = torch.empty(size_mb * MB, dtype=torch.uint8, device=f"cuda:{d}")

            def timed(fn):
                fn()
                torch.cuda.synchronize(s)
                torch.cuda.synchronize(d)
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize(s)
                torch.cuda.synchronize(d)
                return size_mb * MB * iters / (time.perf_counter() - t0) / 1e9

            direct = timed(lambda: b.copy_(a, non_blocking=True))
            d2h = timed(lambda: host.copy_(a, non_blocking=True))
            h2d = timed(lambda: b.copy_(host, non_blocking=True))
            staged = 1 / (1 / d2h + 1 / h2d)
            print(f"{s}->{d:>5} {direct:8.1f} {staged:9.1f} {d2h:6.1f} {h2d:6.1f}")
            del a, b


def nccl_worker(rank, world, q):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device(f"cuda:{rank}"))
    out = []
    for size in SIZES:
        x = torch.ones(size // 2, dtype=torch.bfloat16, device=f"cuda:{rank}")
        iters = 200 if size <= 2 * MB else 20
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        out.append((size, dt * 1e6, size * 2 * (world - 1) / world / dt / 1e9))
    if rank == 0:
        q.put(out)
    dist.destroy_process_group()


def nccl_sweep(p2p_disable, port):
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "NCCL_DEBUG": "INFO"})
    if p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    else:
        os.environ.pop("NCCL_P2P_DISABLE", None)
    q = mp.get_context("spawn").Queue()
    mp.start_processes(nccl_worker, args=(2, q), nprocs=2, start_method="spawn", join=True)
    return q.get()


def main():
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, NCCL {torch.cuda.nccl.version()}, GPUs {n}")
    for i in range(n):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")
    print("\n== 1. CUDA peer access (can_device_access_peer)")
    for s in range(n):
        print(f"{s:>4} " + " ".join(
            f"{'-' if s == d else ('yes' if torch.cuda.can_device_access_peer(s, d) else 'NO'):>4}" for d in range(n)))
    copy_bandwidth()
    print("\n== 3. NCCL all-reduce, 2 GPUs, bf16. Transport: grep ' via ' in this log")
    results = {}
    for i, p2p_disable in enumerate((False, True)):
        results[p2p_disable] = nccl_sweep(p2p_disable, 29510 + i)
    print(f"\n{'size':>10} | {'P2P on: us':>11} {'GB/s':>7} | {'P2P off: us':>12} {'GB/s':>7}")
    for (size, us_on, bw_on), (_, us_off, bw_off) in zip(results[False], results[True]):
        label = f"{size // KB} KiB" if size < MB else f"{size // MB} MiB"
        print(f"{label:>10} | {us_on:11.1f} {bw_on:7.2f} | {us_off:12.1f} {bw_off:7.2f}")
    print(">>> done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
