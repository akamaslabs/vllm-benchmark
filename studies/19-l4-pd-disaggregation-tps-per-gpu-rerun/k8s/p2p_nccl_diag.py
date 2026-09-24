"""Interconnect diagnosis for the 4x L4 node (g6.12xlarge), run once per GPU-node window.

Answers, with numbers, the question both study 18 and the study-16 TP analysis depend on:
does GPU-to-GPU traffic on this node go peer-to-peer over PCIe, or through host memory?

  1. CUDA peer access matrix (torch.cuda.can_device_access_peer), i.e. what the hypervisor
     exposes. `nvidia-smi topo -m` / `-p2p r` are printed by diag-pod.yaml before this runs.
  2. Copy bandwidth for every GPU pair: direct device-to-device, and staged through pinned
     host memory. If "direct" is no faster than "via host", the driver is already staging.
  3. NCCL all-reduce on 2 and 4 GPUs, default settings vs NCCL_P2P_DISABLE=1. With
     NCCL_DEBUG=INFO, the "via P2P/..." / "via SHM/..." lines in the log name the
     transport NCCL picked. If the two runs give the same bus bandwidth, NCCL was not
     using P2P in the first place.

Only what vllm/vllm-openai:v0.29.0 already ships (torch + NCCL). No CUDA samples to build.
Study 18 itself does not use NCCL (every vLLM instance is TP1). Its KV transfer is
NIXL/UCX, whose cuda_ipc path needs the same peer access that section 1 reports.
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MB = 1 << 20


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


def nccl_worker(rank, world, size_mb, iters, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device(f"cuda:{rank}"))
    x = torch.ones(size_mb * MB // 4, dtype=torch.float32, device=f"cuda:{rank}")
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    if rank == 0:
        # nccl-tests "bus bandwidth" for all-reduce: bytes * 2*(n-1)/n / time
        q.put(size_mb * MB * 2 * (world - 1) / world / dt / 1e9)
    dist.destroy_process_group()


def nccl_allreduce(world, p2p_disable, size_mb=64, iters=20, port=29500):
    env = {"NCCL_DEBUG": "INFO", "MASTER_PORT": str(port)}
    if p2p_disable:
        env["NCCL_P2P_DISABLE"] = "1"
    old = {k: os.environ.get(k) for k in list(env) + ["NCCL_P2P_DISABLE"]}
    os.environ.update(env)
    if not p2p_disable:
        os.environ.pop("NCCL_P2P_DISABLE", None)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.start_processes(nccl_worker, args=(world, size_mb, iters, q), nprocs=world, start_method="spawn", join=True)
    bw = q.get()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return bw


def main():
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, NCCL {torch.cuda.nccl.version()}, GPUs {n}")
    for i in range(n):
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}")
    print("\n== 1. CUDA peer access (can_device_access_peer)")
    print("     " + " ".join(f"{d:>4}" for d in range(n)))
    for s in range(n):
        print(f"{s:>4} " + " ".join(f"{'-' if s == d else ('yes' if torch.cuda.can_device_access_peer(s, d) else 'NO'):>4}" for d in range(n)))
    copy_bandwidth()
    print("\n== 3. NCCL all-reduce, 64 MiB x 20 (bus bandwidth GB/s). Transport lines: grep ' via ' in this log")
    port = 29500
    for world in [w for w in (2, 4) if w <= n]:
        for p2p_disable in (False, True):
            port += 1
            bw = nccl_allreduce(world, p2p_disable, port=port)
            print(f">>> NCCL all-reduce world={world} NCCL_P2P_DISABLE={'1' if p2p_disable else 'unset'}: {bw:.2f} GB/s", flush=True)
    print("\nReading it: if 'direct' ~ 'via host' in section 2, and world=2 bandwidth is the same with and "
          "without NCCL_P2P_DISABLE, then GPU-to-GPU traffic already goes through host memory on this node.")


if __name__ == "__main__":
    sys.exit(main())
