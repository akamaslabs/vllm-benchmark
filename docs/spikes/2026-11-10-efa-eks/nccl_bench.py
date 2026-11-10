# Two-rank NCCL micro-benchmark over torch.distributed (the stack vLLM uses), one rank per node.
# Env: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT. Run with NCCL_DEBUG=INFO to see the transport
# (NET/OFI = aws-ofi-nccl over EFA, NET/Socket = TCP). Rank 0 prints the table.
# Small messages give the per-call latency (what a per-layer all-reduce pays), large ones the
# bandwidth. With 2 ranks an all_reduce moves the buffer once in each direction.
import os, time, torch, torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", init_method="tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ.get("MASTER_PORT", "29400")),
                        rank=rank, world_size=world)
dev = torch.device("cuda:0")

def bench(name, fn, nbytes):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    iters = max(5, min(200, int(2e9 // max(nbytes, 1e7))))
    t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    if rank == 0:
        print("BENCH %-14s %10d bytes  %9.3f ms/call  %7.2f Gbit/s per direction" % (name, nbytes, dt * 1e3, nbytes * 8 / dt / 1e9), flush=True)

for nbytes in (4 << 10, 64 << 10, 1 << 20, 16 << 20, 128 << 20, 1 << 30):
    x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
    bench("all_reduce", lambda: dist.all_reduce(x), nbytes)
    del x
for nbytes in (64 << 10, 16 << 20, 256 << 20):
    x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
    outs = [torch.empty_like(x) for _ in range(world)]
    bench("all_gather", lambda: dist.all_gather(outs, x), nbytes)
    del x, outs
if rank == 0: print("BENCH done", flush=True)
dist.destroy_process_group()
