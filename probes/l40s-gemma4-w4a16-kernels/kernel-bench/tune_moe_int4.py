"""Tune vLLM's Triton WNA16 fused-MoE kernel for Gemma 4 26B-A4B W4A16 on one L40S (W4A16 kernel probe).

Runs inside vllm/vllm-openai:v0.29.0 (tune-pod.yaml). Adapted from study 2's tune_moe.py
(akamas-gatling-llms-optimization, FP8 checkpoint): same two-stage search, same CUDA-graph
timing as vLLM's benchmarks/kernels/benchmark_moe.py, but the int4 weights of
cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (compressed-tensors, symmetric, group 32, no zero point)
and Gemma 4's gelu_tanh.

What the file changes in vLLM 0.29.0 (fused_moe.py):
- The file is E=128,N=704,device_name=NVIDIA_L40S,dtype=int4_w4a16.json (block_shape [0, 32]
  has a zero, so the name carries no block_shape part); vLLM ships none for this shape.
- With int4 weights and group 32, a step whose routed rows per expert are <= 6
  (M * top_k / E <= 6, i.e. M <= 96 tokens) runs a CUDA kernel (moe_wna16 CUDA), which takes
  only BLOCK_SIZE_M from the config and computes its own N/K tiles unless BLOCK_SIZE_N/K are in
  it. Above, the Triton GPTQ/AWQ kernel runs, with BLOCK_SIZE_N 64 / BLOCK_SIZE_K 32 /
  BLOCK_SIZE_M 64 by default (get_default_config + get_moe_wna16_block_config).
- So only M > 96 is tuned: the prefill-heavy steps. For M <= 96 the file carries vLLM's own
  default CUDA-path entry (BLOCK_SIZE_M only), because vLLM picks the entry of the closest M:
  without them a decode step would get a large-M Triton tile forced on the CUDA kernel.
Routing: random gating, as vLLM's benchmark (study 2 found tunings on random and on real
routing equivalent end to end, ~+6 % on ShareGPT for both).
"""

import argparse
import itertools
import json
import os
import time

import torch
from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, get_config_file_name
from vllm.triton_utils import triton

# Gemma 4 26B-A4B text_config: 128 experts, top-8, moe_intermediate_size 704, hidden 2816.
E, TOPK, N, K, GROUP = 128, 8, 704, 2816, 32
CUDA_PATH_MAX_M = 96          # M * TOPK / E <= 6
SMALL_M = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]
# 16384 left out (~1 h alone on this GPU): vLLM uses the closest entry, 8192.
TUNED_M = [128, 256, 512, 1024, 2048, 4096, 8192]
SPACE = [dict(zip(("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"), v))
         for v in itertools.product([16, 32, 64, 128, 256], [32, 64, 128, 256], [32, 64, 128, 256],
                                    [1, 16, 64], [4, 8], [2, 3, 4])]
SCREEN_ITERS, FINALISTS = 2, 15


def make_inputs(m):
    dtype = torch.bfloat16
    x = torch.randn(m, K, dtype=dtype)
    # int4 packed two per byte along K (layout of benchmark_moe.py's int4 path).
    w1 = torch.randint(0, 255, (E, 2 * N, K // 2), dtype=torch.uint8)
    w2 = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8)
    w1_scale = torch.rand((E, 2 * N, K // GROUP), dtype=dtype) * 1e-2
    w2_scale = torch.rand((E, K, N // GROUP), dtype=dtype) * 1e-2
    qc = int4_w4a16_moe_quant_config(w1_scale=w1_scale, w2_scale=w2_scale, block_shape=[0, GROUP])
    return x, w1, w2, qc


def bench_one(config, m, inputs, num_iters=20):
    """Mean time (us) of one fused-MoE call with config, CUDA graph of 10 calls."""
    x, w1, w2, qc = inputs
    gating = torch.randn(num_iters, m, E, dtype=torch.float32)
    input_gating = torch.empty(m, E, dtype=torch.float32)

    def run():
        with override_config(config):
            tw, ti, _ = fused_topk(x, input_gating, TOPK, renormalize=True)
            return fused_experts(x, w1, w2, tw, ti, activation=MoEActivation.GELU_TANH, quant_config=qc)

    input_gating.copy_(gating[0])
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lat = []
    for i in range(num_iters):
        input_gating.copy_(gating[i])
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        lat.append(start.elapsed_time(end))
    graph.reset()
    return sum(lat) / (num_iters * 10) * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-path", default="/out")
    p.add_argument("--batch-sizes", default=",".join(map(str, TUNED_M)))
    p.add_argument("--max-configs", type=int, default=0, help="first N configs only (smoke run)")
    args = p.parse_args()
    torch.set_default_device("cuda")
    from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config

    space = SPACE[: args.max_configs] if args.max_configs else SPACE
    name = get_config_file_name(E, N, "int4_w4a16", [0, GROUP])
    os.makedirs(args.save_path, exist_ok=True)
    out_file = os.path.join(args.save_path, name)
    print(f"device {torch.cuda.get_device_name()}, {len(space)} configs -> {name}", flush=True)

    # Small M: vLLM's own default, so the file never changes the CUDA-path steps.
    best = {m: get_default_config(m, E, N, K, TOPK, "int4_w4a16", [0, GROUP]) for m in SMALL_M}
    t_all = time.time()
    for m in [int(x) for x in args.batch_sizes.split(",")]:
        assert m > CUDA_PATH_MAX_M, f"M={m} runs the CUDA kernel, nothing to tune"
        t0 = time.time()
        inputs = make_inputs(m)
        dflt = get_default_config(m, E, N, K, TOPK, "int4_w4a16", [0, GROUP])
        d_us = bench_one(dflt, m, inputs)
        screened = []
        for c in space:
            # From M=1024 a 16/32-row tile cannot win (the default is 64) and costs the most time.
            if m >= 1024 and c["BLOCK_SIZE_M"] <= 32:
                continue
            c = dict(c, SPLIT_K=1)
            try:
                screened.append((bench_one(c, m, inputs, num_iters=SCREEN_ITERS), c))
            except Exception:  # noqa: BLE001  out of shared memory, an unsupported tile, ...
                torch.cuda.synchronize()
                continue
        screened.sort(key=lambda tc: tc[0])
        best_t, best_c = min(((bench_one(c, m, inputs), c) for _, c in screened[:FINALISTS]), key=lambda tc: tc[0])
        if best_t < d_us:
            best[m] = best_c
        print(f"M={m}: {best_c} {best_t:.1f} us, default {dflt} {d_us:.1f} us "
              f"({(best_t / d_us - 1) * 100:+.1f} %), {len(screened)} configs ran ({time.time() - t0:.0f} s)",
              flush=True)
        with open(out_file, "w") as f:
            json.dump({str(k): v for k, v in sorted(best.items())}, f, indent=4)
            f.write("\n")
    print(f"done in {time.time() - t_all:.0f} s: {out_file}", flush=True)


if __name__ == "__main__":
    main()
