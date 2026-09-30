"""Tune vLLM's Triton W8A8 block-FP8 GEMM for Qwen3-8B-FP8 at TP=1 (study 24).

Runs inside the vllm/vllm-openai:v0.29.0 image and reuses vLLM's own tuning script
(/vllm-workspace/benchmarks/kernels/benchmark_w8a8_block_fp8.py): its kernel copy,
search space, tune() and save_configs(). It does NOT use that script's main():
- main() tunes DeepSeek-V3 shapes (get_weight_shapes is hardcoded);
- main() splits the batch sizes across GPUs, but every GPU process writes the same
  file per shape with only its own batch sizes (save_configs opens it with "w"), so the
  last process to finish overwrites the others.
Here the (shape, M) jobs are spread over the GPUs, the results come back to the parent,
and the parent writes each file once, with every batch size.

Output: one JSON per shape, named as vLLM expects
(N=<N>,K=<K>,device_name=<GPU>,dtype=fp8_w8a8,block_shape=[128,128].json), mapping M to
{BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages}.
Copy the files to studies/24-l4-pd-kernels/k8s/tuned-configs/.
"""

import argparse
import multiprocessing as mp
import sys
import time

sys.path.insert(0, "/vllm-workspace/benchmarks/kernels")
import benchmark_w8a8_block_fp8 as bench  # noqa: E402

BLOCK = [128, 128]  # Qwen3-8B-FP8 weight_block_size

# Qwen3-8B config.json: hidden_size 4096, intermediate_size 12288,
# num_attention_heads 32, num_key_value_heads 8, head_dim 128. TP=1, so no split.
# (N, K): the kernel computes out[M, N] = in[M, K] @ W[K, N].
QWEN3_8B_TP1_SHAPES = [
    (6144, 4096),   # qkv_proj: (32 + 2 * 8) heads * 128
    (4096, 4096),   # o_proj: 32 * 128 -> hidden
    (24576, 4096),  # gate_up_proj: 2 * 12288
    (4096, 12288),  # down_proj: intermediate -> hidden
]

# vLLM's default grid (1..4096) plus prefill steps up to 16384 tokens: study 24 allows
# max_num_batched_tokens up to 16384, and vLLM picks the entry with the closest M.
DEFAULT_BATCH_SIZES = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
    6144, 8192, 12288, 16384,
]


def worker(task):
    gpu, jobs, out_dtype_name = task
    import torch

    torch.accelerator.set_device_index(gpu)
    out_dtype = bench.DTYPE_MAP[out_dtype_name]
    space = [c for c in bench.get_configs_compute_bound() if BLOCK[1] % c["BLOCK_SIZE_K"] == 0]
    results = []
    for n, k, m in jobs:
        start = time.time()
        cfg = bench.tune(m, n, k, BLOCK, out_dtype, space, "fp8")
        print(f"[GPU {gpu}] N={n} K={k} M={m}: {cfg} ({time.time() - start:.0f} s)", flush=True)
        results.append((n, k, m, cfg))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-path", default="/out")
    parser.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)),
                        help="comma-separated M values; use a short list for a smoke run")
    parser.add_argument("--out-dtype", default="bfloat16", choices=["bfloat16", "float16"],
                        help="vLLM calls the kernel with bfloat16 output for Qwen3-8B-FP8")
    parser.add_argument("--gpus", type=int, default=None, help="default: all visible GPUs")
    args = parser.parse_args()

    import torch

    n_gpus = args.gpus or torch.accelerator.device_count()
    if n_gpus < 1:
        raise RuntimeError("no GPU visible")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Largest jobs first, each to the GPU with the least work so far (cost ~ M * N * K),
    # so every GPU gets a similar amount of work. Round-robin gave the first GPU ~1.6x the
    # work of the last on the default grid.
    jobs = sorted(((n, k, m) for n, k in QWEN3_8B_TP1_SHAPES for m in batch_sizes),
                  key=lambda j: j[0] * j[1] * j[2], reverse=True)
    per_gpu = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for job in jobs:
        g = load.index(min(load))
        per_gpu[g].append(job)
        load[g] += job[0] * job[1] * job[2]
    print(f"{len(jobs)} jobs on {n_gpus} GPUs, device {bench.get_device_name_as_file_name()}, "
          f"out dtype {args.out_dtype}", flush=True)

    start = time.time()
    with mp.get_context("spawn").Pool(n_gpus) as pool:
        results = [r for part in pool.map(worker, [(g, per_gpu[g], args.out_dtype) for g in range(n_gpus)])
                   for r in part]

    for n, k in QWEN3_8B_TP1_SHAPES:
        configs = {m: cfg for rn, rk, m, cfg in sorted(results, key=lambda r: r[2]) if (rn, rk) == (n, k)}
        bench.save_configs(n, k, BLOCK[0], BLOCK[1], configs, args.save_path, "fp8")
    print(f"done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
