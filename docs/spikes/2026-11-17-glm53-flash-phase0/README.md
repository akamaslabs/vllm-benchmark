# Spike: GLM-5.3-Flash phase 0 on 2x RTX PRO 6000 (2026-11-17)

Throwaway spike before the GLM-5.3-Flash study: which kernels vLLM picks for the 4-bit checkpoints
on SM120 with TP2, whether the two GPUs talk peer-to-peer, and what a start needs. Manifests in
`k8s/`. Not an Akamas study.

Target workload: **coding agents with long contexts** (multi-turn sessions whose history is
re-sent every turn). So prefix caching on this hybrid model (KDA recurrent state, not only KV
blocks) and CPU KV offload matter more than peak batch size.

**Status:** paused on 2026-11-17 ~09:40 UTC. Node group scaled to 0 (still defined), capacity
reservation cancelled. Next inputs: the target deployment's start-up log and `/metrics`, and the
H200 decision. Checkpoints and compile caches lived on the node's hostPath and are gone with it.

## Node

- Node group `llm-serving-g7e-12xl` (eksctl, one g7e.12xlarge, us-east-2a only, gp3 800 GB, tag
  AlwaysOn=true), created 2026-11-17 07:57 UTC on an open capacity reservation in 2a. 8.29 USD/h.
  eksctl config in `infra/nodegroup-g7e-12xl.yaml` (back up with `aws eks update-nodegroup-config
  ... desiredSize=1` after a capacity probe; it only spans us-east-2a).
- 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W, driver 580.178.04,
  CUDA 13.0. 48 vCPU (1 socket, Xeon Platinum 8559C, 1 NUMA node), ~492 GiB allocatable.
- dcgm-exporter covers the node role (Helm revision 29, values in `infra/dcgm-exporter-values.yaml`).

## GPU-to-GPU interconnect (`k8s/nccl_diag.py`, job `nccl-diag`, vLLM v0.34.0 image)

- `nvidia-smi topo -m`: **PIX** (one PCIe bridge between the GPUs), CPU and NUMA affinity 0-47 / 0.
  `topo -p2p r` and `w`: OK. CUDA peer access: yes both ways.
- Copy, 256 MiB: **54.5 GB/s direct**, 28.0 GB/s staged through host (D2H 56.5, H2D 55.5). Real
  peer-to-peer over PCIe Gen5 x16.
- NCCL (2.31.2 at runtime, torch built with 2.30.7) picks **P2P/CUMEM**. All-reduce, 2 GPUs, bf16:

  | Size | P2P: us | GB/s | P2P disabled (SHM): us | GB/s |
  |---|---|---|---|---|
  | 8 KiB | 21.6 | 0.38 | 26.8 | 0.31 |
  | 32 KiB | 21.2 | 1.55 | 31.8 | 1.03 |
  | 128 KiB | 21.3 | 6.14 | 34.1 | 3.84 |
  | 512 KiB | 37.7 | 13.92 | 59.7 | 8.78 |
  | 2 MiB | 82.4 | 25.45 | 161.1 | 13.02 |
  | 8 MiB | 298.7 | 28.08 | 562.4 | 14.92 |
  | 64 MiB | 2282.9 | 29.40 | 4504.3 | 14.90 |
  | 256 MiB | 8633.2 | 31.09 | 17887.4 | 15.01 |

- Reading: a decode all-reduce (4096 x batch x 2 bytes = 8 KiB to 1 MiB) costs ~21-40 us through
  NCCL. TP2 does tens of them per step, so whether vLLM uses its own custom all-reduce (CUDA IPC,
  usually faster at these sizes) shows up directly in ITL at low batch. Check it in the start log.

## Checkpoint and engine

- `nvidia/GLM-5.3-Flash-FP4-Local` (public, 177.5 GB, NVFP4 routed experts via ModelOpt QAD),
  downloaded to the node's `/var/lib/hf-cache-glm53` (job `model-download-fp4`).
- The model card's recipe for 2x RTX PRO 6000 uses a **community container**,
  `ghcr.io/local-inference-lab/vllm:release-10152026`, "until all features are available in
  upstream vLLM": ~40 `B12X_*` / `VLLM_GLM53_*` env vars, `--attention-backend B12X`,
  `--moe-backend b12x`, `--linear-backend b12x`, `--quantization modelopt_mixed`,
  `--load-format instanttensor`, `--decode-context-parallel-size 2`, `--block-size 256`,
  `--max-num-seqs 8`, `--max-model-len 1048576`, MTP 3 tokens, a b12x PCIe all-reduce
  (`VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_ALLREDUCE_BACKEND=b12x`).
- Upstream **vLLM v0.34.0** (released 2026-11-16, contains the NVFP4 loader #64312) has
  `modelopt_mixed`, the `B12X` attention backend (a paged, non-MLA backend), `instanttensor`, the
  `b12x` MoE and linear backends, DCP, `mamba_cache_mode`, `kv_cache_memory_bytes`,
  `kda_prefill_backend`. It lacks `--prefill-compute-share`, `--recurrent-checkpoint-policy`,
  `--max-parallel-prefills` and `VLLM_ENABLE_PCIE_ALLREDUCE`. SM12x MLA priority:
  `TRITON_MLA`, `FLASHINFER_MLA_SPARSE_SM120`.
- Pipeline parallelism: not tried. Upstream PRs #66828 (applies on nightly 8ca7663, not on
  v0.34.0) and #57639 (applies on v0.34.0, not on nightly) are both open; dropped on 2026-11-17.

## Runs

1. `k8s/01-vllm-upstream.yaml`: v0.34.0, FP4-Local, TP2, `--max-model-len 262144`, prefix
   caching, everything else auto. **Failed at weight load after 50 s** (08:16 UTC):
   `FileNotFoundError: Weight files referenced in index but missing: {.../tensors/native-QB.safetensors}`.
   The file is there (a 754 MB symlink into the xet blob store). The default loader globs
   `*.safetensors` in the model folder only (`default_loader.py` `_prepare_weights`, then
   `weight_utils.filter_duplicate_safetensors_files`), and `instanttensor` uses the same pattern;
   the GLM-5.3-Flash model class sets no `allow_patterns_overrides`. The file holds 214 plain BF16
   KDA projections (`self_attn.{b,f_a,f_b,g_a,g_b}_proj.weight`). Before failing, the log showed
   what upstream picks on SM120:
   - all-reduce for the TP group: `['CUSTOM', 'PYNCCL']` (vLLM's custom all-reduce first);
     `SymmMemCommunicator` not available on capability 12.0; FlashInfer all-reduce disabled for
     world_size 2; `FLASHINFER_PCIE_IPC` listed as possible but not picked;
   - attention: `FLASHINFER_MLA_SPARSE_SM120`, the only candidate, KV cache `fp8_ds_mla`
     (`kv_cache_dtype=auto`); "Sparse MLA layer has no dense-MHA prefill path; using the top-k MQA
     path only";
   - MoE NVFP4: `FLASHINFER_CUTLASS` out of FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL,
     FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN, HUMMING, EMULATION;
   - vision tower: `FLASH_ATTN`; quantization detected as `modelopt_mixed`; CUDA graphs
     `FULL_AND_PIECEWISE` up to 512.
2. Same, on `/hf-cache/glm53-fp4-local-flat`: symlinks to every file, the `tensors/` file moved to
   the top level, index rewritten (214 entries). No vLLM change. **Failed at weight load after
   50 s** (08:19 UTC), in the vision tower: the checkpoint's `blocks.*.attn.proj` are FP8 with a
   `weight_scale`, upstream's `Glm5NextVisionTransformer` expects BF16 ("There is no module or
   parameter named 'blocks.0.attn.proj.weight_scale'"). Loading itself reports checkpoint 165.28
   GiB on XFS.
3. Same, plus `--language-model-only` (text only, as a coding workload needs): the vision tower
   becomes a `StageMissingLayer` and its weights are skipped. "Mamba cache mode is set to 'align'
   ... by default when prefix caching is enabled". **Failed at weight load** (08:21 UTC):
   `KeyError: 'layers.0.self_attn.in_proj_qkvbfg_a.weight_scale'` (`glm5next/common/model.py`
   line 1103). In the KDA layers the checkpoint has `q/k/v_proj` in FP8 with `weight_scale` and
   `b/f_a/g_a_proj` in BF16 (the `tensors/` file); upstream fuses all six into one
   `in_proj_qkvbfg_a` with a single precision. Fixing it means changing the loader or rewriting
   the checkpoint, so measuring something the customer does not run: **upstream v0.34.0 stops
   here for FP4-Local.**
4. `k8s/02-vllm-community.yaml`: the community container with the card's preset
   (`PRESET=glm53-tp2`). Not started: running a third-party image on the cluster needs an
   explicit go, and the target deployment turned out to run upstream vLLM (below).

**Target deployment (2026-11-17):** the official FP8 checkpoint with TP4 on H200, upstream vLLM,
MTP 5 tokens, `--kv-cache-dtype fp8`, `--max-model-len 262144`, `--gpu-memory-utilization 0.95`,
glm47 parsers. The FP8 checkpoint (328 GB) does not fit in 2x 96 GB, so the proxy uses a 4-bit
checkpoint and `k8s/04-vllm-customer-like.yaml` keeps the target's container shape and arguments
(differences listed in its header).

5. `k8s/04-vllm-customer-like.yaml` with `canada-quant/GLM-5.3-Flash-W4A16-MTP` (190.9 GB,
   compressed-tensors; only routed experts 4-bit, every self_attn projection BF16, MTP head BF16),
   v0.34.0, TP2, target arguments incl. MTP 5. Weights loaded in 26.7 s (177.72 GiB); routed MoE
   on **`MARLIN` WNA16**, the MTP head's BF16 MoE on FlashInfer CUTLASS unquantized. **OOM** (08:59
   UTC) converting the draft MoE (`swap_w13_to_w31`, 4.50 GiB more with 3.41 GiB free, 91.55 GiB
   in use per GPU). The MTP head's BF16 experts take ~6.75 GiB per GPU at TP2.
6. Same without MTP (also: vllm#59724, MTP 0% acceptance on SM120's sparse MLA backend, open).
   Model loading **82.64 GiB per GPU** in 30-32 s. KV block size 256 for the indexer / KPOOL_TAIL /
   sparse MLA backends; "Setting attention block size to **4352 tokens** to ensure that attention
   page size is >= mamba page size" and mamba page padded by 2.64%: one KDA state block equals 4352
   tokens of KV, so prefix-cache hits (mamba cache mode `align`) land on 4352-token boundaries.
   **Available KV cache memory 2.85 GiB** at gmu 0.95, i.e. **114 Mamba blocks**. Failed
   (09:02 UTC): "max_num_seqs (1024) exceeds available Mamba cache blocks (114)"; 1024 is
   v0.34.0's default here.
7. Same plus `--max-num-seqs 64`. Got through FlashInfer autotune (~5 min, buckets 1-8192 tokens)
   and CUDA graph capture (PIECEWISE 19 sizes + FULL 11, 8 s, 0.46 GiB). **Failed** (09:10 UTC):
   "No available memory for the cache blocks". v0.34.0's extensible KV cache sizes the cache after
   warmup: within gmu 0.95 (90.22 GiB), 83.69 GiB weights + non-torch, 3.08 GiB transient
   activations, 0.46 GiB graphs, 0.05 GiB committed to KV; ~5 GiB still free on the device.
   Start-up to that point: ~7 min.
8. Same with `--gpu-memory-utilization 0.98`. Pre-warmup estimate "Available KV cache memory:
   6.3 GiB"; after warmup "5.04 GiB free, 0.05 GiB committed to the KV cache, 3.08 GiB kept for
   transient activations (... 2.42 GiB of torch reservation growth); 0 KV cache blocks fit within
   the requested 93.07 GiB". **Failed** (09:18 UTC), same error.
9. `--gpu-memory-utilization 1.0` (v0.34.0's default with the extensible KV cache) and
   `--max-num-batched-tokens 4096` (smaller activation peak). Pre-warmup estimate 9.58 GiB; after
   warmup 82.79 GiB weights + non-torch, 2.59 GiB activations, 0.46 GiB graphs, "**60-61 KV cache
   blocks (1.5 GiB)**". **Failed** (09:26 UTC): one 262144-token request needs 1.78 GiB, "the
   estimated maximum model length is 213248". So ~7.1 KB per token per GPU (KV + KDA state), and
   **the whole cache holds less than one full-length context**.
   Start-up time, read from run 7's timestamps: weights ~30 s, **TileLang JIT warmup 213 s**
   (mHC kernels, 210 compile keys), FlashInfer sparse-MLA and MoE autotune ~70 s, graphs 8 s.
10. `--max-model-len 196608`, compile caches (`/root/.cache`, `/root/.tilelang`, `/root/.triton`)
    on a node hostPath. **Up** at 09:33:54 UTC after 6 min 47 s (weights 26 s, TileLang JIT 212 s,
    first start with the cache directories). "**GPU KV cache size: 210,651 tokens**, Maximum
    concurrency for 196,608 tokens per request: 1.07x". Smoke, one request at a time, temperature 0:
    - coding prompt, 3000 max tokens: 618 completion tokens (581 reasoning, glm47 reasoning parser
      separates them), correct code, 7.2 s, **~85 tokens/s single stream** (no MTP);
    - one tool (`read_file`): `finish_reason: tool_calls`, well-formed arguments, 2.8 s (glm47 tool
      parser works).

**What is left of the target's arguments on this rig:** `--kv-cache-dtype fp8` and the glm47
parsers. MTP is off; gmu (1.0), max-num-seqs (64), max-num-batched-tokens (4096) and
max-model-len (196608) differ; checkpoint, TP, GPU architecture and vLLM version differ. The rig is
cache-starved (~1.5 GiB of KV + KDA state per GPU); a TP4 H200 pod with the FP8 checkpoint very
likely is not.
