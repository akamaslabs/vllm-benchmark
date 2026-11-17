# 52-g7e-glm53-coding-sessions-within-slo

**Status:** PREPARED (2026-11-17). Waiting for the us-east-2 spot G vCPU quota (64 -> 192
requested, AWS case 179491438900392) and the smoke on the node.

- **Design:** `docs/superpowers/specs/2026-11-17-study-52-glm53-coding-sessions-design.md`.
- **Plan:** `docs/superpowers/plans/2026-11-17-study-52-glm53-coding-sessions.md`.
- **Follows the spike:** `docs/spikes/2026-11-17-glm53-flash-phase0/README.md`.

## Objective

A target deployment serves `zai-org/GLM-5.3-Flash` to coding agents with long contexts: one vLLM
pod per replica, TP4 on H200, the official FP8 checkpoint, and

```
--kv-cache-dtype fp8 --max-model-len 262144 --gpu-memory-utilization 0.95
--speculative-config.method mtp --speculative-config.num_speculative_tokens 5
--tool-call-parser glm47 --reasoning-parser glm47 --enable-auto-tool-choice --tensor-parallel-size 4
```

Study 52 asks **how many concurrent coding sessions one replica serves within client TTFT p95
<= 5 s and client ITL p95 <= 50 ms**, with those arguments and with the levers they leave unused
(CPU KV offload, decode context parallelism, MTP depth, scheduler sizing).

It runs on **4x RTX PRO 6000 Blackwell (one g7e.24xlarge spot)**: the target's FP8 checkpoint fits
with the target's TP4. The GPU architecture differs (SM120 here, SM90 on H200), so kernels and
absolute numbers do not transfer; which levers matter, and roughly how much, does. Results are
sessions on this rig, never a sessions number for the target.

## Method

- **Sessions (AIPerf 0.11.0, user-centric multi-turn, history re-sent):** a 20k-token system
  prompt shared by every session, 10k tokens of per-session context, 16 turns of 4k input and
  400 output tokens (`ignore_eos`), one turn per session every 30 s. A session grows from ~34k to
  ~100k tokens. Two AIPerf clients on a CPU node, half the sessions each.
- **Ramp task:** levels of 4, 8, 12, 16, 24, 32, 48 and 64 sessions, each a fresh 960 s run + 60 s
  grace, judged on requests started >= 500 s after the first one. A level passes when client
  TTFT p95 <= 5000 ms and client ITL p95 <= 50 ms (each the worse of the two clients), errors
  < 1 %, >= 95 % of sent requests completed and >= 95 % of expected requests sent. The ramp stops
  at the first failing level. Each level also prints the worst session's TTFT p95 and the number
  of sessions above 5 s.
- **Hold task:** the last passing level again, fresh, 1080 s; Akamas trim `[520s, 90s]`.
- **Score:** `30 x vllm.request_success_rate` = sessions held. No TTFT tie-break (the user's
  call): within a level, compare TTFT and ITL directly.
- **Constraints (Hold, safety net):** server TTFT p95 <= 6 s, server ITL p95 <= 60 ms.
- **Steps:** `baseline` (the target's arguments), presets `Offload 256`, `DCP four`, `MTP three`,
  `MTP one` (only those the smoke confirms), `Baseline repeat`, then `optimize` (10 experiments).

## Stack & versions

- **Hardware:** one g7e.24xlarge **spot** in us-east-2c (score 8, ~8.83 USD/h on 2026-11-17;
  on-demand 16.57), node group `llm-serving-g7e-24xl-spot` (`infra/eks/`), cluster `vllm-bench`
  (EKS, us-east-2).
- **Server:** `vllm/vllm-openai:v0.34.0`, `zai-org/GLM-5.3-Flash` (FP8, 328 GB) served as
  `zai-org/GLM-5.3-Flash`, TP4, `--language-model-only`; the target's container shape
  (`vllm serve /mnt/models ... ${VLLM_ADDITIONAL_ARGS}`); namespace `glm53-coding`.
- **Load:** AIPerf 0.11.0 (`k8s/workload.env`).
- **Akamas:** 4.1 (akamas-41); packs: TODO, checked with `akamas describe optimization-pack` before
  the YAML is written.
- **Telemetry:** Prometheus; the full catalog (CLAUDE.md rule 9): vLLM, one GPU component per
  GPU, Kubernetes Node, Container (vLLM and AIPerf), Workload, Cluster.

## Parameters tuned

| Parameter | Domain | Baseline |
|---|---|---|
| `vllm.kv_offloading_size` | 0–512 GiB | 0 |
| `vllm.gpu_memory_utilization` | 0.85–0.97 | 0.95 |
| `vllm.max_num_seqs` | 8–128 | TODO (smoke: under the Mamba block count) |
| `vllm.max_num_batched_tokens` | 2048–16384 | default |
| `vllm.spec_tokens` (MTP) | 0–5 | 5 (0 if the smoke shows no acceptance) |
| `vllm.decode_context_parallel_size` | 1, 2, 4 | 1 |
| `vllm.moe_backend` | auto + the smoke's FP8 candidates | auto |
| `vllm.max_cudagraph_capture_size` | 16–512 | default |
| `vllm.async_scheduling` | true, false | default |
| `vllm.performance_mode` | as in study 38 | default |
| `vllm.optimization_level` | 1–3 | default |

Not levers here: `kv_cache_dtype` and `block_size` (forced by the SM120 sparse MLA backend and
the KDA state), `tensor_parallel_size` (the target's TP4).

## TODO

- Pack gaps (raise with whoever manages packs): `mamba_ssm_cache_dtype` (KDA state fp32/bf16),
  `kv_cache_dtype` value `fp8_ds_mla`, `language_model_only`, and no per-container or per-pod
  network metrics in the Kubernetes pack.
- Smoke on the node (`k8s/smoke.sh`) with the go / no-go rules of the design.
- Spot quota: raised to 192 vCPU on 2026-11-17 (case 179491438900392 closed); spot node up in us-east-2c.
- **After the smoke, before the study (agreed with the user 2026-11-17):** move the checkpoint and
  the compile caches from the node's hostPath to an EBS volume of their own (PVC, gp3 ~600 GB in
  us-east-2c, 1000 MB/s), copied from what the smoke downloaded, so a reclaimed spot node is
  replaced without re-downloading 328 GB or recompiling. Update 01-vllm_template.yaml and its tests;
  delete the volume at the end of the study.

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
