#!/usr/bin/env python3
"""Writes the main and Smoke study manifests of study 41 from one definition, so the two
stay identical except for name, description, workflow and steps. Run: python3 gen_study.py"""
import copy, yaml

SYSTEM = "vLLM_Benchmark_41_L4_DSV2_Lite_CP"
MAIN = "41-L4-DSV2-Lite-Context-Parallel"

# Baseline: TP4 (the usual way to serve a model that does not fit one 24 GB L4 on this
# node), everything else at vLLM 0.32.0's defaults on an L4 (OpenAI
# server, device < 70 GiB: max_num_seqs 256, max_num_batched_tokens 2048, arg_utils.py
# get_batch_defaults; EP off, KV dtype auto = bf16), except gpu_memory_utilization 0.90
# (repo convention; default 0.92). Prefix caching is not a parameter: fixed at vLLM's
# default (on) in k8s/params.env.template.
BASE = {
    "vllm.tensor_parallel_size": 4,
    "vllm.data_parallel_size": 1,
    "vllm.decode_context_parallel_size": 1,
    "vllm.prefill_context_parallel_size": 1,
    "vllm.enable_expert_parallel": "false",
    "vllm.max_num_seqs": 256,
    "vllm.max_num_batched_tokens": 2048,
    "vllm.gpu_memory_utilization": 0.90,
}

def preset(name, **over):
    v = dict(BASE)
    v.update({f"vllm.{k}": val for k, val in over.items()})
    return {"name": name, "type": "preset", "values": v}

COMMON = {
    "system": SYSTEM,
    "goal": {
        "objective": "maximize",
        # Generation tokens/s, not total: every turn resends its whole history (~14k tokens,
        # mostly prefix-cache hits) for 256 generated, and vllm:prompt_tokens counts cached
        # tokens too, so total tokens/s would be ~98 % resent history. With ignore_eos and a
        # fixed 256-token reply, generation tokens/s = completed turns/s x 256: the closest
        # measure of users served within the SLA without the AIPerf pack (decided 2026-10-22).
        "function": {"formula": "vllm.decode_token_throughput", "unit": "tokens per second"},
        # ms (the telemetry multiplies vLLM's seconds by 1000). :max over the scored window
        # avoids scoring a window that ends just after the breach.
        "constraints": {"absolute": [
            {"name": "TTFT P95 over 150 s", "formula": "vllm.time_to_first_token_p95:max <= 10000"},
            {"name": "TPOT P95 over 150 s", "formula": "vllm.time_per_output_token_p95:max <= 100"},
        ]},
    },
    "windowing": {"type": "stability", "stability": {
        "metric": "vllm.decode_token_throughput", "width": 6, "maxStdDev": 300000000,
        "when": {"metric": "vllm.decode_token_throughput", "is": "max"}}},
    "numberOfTrials": 1,
    # Domains inside the vLLM pack 1.14.0's (installed on Akamas 4.1, `akamas describe
    # optimization-pack vLLM`, 2026-10-22): tensor_parallel_size [1,16],
    # data_parallel_size [1,8], decode_context_parallel_size [1,16],
    # prefill_context_parallel_size [1,8], max_num_seqs [1,4096],
    # max_num_batched_tokens [1,65536], gpu_memory_utilization [0,1], kv_cache_dtype
    # {auto, fp8, fp8_e4m3, fp8_e5m2}, enable_expert_parallel {true, false}. The integer
    # sizes are [1, 4] with 3 excluded by parameterConstraints (study 16's precedent).
    "parametersSelection": [
        {"name": "vllm.tensor_parallel_size", "domain": [1, 4]},
        {"name": "vllm.data_parallel_size", "domain": [1, 4]},
        {"name": "vllm.decode_context_parallel_size", "domain": [1, 4]},
        {"name": "vllm.prefill_context_parallel_size", "domain": [1, 4]},
        {"name": "vllm.enable_expert_parallel", "categories": ["true", "false"]},
        # kv_cache_dtype is NOT tuned: fp8 crashes TRITON_MLA on the L4 (Triton OutOfResources,
        # shared memory; Smoke experiment 3, 2026-10-22). Pinned to vLLM's default (auto).
        {"name": "vllm.max_num_seqs", "domain": [16, 512]},
        {"name": "vllm.max_num_batched_tokens", "domain": [2048, 32768]},
        {"name": "vllm.gpu_memory_utilization", "domain": [0.80, 0.94]},
    ],
    # Every rule is also enforced by k8s/render_statefulset.sh (exit 2 before any cluster
    # call); check_offline.py verifies that the two agree on every combination of the sizes.
    # No modulo operator: with sizes in {1, 2, 4}, tp % dcp == 0 is exactly dcp <= tp.
    "parameterConstraints": [
        {"name": "tensor_parallel_size is 1, 2 or 4", "formula": "vllm.tensor_parallel_size != 3"},
        {"name": "data_parallel_size is 1, 2 or 4", "formula": "vllm.data_parallel_size != 3"},
        {"name": "decode_context_parallel_size is 1, 2 or 4", "formula": "vllm.decode_context_parallel_size != 3"},
        {"name": "prefill_context_parallel_size is 1, 2 or 4", "formula": "vllm.prefill_context_parallel_size != 3"},
        # Parity: the pod holds all four L4s and vLLM must use all of them (DCP reuses TP ranks).
        {"name": "TP x PCP x DP uses exactly the 4 GPUs",
         "formula": "vllm.tensor_parallel_size * vllm.prefill_context_parallel_size * vllm.data_parallel_size == 4"},
        # vllm/config/parallel.py: tp % dcp == 0 (pcp == 1); MLA has one KV head, so dcp <= tp.
        {"name": "DCP divides TP", "formula": "vllm.decode_context_parallel_size <= vllm.tensor_parallel_size"},
        # v1/worker/gpu/pcp_manager.py: "MRV2 PCP + DCP supports sparse MLA models only".
        {"name": "PCP needs DCP 1 on dense MLA",
         "formula": "vllm.prefill_context_parallel_size == 1 || vllm.decode_context_parallel_size == 1"},
        # Smoke run 1: TP2xPCP2 without EP crashed (topk_softmax is_padding size mismatch).
        {"name": "PCP needs expert parallelism",
         "formula": "vllm.prefill_context_parallel_size == 1 || vllm.enable_expert_parallel == \"true\""},
        # PCP + DP: unverified upstream (PR #49246 open).
        {"name": "PCP without DP",
         "formula": "vllm.prefill_context_parallel_size == 1 || vllm.data_parallel_size == 1"},
        # No "TP 1 needs EP" rule: without EP vLLM 0.32.0 shards the experts TP-style across all
        # dp x pcp x tp ranks (fused_moe/config.py), so DP4 without EP fits (dropped 2026-10-22).
        # vllm/config/scheduler.py raises otherwise.
        {"name": "max_num_batched_tokens must be at least max_num_seqs",
         "formula": "vllm.max_num_batched_tokens >= vllm.max_num_seqs"},
    ],
    "kpis": [
        {"name": "Generation throughput", "formula": "vllm.decode_token_throughput", "direction": "maximize"},
        {"name": "TTFT P95 150s", "formula": "vllm.time_to_first_token_p95", "direction": "minimize", "aggregation": "max"},
        {"name": "TPOT P95 150s", "formula": "vllm.time_per_output_token_p95", "direction": "minimize", "aggregation": "max"},
        {"name": "Completed requests", "formula": "vllm.request_success_rate", "direction": "maximize"},
        {"name": "KV cache usage", "formula": "vllm.kv_cache_usage_max", "direction": "minimize", "aggregation": "max"},
        {"name": "Preemptions", "formula": "vllm.preemption_rate", "direction": "minimize"},
        {"name": "Waiting requests", "formula": "vllm.num_requests_waiting", "direction": "minimize", "aggregation": "max"},
        {"name": "Prefix cache hit rate", "formula": "vllm.prefix_cache_hit_rate", "direction": "maximize"},
    ],
}

BASELINE = {"name": "baseline", "type": "baseline", "numberOfTrials": 2, "values": dict(BASE)}
MAIN_STEPS = [
    BASELINE,
    preset("tp4 dcp2", decode_context_parallel_size=2),
    preset("tp4 dcp4", decode_context_parallel_size=4),
    preset("tp1 pcp4 ep", tensor_parallel_size=1, prefill_context_parallel_size=4, enable_expert_parallel="true"),
    preset("tp2 pcp2 ep", tensor_parallel_size=2, prefill_context_parallel_size=2, enable_expert_parallel="true"),
    preset("dp4 ep", tensor_parallel_size=1, data_parallel_size=4, enable_expert_parallel="true"),
    {"name": "optimize", "type": "optimize", "optimizer": "AKAMAS",
     "numberOfInitExperiments": 0, "numberOfExperiments": 60, "maxFailedExperiments": 20},
]
# Calibration on the Smoke workflow's steeper ramp: the baseline, TP4+DCP4 (the two knees that
# size the main ramp), plus the two combinations never run under load: DCP with fp8 KV and PCP
# with prefix caching (decided 2026-10-22).
SMOKE_STEPS = [dict(BASELINE, numberOfTrials=1),
               preset("tp4 dcp4", decode_context_parallel_size=4),
               preset("tp4 dcp4 kv fp8", decode_context_parallel_size=4, kv_cache_dtype="fp8"),
               preset("tp1 pcp4 ep", tensor_parallel_size=1, prefill_context_parallel_size=4,
                      enable_expert_parallel="true")]

def doc(name, description, workflow, steps):
    d = {"kind": "study", "name": name, "description": description, "workflow": workflow}
    d.update(copy.deepcopy(COMMON))
    d["steps"] = copy.deepcopy(steps)
    return d

main = doc(MAIN,
    "Study 41: does decode context parallelism (which removes TP's KV duplication on an MLA "
    "model) or prefill context parallelism give more long-context throughput at parity of GPUs? "
    "deepseek-ai/DeepSeek-V2-Lite-Chat (MLA + MoE, bf16) on 4x L4, vLLM 0.32.0, always on all "
    "four GPUs (TP x PCP x DP = 4). Goal: generation tokens/s (= turns/s x 256) under TTFT p95 <= 10 s and TPOT p95 <= "
    "100 ms (150 s p95, :max). Load: closed-loop AIPerf users ramp of multi-turn sessions with "
    "8k-token shared prefixes, ended by a watchdog past 2 x the SLO. Steps: TP4 baseline (2 "
    "trials), 6 layout presets, 60 AKAMAS experiments. Needs vLLM pack 1.14.0 and GPU pack 1.4.0.",
    MAIN + "-Workflow", MAIN_STEPS)
smoke = doc(MAIN + "-Smoke",
    "Calibration run for study 41: TP4 baseline, TP4+DCP4, TP4+DCP4 fp8 KV and TP1xPCP4+EP, one trial each, with the Smoke "
    "workflow's steeper users ramp, same system, goal, constraints, windowing, parameters and KPIs "
    "as 41-L4-DSV2-Lite-Context-Parallel. It checks the pipeline end to end, prefix caching with "
    "DCP and PCP, fp8 KV with DCP, and places the knees, which set the main workflow's RT_USERS / RT_RAMP_S. Delete it "
    "afterwards.",
    MAIN + "-Smoke-Workflow", SMOKE_STEPS)

HEADER = ("# GENERATED by gen_study.py -- edit that file and re-run it, not this one.\n"
          "# Design and reasons: ../README.md and the comments in gen_study.py.\n")
if __name__ == "__main__":
    # The Smoke manifest is not rewritten: it ran on 2026-10-22 (with the 50 ms TPOT SLA and the
    # fp8 preset) and stays in git as the record of what was calibrated.
    for d in (main,):
        with open(d["name"] + ".yaml", "w") as f:
            f.write(HEADER)
            yaml.safe_dump(d, f, sort_keys=False, width=100)
    print("written:", main["name"] + ".yaml")
