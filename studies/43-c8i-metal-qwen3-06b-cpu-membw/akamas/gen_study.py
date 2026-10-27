#!/usr/bin/env python3
"""Study 43 = study 42 (itself an exact repeat of study 39) moved to a bare-metal c8i.metal-48xl, so
membw.py can read the memory-controller (uncore IMC) counters while the study runs; same goal, load,
parameters and steps. Writes the main study manifest; no Smoke study. Originally: writes the main and Smoke study manifests of study 39 from one definition, so the two
stay identical except for name, description, workflow and steps. Run: python3 gen_study.py"""
import copy, yaml

SYSTEM = "vLLM_Benchmark_43_C8i_Metal_Qwen3_06B_CPU_Membw"
MAIN = "43-C8i-Metal-Qwen3-06B-CPU-Membw"

# vLLM 0.31.0 defaults on the CPU backend (online serving, world size 1), except the KV
# budget: gpu_memory_utilization 0.90 instead of the default 0.92 of the cgroup-clamped RAM
# (0.92 leaves ~0-0.5 GiB of the 48 GiB limit at vLLM's KV-allocation check, counting the
# EngineCore process; see
# k8s/params.env.template). Chunked prefill (on) and stream_interval (1) are fixed in the
# template, not tuned: off-chunked cannot start with max_model_len 16384 > the batched-token
# domain, and stream_interval would hide latency from the server-side SLO metrics. max_num_seqs 128,
# max_num_batched_tokens 2048, block_size 128 (AMX path needs % 32 == 0), prefix caching on.
BASE = {
    "vllm.gpu_memory_utilization": 0.90,
    "vllm.max_num_seqs": 128,
    "vllm.max_num_batched_tokens": 2048,
    "vllm.block_size": "128",
    "vllm.kv_cache_dtype": "auto",
    "vllm.enable_prefix_caching": "true",
    "vllm.long_prefill_token_threshold": 0,
}

def preset(name, **over):
    v = dict(BASE)
    v.update({f"vllm.{k}": val for k, val in over.items()})
    return {"name": name, "type": "preset", "values": v}

COMMON = {
    "system": SYSTEM,
    "goal": {
        "objective": "maximize",
        "function": {"formula": "vllm.total_token_throughput", "unit": "tokens per second"},
        "constraints": {"absolute": [
            {"name": "TTFT P95 over 150 s", "formula": "vllm.time_to_first_token_p95:max <= 3000"},
            {"name": "TPOT P95 over 150 s", "formula": "vllm.time_per_output_token_p95:max <= 100"},
        ]},
    },
    "windowing": {"type": "stability", "stability": {
        "metric": "vllm.total_token_throughput", "width": 6, "maxStdDev": 300000000,
        "when": {"metric": "vllm.total_token_throughput", "is": "max"}}},
    "numberOfTrials": 1,
    "parametersSelection": [
        # On CPU this is the KV-cache share of RAM (cgroup limit 48 GiB), not GPU memory.
        # Domain from k8s/params.env.template's KV sizing table (to confirm with the probe's RSS).
        {"name": "vllm.gpu_memory_utilization", "domain": [0.15, 0.90]},
        {"name": "vllm.max_num_seqs", "domain": [8, 256]},
        {"name": "vllm.max_num_batched_tokens", "domain": [512, 8192]},
        {"name": "vllm.block_size", "categories": ["32", "64", "96", "128"]},
        {"name": "vllm.kv_cache_dtype", "categories": ["auto", "fp8"]},
        {"name": "vllm.enable_prefix_caching", "categories": ["true", "false"]},
        {"name": "vllm.long_prefill_token_threshold", "domain": [0, 4096]},
    ],
    "parameterConstraints": [
        {"name": "max_num_batched_tokens must be at least max_num_seqs",
         "formula": "vllm.max_num_batched_tokens >= vllm.max_num_seqs"},
    ],
    "kpis": [
        {"name": "Total throughput", "formula": "vllm.total_token_throughput", "direction": "maximize"},
        {"name": "Completed requests", "formula": "vllm.request_success_rate", "direction": "maximize"},
        {"name": "TTFT P95 150s", "formula": "vllm.time_to_first_token_p95", "direction": "minimize", "aggregation": "max"},
        {"name": "TPOT P95 150s", "formula": "vllm.time_per_output_token_p95", "direction": "minimize", "aggregation": "max"},
        {"name": "Prefix cache hit rate", "formula": "vllm.prefix_cache_hit_rate", "direction": "maximize"},
        {"name": "KV cache usage", "formula": "vllm.kv_cache_usage_avg", "direction": "minimize"},
        {"name": "Preemptions", "formula": "vllm.preemption_rate", "direction": "minimize"},
        {"name": "Running requests", "formula": "vllm.num_requests_running", "direction": "maximize", "aggregation": "max"},
    ],
}

MAIN_STEPS = [
    {"name": "baseline", "type": "baseline", "numberOfTrials": 1, "values": dict(BASE)},
    preset("baseline repeat"),
    preset("kv fp8", kv_cache_dtype="fp8"),
    preset("no prefix cache", enable_prefix_caching="false"),
    preset("small kv budget", gpu_memory_utilization=0.15),
    {"name": "optimize", "type": "optimize", "optimizer": "AKAMAS",
     "numberOfInitExperiments": 0, "numberOfExperiments": 60, "maxFailedExperiments": 20},
]

def doc(name, description, workflow, steps):
    d = {"kind": "study", "name": name, "description": description, "workflow": workflow}
    d.update(copy.deepcopy(COMMON))
    d["steps"] = steps
    return d

main = doc(MAIN,
    "Study 43 (study 39/42 on bare metal, to read memory bandwidth with membw.py; meant to run long "
    "and be stopped by hand): vLLM CPU-backend tuning of Qwen/Qwen3-0.6B (bf16, thinking off) on one "
    "c8i.metal-48xl (the vLLM pod pinned to one of its 3 NUMA nodes, 22 cores), vLLM 0.31.0, no GPU. Goal: total tokens/s (prompt + generation) under TTFT "
    "p95 <= 3000 ms and TPOT p95 <= 100 ms (150 s p95, :max). Load: open-loop AIPerf rate ramp "
    "of multi-turn sessions sharing a pool of prompt prefixes, ended by a watchdog past the SLO. Steps: "
    "vLLM CPU-default baseline twice, kv fp8, no prefix cache, small kv budget, 60 AKAMAS "
    "experiments. Needs vLLM pack 1.14.0.",
    MAIN + "-Workflow", MAIN_STEPS)
smoke = doc(MAIN + "-Smoke",
    "Smoke run for study 39: one trial of the vLLM CPU-default baseline with a steep ramp, same "
    "system, goal, constraints, windowing, parameters and KPIs as 43-C8i-Metal-Qwen3-06B-CPU-Membw. It "
    "checks the pipeline end to end and locates the baseline's knee, which sets the main "
    "workflow's RT_RATE / RT_RAMP_S. Delete it afterwards.",
    MAIN + "-Smoke-Workflow", [MAIN_STEPS[0]])

HEADER = ("# GENERATED by gen_study.py -- edit that file and re-run it, not this one.\n"
          "# Design and reasons: ../README.md and the comments in gen_study.py.\n")
for d in (main,):
    with open(d["name"] + ".yaml", "w") as f:
        f.write(HEADER)
        yaml.safe_dump(d, f, sort_keys=False, width=100)
print("written:", main["name"] + ".yaml")
