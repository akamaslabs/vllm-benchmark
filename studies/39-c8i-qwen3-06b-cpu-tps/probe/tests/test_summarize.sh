#!/bin/bash
# summarize.py on synthetic inputs: --parse reads vLLM 0.31.0's CPU log lines (formats from
# cpu_worker.py, kv_cache_utils.py, ompmultiprocessing.py) and node_facts.sh output; the
# summary finds the knee on a synthetic sweep and proposes RT_RATE = 2 x knee.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); P=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
chk() { if grep -qE -- "$2" <<<"$3"; then echo "ok   $1"; else echo "FAIL $1"; FAILS=$((FAILS + 1)); fi; }
cat > "$TMP/B-default.log" <<'L'
02:00:01 apply_config: parameters: GPU_MEMORY_UTILIZATION=0.92 MAX_NUM_SEQS=128
(EngineCore pid=120) INFO 10-16 02:03:00 [ompmultiprocessing.py:180] OpenMP thread binding info:
	VLLM_CPU_OMP_THREADS_BIND='auto', auto_setup=True, skip_setup=False
	local_world_size=1, reserve_cpu_num=2
	local_rank=0, core ids=[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
	reserved_cpus=[46, 47]
(EngineCore pid=120) INFO 10-16 02:08:00 [cpu_worker.py:260] Auto set (39.01/48.0) GiB for KV cache on node 0, with 44.16 GiB requested memory for the worker. 5.15 GiB memory was consumed by non-kv usages.
(EngineCore pid=120) INFO 10-16 02:08:01 [kv_cache_utils.py:2465] CPU KV cache size: 365,312 tokens, Maximum concurrency for 16,384 tokens per request: 22.30x
L
cat > "$TMP/B-default.facts.txt" <<'F'
### lscpu
NUMA node(s):                         1
Flags: fpu avx512f amx_bf16 amx_tile
### cpuset / affinity of this container
Cpus_allowed_list:	2-23,26-47
### cgroup memory (bytes)
memory.max: 51539607552
memory.current: 49000000000
memory.peak: 49500000000
F
cat > "$TMP/C-nochunk.log" <<'L'
error: enable_chunked_prefill false needs max_num_batched_tokens >= max_model_len 16384 (got 2048): vLLM refuses to start
error: invalid parameters in /tmp/probe39/results/C-nochunk.params.env
L
cat > "$TMP/E-gmu-low.log" <<'L'
(EngineCore pid=99) ERROR 10-16 03:00:00 [core.py:1] ValueError: Available memory on node 0 (40.1/48.0 GiB) on kv cache allocation is less than requested memory for kv (-0.5/7.2 GiB). Reduce CPU memory used by other processes.
error: vllm-0 restarted 2 times during startup — crash loop, failing now
L
python3 "$P/summarize.py" --parse B-default - 0 412 "$TMP" > "$TMP/B-default.json"
python3 "$P/summarize.py" --parse C-nochunk ENABLE_CHUNKED_PREFILL=false 2 1 "$TMP" > "$TMP/C-nochunk.json"
python3 "$P/summarize.py" --parse E-gmu-low GPU_MEMORY_UTILIZATION=0.15 4 300 "$TMP" > "$TMP/E-gmu-low.json"
J=$(cat "$TMP/B-default.json")
chk "kv tokens"           '"kv_tokens": 365312' "$J"
chk "kv GiB"              '"kv_gib": 39.01' "$J"
chk "requested GiB"       '"requested_gib": 44.16' "$J"
chk "non-kv GiB"          '"non_kv_gib": 5.15' "$J"
chk "max concurrency"     '"max_concurrency_x": 22.3' "$J"
chk "20 OMP threads"      '"omp_threads": 20' "$J"
chk "2 reserved"          '"omp_reserved": 2' "$J"
chk "one NUMA node"       '"numa_nodes": 1' "$J"
chk "AMX"                 '"amx": true' "$J"
chk "memory.peak"         '"memory_peak": 49500000000' "$J"
chk "LD_PRELOAD ok"       '"ld_preload_missing": false' "$J"
chk "renderer refusal"    '"rejected_by_renderer": true' "$(cat "$TMP/C-nochunk.json")"
chk "startup error kept"  '"error": "error: vllm-0 restarted' "$(cat "$TMP/E-gmu-low.json")"
# Sweep: 1 and 2 req/s within the SLO, 4 over on TPOT (AIPerf p95s) -> knee 2, RT_RATE 4.
cat > "$TMP/sweep.jsonl" <<'S'
{"rate": 1.0, "req_s": 1.0, "total_tok_s": 900, "ttft_p95_ms": 300, "tpot_p95_ms": 40}
{"rate": 2.0, "req_s": 2.0, "total_tok_s": 1800, "ttft_p95_ms": 900, "tpot_p95_ms": 85, "prom_prefix_hit_pct": 61.5}
{"rate": 4.0, "req_s": 3.1, "total_tok_s": 2500, "ttft_p95_ms": 2500, "tpot_p95_ms": 140}
S
OUT=$(python3 "$P/summarize.py" "$TMP")
chk "phase A row"         '^B-default +0 +412 +365312 +39\.01 +5\.15 +46\.1 +20 +2 +1 ' "$OUT"
chk "refused flagged R"   '^C-nochunk +2 .* R ' "$OUT"
chk "error line printed"  'crash loop' "$OUT"
chk "lower bound line"    'f >= ' "$OUT"
chk "over row"            '^ +4 .* over$' "$OUT"
chk "knee 2 req/s"        '^KNEE: 2 req/s' "$OUT"
chk "proposal"            'RT_RATE=4 RT_RAMP_S=3000' "$OUT"
# A sweep whose first step is already over the SLO.
printf '%s\n' '{"rate": 1.0, "ttft_p95_ms": 5000, "tpot_p95_ms": 40}' > "$TMP/sweep.jsonl"
chk "knee below the sweep" 'below the lowest rate' "$(python3 "$P/summarize.py" "$TMP")"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
