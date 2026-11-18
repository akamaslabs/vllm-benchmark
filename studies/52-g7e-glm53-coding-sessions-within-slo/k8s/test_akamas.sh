#!/bin/bash
# Local checks of the study YAML and the smoke's histogram helper (no Akamas, no cluster).
# The ITL safety net reads the per-request TPOT (the server-side equivalent of AIPerf's ITL), not
# vLLM's per-engine-step ITL (final review I1).
HERE=$(cd "$(dirname "$0")" && pwd)
Y="$HERE/../akamas/52-G7E-GLM53-Coding-Sessions.yaml"
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
grep -q 'formula: vllm.time_per_output_token_p95 <= 80' "$Y" && ok || ko "TPOT safety net at 80 ms (smoke 2026-11-18: server TPOT p95 53.6 ms vs client ITL p95 19.5 ms)"
grep -q 'vllm:request_time_per_output_token_seconds' "$HERE/smoke.sh" && ok || ko "smoke reads the v0.34 TPOT histogram name"
grep -q 'formula: vllm.inter_token_latency_p95 <=' "$Y" && ko "no constraint on the per-step inter_token_latency_p95" || ok
# DCP is out (no MLA backend supports it on SM120), and the MoE candidates are the W4A16 ones.
grep -q 'decode_context_parallel_size' "$Y" && ko "no decode_context_parallel_size in the study" || ok
grep -q 'categories: \[auto, marlin, humming, triton\]' "$Y" && ok || ko "moe_backend categories = the W4A16 candidates"
# Smoke helper: interpolated histogram quantile between two /metrics snapshots.
T=$(mktemp -d)
printf '%s\n' 'vllm:time_per_output_token_seconds_bucket{le="0.025"} 0' 'vllm:time_per_output_token_seconds_bucket{le="0.05"} 0' \
  'vllm:time_per_output_token_seconds_bucket{le="0.075"} 0' 'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 0' > "$T/a"
printf '%s\n' 'vllm:time_per_output_token_seconds_bucket{le="0.025"} 50' 'vllm:time_per_output_token_seconds_bucket{le="0.05"} 90' \
  'vllm:time_per_output_token_seconds_bucket{le="0.075"} 100' 'vllm:time_per_output_token_seconds_bucket{le="+Inf"} 100' > "$T/b"
O=$(python3 "$HERE/hist_quantile.py" vllm:time_per_output_token_seconds "$T/a" "$T/b" 0.95 2>&1)
[ "$O" = "62.5 ms" ] && ok || ko "hist_quantile interpolates inside the 50-75 ms bucket ($O)"
rm -rf "$T"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
