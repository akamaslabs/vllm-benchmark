#!/bin/bash
# Tests for ../summarize.py on fixture results.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); KP=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mk() {  # name linear attention kv prefill tpot1 tpot30
  printf '{"name":"%s","linear":"%s","attention":"%s","kv":"%s","started":true,"startup_s":300,"bench":{"summary":{"prefill_2k_mean_s":%s,"tpot_single_ms":%s,"decode_c30_tpot_ms":%s,"decode_c30_gen_tok_per_s":900}}}\n' "$@" > "$TMP/$1.json"
}
mk L-auto auto FLASHINFER auto 0.200 20.0 30.0
mk L-fast cutlass FLASHINFER auto 0.180 19.0 29.0
mk L-slow triton FLASHINFER auto 0.300 25.0 40.0
# A faster fp8-KV row must not push the bf16 linear rows out: each group (L-, A-, K-) is
# ranked against its own best.
mk K-fast auto FLASHINFER fp8 0.150 15.0 20.0
mk A-triton auto TRITON_ATTN auto 0.260 22.0 33.0
echo '{"name":"L-broken","linear":"deep_gemm","attention":"FLASHINFER","kv":"auto","started":false,"apply_exit":4,"startup_s":90}' > "$TMP/L-broken.json"
OUT=$(python3 "$KP/summarize.py" "$TMP")
FAILS=0
chk() { if grep -qE "$2" <<<"$OUT"; then echo "ok   $1"; else echo "FAIL $1"; FAILS=$((FAILS + 1)); fi; }
chk "fast within 15 %" '^L-fast .* yes$'
chk "auto within 15 %" '^L-auto .* yes$'
chk "slow outside 15 %" '^L-slow .* no$'
chk "fp8 row ranked in its own group" '^K-fast .* yes$'
chk "attention row ranked in its own group" '^A-triton .* yes$'
chk "broken reported" '^L-broken .*did not start \(apply exit 4\)'
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
