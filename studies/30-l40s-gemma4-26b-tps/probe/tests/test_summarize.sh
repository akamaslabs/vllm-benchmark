#!/bin/bash
# summarize.py on synthetic results: ranks A-/L- within their group, B-default counts in both.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); P=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mk() {  # name prefill_s tpot64_ms
  printf '{"name":"%s","overrides":"-","started":true,"startup_s":200,"bench":{"summary":{"prefill_mean_s":%s,"tpot_single_ms":10,"decode_c64_tpot_ms":%s,"decode_c64_gen_tok_per_s":3000}}}\n' "$1" "$2" "$3" > "$TMP/$1.json"
}
mk B-default 0.30 30; mk L-triton 0.33 31; mk L-marlin 0.50 30; mk A-triton 0.30 30
printf '{"name":"A-fa","overrides":"ATTENTION_BACKEND=FLASH_ATTN","started":false,"apply_exit":4,"startup_s":90}\n' > "$TMP/A-fa.json"
OUT=$(python3 "$P/summarize.py" "$TMP")
FAILS=0
chk() { if grep -qE "$2" <<<"$OUT"; then echo "ok   $1"; else echo "FAIL $1"; FAILS=$((FAILS + 1)); fi; }
chk "triton within 15 % of auto"  '^L-triton .* yes '
chk "marlin out (+67 % prefill)"  '^L-marlin .* no '
chk "A-fa did not start"          '^A-fa +did not start'
chk "B-default not ranked"        '^B-default .* - '
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
