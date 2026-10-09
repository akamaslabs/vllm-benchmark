#!/bin/bash
# summarize.py on synthetic results: A-/L- ranked within their group with B-compose, M- against
# the same configuration without MTP.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); P=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mk() {  # name prefill_s tok/s@1 tok/s@64 [acceptance]
  local a=${5:-null}
  printf '{"name":"%s","overrides":"-","started":true,"startup_s":200,"bench":{"summary":{"prefill_mean_s":%s,"tok_per_s_single":%s,"c64_gen_tok_per_s":%s,"c64_per_request_tok_per_s":40,"c32_it_gen_tok_per_s":1500,"acceptance_single_en":%s,"acceptance_c64_en":%s,"acceptance_c32_it":%s}}}\n' \
    "$1" "$2" "$3" "$4" "$a" "$a" "$a" > "$TMP/$1.json"
}
mk B-compose 0.30 100 3000; mk K-fp8 0.31 100 3300; mk L-triton 0.33 98 2900; mk L-marlin 0.50 99 3000
mk A-triton 0.30 100 3000; mk M-mtp2 0.31 180 2700 0.7; mk M-mtp2-fp8 0.31 170 3300 0.7
printf '{"name":"A-fa","overrides":"ATTENTION_BACKEND=FLASH_ATTN","started":false,"apply_exit":4,"startup_s":90}\n' > "$TMP/A-fa.json"
OUT=$(python3 "$P/summarize.py" "$TMP")
FAILS=0
chk() { if grep -qE -- "$2" <<<"$OUT"; then echo "ok   $1"; else echo "FAIL $1"; FAILS=$((FAILS + 1)); fi; }
chk "triton within 15 % of auto"      '^L-triton .* yes '
chk "marlin out (+67 % prefill)"      '^L-marlin .* no '
chk "A-fa did not start"              '^A-fa +did not start'
chk "B-compose not ranked"            '^B-compose .* - '
chk "mtp2 vs B-compose: +80 %/-10 %"  '^M-mtp2 .* vs ref \+80%/-10% '
chk "mtp2-fp8 vs K-fp8: +70 %/+0 %"   '^M-mtp2-fp8 .* vs ref \+70%/\+0% '
chk "acceptance printed"              '^M-mtp2 .* 0\.70/0\.70/0\.70 '
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
