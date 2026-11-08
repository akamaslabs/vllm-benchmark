#!/bin/bash
# Local tests of lib_render.sh: the tuned flags must land in the customer container's args of
# the rendered 01-vllm.yaml, as YAML list items, never in a comment. Study 49 ran all 23
# experiments with the customer at vLLM's defaults because the template's header comment
# mentioned @TUNED_ARGS@ too: the indent was taken from that comment line, so every flag was
# rendered as a comment and the pod got none of them.
# shellcheck disable=SC2015
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
# shellcheck disable=SC1091
source "$HERE/lib_render.sh"

ARGS=(--gpu-memory-utilization=0.80 --max-num-seqs=16 --kv-cache-dtype=fp8 --no-async-scheduling)

# 1. The real template: every flag is a list item, indented like the fixed args, in the customer
#    container only (the neighbour keeps its fixed --gpu-memory-utilization=0.90).
render_vllm_manifest "$HERE/01-vllm_template.yaml" mig-2g48gb 1 llm-serving-g7e-4xl 7000 56 "${ARGS[@]}" > "$TMP/01-vllm.yaml"; RC=$?
CUSTOMER=$(awk '/name: vllm-customer, namespace/,/name: vllm-neighbour, namespace/' "$TMP/01-vllm.yaml")
[ $RC = 0 ] && for a in "${ARGS[@]}"; do grep -qx "        - \"$a\"" <<< "$CUSTOMER" || exit_flag=1; done \
  && [ -z "${exit_flag:-}" ] && ok || ko "real template: flags as list items in the customer args"
[ "$(grep -c -- '--gpu-memory-utilization=' "$TMP/01-vllm.yaml")" = 2 ] && ok || ko "exactly two gpu-memory-utilization lines (customer + neighbour)"
! grep -q '@[A-Z_]*@' "$TMP/01-vllm.yaml" && grep -q 'resourceClaimTemplateName: mig-2g48gb' "$TMP/01-vllm.yaml" \
  && grep -q 'replicas: 1$' "$TMP/01-vllm.yaml" && grep -q 'cpu: "7000m"' "$TMP/01-vllm.yaml" && ok || ko "other tokens substituted"
! grep -E '^#.*- "--' "$TMP/01-vllm.yaml" >/dev/null && ok || ko "no flag rendered inside a comment"

# 2. A template whose comment mentions the token (study 49's header) must not take its indent
#    from the comment.
printf '# tuned flags (@TUNED_ARGS@)\nargs:\n- "--model=x"\n@TUNED_ARGS@\nx: 1\n' > "$TMP/t.yaml"
render_vllm_manifest "$TMP/t.yaml" c 0 r 1 1 "${ARGS[@]}" > "$TMP/o.yaml"; RC=$?
[ $RC = 0 ] && grep -qx -- '- "--kv-cache-dtype=fp8"' "$TMP/o.yaml" && grep -qx '# tuned flags (@TUNED_ARGS@)' "$TMP/o.yaml" && ok || ko "comment mentioning the token is left alone"

# 3. A template without the placeholder line fails loudly.
printf 'args:\n- "--model=x"\n' > "$TMP/n.yaml"
render_vllm_manifest "$TMP/n.yaml" c 0 r 1 1 "${ARGS[@]}" > "$TMP/o.yaml" 2>/dev/null; RC=$?
[ $RC != 0 ] && ok || ko "no placeholder line -> non-zero rc"

# 4. check_rendered_args: passes on the good render, fails when a flag is missing or commented.
check_rendered_args "$TMP/01-vllm.yaml" "${ARGS[@]}" && ok || ko "check passes on a good render"
sed 's/^\(        - "--kv-cache-dtype=fp8"\)/#\1/' "$TMP/01-vllm.yaml" > "$TMP/bad.yaml"
! check_rendered_args "$TMP/bad.yaml" "${ARGS[@]}" 2>/dev/null && ok || ko "check fails on a commented flag"
# A flag present only in the neighbour's fixed args must not satisfy the customer's check.
render_vllm_manifest "$HERE/01-vllm_template.yaml" mig-2g48gb 1 r 1 1 --max-num-seqs=16 > "$TMP/n2.yaml"
! check_rendered_args "$TMP/n2.yaml" --gpu-memory-utilization=0.90 2>/dev/null && ok || ko "check ignores the neighbour's flags"

# 5. args_carry_flags: the pod's args as kubectl prints them.
JSON='["--model=Qwen/Qwen3-8B-FP8","--max-model-len=4096","--gpu-memory-utilization=0.80","--max-num-seqs=16","--kv-cache-dtype=fp8","--no-async-scheduling"]'
args_carry_flags "$JSON" "${ARGS[@]}" && ok || ko "pod args carry every flag"
! args_carry_flags '["--model=Qwen/Qwen3-8B-FP8","--max-model-len=4096"]' "${ARGS[@]}" 2>/dev/null && ok || ko "pod at defaults -> fail"
! args_carry_flags "$JSON" --max-num-seqs=1 2>/dev/null && ok || ko "a flag that is only a substring of another does not count"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
