# shellcheck shell=bash
# Rendering of 01-vllm_template.yaml, sourced by apply_config.sh (and test_render.sh).
# Study 49's first run (23 experiments, 2026-11-07/08) ran the customer at vLLM's defaults: the
# indent of the tuned flags was read from the first line matching @TUNED_ARGS@, which was the
# template's header comment, so every flag was written as a comment and the pod got none of
# them (found at the recap from vllm:cache_config_info). Hence: the placeholder is the line that
# holds the token alone, the rendered customer block is checked before kubectl apply, and the
# running pod's args are checked after the rollout.

# render_vllm_manifest TEMPLATE CLAIM NEIGHBOURS NODE_ROLE CPU MEM [FLAG...]: the manifest on
# stdout. The placeholder line (whitespace + @TUNED_ARGS@ only) is replaced by one YAML list
# item per flag at the placeholder's indent; a mention of the token inside a comment is left
# alone. rc 1 when the template has no placeholder line.
render_vllm_manifest() {
  local template=$1 claim=$2 neighbours=$3 node_role=$4 cpu=$5 mem=$6
  shift 6
  local placeholder='^[[:space:]]*@TUNED_ARGS@[[:space:]]*$'
  grep -qE "$placeholder" "$template" || { echo "error: $template has no line holding only @TUNED_ARGS@" >&2; return 1; }
  local indent items="" a line
  indent=$(grep -m1 -E "$placeholder" "$template" | sed 's/@TUNED_ARGS@.*//')
  for a in "$@"; do items+="$indent- \"$a\""$'\n'; done
  # Every line but the placeholder, which becomes the flag items (a bash loop: BSD awk rejects
  # newlines in -v values, and sed's r+d needs a temp file).
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line =~ $placeholder ]]; then printf '%s' "$items"; else printf '%s\n' "$line"; fi
  done < "$template" \
    | sed -e "s/@CLAIM@/$claim/g" -e "s/@NEIGHBOURS@/$neighbours/g" -e "s/@NODE_ROLE@/$node_role/g" \
          -e "s/@CPU@/$cpu/g" -e "s/@MEM@/$mem/g"
}

# customer_block FILE: the rendered customer StatefulSet (from its metadata line to the
# neighbour's), where the tuned flags must be.
customer_block() { awk '/name: vllm-customer, namespace/,/name: vllm-neighbour, namespace/' "$1"; }

# check_rendered_args FILE [FLAG...]: every flag must appear in the customer block as a YAML
# list item on its own line (not inside a comment, not only in the neighbour's fixed args) and
# no @TOKEN@ may be left anywhere. Prints the problem and returns 1.
check_rendered_args() {
  local file=$1; shift
  local block a
  block=$(customer_block "$file")
  for a in "$@"; do
    grep -qE "^[[:space:]]*- \"$(printf '%s' "$a" | sed 's/[][\.*^$]/\\&/g')\"[[:space:]]*$" <<< "$block" \
      || { echo "error: $file: the customer StatefulSet does not carry $a as a list item (flag rendered as a comment or dropped)" >&2; return 1; }
  done
  if grep -q '@[A-Z_]*@' "$file"; then
    echo "error: $file still has a token: $(grep -o '@[A-Z_]*@' "$file" | sort -u | tr '\n' ' ')" >&2; return 1
  fi
}

# args_carry_flags ARGS_JSON [FLAG...]: ARGS_JSON is a container's args as kubectl prints it
# (`-o jsonpath='{.spec.containers[0].args}'`, a JSON array of strings). Every flag must be one
# of its elements. Prints the first missing flag and returns 1. This is the check that proves
# the experiment, not the file: run it on the pod after the rollout.
args_carry_flags() {
  local json=$1; shift
  local a
  for a in "$@"; do
    [[ $json == *"\"$a\""* ]] || { echo "error: the running customer pod has no arg $a (args: $json)" >&2; return 1; }
  done
}
