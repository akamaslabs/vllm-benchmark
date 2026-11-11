# shellcheck shell=bash
# Rendering of 01-vllm_template.yaml and the checks of what was really applied, sourced by
# apply_config.sh (and test_render.sh).
# Why the checks: study 49's first run (23 experiments) ran at vLLM's defaults because the indent
# of the tuned flags was read from a template comment that mentioned the token, so every flag was
# written as a comment. Hence: a placeholder is a line that holds the token alone, the rendered
# file is checked before kubectl apply, and the running pod's args and cache_config_info are
# checked after the rollout.

# render_vllm_manifest TEMPLATE ROLE SERVING LAYOUT APPLIED_AT EXTRA_ENV [FLAG...]: the manifest
# of one pod (ROLE a or b) on stdout. The TUNED_ARGS placeholder line becomes one YAML list item
# per flag, the EXTRA_ENV placeholder line one env item per NAME=value word of EXTRA_ENV (may be
# empty), both at their placeholder's indent. rc 1 if a placeholder line is missing or a word of
# EXTRA_ENV is not NAME=value.
render_vllm_manifest() {
  local template=$1 role=$2 serving=$3 layout=$4 applied_at=$5 extra_env=$6
  shift 6
  local ph_args='^[[:space:]]*@TUNED_ARGS@[[:space:]]*$' ph_env='^[[:space:]]*@EXTRA_ENV@[[:space:]]*$'
  grep -qE "$ph_args" "$template" || { echo "error: $template has no line holding only @TUNED_ARGS@" >&2; return 1; }
  grep -qE "$ph_env" "$template" || { echo "error: $template has no line holding only @EXTRA_ENV@" >&2; return 1; }
  local ind_args ind_env items="" envs="" a kv line
  ind_args=$(grep -m1 -E "$ph_args" "$template" | sed 's/@TUNED_ARGS@.*//')
  ind_env=$(grep -m1 -E "$ph_env" "$template" | sed 's/@EXTRA_ENV@.*//')
  for a in "$@"; do items+="$ind_args- \"$a\""$'\n'; done
  for kv in $extra_env; do
    [[ $kv =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_.,:/-]*$ ]] || { echo "error: extra env '$kv' is not NAME=value" >&2; return 1; }
    envs+="$ind_env- {name: ${kv%%=*}, value: \"${kv#*=}\"}"$'\n'
  done
  # A bash loop: BSD awk rejects newlines in -v values, and sed's r+d needs a temp file.
  while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line =~ $ph_args ]]; then printf '%s' "$items"
    elif [[ $line =~ $ph_env ]]; then printf '%s' "$envs"
    else printf '%s\n' "$line"; fi
  done < "$template" \
    | sed -e "s/@ROLE@/$role/g" -e "s/@SERVING@/$serving/g" -e "s/@LAYOUT@/$layout/g" -e "s/@APPLIED_AT@/$applied_at/g"
}

# check_rendered FILE [FLAG...]: every flag must be a YAML list item on its own line (not inside
# a comment) and no @TOKEN@ may be left. Prints the problem and returns 1.
check_rendered() {
  local file=$1 a body; shift
  body=$(sed 's/^[[:space:]]*//' "$file")
  for a in "$@"; do
    # Fixed-string match: --master-addr=$(POD_IP) has regex characters.
    grep -qxF -- "- \"$a\"" <<< "$body" \
      || { echo "error: $file does not carry $a as a list item (flag rendered as a comment or dropped)" >&2; return 1; }
  done
  if grep -q '@[A-Z_]*@' "$file"; then
    echo "error: $file still has a token: $(grep -o '@[A-Z_]*@' "$file" | sort -u | tr '\n' ' ')" >&2; return 1
  fi
}

# The three checks below take a container's args as kubectl prints them
# (`-o jsonpath='{.spec.containers[0].args}'`: a compact JSON array of strings). They prove the
# experiment, not the file: run them on the pod after the rollout.
# args_carry_flags ARGS_JSON [FLAG...]: every flag is one of the elements.
args_carry_flags() {
  local json=$1 a; shift
  for a in "$@"; do
    [[ $json == *"\"$a\""* ]] || { echo "error: the running pod has no arg $a (args: $json)" >&2; return 1; }
  done
}
# args_lack_flags ARGS_JSON [FLAG...]: none of the flags is an element.
args_lack_flags() {
  local json=$1 a; shift
  for a in "$@"; do
    [[ $json != *"\"$a\""* ]] || { echo "error: the running pod has the arg $a, which this layout must not have (args: $json)" >&2; return 1; }
  done
}
# args_count ARGS_JSON N: the array has exactly N elements (no stray flag from an old template).
args_count() {
  local n
  n=$(awk -F'","' '{ print NF }' <<< "$1")
  [ "$n" = "$2" ] || { echo "error: the running pod has $n args, expected $2 (args: $1)" >&2; return 1; }
}

# check_cache_config LINES GMU DTYPE: LINES are the vllm:cache_config_info lines of one API
# server (one per engine). Every line must carry gpu_memory_utilization equal to GMU (numeric
# compare, skipped when GMU is empty) and cache_dtype equal to DTYPE (skipped when DTYPE is empty
# or auto: vLLM's default, nothing to prove). rc 1 with the problem on stderr, also when there is
# no line at all.
check_cache_config() {
  local lines=$1 gmu=$2 dtype=$3 l v n=0
  while IFS= read -r l; do
    [ -n "$l" ] || continue
    n=$((n + 1))
    if [ -n "$gmu" ]; then
      v=$(sed -n 's/.*gpu_memory_utilization="\([^"]*\)".*/\1/p' <<< "$l")
      awk -v a="$v" -v b="$gmu" 'BEGIN { exit !(a != "" && a + 0 == b + 0) }' \
        || { echo "error: cache_config_info has gpu_memory_utilization=\"$v\", the experiment asked for $gmu" >&2; return 1; }
    fi
    if [ -n "$dtype" ] && [ "$dtype" != auto ]; then
      v=$(sed -n 's/.*cache_dtype="\([^"]*\)".*/\1/p' <<< "$l")
      [ "$v" = "$dtype" ] || { echo "error: cache_config_info has cache_dtype=\"$v\", the experiment asked for $dtype" >&2; return 1; }
    fi
  done <<< "$lines"
  [ "$n" -gt 0 ] || { echo "error: no vllm:cache_config_info line" >&2; return 1; }
}

# nccl_transport LOGFILE: the distinct NCCL transports in a pod log ("Libfabric", "Socket", ...),
# space separated, empty if NCCL logged none (NCCL_DEBUG=INFO prints "Using network <name>").
nccl_transport() { grep -o 'Using network [A-Za-z]*' "$1" 2>/dev/null | awk '{ print $3 }' | sort -u | tr '\n' ' ' | sed 's/ $//'; }

# check_transport SEEN EXPECTED MODE: MODE required = SEEN must be exactly EXPECTED; warn = empty
# is accepted (rc 0, a WARNING on stdout), anything else must be exactly EXPECTED; none = rc 0.
check_transport() {
  local seen=$1 expected=$2 mode=$3
  [ "$mode" != none ] || return 0
  if [ -z "$seen" ]; then
    [ "$mode" = warn ] && { echo "WARNING: NCCL logged no transport (no exchange between the nodes?)"; return 0; }
    echo "error: NCCL logged no transport, expected $expected" >&2; return 1
  fi
  [ "$seen" = "$expected" ] || { echo "error: NCCL transport is '$seen', expected $expected" >&2; return 1; }
}
