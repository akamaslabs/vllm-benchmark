#!/bin/bash
# Local tests of lib_render.sh: the rendering of 01-vllm_template.yaml for both pods and the
# checks apply_config.sh runs on the rendered files, the pods' args, cache_config_info and the
# NCCL transport lines. No cluster needed, except the last block (kubectl client dry-run),
# skipped when kubectl cannot reach an API server.
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1091
source "$HERE/lib_render.sh"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ok() { PASS=$((PASS + 1)); }
ko() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }
yes_() { local n=$1; shift; "$@" >/dev/null 2>&1 && ok || ko "$n"; }      # the command must succeed
no_()  { local n=$1; shift; "$@" >/dev/null 2>&1 && ko "$n" || ok; }      # the command must fail
T=$HERE/01-vllm_template.yaml
A=(--gpu-memory-utilization=0.90 --max-num-seqs=512 --tensor-parallel-size=2 --enable-expert-parallel --nnodes=2 --node-rank=0 '--master-addr=$(POD_IP)')
B=(--gpu-memory-utilization=0.90 --max-num-seqs=512 --tensor-parallel-size=2 --enable-expert-parallel --nnodes=2 --node-rank=1 --master-addr=10.0.1.7 --headless)

# --- rendering -----------------------------------------------------------------------
render_vllm_manifest "$T" a true tp2ep 20261110T220000Z "" "${A[@]}" > "$TMP/a.yaml" && ok || ko "render a"
render_vllm_manifest "$T" b false tp2ep 20261110T220000Z "" "${B[@]}" > "$TMP/b.yaml" && ok || ko "render b"
yes_ "check_rendered a" check_rendered "$TMP/a.yaml" "${A[@]}"
yes_ "check_rendered b" check_rendered "$TMP/b.yaml" "${B[@]}"
no_  "check_rendered: a flag that is not there" check_rendered "$TMP/a.yaml" --headless
yes_ "a: StatefulSet name" grep -q 'name: vllm-a, namespace: moe-tcp' "$TMP/a.yaml"
yes_ "b: StatefulSet name" grep -q 'name: vllm-b, namespace: moe-tcp' "$TMP/b.yaml"
yes_ "a: serving label" grep -q 'study51/serving: "true", study51/layout: "tp2ep"' "$TMP/a.yaml"
yes_ "b: not serving" grep -q 'study51/serving: "false"' "$TMP/b.yaml"
yes_ "b: SERVING env" grep -qF -- '- {name: SERVING, value: "false"}' "$TMP/b.yaml"
yes_ "b: node selector" grep -q 'nodeSelector: {node-role: llm-serving-g7e-4xl-tcp, study51/role: "b"}' "$TMP/b.yaml"
yes_ "a: applied-at" grep -q 'study51/applied-at: "20261110T220000Z"' "$TMP/a.yaml"
yes_ "a: POD_IP kept literal" grep -qxF -- '        - "--master-addr=$(POD_IP)"' "$TMP/a.yaml"
yes_ "a: flags right after the fixed args" sh -c "grep -A1 -F -- '--enable-prefix-caching' '$TMP/a.yaml' | grep -qF -- '--gpu-memory-utilization=0.90'"
# Fixed args + flags, nothing else under args.
[ "$(grep -cE '^[[:space:]]*- "--' "$TMP/a.yaml")" = $((5 + ${#A[@]})) ] && ok || ko "a: number of args"
[ "$(grep -cE '^[[:space:]]*- "--' "$TMP/b.yaml")" = $((5 + ${#B[@]})) ] && ok || ko "b: number of args"
[ "$(grep -cE '^[[:space:]]*- "--' "$T")" = 5 ] && ok || ko "template: five fixed args (apply_config.sh counts them)"
yes_ "NCCL pinned to TCP" grep -qxF -- '        - {name: NCCL_NET, value: Socket}' "$TMP/a.yaml"
no_ "no extra env by default" grep -q NCCL_SOCKET_NTHREADS "$TMP/a.yaml"

# No flags at all (vLLM defaults): still a valid manifest with no token left.
render_vllm_manifest "$T" a true single x "" > "$TMP/none.yaml" && ok || ko "render with no flags"
yes_ "check_rendered with no flags" check_rendered "$TMP/none.yaml"

# NCCL socket tuning by hand: extra env items at the placeholder's indent.
render_vllm_manifest "$T" a true tp2 x "NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4" "${A[@]}" > "$TMP/tcp.yaml" && ok || ko "render tuning"
yes_ "tuning: first item" grep -qxF -- '        - {name: NCCL_SOCKET_NTHREADS, value: "4"}' "$TMP/tcp.yaml"
yes_ "tuning: second item" grep -qxF -- '        - {name: NCCL_NSOCKS_PERTHREAD, value: "4"}' "$TMP/tcp.yaml"
yes_ "tuning: check_rendered" check_rendered "$TMP/tcp.yaml" "${A[@]}"
no_  "bad env word" render_vllm_manifest "$T" a true tp2 x "nccl_net=Socket" "${A[@]}"
no_  "env word with a quote" render_vllm_manifest "$T" a true tp2 x 'NCCL_NET=So"cket' "${A[@]}"

# Study 49's bug: a comment that mentions the token. The placeholder is only the line that holds
# the token alone, so the flags are real list items; the token left in the comment is then caught.
sed 's/^apiVersion: apps\/v1$/# flags go where @TUNED_ARGS@ is\napiVersion: apps\/v1/' "$T" > "$TMP/comment.yaml"
grep -q '^# flags go where' "$TMP/comment.yaml" || printf '%s\n' '# flags go where @TUNED_ARGS@ is' | cat - "$T" > "$TMP/comment.yaml"
render_vllm_manifest "$TMP/comment.yaml" a true tp2 x "" "${A[@]}" > "$TMP/c.yaml" && ok || ko "render with the token in a comment"
yes_ "token in a comment: flags are still list items" grep -qxF -- '        - "--max-num-seqs=512"' "$TMP/c.yaml"
no_  "token in a comment: check_rendered refuses the leftover" check_rendered "$TMP/c.yaml" "${A[@]}"
grep -v '^[[:space:]]*@TUNED_ARGS@[[:space:]]*$' "$T" > "$TMP/noph.yaml"
no_  "no placeholder line" render_vllm_manifest "$TMP/noph.yaml" a true tp2 x "" "${A[@]}"
grep -v '^[[:space:]]*@EXTRA_ENV@[[:space:]]*$' "$T" > "$TMP/noenv.yaml"
no_  "no env placeholder line" render_vllm_manifest "$TMP/noenv.yaml" a true tp2 x "" "${A[@]}"
# A flag written as a comment (what study 49 rendered) is not a list item.
sed 's/^\([[:space:]]*\)- "--max-num-seqs=512"$/\1# - "--max-num-seqs=512"/' "$TMP/a.yaml" > "$TMP/commented.yaml"
no_  "flag rendered as a comment" check_rendered "$TMP/commented.yaml" "${A[@]}"

# --- the running pod's args (kubectl jsonpath prints a compact JSON array) --------------
J='["--served-model-name=gpt-oss-120b","--port=8000","--host=0.0.0.0","--max-model-len=16384","--enable-prefix-caching","--gpu-memory-utilization=0.90","--max-num-seqs=512","--tensor-parallel-size=2","--enable-expert-parallel","--nnodes=2","--node-rank=0","--master-addr=$(POD_IP)"]'
yes_ "args carry the flags" args_carry_flags "$J" "${A[@]}"
no_  "args: a flag is missing" args_carry_flags "$J" --headless
no_  "args: a different value" args_carry_flags "$J" --max-num-seqs=51
yes_ "args lack --headless" args_lack_flags "$J" --headless --data-parallel-external-lb
no_  "args lack: it is there" args_lack_flags "$J" --enable-expert-parallel
yes_ "args count" args_count "$J" 12
no_  "args count: a stray flag" args_count "$J" 11
yes_ "args count: one arg" args_count '["--port=8000"]' 1

# --- vllm:cache_config_info ----------------------------------------------------------
C1='vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",gpu_memory_utilization="0.9",num_gpu_blocks="51234"} 1.0'
C2=$C1$'\n''vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="1",gpu_memory_utilization="0.9",num_gpu_blocks="51234"} 1.0'
C3=$C1$'\n''vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="1",gpu_memory_utilization="0.92",num_gpu_blocks="51234"} 1.0'
yes_ "cache config: one engine" check_cache_config "$C1" 0.90 auto
yes_ "cache config: two engines" check_cache_config "$C2" 0.900000 auto
no_  "cache config: default gmu instead of the experiment's" check_cache_config "$C1" 0.85 auto
no_  "cache config: the second engine differs" check_cache_config "$C3" 0.90 auto
no_  "cache config: dtype not applied" check_cache_config "$C1" 0.90 fp8
yes_ "cache config: dtype applied" check_cache_config "${C1/auto/fp8}" 0.90 fp8
yes_ "cache config: nothing asked" check_cache_config "$C1" "" ""
no_  "cache config: no line" check_cache_config "" 0.90 auto

# --- NCCL transport ------------------------------------------------------------------
printf '%s\n' 'vllm-a-0:1:1 [0] NCCL INFO NET/OFI Selected provider is efa, fabric is efa (found 1 nics)' \
  'vllm-a-0:1:1 [0] NCCL INFO Using network Libfabric' 'vllm-a-0:1:2 [0] NCCL INFO Using network Libfabric' > "$TMP/efa.log"
printf '%s\n' 'x NCCL INFO Using network Socket' > "$TMP/tcp.log"
cat "$TMP/efa.log" "$TMP/tcp.log" > "$TMP/both.log"
printf '%s\n' 'INFO no nccl here' > "$TMP/none.log"
[ "$(nccl_transport "$TMP/efa.log")" = Libfabric ] && ok || ko "transport efa: '$(nccl_transport "$TMP/efa.log")'"
[ "$(nccl_transport "$TMP/tcp.log")" = Socket ] && ok || ko "transport tcp"
[ "$(nccl_transport "$TMP/both.log")" = "Libfabric Socket" ] && ok || ko "transport both: '$(nccl_transport "$TMP/both.log")'"
[ -z "$(nccl_transport "$TMP/none.log")" ] && ok || ko "transport none"
[ -z "$(nccl_transport "$TMP/missing.log")" ] && ok || ko "transport: no log file"
yes_ "transport required, TCP" check_transport Socket Socket required
no_  "transport required, OFI instead of TCP" check_transport Libfabric Socket required
no_  "transport required, both" check_transport "Libfabric Socket" Socket required
no_  "transport required, none logged" check_transport "" Socket required
yes_ "transport warn, none logged" check_transport "" Socket warn
no_  "transport warn, OFI instead of TCP" check_transport Libfabric Socket warn
yes_ "transport none" check_transport Libfabric Socket none

# --- the rendered manifests are valid Kubernetes objects -------------------------------
if kubectl version --request-timeout=5s >/dev/null 2>&1; then
  for f in a b none tcp; do
    kubectl apply --dry-run=client -f "$TMP/$f.yaml" >/dev/null 2>"$TMP/err" && ok || ko "kubectl dry-run of $f.yaml: $(cat "$TMP/err")"
  done
else
  echo "SKIP: kubectl has no API server, dry-run of the rendered manifests not done"
fi

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
