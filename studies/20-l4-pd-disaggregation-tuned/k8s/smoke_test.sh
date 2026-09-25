#!/bin/bash
# Study 18 smoke test (phase B): run BY HAND on the toolbox, with the llm-serving-l4 node
# up, BEFORE `akamas start study`. It checks the assumptions that cannot be verified
# offline and calibrates the ramp levels and the SLA (done 2026-09-24, see the README).
#
#   bash smoke_test.sh up <preset>     render + apply one preset through apply_config.sh
#   bash smoke_test.sh probe           10 streamed requests via the router, then the checks
#   bash smoke_test.sh calibrate       short AIPerf ramp (60 s per level) on what is running
#   bash smoke_test.sh down            delete the serving Deployment (node stays up)
#   bash smoke_test.sh diag            interconnect diagnosis: P2P matrix, copy bandwidth
#                                      direct vs via host, NCCL transport + all-reduce with and
#                                      without NCCL_P2P_DISABLE (~5 min, needs the 4 GPUs free)
#
# Suggested order (~1-1.5 h of node time):
#   diag                    FIRST, while the GPUs are free: answers "P2P or through the host?"
#                           for both this study (NIXL cuda_ipc) and study 16's TP penalty
#   (study 20 presets: baseline, A1-A3 aggregated, D1-D6 disaggregated)
#   up S3 -> probe          1P1D works end to end? which UCX transport? metrics labelled?
#   up S4 -> probe          host-buffer path starts and transfers
#   up S5 -> probe          fp8 KV accepted on both roles (no dynamic-scale error)
#   up S9 -> probe          4 processes start together (3P1D) within the deadline
#   up baseline -> calibrate   aggregated knee
#   up S3 -> calibrate         disaggregated knee
# then set CONCURRENCY_LIST in 05-job.yaml and the SLA in the study manifest + 05-job.yaml.
set -euo pipefail
K8S=/work/vllm-benchmark/studies/20-l4-pd-disaggregation-tuned/k8s
NS=llm-serving

# name -> P D buffer | prefill seqs mnbt kv | decode seqs mnbt kv   (same values as the study manifest)
preset() {
  case "$1" in
    # baseline: "-" = not rendered, vLLM picks its own default (as the study's doNotRenderParameters)
    baseline) echo "0 2 cpu - - - - - -" ;;
    A1) echo "0 3 cpu 56 8214 fp8 460 4142 fp8" ;;
    A2) echo "0 2 cpu 56 8214 fp8 460 4142 fp8" ;;
    A3) echo "0 4 cpu 56 8214 fp8 460 4142 fp8" ;;
    D1) echo "1 1 cpu 16 4096 fp8 24 2048 fp8" ;;
    D2) echo "2 1 cpu 16 4096 fp8 24 2048 fp8" ;;
    D3) echo "3 1 cpu 16 4096 fp8 24 2048 fp8" ;;
    D4) echo "2 2 cpu 16 4096 fp8 24 2048 fp8" ;;
    D5) echo "1 2 cpu 16 4096 fp8 24 2048 fp8" ;;
    D6) echo "3 1 cpu 16 4096 auto 12 2048 auto" ;;
    *) echo "unknown preset $1" >&2; exit 1 ;;
  esac
}

router_py() { kubectl exec deployment/vllm-pd -n "$NS" -c router -- python3 -c "$1"; }

case "${1:-}" in
  up)
    read -r P D BUF PS PM PK DS DM DK <<<"$(preset "${2:?preset name}")"
    r() { [ "$2" = "-" ] && echo "-e /\${$1}/d" || echo "-e s/\${$1}/$2/"; }
    # shellcheck disable=SC2046
    sed -e "s/\${pd_topology.pd_prefill_instances}/$P/g" -e "s/\${pd_topology.pd_decode_instances}/$D/g" \
        -e "s/\${pd_topology.pd_kv_connector}/NixlConnector/" -e "s/\${pd_topology.pd_kv_buffer_device}/$BUF/" \
        -e "s/\${vllm_prefill.gpu_memory_utilization}/0.9/" -e "s/\${vllm_decode.gpu_memory_utilization}/0.9/" \
        $(r vllm_prefill.max_num_seqs "$PS") $(r vllm_prefill.max_num_batched_tokens "$PM") $(r vllm_prefill.kv_cache_dtype "$PK") \
        $(r vllm_decode.max_num_seqs "$DS") $(r vllm_decode.max_num_batched_tokens "$DM") $(r vllm_decode.kv_cache_dtype "$DK") \
        "$K8S/01-deployment_template.yaml" > "$K8S/01-deployment.yaml"
    START=$SECONDS
    bash "$K8S/apply_config.sh"
    echo "== ready in $((SECONDS - START)) s (progressDeadlineSeconds is 1200: tighten it if this is far below)"
    ;;

  probe)
    echo "== 10 streamed chat requests through the router (4096-token prompt, 64 output tokens)"
    router_py '
import json,time,urllib.request
prompt=" ".join(["hello"]*4000)
for i in range(10):
    body=json.dumps({"model":"qwen3-8b","messages":[{"role":"user","content":prompt}],"max_tokens":64,"ignore_eos":True,"stream":True}).encode()
    t0=time.time(); first=None; n=0
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=300) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                n+=1; first=first or time.time()
    print(f"req {i}: ttft {first-t0:.3f}s  e2e {time.time()-t0:.3f}s  chunks {n}")
'
    echo "== router series (model_name must be qwen3-8b-router)"
    router_py 'import urllib.request;[print(l) for l in urllib.request.urlopen("http://127.0.0.1:8000/metrics").read().decode().splitlines() if l.startswith(("vllm:","pd_router")) and ("_count" in l or "_total" in l) and "created" not in l]'
    echo "== engine series: model_name per port, NIXL metrics, prompt tokens by source"
    for port in 8100 8101 8102 8200 8201 8202 8203; do
      router_py "
import urllib.request
try: t=urllib.request.urlopen('http://127.0.0.1:$port/metrics',timeout=3).read().decode()
except Exception: raise SystemExit
names=sorted({l.split('model_name=\"')[1].split('\"')[0] for l in t.splitlines() if 'model_name=\"' in l})
print('port $port model_name', names)
[print('   ',l) for l in t.splitlines() if l.startswith(('vllm:nixl_xfer_time_seconds_count','vllm:nixl_bytes_transferred_sum','vllm:nixl_num_failed_transfers_total','vllm:prompt_tokens_by_source_total'))]
" || true
    done
    echo "== UCX transport chosen (cuda_ipc = GPU-to-GPU over PCIe P2P; cuda_copy/sm/tcp = staged)"
    kubectl logs deployment/vllm-pd -n "$NS" -c engine --tail=-1 | grep -iE 'ucp|cuda_ipc|cuda_copy|proto|transport' | head -40 || true
    echo "== GPU P2P matrix printed by the launcher"
    kubectl logs deployment/vllm-pd -n "$NS" -c engine --tail=-1 | sed -n '/topo -p2p\|P2P\|GPU0/,+8p' | head -30 || true
    echo "== vLLM KV transfer log lines"
    kubectl logs deployment/vllm-pd -n "$NS" -c engine --tail=-1 | grep -iE 'KV Transfer metrics|nixl' | tail -15 || true
    echo "== the study telemetry's own queries, as Prometheus answers them (wait ~30 s after the requests)"
    PROM=/api/v1/namespaces/monitoring/services/kube-prometheus-stack-prometheus:9090/proxy/api/v1/query
    for q in 'count(DCGM_FI_DEV_FB_USED{exported_pod=~"^vllm-pd-.*"} > 1024)' \
             'sum by (model_name)(vllm:generation_tokens_total{pod=~"^vllm-pd-.*"})' \
             'sum by (model_name)(vllm:time_to_first_token_seconds_count{pod=~"^vllm-pd-.*"})' \
             'count by (endpoint)(up{pod=~"^vllm-pd-.*"} == 1)'; do
      echo "-- $q"
      kubectl get --raw "$PROM?query=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$q")" \
        | python3 -c 'import json,sys;[print("  ",r["metric"],r["value"][1]) for r in json.load(sys.stdin)["data"]["result"]] or print("   (no data)")'
    done
    echo "   expected: active GPUs = P + D; generation tokens under qwen3-8b-router and -decode (-prefill ~1/request); TTFT counts under all present roles"
    ;;

  calibrate)
    echo "== short AIPerf ramp: 60 s per level, wider level list, same prompts as the study"
    sed -e 's/--benchmark-duration 600/--benchmark-duration 60/' \
        -e 's/CONCURRENCY_LIST="[^"]*"/CONCURRENCY_LIST="2,4,8,12,16,24,32,48,64,96"/' \
        "$K8S/05-job.yaml" > /tmp/aiperf-calibrate.yaml
    kubectl delete -f /tmp/aiperf-calibrate.yaml --ignore-not-found
    kubectl apply -f /tmp/aiperf-calibrate.yaml
    kubectl wait --for=condition=complete job/aiperf-benchmark -n llm-benchmark --timeout=3000s || true
    kubectl logs job/aiperf-benchmark -n llm-benchmark -c aiperf --tail=-1
    echo "== read per level: TTFT p95, ITL p95, output token throughput. The knee is where TTFT p95"
    echo "   leaves the flat region. Pick 6 levels bracketing the aggregated and disaggregated knees."
    ;;

  down)
    kubectl delete deployment vllm-pd -n "$NS" --ignore-not-found
    ;;

  diag)
    if kubectl get deployment vllm-pd -n "$NS" >/dev/null 2>&1; then
      echo "vllm-pd is deployed and holds the GPUs: run 'bash smoke_test.sh down' first"; exit 1
    fi
    kubectl create configmap p2p-nccl-diag -n "$NS" --from-file=p2p_nccl_diag.py="$K8S/p2p_nccl_diag.py" \
      --dry-run=client -o yaml | kubectl apply -f -
    kubectl delete pod p2p-nccl-diag -n "$NS" --ignore-not-found
    kubectl apply -f "$K8S/diag-pod.yaml"
    kubectl wait --for=condition=Ready pod/p2p-nccl-diag -n "$NS" --timeout=900s || true
    kubectl logs -f pod/p2p-nccl-diag -n "$NS" || true
    kubectl logs pod/p2p-nccl-diag -n "$NS" > "/tmp/p2p-nccl-diag-$(date +%Y%m%d-%H%M).log" 2>&1 || true
    echo "== full log saved under /tmp/p2p-nccl-diag-*.log; copy it into the study's results/ folder"
    kubectl delete pod p2p-nccl-diag -n "$NS" --ignore-not-found
    ;;

  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
