# k8s/ — the files study 18 runs

The workflow `18-L4-PD-Disaggregation-Workflow` references these by absolute path on the
toolbox (`/work/vllm-benchmark/studies/18-l4-pd-disaggregation-tps-per-gpu/k8s/`), so the
repo must be pulled there before the study starts. Editing a file here changes what the
next trial runs.

## What runs

One pod (Deployment `vllm-pd`, namespace `llm-serving`, node `llm-serving-l4`, all 4 L4s):

```
                     ┌──────────────────── pod vllm-pd ────────────────────────┐
AIPerf ──► Service   │ router :8000 ──► prefill-i :8100+i  (GPU i,   kv_producer) │
         vllm-pd:8000│ (pd_router.py)    │ NIXL (KV pull)                           │
                     │                ──► decode-j  :8200+j (GPU P+j, kv_consumer) │
                     │  engine container: launcher.sh starts every vLLM process    │
                     └──────────────────────────────────────────────────────────────┘
```

Aggregated presets (`pd_prefill_instances = 0`): no prefill processes. The router
round-robins whole requests over the decode replicas.

| File | Role |
|---|---|
| `01-deployment_template.yaml` | Rendered per trial (FileConfigurator): ConfigMap `pd-config` (topology + per-role flags) + Deployment `vllm-pd` |
| `launcher.sh` | Engine container entrypoint: starts P prefill + D decode `vllm serve` processes, one GPU each, and exits if any dies |
| `pd_router.py` | Router: NIXL P/D protocol (vLLM's toy proxy) or round-robin; exports client-side TTFT/ITL/E2E and delivered tokens under vLLM's own metric names, `model_name=qwen3-8b-router` |
| `apply_config.sh` | Workflow task 2: strips unrendered flags, regenerates ConfigMap `pd-scripts` from the two scripts, forces a fresh pod, waits with a crash-loop guard, dumps all logs |
| `run_test_tps.sh` | Workflow task 3: AIPerf Job, full logs, router counters, KV-transfer log lines, restart guard |
| `05-job.yaml` | AIPerf 0.11.0: synthetic 4096 in / 256 out, 6 levels x 600 s |
| `smoke_test.sh` | Phase B, by hand: `diag`, `up <preset>`, `probe`, `calibrate`, `down` |
| `p2p_nccl_diag.py`, `diag-pod.yaml` | `smoke_test.sh diag`: P2P matrix, copy bandwidth direct vs via host, NCCL transport + all-reduce with/without `NCCL_P2P_DISABLE` |
| `02-service.yaml`, `monitoring/servicemonitor.yaml` | Service + scrape of the router and each instance port (label `app: vllm-pd`, not `app: vllm`, so the old `vllm` ServiceMonitor cannot double-scrape the router) |
| `00-pvc.yaml`, `06-hf-cache-pvc.yaml` | AIPerf results / HF cache, same claims as earlier studies |
| `01-pvc-model-cache.yaml` | `vllm-model-cache-qwen3-8b`, 30Gi, own claim |
| `monitoring/dcgm-exporter-values.yaml` | DCGM Exporter pointed at `llm-serving-l4` |

## One-time setup (before the smoke test)

```bash
cd /work/vllm-benchmark/studies/18-l4-pd-disaggregation-tps-per-gpu/k8s
kubectl apply -f 00-pvc.yaml -f 06-hf-cache-pvc.yaml     # existing claims: no-op if present
kubectl apply -f 01-pvc-model-cache.yaml                  # binds on first use (WaitForFirstConsumer)
kubectl apply -f 02-service.yaml
kubectl apply -f monitoring/servicemonitor.yaml
helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring --reuse-values \
  --set nodeSelector.node-role=llm-serving-l4
```

`01-deployment_template.yaml` is never applied by hand. `pd-scripts` needs no manual step,
because `apply_config.sh` (and so `smoke_test.sh up`) creates it.

## Checked offline (2026-09-24)

- **Router:** tested against mock prefill/decode servers in both modes. Checked: the NIXL
  request shape (prefill `max_tokens=1`, `min_tokens` stripped, `kv_transfer_params`
  forwarded), the same `X-Request-Id` on both legs, the usage chunk swallowed or forwarded,
  exact TTFT/ITL/E2E counts and token counters, and `/health` returning 503 when a backend
  dies.
- **Launcher:** tested in the toolbox (bash 5) with stub binaries for 1P1D host-buffer,
  2P2D and aggregated-4: GPU and port layout, `kv_role` and `--kv-transfer-config` only when
  disaggregated. It refuses 3P2D.
- **Template:** rendered with all 10 presets' values. The tokens match `parametersSelection`
  exactly, and `kubectl apply --dry-run=server` accepts the result, the Service, the PVC,
  the Job and the ServiceMonitor. `smoke_test.sh up` renders identically to the study.

- **Telemetry:** all 116 queries of `akamas/telemetry/prometheus.yaml`, with the component
  placeholders substituted, are accepted by the live Prometheus (syntax only, since no
  study-18 series exist yet). dcgm-exporter fills `exported_pod` with the workload pod
  name, so `active_gpus` will match `^vllm-pd-.*`.

Not checkable offline: see the phase-B list in the study README.
