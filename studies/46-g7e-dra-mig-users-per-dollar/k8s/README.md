# k8s/ — study 46

These files run on the akamas-41 toolbox, called by the workflow in `../akamas/`. All of
them work in namespace `dra-mig`.

| File | Role |
|---|---|
| `params.env.template` | FileConfigurator source: the 5 tuned parameters |
| `00-claims.yaml` | Namespace plus ResourceClaimTemplates: `gpu-full`, `mig-2g48gb`, `mig-1g24gb`, `mig-4g96gb` |
| `01-statefulset_template.yaml` | vLLM StatefulSet + headless Service. Each pod gets its own claim. CPU and memory are split evenly across the node |
| `02-aiperf-clients_template.yaml` | One long-lived AIPerf client pod per replica, on `m8a-2xlarge` |
| `apply_config.sh` | **Apply config** task. Deletes the old pods and claims, picks the claim template and replica count from `mig_profile`, renders and applies, waits (fails fast on a crash loop or an unschedulable claim), dumps the logs, restarts dcgm-exporter, waits for the clients |
| `run_ramp.sh` | **Ramp** task. Per-replica concurrency levels, all replicas together, stops at the first SLA failure, writes `level.current` |
| `run_hold.sh` | **Hold** task. Holds `level.current` for 240 s, the window Akamas scores, and dumps the AIPerf logs |
| `lib_load.sh` | Shared by Ramp and Hold: `run_level`, AIPerf CSV parser, per-replica SLA check, restart guard |
| `monitoring/dcgm-exporter-values.yaml` | dcgm-exporter Helm values as deployed (g7e node groups added, `KUBERNETES_ENABLE_DRA=true`) |
| `spike/` | Phase 0 throwaway scripts (claims probe, vLLM up, CPU check) |

Rendered files (`params.env`, `01-statefulset.yaml`, `02-aiperf-clients.yaml`,
`replicas.current`, `level.current`) are git-ignored.
