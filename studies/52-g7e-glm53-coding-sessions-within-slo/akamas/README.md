# akamas/ — this study's Akamas resources

Generate everything in this folder with the `akamas-study-manager` plugin
(`/akamas-study-manager:build`) — don't hand-write Akamas YAML. It handles the build
order (System → Component → Telemetry → Workflow → Study) and the resource shapes
(`system.yaml`, `components/*.yaml`, `telemetry/*.yaml`, the workflow file,
`study.yaml`).

Optimization packs (which component types/parameters/metrics exist) are **not**
configured here — they're managed outside this repo, via the `akamas-optimization-pack`
plugin against the pack's own repo (e.g.
<https://gitlab.com/akamas/optimization-packs/vllm> for vLLM).

## Components and telemetry: always the full catalog (from 2026-11-16)

Every study models **every component and collects every useful metric**, even the ones the
goal does not use and no parameter touches, so the bottleneck can be found afterwards.
Studies 45, 47 and 48 had no GPU or Kubernetes Node component; start from study 51's
catalog (`studies/51-g7e-tcp-moe-parallelism-users-ramp/akamas/telemetry/prometheus.yaml`)
and add what the installed packs offer (`akamas describe optimization-pack <name>`).

| Component | One per | Metrics, at least |
|---|---|---|
| vLLM | serving deployment | the full catalog, plus the families of the features in use (KV offload, speculative decoding acceptance, combined prefix hit rate) |
| GPU | physical GPU (`gpu0`, `gpu1`, …; MIG: per instance too) | all DCGM: SM / tensor / DRAM active, power, clocks, throttle reasons, violations, XID, PCIe tx/rx, NVLink where present |
| Kubernetes Node | GPU node and load-generator node | CPU, PSI (cpu / memory / io), load average, memory and swap, disk (IOPS, latency, bytes), filesystem, network per interface (`*_details`), TCP retransmits |
| Kubernetes Container | vLLM, load generator | CPU, throttling, PSI, memory, OOM kills, restarts |
| Kubernetes Workload | serving workload | desired / running / ready pods, CPU, memory |
| Kubernetes Cluster | cluster | CPU, memory, PSI, nodes |

- Extra metrics go in the telemetry instance, not in the KPIs (Akamas caps KPIs at 8).
- Metrics no pack has (per-container or per-pod network, `/dev/shm` usage, NCCL, NUMA) are a
  TODO in the study README, raised with whoever manages the packs.
- Before the start, check that every component's queries return data (study 51's smoke:
  111 of 113 queries with data). At the end, export the series from Prometheus to `results/`.

This is a template placeholder — real studies replace this README with their actual
YAML files (this note can stay or go once populated).
