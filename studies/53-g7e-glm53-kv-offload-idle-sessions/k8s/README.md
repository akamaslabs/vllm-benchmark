# k8s/ — this study's Kubernetes manifests

Whatever this study needs to apply *after* `../infra/`'s cluster-bootstrap layer is
up: the workload under test (e.g. a serving Deployment/Service, or a `.yaml.templ`
template with `${Component.parameter}` tokens for Akamas' FileConfigurator to render),
the PVCs it needs (applied once by `../infra/eks/provision.sh`, not per-trial), the
load-test job/manifest for whichever tool this study uses, and this study's own
monitoring setup (see `monitoring/` below) — **every study is atomic** (decided
2026-07-15), so don't assume any of this is "already covered by shared cluster infra."

- **`monitoring/`** — Helm values for the DCGM Exporter and kube-prometheus-stack,
  plus the `ServiceMonitor` that tells Prometheus to scrape this study's workload's
  `/metrics` endpoint, and (optional) Grafana dashboard JSON. Mirror
  `studies/0-explorative/k8s/monitoring/` and adapt node selectors/labels to this
  study's actual node group names.

**The load generator is a per-study choice** — this repo has used GuideLLM and is
moving to `kubernetes-sigs/inference-perf` (see `ROADMAP.md` Q2 and Section D). Put
whichever manifest this study actually uses here; don't copy another study's job
definition assuming the tool is the same.

This is a template placeholder — real studies replace this README with their actual
manifests (this note can stay or go once populated).
