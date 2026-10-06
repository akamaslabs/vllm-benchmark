# vllm-benchmark — Offline Akamas Optimization Studies

Akamas **offline** optimization studies for tuning AI inference workloads (vLLM serving,
and related Kubernetes/GPU infrastructure) on Kubernetes. This repo holds the studies
and the Claude Code tooling for configuring them and analyzing their results — see
`README.md` for the full picture; this file is the short operational contract.

## Akamas version

- **Target: Akamas Studio 4.1.x** from 2026-10-06 (4.1.0, chart 1.9.0-rc14, at
  <https://akamas41.lab.akamas.io>, namespace `akamas-41` on `vllm-bench`; CLI via
  `kubectl -n akamas-41 exec deploy/toolbox -c toolbox -- akamas ...`, SSH via
  `toolbox-ssh vllm-bench akamas-41`). It is the version used at customers. The old 3.7.1
  (<https://akamas.lab.akamas.io>, namespace `akamas`) is kept as it was for the studies run
  there (studies 27, 28, 29 and 1-Gatling were imported into 4.1).
- Docs: `llms.txt` (`https://docs.akamas.io/akamas-docs/llms.txt`) lists versioned trees only
  up to 3.7 (<https://docs.akamas.io/akamas-docs/3.7/>); for 4.1 use the unversioned pages.
  Append `.md` to any URL for raw markdown. The 3.7-specific rules in
  `.claude/rules/akamas-yaml.md` were written on 3.7: re-check one before relying on it on
  4.1 (study 30 was accepted on 4.1 with 8 KPIs, mixed categorical/numeric constraints and
  `when` nested under `stability`).
- **Optimization packs (which component types/parameters/metrics exist — vLLM,
  Kubernetes, GPU, etc.) are managed outside this repo**, via the `akamas-optimization-pack`
  plugin (see the vLLM pack itself at
  <https://gitlab.com/akamas/optimization-packs/vllm>). Check what's installed with
  `akamas list optimization-pack` before modeling a study's System — don't assume.

## Layout

```
studies/README.md  Recap index: one row per study — folder, status, how it was
                    configured, what was observed. Factual record, not a plan.
studies/<name>/  One folder per study, fully self-contained: its own cluster
                 provisioning (infra/), Akamas resources (system, components,
                 telemetry, workflow, study), Kubernetes manifests (k8s/), results,
                 and README.md. Duplication across studies is intentional — see
                 studies/_TEMPLATE/ for the expected shape.
knowledge/       Distilled tuning knowledge (papers, articles, docs alike), shared across
                 studies (index: knowledge/README.md).
ROADMAP.md       Cross-study living plan: hypotheses, backlog, learnings — what to try
                 next (complements studies/README.md, which is what's already been done).
.claude/skills/  study-recap (close out a study). Akamas resource generation
                 (system/component/telemetry/workflow/study) and optimization-pack
                 work are handled by the akamas-study-manager and akamas-optimization-pack
                 plugins, not a local skill — see "Working rules" below.
```

**Cluster/environment provisioning is atomic per study** (decided 2026-07-15,
superseding the earlier "not owned by this repo's tooling" stance): each study's own
`infra/` folder (`eksctl` cluster config + provisioning script + Kubernetes bootstrap
manifests) takes an empty AWS account to a cluster ready for that study's Akamas
resources — see `studies/0-explorative/infra/README.md` for the reference shape, and
`studies/_TEMPLATE/infra/` for what a new study should scaffold. Studies don't share a
cluster config; a future study with different hardware gets its own `infra/eks/
cluster.yaml`.

## Key commands

```bash
akamas create -f studies/<name>/akamas/<dir-or-file>   # every YAML needs the `kind` key
akamas start study "<name>"
akamas list studies / akamas describe study "<name>"
akamas export study "<name>" studies/<name>/results/export.tar.gz
kubectl rollout status deployment/<workload> -n <namespace> --timeout=15m
```

## Working rules

1. **Before planning a study**: read `ROADMAP.md`, `studies/README.md`, and
   `knowledge/README.md`.
2. **For any Akamas configuration** (system, component, telemetry, workflow, study)
   inside a study's `akamas/` folder: use the **`akamas-study-manager`** plugin
   (`/akamas-study-manager:build`) — never write Akamas YAML from memory. To scaffold or
   extend an optimization pack itself, use the **`akamas-optimization-pack`** plugin
   (`/akamas-optimization-pack:build`) instead — but pack lifecycle for the packs this
   repo actually uses is managed outside this repo (see `Akamas version` above); don't
   add pack files under a study.
3. **At the end of a study**: use the **`study-recap`** skill (writes
   `studies/<name>/README.md`, updates the `studies/README.md` recap table and `ROADMAP.md`).
4. **To add tuning knowledge**: use **`/update-knowledge`** for any paper, article, blog
   post, or vendor doc — not paper-only. Never paste a raw source into a study.
5. Plans live in `ROADMAP.md` (versioned), not in Claude's memory.
6. **Don't assume the stack** (load generator, model, hardware) is the same across
   studies — it isn't, by design. Read the specific study's `README.md` for what it
   actually used.
7. **Never commit secrets**: no SSH keys, HF tokens, or passwords in manifests. This
   repo's git history already contains one compromised private key (see `ROADMAP.md`
   security debt) — don't repeat that mistake.
8. Validate Akamas YAML with the CLI before declaring work "done" (see
   `.claude/rules/akamas-yaml.md`).
