# Study 28 (MIG min-cost, fixed load) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `studies/28-g7-4500-mig-min-cost-fixed-load/` so that one Akamas study finds the cheapest deployment (MIG slice + CPU + RAM + vLLM settings) of Qwen3-8B-FP8 that holds a fixed open-loop chat load within the SLO, preceded by a kernel probe and a calibration study.

**Architecture:** Study 26's MIG machinery (privileged `gpu-admin` DaemonSet, node-scoped device plugin, StatefulSet with one pod per MIG instance) serves the model; the Akamas FileConfigurator renders `params.env`, `apply_config.sh` validates it, sets the MIG layout and renders the StatefulSet through `render_statefulset.sh`; `run_test.sh` renders one AIPerf Job per pod through `render_job.sh` (study 27's open-loop generator, at a fixed rate or as a calibration ramp) and watches it with a watchdog whose decisions live in `lib_watchdog.sh`. The pure pieces (renderers, watchdog, probe summary) are unit-tested locally with bash tests; the cluster pieces are exercised by the kernel probe and the calibration study.

**Tech Stack:** bash, sed, python3 (stdlib only on the toolbox), yq v4 and shellcheck (local tests only), kubectl, AIPerf 0.11.0, vLLM v0.29.0, Akamas 3.7.x (`akamas-study-manager` plugin for every Akamas YAML), Prometheus.

**Spec:** `studies/28-g7-4500-mig-min-cost-fixed-load/README.md` (approved 2026-10-02). Read it before any task.

**Executed 2026-10-02 (Tasks 1-7b), then reviewed.** The final review's fixes live in the files, not in this plan's code blocks: `render_statefulset.sh` accepts the Kubernetes pack's FileConfigurator units (`7000m`, `28000M`), `check_offline.py` renders every baseline/preset as the FileConfigurator writes it, `run_test.sh` requires every asked-for replica Ready, `apply_config.sh` fails a trial whose replica restarted or failed its warm-up (`lib_health.sh`), the kernel-probe summary ranks within each group, the create commands use `akamas create -f <file>`, and the memory KPI reads the working set. The files are the reference from Task 8 on.

## Global Constraints

- Study folder: `studies/28-g7-4500-mig-min-cost-fixed-load/`; toolbox path `/work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load`.
- Akamas names: system `vLLM_Benchmark_28_G7_4500_MIG_Min_Cost`, telemetry `Prometheus_28_G7_4500_MIG_Min_Cost`, workflows `28-G7-4500-MIG-Min-Cost-Workflow` and `28-G7-4500-MIG-Min-Cost-Calibration-Workflow`, studies `28-G7-4500-MIG-Min-Cost` and `28-G7-4500-MIG-Min-Cost-Calibration`.
- Component names match `^[a-zA-Z][a-zA-Z0-9_]*$`; step names match `^[a-zA-Z\s][a-zA-Z0-9_\s]*$` (no leading digit, no hyphen, dot or parenthesis).
- Telemetry placeholder keys have no underscore (`$GPUMODEL$`, `$NODEROLE$`): Akamas 3.7 does not substitute keys with one.
- At most 8 KPIs per study (Akamas 3.7 rejects more at `akamas create`). KPI names in Italian, everything else in English.
- Packs: GPU 1.4.0 (`mig_profile`), vLLM 1.12.0, Kubernetes (installed build; `Kubernetes Container` must expose `cpu_limit` / `memory_limit`). Domains must fit inside the pack's.
- Serving: `vllm/vllm-openai:v0.29.0`, `--model=Qwen/Qwen3-8B-FP8`, `--served-model-name=qwen3-8b-mig`, `--max-model-len=4096`, prefix caching off, namespace `gpu-sharing`, StatefulSet `vllm`, headless Service `vllm-headless`.
- Load: AIPerf 0.11.0, ShareGPT cache, gamma arrivals smoothness 4, seed 28 (`vllm-0`) / 29 (`vllm-1`), fixed mode = 60 s warm-up at concurrency 4 + 780 s at R (default 3.3 req/s); ramp mode = 60 s warm-up + 0 -> 12 req/s over 2400 s. Env `AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10` always set.
- Watchdog: TTFT p95 (150 s) > 3000 ms or ITL p95 (150 s) > 600 ms on `vllm-0` for 120 s, armed 150 s after `MEASURED RUN START`; ends the test with exit 0.
- SLO constraints: `vllm_r0.time_to_first_token_p95:max <= 1500`, `vllm_r0.inter_token_latency_p95:max <= 300` (both redefined over `[150s]` in this study's telemetry), `vllm_r0.request_success_rate:avg >= 3.135`.
- Goal (minimize, USD/h): `2.0683 * vllm_r0.active_gpus + 0.04522 * container.container_cpu_limit / 1000 + 0.0043325 * container.container_memory_limit / 1073741824`.
- Windowing (main study): `stability` on `vllm_r0.request_success_rate`, `width` 24, `maxStdDev` 300000000, `when: max`.
- The toolbox image has no `envsubst`: render with `sed` / `python3` only.
- **No push, no `git pull` on the toolbox, no `akamas create` / `start`, no node scaling without the user's explicit OK** (Tasks 8-11 start with a STOP). Commits in this repo are the user's call: each task ends with the commit command to run once the user agrees.
- Never commit secrets; the workflows reference the toolbox key path `/home/akamas/.ssh/id_rsa` only.

## Review Focus

1. A `params.env` with an empty value (Akamas renders `doNotRenderParameters` as `""`) or a leftover `${...}` token must stop `apply_config.sh` before any `kubectl` call touches the GPU (test in Task 3, Step 1).
2. With `mig_profile` `none` there is one pod: `run_test.sh` must start exactly one AIPerf Job, never a neighbour Job against a missing `vllm-1` (test in Task 5, Step 5).
3. The watchdog must not fire on the warm-up's or the cold start's latency: nothing before `MEASURED RUN START` + 150 s may count (tests in Task 5, Step 1).
4. CPU and memory are rendered with units (`<n>m`, `<n>M`) and requests equal limits for cpu, memory and GPU, or the pod is not Guaranteed and the cost the goal reads is not what the pod holds (test in Task 2, Step 1).
5. `FLASH_ATTN` with an fp8 KV cache must never be deployed by the study (vLLM refuses to start, ~8 min lost per trial): rejected by the renderer and excluded by a parameter constraint (tests in Task 2, Step 1 and Task 7, Step 3).

---

### Task 1: Scaffold the study folder and the static files

**Files:**
- Create: `studies/28-g7-4500-mig-min-cost-fixed-load/{akamas/,k8s/,k8s/tests/,k8s/monitoring/,kernel-probe/,infra/,results/.gitkeep}`
- Copy from study 26 (unchanged content, then the edits below): `infra/**`, `k8s/00-pvc.yaml`, `k8s/02-service.yaml`, `k8s/03-hf-secret.yaml`, `k8s/06-hf-cache-pvc.yaml`, `k8s/monitoring/*`
- Modify: `studies/README.md` (recap table), `ROADMAP.md` (section B table, section D study #4 update line), `.gitignore`

These steps are what `/new-study` does (folder shape from `_TEMPLATE`, README, both index rows); the README already exists, written during the design.

**Interfaces:**
- Produces: the folder layout every later task writes into; Service `vllm-headless` (from `02-service.yaml`) used by Task 4's Job URLs.

- [ ] **Step 1: Copy the files**

```bash
cd /Users/stefano/akamas/offline/vllm-benchmark/studies
S26=26-g7-4500-gpu-slice-right-sizing; S28=28-g7-4500-mig-min-cost-fixed-load
mkdir -p $S28/akamas $S28/k8s/tests $S28/k8s/monitoring $S28/kernel-probe $S28/results
touch $S28/results/.gitkeep
cp -R $S26/infra $S28/
cp $S26/k8s/00-pvc.yaml $S26/k8s/02-service.yaml $S26/k8s/03-hf-secret.yaml $S26/k8s/06-hf-cache-pvc.yaml $S28/k8s/
cp $S26/k8s/monitoring/* $S28/k8s/monitoring/
```

- [ ] **Step 2: Point the copies at study 28**

In every copied file, replace the study 26 folder name and title, and nothing else:

```bash
cd /Users/stefano/akamas/offline/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load
grep -rl '26-g7-4500-gpu-slice-right-sizing' infra k8s | xargs sed -i '' 's/26-g7-4500-gpu-slice-right-sizing/28-g7-4500-mig-min-cost-fixed-load/g'
```

Then edit `infra/README.md`'s first paragraph to read: "This study runs on the same node group, namespace and GPU sharing layer as studies 25 and 26 (`../../26-g7-4500-gpu-slice-right-sizing/infra/`); the files below are a copy, as every study here is self-contained, and every script is idempotent. Studies 25, 26 and 28 must not run at the same time."

- [ ] **Step 3: Verify the copy**

Run:
```bash
cd /Users/stefano/akamas/offline/vllm-benchmark/studies
diff -r 26-g7-4500-gpu-slice-right-sizing/infra 28-g7-4500-mig-min-cost-fixed-load/infra | grep '^[<>]' | grep -v -e '26-g7-4500' -e '28-g7-4500' -e 'studies 25' -e 'study 25' ; echo "exit=$?"
shellcheck -S warning 28-g7-4500-mig-min-cost-fixed-load/infra/eks/*.sh 28-g7-4500-mig-min-cost-fixed-load/infra/gpu-sharing/*.sh
grep -rn '26-g7-4500' 28-g7-4500-mig-min-cost-fixed-load/infra 28-g7-4500-mig-min-cost-fixed-load/k8s
```
Expected: the diff shows only the renamed lines and the README paragraph (the `grep -v` leaves nothing beyond the paragraph's lines); shellcheck prints the same warnings as on study 26's copies (none new); the last grep prints nothing except references that point at study 26 on purpose (the README paragraph).

- [ ] **Step 4: Index rows**

In `studies/README.md`, add after study 26's row:

```markdown
| [28-g7-4500-mig-min-cost-fixed-load](28-g7-4500-mig-min-cost-fixed-load/README.md) | TODO (designed 2026-10-02) | | |
```

In `ROADMAP.md` section B's table add a row for study 28 (status `TODO`, link to the README), and in section D "Study #4 — MIG Right-Sizing" append to the **Update 2026-09-30** paragraph: "**Study 28 (designed 2026-10-02):** the inverted shape on the same GPU — minimize the tenant's hourly cost (slice + CPU + RAM, vLLM settings tuned) under a fixed open-loop ShareGPT load (3.3 req/s), Qwen3-8B-FP8 (tight on a `1g.16gb` slice), busy neighbour (`studies/28-g7-4500-mig-min-cost-fixed-load/README.md`)."

- [ ] **Step 5: Ignore what the toolbox renders into the tree**

Append to `.gitignore`, under the "Rendered per trial on the toolbox" block:

```
studies/28-g7-4500-mig-min-cost-fixed-load/k8s/params.env
studies/28-g7-4500-mig-min-cost-fixed-load/k8s/01-statefulset.yaml
studies/28-g7-4500-mig-min-cost-fixed-load/k8s/01-statefulset.yaml.check
```

Run: `git check-ignore -v studies/28-g7-4500-mig-min-cost-fixed-load/k8s/params.env studies/28-g7-4500-mig-min-cost-fixed-load/k8s/01-statefulset.yaml`
Expected: both paths reported as ignored by `.gitignore`.

- [ ] **Step 6: Commit (after the user agrees)**

```bash
git add .gitignore studies/28-g7-4500-mig-min-cost-fixed-load studies/README.md ROADMAP.md
git commit -m "Study 28: scaffold (infra and static k8s from study 26), design README, implementation plan"
```

---

### Task 2: StatefulSet template, params template and `render_statefulset.sh`

**Files:**
- Create: `k8s/params.env.template`, `k8s/01-statefulset_template.yaml`, `k8s/render_statefulset.sh`
- Test: `k8s/tests/test_render_statefulset.sh`

(All paths below are relative to `studies/28-g7-4500-mig-min-cost-fixed-load/`.)

**Interfaces:**
- Produces: `render_statefulset.sh <params.env> <template> <output> <replicas>` — exit 0 and writes `<output>`; exit 2 and writes nothing on an invalid parameter. Env `RENDER_ALLOW_FA_FP8=1` lifts the FLASH_ATTN + fp8 guard (kernel probe only).
- Produces: the `params.env` variable names `MIG_PROFILE CPU_LIMIT MEMORY_LIMIT GPU_MEMORY_UTILIZATION KV_CACHE_DTYPE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS LINEAR_BACKEND ATTENTION_BACKEND` (Task 3 sources them, Task 6 writes them).
- Produces: template tokens `@REPLICAS@ @CPU_LIMIT@ @MEMORY_LIMIT@ @GMU@ @KV_CACHE_DTYPE@ @MAX_NUM_SEQS@ @MAX_NUM_BATCHED_TOKENS@ @LINEAR_BACKEND@ @ATTENTION_BACKEND@`.

- [ ] **Step 1: Write the failing test**

`k8s/tests/test_render_statefulset.sh`:

```bash
#!/bin/bash
# Tests for ../render_statefulset.sh. Run: bash k8s/tests/test_render_statefulset.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
R=$K8S/render_statefulset.sh; T=$K8S/01-statefulset_template.yaml
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
params() {  # a valid params.env, then KEY=VALUE overrides
  cat > "$TMP/params.env" <<'EOF'
MIG_PROFILE=1g.16gb
CPU_LIMIT=2500
MEMORY_LIMIT=12000
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=4096
LINEAR_BACKEND=cutlass
ATTENTION_BACKEND=FLASHINFER
EOF
  local kv
  for kv in "$@"; do sed -i.bak "s|^${kv%%=*}=.*|$kv|" "$TMP/params.env"; done
}
C='.spec.template.spec.containers[0]'

# 1. A valid params.env renders every value; requests == limits with units.
params
if bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 2 >/dev/null 2>&1; then
  Y=$TMP/out.yaml
  [ "$(yq '.spec.replicas' "$Y")" = 2 ] && ok "replicas" || ko "replicas"
  [ "$(yq "$C.resources.requests.cpu" "$Y")" = 2500m ] && [ "$(yq "$C.resources.limits.cpu" "$Y")" = 2500m ] \
    && ok "cpu request = limit = 2500m" || ko "cpu request = limit = 2500m"
  [ "$(yq "$C.resources.requests.memory" "$Y")" = 12000M ] && [ "$(yq "$C.resources.limits.memory" "$Y")" = 12000M ] \
    && ok "memory request = limit = 12000M" || ko "memory request = limit = 12000M"
  [ "$(yq "$C.resources.requests.\"nvidia.com/gpu\"" "$Y")" = 1 ] && [ "$(yq "$C.resources.limits.\"nvidia.com/gpu\"" "$Y")" = 1 ] \
    && ok "gpu request = limit = 1" || ko "gpu request = limit = 1"
  for a in --gpu-memory-utilization=0.92 --kv-cache-dtype=fp8 --max-num-seqs=64 --max-num-batched-tokens=4096 \
           --linear-backend=cutlass --attention-backend=FLASHINFER --served-model-name=qwen3-8b-mig --model=Qwen/Qwen3-8B-FP8; do
    yq "$C.args[]" "$Y" | grep -qx -- "$a" && ok "arg $a" || ko "arg $a"
  done
  grep -q '@[A-Z_]*@' "$Y" && ko "no token left" || ok "no token left"
else
  ko "valid params render"
fi

# 2. Invalid inputs exit 2 and write nothing.
expect_reject() {
  local name=$1; shift
  params "$@"; rm -f "$TMP/out.yaml"
  bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/out.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
expect_reject "unsubstituted token" 'CPU_LIMIT=${container.cpu_limit}'
expect_reject "empty value" 'LINEAR_BACKEND='
expect_reject "MIG profile outside the study" 'MIG_PROFILE=2g.32gb'
expect_reject "non-integer cpu" 'CPU_LIMIT=2.5'
expect_reject "non-integer memory" 'MEMORY_LIMIT=12G'
expect_reject "gpu_memory_utilization not a fraction" 'GPU_MEMORY_UTILIZATION=92'
expect_reject "unknown kv dtype" 'KV_CACHE_DTYPE=fp8_e5m2'
expect_reject "unknown attention backend" 'ATTENTION_BACKEND=XFORMERS'
expect_reject "FLASH_ATTN with fp8 KV" 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
params; rm -f "$TMP/out.yaml"
bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 3 >/dev/null 2>&1; rc=$?
{ [ $rc = 2 ] && [ ! -f "$TMP/out.yaml" ]; } && ok "reject: 3 replicas" || ko "reject: 3 replicas (rc=$rc)"

# 3. Accepted edge cases.
params 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=auto'
bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1 && ok "FLASH_ATTN with auto KV" || ko "FLASH_ATTN with auto KV"
params 'ATTENTION_BACKEND=FLASH_ATTN' 'KV_CACHE_DTYPE=fp8'
RENDER_ALLOW_FA_FP8=1 bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1 \
  && ok "FLASH_ATTN with fp8 KV when the probe allows it" || ko "FLASH_ATTN with fp8 KV when the probe allows it"
params 'MIG_PROFILE=none' 'LINEAR_BACKEND=auto' 'ATTENTION_BACKEND=auto' 'KV_CACHE_DTYPE=auto'
bash "$R" "$TMP/params.env" "$T" "$TMP/out.yaml" 1 >/dev/null 2>&1 && ok "none / auto everywhere" || ko "none / auto everywhere"

echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bash k8s/tests/test_render_statefulset.sh`
Expected: `FAIL valid params render` and every `reject:` line FAIL (the script does not exist yet: `bash` exits 127), non-zero exit.

- [ ] **Step 3: Write `k8s/params.env.template`**

```bash
# Rendered by the Akamas FileConfigurator task into params.env, then sourced by
# apply_config.sh. The parameters are NOT rendered straight into the StatefulSet: the
# MIG profile decides the MIG layout, the device-plugin config and the replica count
# first. Every token must be in the study's parametersSelection (ignoreUnsubstitutedTokens
# false), and render_statefulset.sh refuses a leftover token or an empty value.
MIG_PROFILE=${gpu0.mig_profile}
CPU_LIMIT=${container.cpu_limit}
MEMORY_LIMIT=${container.memory_limit}
GPU_MEMORY_UTILIZATION=${vllm.gpu_memory_utilization}
KV_CACHE_DTYPE=${vllm.kv_cache_dtype}
MAX_NUM_SEQS=${vllm.max_num_seqs}
MAX_NUM_BATCHED_TOKENS=${vllm.max_num_batched_tokens}
LINEAR_BACKEND=${vllm.linear_backend}
ATTENTION_BACKEND=${vllm.attention_backend}
```

- [ ] **Step 4: Write `k8s/01-statefulset_template.yaml`**

```yaml
# vLLM serving for 28-g7-4500-mig-min-cost-fixed-load. Rendered by render_statefulset.sh
# (called by apply_config.sh), NOT by the Akamas FileConfigurator: the replica count
# follows from the MIG profile. Not valid YAML until rendered. Tokens (each between two
# at-signs): REPLICAS CPU_LIMIT MEMORY_LIMIT GMU KV_CACHE_DTYPE MAX_NUM_SEQS
# MAX_NUM_BATCHED_TOKENS LINEAR_BACKEND ATTENTION_BACKEND.
#
# Study 26's StatefulSet with the 8B model and the pod's CPU / memory as parameters:
#   - stable pod names: vllm-0 is the tenant under test, vllm-1 the busy neighbour
#     (mig_profile 1g.16gb only); the Akamas components filter on these exact names;
#   - OrderedReady: vllm-1 starts once vllm-0 is Ready;
#   - nvidia.com/gpu: 1 is the whole GPU (none) or one 1g.16gb slice, as decided by the
#     study's own device plugin (infra/gpu-sharing/).
# requests == limits for cpu, memory and the GPU: Guaranteed QoS, and the limits the goal
# reads (kube-state-metrics) are what the pod really holds.
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vllm
  namespace: gpu-sharing
spec:
  serviceName: vllm-headless
  replicas: @REPLICAS@
  podManagementPolicy: OrderedReady
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      # Kubernetes would otherwise inject VLLM_PORT=tcp://... from the Service, which
      # collides with vLLM's own VLLM_PORT variable and crashes it at startup.
      enableServiceLinks: false
      nodeSelector:
        node-role: llm-serving-g7-4500
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        - key: akamas.io/gpu-sharing
          operator: Exists
          effect: NoSchedule
      terminationGracePeriodSeconds: 30
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.29.0
          args:
            - "--model=Qwen/Qwen3-8B-FP8"
            # A name no other study's telemetry filters on.
            - "--served-model-name=qwen3-8b-mig"
            - "--port=8000"
            - "--host=0.0.0.0"
            - "--no-enable-prefix-caching"
            - "--max-model-len=4096"
            # --- Tuned (parametersSelection) ---
            # Of the device vLLM sees: the whole GPU (none) or the MIG instance.
            - "--gpu-memory-utilization=@GMU@"
            - "--kv-cache-dtype=@KV_CACHE_DTYPE@"
            - "--max-num-seqs=@MAX_NUM_SEQS@"
            - "--max-num-batched-tokens=@MAX_NUM_BATCHED_TOKENS@"
            - "--linear-backend=@LINEAR_BACKEND@"
            - "--attention-backend=@ATTENTION_BACKEND@"
          env:
            - name: HF_HOME
              value: /hf-cache
            - name: VLLM_USE_V2_MODEL_RUNNER
              value: "0"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
                  optional: true
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          resources:
            requests:
              nvidia.com/gpu: "1"
              cpu: "@CPU_LIMIT@m"
              memory: "@MEMORY_LIMIT@M"
            limits:
              nvidia.com/gpu: "1"
              cpu: "@CPU_LIMIT@m"
              memory: "@MEMORY_LIMIT@M"
          volumeMounts:
            - name: hf-cache
              mountPath: /hf-cache
            - name: shm
              mountPath: /dev/shm
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
            failureThreshold: 120   # 20 min: cold image pull (~10 GB) + 8B download + graphs
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
            failureThreshold: 3
      volumes:
        # hostPath, not a PVC (study 26): the node group spans three AZs and an EBS volume
        # would pin the study to one. A node scaled up from 0 starts empty: the first start
        # downloads the model (~9-10 GiB); vllm-1 reuses what vllm-0 fetched (OrderedReady).
        - name: hf-cache
          hostPath:
            path: /var/lib/hf-cache-study28
            type: DirectoryOrCreate
        # Memory-backed: counts against the container's memory limit, so kept small (TP 1
        # uses almost no shared memory).
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 1Gi
```

- [ ] **Step 5: Write `k8s/render_statefulset.sh`**

```bash
#!/bin/bash
# Validates params.env and renders the StatefulSet template (28-g7-4500-mig-min-cost-fixed-load).
# Usage: render_statefulset.sh <params.env> <template> <output> <replicas 1|2>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
# RENDER_ALLOW_FA_FP8=1 lifts the FLASH_ATTN + fp8 guard: the kernel probe uses it to check
# that guard on this GPU; the study never sets it.
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 4 ] || die "usage: render_statefulset.sh <params.env> <template> <output> <replicas>"
PARAMS=$1 TEMPLATE=$2 OUT=$3 REPLICAS=$4
[ -f "$PARAMS" ] || die "no params file $PARAMS"
[ -f "$TEMPLATE" ] || die "no template $TEMPLATE"
# shellcheck disable=SC2016  # a literal ${ is what an unsubstituted Akamas token looks like
if grep -q '\${' "$PARAMS"; then
  die "params.env still has unsubstituted tokens (a parameter is missing from parametersSelection): $(grep '\${' "$PARAMS" | tr '\n' ' ')"
fi
# shellcheck disable=SC1090
source "$PARAMS"
for v in MIG_PROFILE CPU_LIMIT MEMORY_LIMIT GPU_MEMORY_UTILIZATION KV_CACHE_DTYPE MAX_NUM_SEQS \
         MAX_NUM_BATCHED_TOKENS LINEAR_BACKEND ATTENTION_BACKEND; do
  [ -n "${!v:-}" ] || die "$v is empty in params.env (doNotRenderParameters renders an empty string, not the token)"
done
[[ "$REPLICAS" =~ ^[12]$ ]] || die "replicas '$REPLICAS' is not 1 or 2"
[[ "$MIG_PROFILE" =~ ^(none|1g\.16gb)$ ]] || die "mig_profile '$MIG_PROFILE' is not none or 1g.16gb"
for v in CPU_LIMIT MEMORY_LIMIT MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v='${!v}' is not an integer"
done
[[ "$GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] || die "gpu_memory_utilization '$GPU_MEMORY_UTILIZATION' is not a fraction"
[[ "$KV_CACHE_DTYPE" =~ ^(auto|fp8)$ ]] || die "kv_cache_dtype '$KV_CACHE_DTYPE' is not auto or fp8"
[[ "$LINEAR_BACKEND" =~ ^[a-z0-9_]+$ ]] || die "linear_backend '$LINEAR_BACKEND' is not a backend name"
[[ "$ATTENTION_BACKEND" =~ ^(auto|FLASHINFER|FLASH_ATTN|TRITON_ATTN)$ ]] || die "attention_backend '$ATTENTION_BACKEND' is not auto, FLASHINFER, FLASH_ATTN or TRITON_ATTN"
# FlashAttention 2 rejects an fp8 KV cache (vLLM 0.29.0 fa_utils.py: fp8 needs FA3 on SM 9.x
# or FA4 on SM 10.x). The kernel probe checks it on SM 12.0; until then, refuse the pair
# instead of losing ~8 min to a vLLM that does not start.
if [ "$ATTENTION_BACKEND" = FLASH_ATTN ] && [ "$KV_CACHE_DTYPE" = fp8 ] && [ "${RENDER_ALLOW_FA_FP8:-0}" != 1 ]; then
  die "FLASH_ATTN with kv_cache_dtype fp8 (FlashAttention 2 rejects an fp8 KV cache)"
fi
sed -e "s|@REPLICAS@|$REPLICAS|g" \
    -e "s|@CPU_LIMIT@|$CPU_LIMIT|g" \
    -e "s|@MEMORY_LIMIT@|$MEMORY_LIMIT|g" \
    -e "s|@GMU@|$GPU_MEMORY_UTILIZATION|g" \
    -e "s|@KV_CACHE_DTYPE@|$KV_CACHE_DTYPE|g" \
    -e "s|@MAX_NUM_SEQS@|$MAX_NUM_SEQS|g" \
    -e "s|@MAX_NUM_BATCHED_TOKENS@|$MAX_NUM_BATCHED_TOKENS|g" \
    -e "s|@LINEAR_BACKEND@|$LINEAR_BACKEND|g" \
    -e "s|@ATTENTION_BACKEND@|$ATTENTION_BACKEND|g" \
    "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then
  LEFT=$(grep -o '@[A-Z_]*@' "$OUT.tmp" | sort -u | tr '\n' ' '); rm -f "$OUT.tmp"
  die "rendered StatefulSet still has tokens: $LEFT"
fi
mv "$OUT.tmp" "$OUT"
```

- [ ] **Step 6: Run the test to verify it passes, and lint**

Run: `bash k8s/tests/test_render_statefulset.sh && shellcheck -x -S warning k8s/render_statefulset.sh k8s/tests/test_render_statefulset.sh`
Expected: every line `ok`, `0 failure(s)`, shellcheck silent.

Then validate a rendered manifest against the real API server (no mutation: `--dry-run=server` checks the quantities `2500m` / `12000M` and the StatefulSet's immutable fields against study 26's existing `vllm` StatefulSet in `gpu-sharing`):

```bash
printf 'MIG_PROFILE=1g.16gb\nCPU_LIMIT=2500\nMEMORY_LIMIT=12000\nGPU_MEMORY_UTILIZATION=0.92\nKV_CACHE_DTYPE=fp8\nMAX_NUM_SEQS=64\nMAX_NUM_BATCHED_TOKENS=4096\nLINEAR_BACKEND=cutlass\nATTENTION_BACKEND=FLASHINFER\n' > /tmp/p28.env
bash k8s/render_statefulset.sh /tmp/p28.env k8s/01-statefulset_template.yaml /tmp/sts28.yaml 2
kubectl apply --dry-run=server -f /tmp/sts28.yaml
```
Expected: `statefulset.apps/vllm configured (server dry run)` (or `created` if study 26's StatefulSet is gone). An error on an immutable field means the selector / serviceName / podManagementPolicy drifted from study 26's: fix the template, not the cluster.

- [ ] **Step 7: Commit (after the user agrees)**

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/k8s/{params.env.template,01-statefulset_template.yaml,render_statefulset.sh,tests/test_render_statefulset.sh}
git commit -m "Study 28: StatefulSet template with CPU/memory/kernel parameters and its validating renderer"
```

---

### Task 3: `apply_config.sh`

**Files:**
- Create: `k8s/apply_config.sh` (copy of study 26's, then the edits below)
- Test: `k8s/tests/test_apply_config_guard.sh`, `k8s/tests/stub/kubectl`

**Interfaces:**
- Consumes: `render_statefulset.sh` (Task 2).
- Produces: `apply_config.sh` with env overrides `STUDY_DIR`, `PARAMS`, `RENDERED`, `REPLICAS_OVERRIDE` (Task 6 uses all four). Exit 0 = every replica Ready and warmed up; 2 = invalid parameters (GPU untouched); 3-5 = cluster state errors (as study 26).
- Produces: `k8s/tests/stub/kubectl`, reused by Task 5's test.

- [ ] **Step 1: Write the stub and the failing test**

`k8s/tests/stub/kubectl` (executable):

```bash
#!/bin/bash
# Stub kubectl for the apply_config.sh / run_test.sh tests. Logs every call to $STUB_LOG and
# answers the calls those scripts make. STUB_REPLICAS: what `get sts vllm` reports ready.
echo "kubectl $*" >> "$STUB_LOG"
args="$*"
case "$args" in
  *"get sts vllm"*readyReplicas*) echo "${STUB_REPLICAS:-1}" ;;
  *"get pods"*"app=vllm"*uid*) echo "vllm-0:uid0:0" ;;
  *"get job aiperf-mig-r"*succeeded*) echo 1 ;;
  *"get job aiperf-mig-r"*failed*) echo "" ;;
  *"apply -f "*) f=${args##*-f }; echo "applied $(grep -m1 '^  name: ' "$f" | awk '{print $2}')" >> "$STUB_LOG" ;;
  logs*|*" logs "*) echo "MEASURED RUN START" ;;
  exec*|*" exec "*) echo 5 ;;
  *) : ;;
esac
exit 0
```

`k8s/tests/test_apply_config_guard.sh`:

```bash
#!/bin/bash
# apply_config.sh must refuse an invalid params.env BEFORE any kubectl call (Review Focus 1).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); STUDY=$(dirname "$K8S")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
run_with() {  # $1 name, $2 params.env content
  printf '%s\n' "$2" > "$TMP/params.env"; : > "$TMP/kubectl.log"
  PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUDY_DIR="$STUDY" PARAMS="$TMP/params.env" \
    RENDERED="$TMP/sts.yaml" bash "$K8S/apply_config.sh" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -s "$TMP/kubectl.log" ]; } && ok "$1: exit 2, no kubectl call" \
    || ko "$1: rc=$rc, kubectl calls: $(wc -l < "$TMP/kubectl.log")"
}
GOOD='MIG_PROFILE=1g.16gb
CPU_LIMIT=2500
MEMORY_LIMIT=12000
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE=fp8
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=4096
LINEAR_BACKEND=cutlass
ATTENTION_BACKEND=FLASHINFER'
with() { printf '%s\n' "$GOOD" | sed "$1"; }   # GOOD with one sed edit
run_with "empty value" "$(with 's|^LINEAR_BACKEND=.*|LINEAR_BACKEND=|')"
# shellcheck disable=SC2016
run_with "leftover token" "$(with 's|^CPU_LIMIT=.*|CPU_LIMIT=${container.cpu_limit}|')"
run_with "FLASH_ATTN with fp8" "$(with 's|^ATTENTION_BACKEND=.*|ATTENTION_BACKEND=FLASH_ATTN|')"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `chmod +x k8s/tests/stub/kubectl && bash k8s/tests/test_apply_config_guard.sh`
Expected: three `FAIL` lines (no `apply_config.sh` yet: rc 127).

- [ ] **Step 3: Copy study 26's script**

```bash
cp ../26-g7-4500-gpu-slice-right-sizing/k8s/apply_config.sh k8s/apply_config.sh
```

- [ ] **Step 4: Edit the header and the paths**

Replace the first comment block (lines 2-17, up to `set -euo pipefail`) with:

```bash
# Apply-config step for 28-g7-4500-mig-min-cost-fixed-load (runs on the toolbox host via the
# Akamas workflow's Executor task, after FileConfigurator has rendered params.env).
#
# One experiment = one MIG layout (gpu0.mig_profile) + the pod's CPU / memory + one vLLM
# configuration. Study 26's switching sequence, unchanged:
#   none     MIG off, one replica on the whole GPU
#   1g.16gb  MIG on, both 1g.16gb instances, two replicas: vllm-0 is the tenant under
#            test, vllm-1 the busy neighbour (same configuration, same traffic)
# Changes against study 26: the parameters are validated and the StatefulSet rendered by
# render_statefulset.sh (step 0 validates before anything touches the GPU); every replica
# gets 8 warm-up requests at the end (first-use kernel JIT, see the README); env overrides
# PARAMS / RENDERED / REPLICAS_OVERRIDE for the kernel probe.
#
# Fail-fast by design (set -e, explicit checks): a half-applied mode would benchmark the
# previous experiment's configuration under the current one's name.
```

Replace the variable block (`STUDY_DIR=...` to `RENDERED=...`) with:

```bash
STUDY_DIR=${STUDY_DIR:-/work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load}   # overridable for manual tests
NS=gpu-sharing
PARAMS=${PARAMS:-$STUDY_DIR/k8s/params.env}
TEMPLATE=$STUDY_DIR/k8s/01-statefulset_template.yaml
RENDERED=${RENDERED:-$STUDY_DIR/k8s/01-statefulset.yaml}
RENDER=$STUDY_DIR/k8s/render_statefulset.sh
```

- [ ] **Step 5: Replace step 0**

Replace everything from `# --- 0. Parameters` down to (and including) the `t "mig_profile=$MIG_PROFILE ..."` line with:

```bash
# --- 0. Parameters -------------------------------------------------------------------
# Validate everything BEFORE touching the GPU: render once with one replica into a scratch
# file (render_statefulset.sh exits 2 on a leftover token, an empty value or a value
# outside the study's domains), then source the values for the steps below.
bash "$RENDER" "$PARAMS" "$TEMPLATE" "$RENDERED.check" 1 || die "invalid parameters in $PARAMS"
rm -f "$RENDERED.check"
# shellcheck disable=SC1090
source "$PARAMS"
t "mig_profile=$MIG_PROFILE cpu=${CPU_LIMIT}m memory=${MEMORY_LIMIT}M gpu_memory_utilization=$GPU_MEMORY_UTILIZATION kv_cache_dtype=$KV_CACHE_DTYPE max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS linear_backend=$LINEAR_BACKEND attention_backend=$ATTENTION_BACKEND"
```

- [ ] **Step 6: Separate MIG instances from replicas in step 3 and 4**

In step 3, change `REPLICAS=1; PLUGIN_CONFIG=exclusive; WANT_STATE=Disabled,Disabled` to `INSTANCES=1; PLUGIN_CONFIG=exclusive; WANT_STATE=Disabled,Disabled`, and the two lines

```bash
  REPLICAS=$(H nvidia-smi -L | grep -c "MIG $MIG_PROFILE")
  [ "$REPLICAS" = "$FREE" ] || die "created $REPLICAS $MIG_PROFILE instances, expected $FREE" 4
```

to

```bash
  INSTANCES=$(H nvidia-smi -L | grep -c "MIG $MIG_PROFILE")
  [ "$INSTANCES" = "$FREE" ] || die "created $INSTANCES $MIG_PROFILE instances, expected $FREE" 4
```

Replace `t "   $REPLICAS x $MIG_PROFILE -> $REPLICAS replica(s)"` with:

```bash
# One replica per instance (the GPU always fully used); the kernel probe asks for one.
REPLICAS=${REPLICAS_OVERRIDE:-$INSTANCES}
[[ "$REPLICAS" =~ ^[12]$ ]] && [ "$REPLICAS" -le "$INSTANCES" ] || die "REPLICAS_OVERRIDE=$REPLICAS with $INSTANCES instance(s)" 2
t "   $INSTANCES x $MIG_PROFILE -> $REPLICAS replica(s)"
```

In step 4, change both `"$REPLICAS"` comparisons of the allocatable loop to `"$INSTANCES"` and the error to `die "node advertises nvidia.com/gpu=$N, expected $INSTANCES for $MIG_PROFILE" 5`.

- [ ] **Step 7: Replace the rendering in step 6**

Replace everything from the `# sed, not envsubst` comment through the `grep -q '\$[A-Z_]\{3,\}' "$RENDERED" && die ...` line with:

```bash
bash "$RENDER" "$PARAMS" "$TEMPLATE" "$RENDERED" "$REPLICAS" || die "rendering the StatefulSet failed"
```

- [ ] **Step 8: Warm-up and kernel lines at the end**

Change the startup-summary `grep -E` pattern to `'Selected .*Kernel|Using .*[Bb]ackend|attention backend|linear|Model loading took|Available KV cache memory|GPU KV cache size|Error|Traceback'`.

Insert before the line `# Full logs of every replica, success or failure,`:

```bash
# Warm-up (README "Load"): the first requests a fresh vLLM serves hit first-use kernel JIT
# (TTFT p95 33.6 s in study 25's phase 0). They happen here, so they are > 150 s old (the
# p95 window) when RunTest's measured run starts after pip install and its warm-up.
if [ $ROLLOUT_EXIT -eq 0 ]; then
  for i in $(seq 0 $((REPLICAS - 1))); do
    echo "--- vllm-$i: 8 warm-up requests ---"
    kubectl -n $NS exec vllm-$i -c vllm -- python3 -c '
import json, time, urllib.request, concurrent.futures as cf
def one(i):
    body = {"model": "qwen3-8b-mig", "max_tokens": 64,
            "messages": [{"role": "user", "content": "Write a short story about the number %d." % i}]}
    s = time.perf_counter()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300).read()
    return time.perf_counter() - s
with cf.ThreadPoolExecutor(4) as ex:
    print("warm-up e2e (s):", " ".join("%.2f" % x for x in ex.map(one, range(8))))
' || echo "warning: warm-up requests failed on vllm-$i (RunTest will fail fast if the server is down)"
  done
fi
```

- [ ] **Step 9: Run the test and lint**

Run: `bash k8s/tests/test_apply_config_guard.sh && shellcheck -x -S warning k8s/apply_config.sh && diff <(sed 's/26-g7-4500-gpu-slice-right-sizing/STUDY/' ../26-g7-4500-gpu-slice-right-sizing/k8s/apply_config.sh) <(sed 's/28-g7-4500-mig-min-cost-fixed-load/STUDY/' k8s/apply_config.sh)`
Expected: `0 failure(s)`; shellcheck reports no new warning against study 26's script; the diff shows only Steps 4-8's edits.

- [ ] **Step 10: Commit (after the user agrees)**

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/k8s/apply_config.sh studies/28-g7-4500-mig-min-cost-fixed-load/k8s/tests/{stub/kubectl,test_apply_config_guard.sh}
git commit -m "Study 28: apply_config.sh from study 26 with validation first, warm-up requests and probe overrides"
```

---

### Task 4: AIPerf Job template and `render_job.sh`

**Files:**
- Create: `k8s/05-job_template.yaml`, `k8s/render_job.sh`
- Test: `k8s/tests/test_render_job.sh`

**Interfaces:**
- Produces: `render_job.sh <replica 0|1> <mode fixed|ramp> <rate> <ramp_s> <output>` — exit 0 and writes the Job `aiperf-mig-r<replica>` (label `app: aiperf-mig`); exit 2 and writes nothing on invalid input.
- Produces: the log line `MEASURED RUN START` printed by the Job right before the measured run (Task 5's watchdog arms on it).

- [ ] **Step 1: Write the failing test**

`k8s/tests/test_render_job.sh`:

```bash
#!/bin/bash
# Tests for ../render_job.sh. Run: bash k8s/tests/test_render_job.sh (needs yq v4).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_job.sh
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
has() { grep -qF -- "$2" "$1"; }

bash "$R" 0 fixed 3.3 0 "$TMP/r0.yaml" >/dev/null 2>&1 || ko "fixed r0 renders"
if [ -f "$TMP/r0.yaml" ]; then
  [ "$(yq '.kind' "$TMP/r0.yaml")" = Job ] && ok "valid YAML" || ko "valid YAML"
  [ "$(yq '.metadata.name' "$TMP/r0.yaml")" = aiperf-mig-r0 ] && ok "name r0" || ko "name r0"
  [ "$(yq '.metadata.labels.app' "$TMP/r0.yaml")" = aiperf-mig ] && ok "label" || ko "label"
  has "$TMP/r0.yaml" "--request-rate 3.3 --arrival-pattern gamma --arrival-smoothness 4 --random-seed 28 --benchmark-duration 780" \
    && ok "fixed load args" || ko "fixed load args"
  has "$TMP/r0.yaml" "ramp-duration" && ko "no ramp in fixed mode" || ok "no ramp in fixed mode"
  has "$TMP/r0.yaml" "http://vllm-0.vllm-headless.gpu-sharing.svc.cluster.local:8000" && ok "r0 targets vllm-0" || ko "r0 targets vllm-0"
  has "$TMP/r0.yaml" '[ "make" = make ]' && ok "r0 makes the cache" || ko "r0 makes the cache"
  has "$TMP/r0.yaml" "MEASURED RUN START" && ok "marker" || ko "marker"
  grep -q '@[A-Z_]*@' "$TMP/r0.yaml" && ko "no token left" || ok "no token left"
fi
bash "$R" 1 fixed 3.3 0 "$TMP/r1.yaml" >/dev/null 2>&1 || ko "fixed r1 renders"
if [ -f "$TMP/r1.yaml" ]; then
  [ "$(yq '.metadata.name' "$TMP/r1.yaml")" = aiperf-mig-r1 ] && ok "name r1" || ko "name r1"
  has "$TMP/r1.yaml" "--random-seed 29" && ok "r1 seed 29" || ko "r1 seed 29"
  has "$TMP/r1.yaml" "http://vllm-1.vllm-headless.gpu-sharing.svc.cluster.local:8000" && ok "r1 targets vllm-1" || ko "r1 targets vllm-1"
  has "$TMP/r1.yaml" '[ "wait" = make ]' && ok "r1 waits for the cache" || ko "r1 waits for the cache"
fi
bash "$R" 0 ramp 12 2400 "$TMP/ramp.yaml" >/dev/null 2>&1 || ko "ramp renders"
[ -f "$TMP/ramp.yaml" ] && has "$TMP/ramp.yaml" "--request-rate 12 --request-rate-ramp-duration 2400 --arrival-pattern gamma --arrival-smoothness 4 --random-seed 28 --benchmark-duration 2400" \
  && ok "ramp load args" || ko "ramp load args"
reject() {
  local name=$1; shift; rm -f "$TMP/x.yaml"
  bash "$R" "$@" "$TMP/x.yaml" >/dev/null 2>&1; local rc=$?
  { [ $rc = 2 ] && [ ! -f "$TMP/x.yaml" ]; } && ok "reject: $name" || ko "reject: $name (rc=$rc)"
}
reject "replica 2" 2 fixed 3.3 0
reject "unknown mode" 0 burst 3.3 0
reject "rate not a number" 0 fixed abc 0
reject "ramp without duration" 0 ramp 12 0
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bash k8s/tests/test_render_job.sh`
Expected: `FAIL fixed r0 renders`, `FAIL fixed r1 renders`, `FAIL ramp renders`, non-zero exit.

- [ ] **Step 3: Write `k8s/05-job_template.yaml`**

```yaml
# AIPerf Job for one vLLM replica of 28-g7-4500-mig-min-cost-fixed-load. Rendered by
# render_job.sh (tokens REPLICA, LOAD_ARGS, CACHE_ROLE between at-signs); one Job per replica, each on its
# own pod through the headless Service (vllm-0 = tenant under test, vllm-1 = neighbour).
#
# ShareGPT replay (study 26's cached inputs.json), OPEN LOOP (study 27's generator):
#   fixed  --request-rate R, gamma arrivals (smoothness 4), 780 s: every configuration
#          receives the same traffic (same seed -> same arrival sequence in every trial)
#   ramp   calibration only: 0 -> R over D s, AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10
#          (study 27: without it the first Poisson wait is drawn at ~0 req/s and the ramp
#          sends nothing for minutes)
# Before the measured run: 60 s at concurrency 4 (discarded). The marker line
# "MEASURED RUN START" arms run_test.sh's watchdog.
# The ShareGPT cache is model-specific (inputs.json embeds the served name); the vllm-0 Job
# makes it once (CACHE_ROLE make), the vllm-1 Job waits for it (CACHE_ROLE wait). Both Jobs
# mount the RWO PVCs from the single system-m8a node.
apiVersion: batch/v1
kind: Job
metadata:
  name: aiperf-mig-r@REPLICA@
  namespace: gpu-sharing
  labels:
    app: aiperf-mig
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: aiperf-mig
    spec:
      restartPolicy: Never
      nodeSelector:
        node-role: system-m8a
      initContainers:
        - name: wait-for-vllm
          image: alpine:3
          command:
            - sh
            - -c
            - |
              until wget -qO- http://vllm-@REPLICA@.vllm-headless.gpu-sharing.svc.cluster.local:8000/health >/dev/null 2>&1; do
                echo "$(date -u +%T) waiting for vllm-@REPLICA@..."
                sleep 10
              done
              echo "vllm-@REPLICA@ is ready."
      containers:
        - name: aiperf
          image: python:3.12-slim
          command: ["sh", "-c"]
          args:
            - |
              set -e
              pip install --quiet aiperf==0.11.0
              MODEL_NAME=qwen3-8b-mig
              TOKENIZER=Qwen/Qwen3-8B-FP8
              URL=http://vllm-@REPLICA@.vllm-headless.gpu-sharing.svc.cluster.local:8000
              OUT=/benchmarks/study28-r@REPLICA@
              rm -rf "$OUT"; mkdir -p "$OUT"
              CACHE_DIR=/benchmarks/sharegpt-cache
              CACHE_FILE=$CACHE_DIR/inputs-$MODEL_NAME.json
              mkdir -p "$CACHE_DIR"
              if [ "@CACHE_ROLE@" = make ] && [ ! -f "$CACHE_FILE" ]; then
                echo "No cached ShareGPT inputs for $MODEL_NAME: generating once (~5 min)..."
                set +e
                aiperf profile --model "$MODEL_NAME" --tokenizer "$TOKENIZER" --url "$URL" \
                  --endpoint-type chat --streaming --public-dataset sharegpt \
                  --concurrency 1 --request-count 1 --ui simple \
                  --output-artifact-dir /tmp/sharegpt-prep
                set -e
                [ -f /tmp/sharegpt-prep/inputs.json ] || { echo "ERROR: prep run did not produce inputs.json" >&2; exit 1; }
                cp /tmp/sharegpt-prep/inputs.json "$CACHE_FILE.tmp" && mv "$CACHE_FILE.tmp" "$CACHE_FILE"
              fi
              i=0
              until [ -f "$CACHE_FILE" ]; do
                [ $i -ge 120 ] && { echo "ERROR: no ShareGPT cache after 20 min" >&2; exit 1; }
                echo "$(date -u +%T) waiting for the ShareGPT cache (made by the vllm-0 Job)..."
                i=$((i + 1)); sleep 10
              done
              if grep -o '"model": *"[^"]*"' "$CACHE_FILE" | grep -qv "\"$MODEL_NAME\""; then
                echo "ERROR: $CACHE_FILE references a model other than $MODEL_NAME" >&2; exit 1
              fi
              echo "$(date -u +%T) warm-up: 60 s at concurrency 4 (discarded)"
              aiperf profile --model "$MODEL_NAME" --tokenizer "$TOKENIZER" --url "$URL" \
                --endpoint-type chat --streaming \
                --input-file "$CACHE_FILE" --custom-dataset-type inputs_json \
                --concurrency 4 --benchmark-duration 60 --ui simple \
                --output-artifact-dir /tmp/warmup > /tmp/warmup.log 2>&1 \
                || { tail -50 /tmp/warmup.log; exit 1; }
              echo "$(date -u +%T) MEASURED RUN START: @LOAD_ARGS@"
              aiperf profile --model "$MODEL_NAME" --tokenizer "$TOKENIZER" --url "$URL" \
                --endpoint-type chat --streaming \
                --input-file "$CACHE_FILE" --custom-dataset-type inputs_json \
                @LOAD_ARGS@ \
                --goodput "time_to_first_token:1500 inter_token_latency:300" \
                --ui simple \
                --output-artifact-dir "$OUT/aiperf-$(date +%s)"
          env:
            - name: AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL
              value: "10"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
                  optional: true
            - name: HF_HOME
              value: /hf-cache
          resources:
            # ~3.3 req/s per Job: far below the ~1.3 cores study 26 measured at ~20 req/s.
            # Two Jobs must fit next to the other pods on system-m8a (3.92 allocatable).
            requests:
              cpu: 500m
              memory: 2Gi
          volumeMounts:
            - name: results
              mountPath: /benchmarks
            - name: hf-cache
              mountPath: /hf-cache
      volumes:
        - name: results
          persistentVolumeClaim:
            claimName: aiperf-results
        - name: hf-cache
          persistentVolumeClaim:
            claimName: hf-cache
```

- [ ] **Step 4: Write `k8s/render_job.sh`**

```bash
#!/bin/bash
# Renders the AIPerf Job for one vLLM replica (28-g7-4500-mig-min-cost-fixed-load).
# Usage: render_job.sh <replica 0|1> <mode fixed|ramp> <rate req/s> <ramp_s> <output>
# Exit codes: 0 rendered; 2 invalid input (nothing written).
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }
[ $# -eq 5 ] || die "usage: render_job.sh <replica> <mode> <rate> <ramp_s> <output>"
REPLICA=$1 MODE=$2 RATE=$3 RAMP_S=$4 OUT=$5
TEMPLATE=$(cd "$(dirname "$0")" && pwd)/05-job_template.yaml
[[ "$REPLICA" =~ ^[01]$ ]] || die "replica '$REPLICA' is not 0 or 1"
[[ "$RATE" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "rate '$RATE' is not a number"
[[ "$RAMP_S" =~ ^[0-9]+$ ]] || die "ramp_s '$RAMP_S' is not an integer"
SEED=$((28 + REPLICA))   # one sequence per slice, the same in every trial
ARRIVALS="--arrival-pattern gamma --arrival-smoothness 4 --random-seed $SEED"
case "$MODE" in
  fixed) LOAD_ARGS="--request-rate $RATE $ARRIVALS --benchmark-duration 780 --benchmark-grace-period 30" ;;
  ramp)  [ "$RAMP_S" -gt 0 ] || die "ramp mode needs ramp_s > 0"
         LOAD_ARGS="--request-rate $RATE --request-rate-ramp-duration $RAMP_S $ARRIVALS --benchmark-duration $RAMP_S --benchmark-grace-period 60" ;;
  *) die "mode '$MODE' is not fixed or ramp" ;;
esac
if [ "$REPLICA" = 0 ]; then CACHE_ROLE="make"; else CACHE_ROLE="wait"; fi
sed -e "s|@REPLICA@|$REPLICA|g" -e "s|@LOAD_ARGS@|$LOAD_ARGS|g" -e "s|@CACHE_ROLE@|$CACHE_ROLE|g" \
  "$TEMPLATE" > "$OUT.tmp"
if grep -q '@[A-Z_]*@' "$OUT.tmp"; then rm -f "$OUT.tmp"; die "unrendered token in $OUT"; fi
mv "$OUT.tmp" "$OUT"
```

- [ ] **Step 5: Run the test to verify it passes, and lint**

Run: `bash k8s/tests/test_render_job.sh && shellcheck -x -S warning k8s/render_job.sh k8s/tests/test_render_job.sh`
Expected: every line `ok`, `0 failure(s)`, shellcheck silent.

Then: `bash k8s/render_job.sh 1 fixed 3.3 0 /tmp/job28.yaml && kubectl apply --dry-run=server -f /tmp/job28.yaml`
Expected: `job.batch/aiperf-mig-r1 created (server dry run)`.

- [ ] **Step 6: Commit (after the user agrees)**

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/k8s/{05-job_template.yaml,render_job.sh,tests/test_render_job.sh}
git commit -m "Study 28: per-replica AIPerf Job, open loop at a fixed rate or as a calibration ramp"
```

---

### Task 5: Watchdog library and `run_test.sh`

**Files:**
- Create: `k8s/lib_watchdog.sh`, `k8s/run_test.sh`
- Test: `k8s/tests/test_lib_watchdog.sh`, `k8s/tests/test_run_test.sh`

**Interfaces:**
- Consumes: `render_job.sh` (Task 4), `k8s/tests/stub/kubectl` (Task 3), the `MEASURED RUN START` marker (Task 4).
- Produces: `lib_watchdog.sh` functions (globals `WD_TTFT_MS WD_ITL_MS WD_HOLD_S WD_ARM_DELAY_S` set by the caller):
  - `wd_over <ttft_ms> <itl_ms>` — 0 if either value is an integer above its threshold; empty = no data.
  - `wd_next_since <now> <over_since> <0 if over now>` — prints the new over-since time (empty when under).
  - `wd_fired <now> <over_since>` — 0 if over for at least `WD_HOLD_S`.
  - `wd_armed <log text>` — 0 once the text contains `MEASURED RUN START`.
  - `wd_ready <now> <marker_seen_at>` — 0 once `WD_ARM_DELAY_S` have passed since the marker.
- Produces: `run_test.sh` (env `RT_MODE RT_RATE RT_RAMP_S RT_WD_* RT_POLL_S RT_STALL_S RT_FIRST_OK_S RT_DEADLINE_S RT_PROM K8S`); exit 0 = Jobs completed or watchdog ended the test; 1 = trial failed; 2 = bad setup.

- [ ] **Step 1: Write the failing watchdog test**

`k8s/tests/test_lib_watchdog.sh`:

```bash
#!/bin/bash
# Tests for ../lib_watchdog.sh (Review Focus 3). Run: bash k8s/tests/test_lib_watchdog.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../lib_watchdog.sh
source "$(dirname "$HERE")/lib_watchdog.sh"
export WD_TTFT_MS=3000 WD_ITL_MS=600 WD_HOLD_S=120 WD_ARM_DELAY_S=150   # read by the sourced library
FAILS=0
t() { local name=$1; shift; if "$@"; then echo "ok   $name"; else echo "FAIL $name"; FAILS=$((FAILS + 1)); fi; }
n() { local name=$1; shift; if "$@"; then echo "FAIL $name"; FAILS=$((FAILS + 1)); else echo "ok   $name"; fi; }
eq() { [ "$1" = "$2" ]; }
n "no data is not over"             wd_over "" ""
n "under both thresholds"           wd_over 3000 600
t "TTFT over"                       wd_over 3001 100
t "ITL over"                        wd_over 100 601
n "non-numeric is no data"          wd_over NaN +Inf
t "starts counting when over"       eq "$(wd_next_since 1000 "" 0)" 1000
t "keeps the first over time"       eq "$(wd_next_since 1100 1000 0)" 1000
t "resets when back under"          eq "$(wd_next_since 1200 1000 1)" ""
n "not fired before the hold"       wd_fired 1119 1000
t "fired at the hold"               wd_fired 1120 1000
n "never fired when under"          wd_fired 5000 ""
n "not armed before the marker"     wd_armed "warm-up: 60 s at concurrency 4"
t "armed by the marker"             wd_armed $'line\n12:00:00 MEASURED RUN START: --request-rate 3.3'
n "not ready without the marker"    wd_ready 5000 ""
n "not ready inside the delay"      wd_ready 1149 1000
t "ready after the delay"           wd_ready 1150 1000
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash k8s/tests/test_lib_watchdog.sh`
Expected: `source` error (no file), every case FAIL or a non-zero exit.

- [ ] **Step 3: Write `k8s/lib_watchdog.sh`**

```bash
# Pure watchdog helpers for run_test.sh (sourced; no kubectl, no network).
# The caller sets WD_TTFT_MS, WD_ITL_MS (ms), WD_HOLD_S and WD_ARM_DELAY_S (s).
# shellcheck shell=bash
WD_MARKER="MEASURED RUN START"

wd_over() {  # $1 TTFT p95 ms, $2 ITL p95 ms (empty or non-integer = no data). 0 if either is over.
  { [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "$1" -gt "$WD_TTFT_MS" ]; } || \
  { [[ "${2:-}" =~ ^[0-9]+$ ]] && [ "$2" -gt "$WD_ITL_MS" ]; }
}

wd_next_since() {  # $1 now, $2 current over-since ("" = under), $3 0 if over now. Prints the new over-since.
  if [ "$3" = 0 ]; then echo "${2:-$1}"; else echo ""; fi
}

wd_fired() {  # $1 now, $2 over-since. 0 once over for at least WD_HOLD_S.
  [ -n "${2:-}" ] && [ $(( $1 - $2 )) -ge "$WD_HOLD_S" ]
}

wd_armed() {  # $1 the vllm-0 Job's log text. 0 once the measured run has started.
  grep -qF "$WD_MARKER" <<<"$1"
}

wd_ready() {  # $1 now, $2 when the marker was first seen ("" = not yet). 0 once the arm delay has passed.
  [ -n "${2:-}" ] && [ $(( $1 - $2 )) -ge "$WD_ARM_DELAY_S" ]
}
```

- [ ] **Step 4: Run the watchdog test to verify it passes**

Run: `bash k8s/tests/test_lib_watchdog.sh && shellcheck -x -S warning k8s/lib_watchdog.sh k8s/tests/test_lib_watchdog.sh`
Expected: every line `ok`, `0 failure(s)`.

- [ ] **Step 5: Write the failing `run_test.sh` test**

`k8s/tests/test_run_test.sh`:

```bash
#!/bin/bash
# run_test.sh must start one AIPerf Job per ready replica, and only those (Review Focus 2).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); export K8S
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
run() {  # $1 ready replicas
  : > "$TMP/kubectl.log"
  PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" STUB_REPLICAS=$1 \
    RT_POLL_S=0 RT_PROM=http://127.0.0.1:9 bash "$K8S/run_test.sh" > "$TMP/out.txt" 2>&1
}
run 1; rc=$?
[ $rc = 0 ] && ok "one replica: exit 0" || ko "one replica: exit $rc ($(tail -3 "$TMP/out.txt"))"
grep -q '^applied aiperf-mig-r0$' "$TMP/kubectl.log" && ok "one replica: r0 Job" || ko "one replica: r0 Job"
grep -q '^applied aiperf-mig-r1$' "$TMP/kubectl.log" && ko "one replica: no r1 Job" || ok "one replica: no r1 Job"
run 2; rc=$?
[ $rc = 0 ] && ok "two replicas: exit 0" || ko "two replicas: exit $rc"
[ "$(grep -c '^applied aiperf-mig-r[01]$' "$TMP/kubectl.log")" = 2 ] && ok "two replicas: r0 and r1 Jobs" || ko "two replicas: r0 and r1 Jobs"
run 0; rc=$?
[ $rc = 2 ] && ok "no replica: exit 2" || ko "no replica: exit $rc"
grep -q '^applied ' "$TMP/kubectl.log" && ko "no replica: no Job" || ok "no replica: no Job"
RT_MODE=burst PATH="$HERE/stub:$PATH" STUB_LOG="$TMP/kubectl.log" bash "$K8S/run_test.sh" >/dev/null 2>&1; rc=$?
[ $rc = 2 ] && ok "unknown mode: exit 2" || ko "unknown mode: exit $rc"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 6: Run it to verify it fails**

Run: `bash k8s/tests/test_run_test.sh`
Expected: FAIL lines (no `run_test.sh`: rc 127).

- [ ] **Step 7: Write `k8s/run_test.sh`**

```bash
#!/bin/bash
# RunTest step for 28-g7-4500-mig-min-cost-fixed-load (toolbox, Akamas Executor task).
#
# One AIPerf Job per vLLM replica, each on its own pod through the headless Service:
# vllm-0 is the tenant under test, vllm-1 (mig_profile 1g.16gb only) the busy neighbour
# with the same traffic. Modes (render_job.sh):
#   RT_MODE=fixed (the study)        60 s warm-up, then 780 s at RT_RATE (default 3.3 req/s)
#   RT_MODE=ramp  (calibration)      60 s warm-up, then 0 -> RT_RATE (default 12 req/s) over
#                                    RT_RAMP_S (default 2400 s)
# The trial FAILS (exit 1) if a Job fails, a vLLM pod restarts or is replaced, vllm-0
# completes nothing for RT_STALL_S, or the deadline passes (study 26/27 guards). The
# watchdog ENDS the test with SUCCESS (exit 0) once vllm-0's TTFT p95 (150 s) >
# RT_WD_TTFT_MS or ITL p95 (150 s) > RT_WD_ITL_MS for RT_WD_HOLD_S: the goal is then INVALID
# by its own constraints. It is armed RT_WD_ARM_DELAY_S after the measured run starts, so
# the warm-up is out of its 150 s view.
set -euo pipefail
K8S=${K8S:-/work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load/k8s}
NS=gpu-sharing
MODEL=qwen3-8b-mig
MODE=${RT_MODE:-fixed}
case "$MODE" in
  # Deadlines: pip (~2 min) + one-time ShareGPT prep (~10 min, first trial only) + 60 s
  # warm-up + the run + grace. The workflow's RunTest timeout must stay above them (45m /
  # 65m), or Akamas kills the task before the log dump.
  fixed) RATE=${RT_RATE:-3.3}; DEADLINE_S=${RT_DEADLINE_S:-2100} ;;
  ramp)  RATE=${RT_RATE:-12};  DEADLINE_S=${RT_DEADLINE_S:-3300} ;;
  *) echo "error: RT_MODE '$MODE' is not fixed or ramp" >&2; exit 2 ;;
esac
RAMP_S=${RT_RAMP_S:-2400}
WD_TTFT_MS=${RT_WD_TTFT_MS:-3000}; WD_ITL_MS=${RT_WD_ITL_MS:-600}
WD_HOLD_S=${RT_WD_HOLD_S:-120}; WD_ARM_DELAY_S=${RT_WD_ARM_DELAY_S:-150}
POLL_S=${RT_POLL_S:-15}; STALL_S=${RT_STALL_S:-900}; FIRST_OK_S=${RT_FIRST_OK_S:-1500}
PROGRESS_EVERY_S=${RT_PROGRESS_EVERY_S:-60}
PROM=${RT_PROM:-http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090}
TMP=$(mktemp -d /tmp/run-test-28-XXXXXX)
# shellcheck source=lib_watchdog.sh
source "$K8S/lib_watchdog.sh"

pods_state() {  # "name:uid:restarts" for every vLLM replica, sorted
  kubectl get pods -n "$NS" -l app=vllm \
    -o jsonpath='{range .items[*]}{.metadata.name}:{.metadata.uid}:{.status.containerStatuses[0].restartCount}{"\n"}{end}' 2>/dev/null | sort
}
r0_successes() {  # requests vllm-0 completed so far (empty if unreadable)
  kubectl exec -n "$NS" vllm-0 -c vllm -- python3 -c \
    "import urllib.request;print(int(sum(float(l.split()[-1]) for l in urllib.request.urlopen('http://127.0.0.1:8000/metrics',timeout=5).read().decode().splitlines() if l.startswith('vllm:request_success_total'))))" 2>/dev/null || true
}
p95_ms() {  # $1 vLLM histogram name: vllm-0's p95 over 150 s in ms (empty if no data)
  python3 - "$PROM" "$1" "$MODEL" <<'EOF' 2>/dev/null || true
import json, sys, urllib.parse, urllib.request
prom, h, model = sys.argv[1:4]
q = ('histogram_quantile(0.95, sum by(le)(rate(vllm:%s_bucket{model_name="%s",pod="vllm-0"}[150s])))*1000'
     % (h, model))
r = json.load(urllib.request.urlopen(prom + '/api/v1/query?' + urllib.parse.urlencode({'query': q}), timeout=10))
v = r['data']['result']
if v and v[0]['value'][1] not in ('NaN', '+Inf'):
    print(int(float(v[0]['value'][1])))
EOF
}

REPLICAS=$(kubectl -n "$NS" get sts vllm -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)
if ! [[ "${REPLICAS:-}" =~ ^[12]$ ]]; then
  echo "error: expected 1 or 2 ready vLLM replicas, found '${REPLICAS:-none}' (Apply config did not leave a server up)" >&2
  exit 2
fi
STATE_BEFORE=$(pods_state)
echo "vLLM replicas before the test ($REPLICAS):"; echo "$STATE_BEFORE" | sed 's/^/  /'
echo "mode=$MODE rate=$RATE ramp_s=$RAMP_S; watchdog: TTFT p95 > $WD_TTFT_MS ms or ITL p95 > $WD_ITL_MS ms for $WD_HOLD_S s, armed $WD_ARM_DELAY_S s after the measured run starts"

kubectl -n "$NS" delete job -l app=aiperf-mig --ignore-not-found --wait=true
JOBS=""
for i in $(seq 0 $((REPLICAS - 1))); do
  bash "$K8S/render_job.sh" "$i" "$MODE" "$RATE" "$RAMP_S" "$TMP/job-r$i.yaml"
  kubectl apply -f "$TMP/job-r$i.yaml"
  JOBS="$JOBS aiperf-mig-r$i"
done

set +e
FAIL_REASON=""; WATCHDOG=""
T0=$SECONDS; LAST_OK=""; LAST_PROGRESS=$SECONDS; NEXT_PROGRESS=$SECONDS; MARKER_AT=""; OVER_SINCE=""
while true; do
  DONE=0
  for j in $JOBS; do
    S=$(kubectl get job "$j" -n "$NS" -o jsonpath='{.status.succeeded}' 2>/dev/null)
    F=$(kubectl get job "$j" -n "$NS" -o jsonpath='{.status.failed}' 2>/dev/null)
    [ "${F:-0}" -ge 1 ] && FAIL_REASON="the AIPerf job $j failed"
    [ "${S:-0}" -ge 1 ] && DONE=$((DONE + 1))
  done
  [ -n "$FAIL_REASON" ] && break
  [ "$DONE" -eq "$REPLICAS" ] && break
  STATE_NOW=$(pods_state)
  if [ "$STATE_NOW" != "$STATE_BEFORE" ]; then
    FAIL_REASON="a vLLM replica restarted or was replaced during the test (before: $(echo "$STATE_BEFORE" | tr '\n' ' ')/ now: $(echo "${STATE_NOW:-none}" | tr '\n' ' '))"; break
  fi
  if [ "$SECONDS" -ge "$NEXT_PROGRESS" ]; then
    NEXT_PROGRESS=$((SECONDS + PROGRESS_EVERY_S))
    OK=$(r0_successes)
    if [[ "$OK" =~ ^[0-9]+$ ]]; then
      if [ "$OK" != "$LAST_OK" ]; then LAST_OK=$OK; LAST_PROGRESS=$SECONDS; fi
      if [ "$OK" -gt 0 ] && [ $((SECONDS - LAST_PROGRESS)) -ge "$STALL_S" ]; then
        FAIL_REASON="stalled: vllm-0 completed no request for $((SECONDS - LAST_PROGRESS)) s (at $OK)"; break
      fi
      if [ "$OK" -eq 0 ] && [ $((SECONDS - T0)) -ge "$FIRST_OK_S" ]; then
        FAIL_REASON="stalled: vllm-0 completed no request $((SECONDS - T0)) s after the start"; break
      fi
    fi
  fi
  # --- Watchdog on vllm-0 ---
  if [ -z "$MARKER_AT" ] && wd_armed "$(kubectl logs job/aiperf-mig-r0 -n "$NS" -c aiperf --tail=400 2>/dev/null)"; then
    MARKER_AT=$SECONDS
    echo "$(date -u +%T) measured run started; watchdog armed in $WD_ARM_DELAY_S s"
  fi
  if wd_ready "$SECONDS" "$MARKER_AT"; then
    TT=$(p95_ms time_to_first_token_seconds); IT=$(p95_ms inter_token_latency_seconds)
    if wd_over "$TT" "$IT"; then NOW_OVER=0; else NOW_OVER=1; fi
    NEW_SINCE=$(wd_next_since "$SECONDS" "$OVER_SINCE" "$NOW_OVER")
    if [ -z "$OVER_SINCE" ] && [ -n "$NEW_SINCE" ]; then echo "$(date -u +%T) watchdog: over (TTFT p95 ${TT:-n/a} ms, ITL p95 ${IT:-n/a} ms)"; fi
    if [ -n "$OVER_SINCE" ] && [ -z "$NEW_SINCE" ]; then echo "$(date -u +%T) watchdog: back under (TTFT p95 ${TT:-n/a} ms, ITL p95 ${IT:-n/a} ms)"; fi
    OVER_SINCE=$NEW_SINCE
    if wd_fired "$SECONDS" "$OVER_SINCE"; then
      WATCHDOG="TTFT p95 ${TT:-n/a} ms, ITL p95 ${IT:-n/a} ms for $((SECONDS - OVER_SINCE)) s"; break
    fi
  fi
  if [ $((SECONDS - T0)) -ge "$DEADLINE_S" ]; then FAIL_REASON="test timeout (${DEADLINE_S} s)"; break; fi
  sleep "$POLL_S"
done
set -e

if [ -n "$FAIL_REASON" ]; then
  echo "error: $FAIL_REASON, after $((SECONDS - T0)) s. Dumping logs, then failing the trial."
elif [ -n "$WATCHDOG" ]; then
  echo "Watchdog ended the test after $((SECONDS - T0)) s: $WATCHDOG. The test is complete (the goal will be INVALID by its constraints)."
else
  echo "AIPerf job(s) completed after $((SECONDS - T0)) s."
fi
for j in $JOBS; do
  echo "--- job/$j: full logs ---"
  kubectl logs "job/$j" -n "$NS" --all-containers --tail=-1 || true
done
if [ -n "$FAIL_REASON" ]; then
  echo "--- vLLM replicas at failure ---"
  kubectl get pods -n "$NS" -l app=vllm -o wide || true
  for p in $(kubectl get pods -n "$NS" -l app=vllm -o name 2>/dev/null); do
    echo "--- $p: last 200 lines (current and previous container) ---"
    kubectl logs -n "$NS" "$p" --tail=200 || true
    kubectl logs -n "$NS" "$p" --tail=200 --previous 2>/dev/null || true
  done
  kubectl -n "$NS" delete job -l app=aiperf-mig --ignore-not-found --wait=false || true
  rm -rf "$TMP"; exit 1
fi
if [ -n "$WATCHDOG" ]; then  # logs dumped: now stop the load
  kubectl -n "$NS" delete job -l app=aiperf-mig --ignore-not-found --wait=false || true
fi
rm -rf "$TMP"
exit 0
```

- [ ] **Step 8: Run both tests and lint**

Run: `bash k8s/tests/test_lib_watchdog.sh && bash k8s/tests/test_run_test.sh && shellcheck -x -S warning k8s/run_test.sh k8s/tests/test_run_test.sh k8s/tests/stub/kubectl`
Expected: `0 failure(s)` twice; shellcheck silent.

- [ ] **Step 9: Commit (after the user agrees)**

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/k8s/{lib_watchdog.sh,run_test.sh,tests/test_lib_watchdog.sh,tests/test_run_test.sh}
git commit -m "Study 28: RunTest with one Job per replica, study 26/27 guards and a tested watchdog"
```

---

### Task 6: Kernel probe

**Files:**
- Create: `kernel-probe/probe.sh`, `kernel-probe/bench_in_pod.py`, `kernel-probe/summarize.py`, `kernel-probe/README.md`
- Test: `kernel-probe/tests/test_summarize.sh`

**Interfaces:**
- Consumes: `apply_config.sh` overrides `STUDY_DIR PARAMS RENDERED REPLICAS_OVERRIDE`, `RENDER_ALLOW_FA_FP8` (Tasks 2-3).
- Produces: `kernel-probe/results/<name>.json` with keys `name linear attention kv started startup_s` and, when started, `bench.summary.{prefill_2k_mean_s,tpot_single_ms,decode_c30_tpot_ms,decode_c30_gen_tok_per_s}`; `results/summary.txt` (Task 9 reads it).

- [ ] **Step 1: Write the failing summary test**

`kernel-probe/tests/test_summarize.sh`:

```bash
#!/bin/bash
# Tests for ../summarize.py on fixture results.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); KP=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mk() {  # name linear attention kv prefill tpot1 tpot30
  printf '{"name":"%s","linear":"%s","attention":"%s","kv":"%s","started":true,"startup_s":300,"bench":{"summary":{"prefill_2k_mean_s":%s,"tpot_single_ms":%s,"decode_c30_tpot_ms":%s,"decode_c30_gen_tok_per_s":900}}}\n' "$@" > "$TMP/$1.json"
}
mk L-auto auto FLASHINFER auto 0.200 20.0 30.0
mk L-fast cutlass FLASHINFER auto 0.180 19.0 29.0
mk L-slow triton FLASHINFER auto 0.300 25.0 40.0
echo '{"name":"L-broken","linear":"deep_gemm","attention":"FLASHINFER","kv":"auto","started":false,"apply_exit":4,"startup_s":90}' > "$TMP/L-broken.json"
OUT=$(python3 "$KP/summarize.py" "$TMP")
FAILS=0
chk() { if grep -qE "$2" <<<"$OUT"; then echo "ok   $1"; else echo "FAIL $1"; FAILS=$((FAILS + 1)); fi; }
chk "fast within 15 %" '^L-fast .* yes$'
chk "auto within 15 %" '^L-auto .* yes$'
chk "slow outside 15 %" '^L-slow .* no$'
chk "broken reported" '^L-broken .*did not start \(apply exit 4\)'
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash kernel-probe/tests/test_summarize.sh`
Expected: four FAIL lines (no `summarize.py`).

- [ ] **Step 3: Write `kernel-probe/summarize.py`**

```python
"""Kernel probe summary for study 28: one row per combination, from results/<name>.json.

'within 15 %' = prefill step AND decode TPOT at 30 sequences both within 15 % of the best
started combination (the README's rule for keeping a backend in the domain).
"""
import glob
import json
import os
import sys

rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.json')))]
ok = [r for r in rows if r.get('started') and r.get('bench')]
best_pf = min((r['bench']['summary']['prefill_2k_mean_s'] for r in ok), default=None)
best_tp = min((r['bench']['summary']['decode_c30_tpot_ms'] for r in ok), default=None)
print('%-14s %-18s %-12s %-5s %7s %9s %9s %10s %s' % (
    'name', 'linear', 'attention', 'kv', 'start_s', 'pf2k_s', 'tpot1_ms', 'tpot30_ms', 'within 15 %'))
for r in rows:
    head = '%-14s %-18s %-12s %-5s' % (r['name'], r['linear'], r['attention'], r['kv'])
    if not (r.get('started') and r.get('bench')):
        print('%s did not start (apply exit %s)' % (head, r.get('apply_exit')))
        continue
    s = r['bench']['summary']
    within = s['prefill_2k_mean_s'] <= 1.15 * best_pf and s['decode_c30_tpot_ms'] <= 1.15 * best_tp
    print('%s %7d %9.3f %9.1f %10.1f %s' % (head, r['startup_s'], s['prefill_2k_mean_s'],
                                            s['tpot_single_ms'], s['decode_c30_tpot_ms'],
                                            'yes' if within else 'no'))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bash kernel-probe/tests/test_summarize.sh`
Expected: four `ok`, `0 failure(s)`.

- [ ] **Step 5: Write `kernel-probe/bench_in_pod.py`**

```python
"""Runs inside vllm-0 of study 28 (kernel probe). Prints one BENCH_RESULT JSON line.

Prefill: a ~2048-token prompt (one scheduler step at max_num_batched_tokens 2048), 1 output
token, mean of 4 after one warm-up: the compute-bound side of the linear kernel.
Decode: 30 concurrent ShareGPT-like requests (~100-token prompts, 256 output tokens): the
study's regime (~25-30 in flight), which fits a 1g.16gb slice's KV even in bf16.
Client-side wall times: they include HTTP and scheduling, so compare combinations, not
absolute numbers.
"""
import json
import random
import statistics as st
import threading
import time
import urllib.request

URL = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'qwen3-8b-mig'
VOCAB = ['alpha', 'river', 'stone', 'quantum', 'market', 'silver', 'engine', 'forest', 'number', 'signal',
         'orange', 'planet', 'memory', 'window', 'garden', 'rocket', 'yellow', 'bridge', 'castle', 'dragon']


def mk(seed, words):  # ~3.9 tokens per word with Qwen3's tokenizer (study 24: 1045 words ~ 4090 tokens)
    rnd = random.Random(seed)
    return ' '.join(rnd.choice(VOCAB) + str(rnd.randint(0, 999)) for _ in range(words))


def post(body):
    return urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                         {'Content-Type': 'application/json'}), timeout=600)


def prefill(seed):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': mk(seed, 523)}], 'max_tokens': 1}
    s = time.perf_counter()
    post(body).read()
    return time.perf_counter() - s


def stream(seed, words, ntok):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': mk(seed, words)}], 'max_tokens': ntok,
            'stream': True, 'ignore_eos': True}
    s = time.perf_counter()
    ttft, n = None, 0
    for line in post(body):
        if line.startswith(b'data:') and b'[DONE]' not in line:
            c = json.loads(line[5:]).get('choices') or []
            if c and (c[0].get('delta') or {}).get('content'):
                n += 1
                if ttft is None:
                    ttft = time.perf_counter() - s
    tot = time.perf_counter() - s
    return {'ttft': ttft, 'tpot': (tot - ttft) / max(n - 1, 1) if ttft else None, 'tokens': n}


out = {}
seed = int(time.time()) % 100000
prefill(seed)
iso = [prefill(seed + 10 + i) for i in range(4)]
out['prefill_2k_s'] = iso
single = [stream(seed + 300 + i, 26, 128) for i in range(3)]
out['single'] = single
time.sleep(2)
res = [None] * 30


def w(i):
    res[i] = stream(seed + 500 + i, 26, 256)


ts = [threading.Thread(target=w, args=(i,)) for i in range(30)]
s = time.perf_counter()
[t.start() for t in ts]
[t.join() for t in ts]
wall = time.perf_counter() - s
okr = [r for r in res if r and r['tokens'] and r['tpot']]
out['decode_c30'] = {'wall_s': wall, 'ok': len(okr)}
out['summary'] = {
    'prefill_2k_mean_s': st.mean(iso),
    'tpot_single_ms': 1000 * st.mean(x['tpot'] for x in single if x['tpot']),
    'decode_c30_tpot_ms': 1000 * st.mean(r['tpot'] for r in okr),
    'decode_c30_gen_tok_per_s': sum(r['tokens'] for r in okr) / wall,
}
print('BENCH_RESULT ' + json.dumps(out))
```

- [ ] **Step 6: Write `kernel-probe/probe.sh`**

```bash
#!/bin/bash
# Kernel probe for 28-g7-4500-mig-min-cost-fixed-load (README "Kernel probe"). Runs on the
# toolbox, outside Akamas, with the GPU node to itself (no study running on it):
#   setsid nohup bash kernel-probe/probe.sh > kernel-probe/probe.log 2>&1 &
# For each combination: write a params.env, run ../k8s/apply_config.sh with ONE replica on a
# 1g.16gb slice (the other slice idle: this ranks kernels, it does not measure capacity),
# then bench_in_pod.py inside vllm-0. KP_COMBOS overrides the list (same 4 columns).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY_DIR=$(dirname "$HERE"); export STUDY_DIR
OUT=${KP_OUT:-$HERE/results}; mkdir -p "$OUT"
NS=gpu-sharing
# name          linear_backend      attention_backend  kv_cache_dtype
COMBOS=${KP_COMBOS:-"
L-auto          auto                FLASHINFER         auto
L-cutlass       cutlass             FLASHINFER         auto
L-fi-cutlass    flashinfer_cutlass  FLASHINFER         auto
L-deepgemm      deep_gemm           FLASHINFER         auto
L-marlin        marlin              FLASHINFER         auto
L-humming       humming             FLASHINFER         auto
L-triton        triton              FLASHINFER         auto
A-fa            auto                FLASH_ATTN         auto
A-triton        auto                TRITON_ATTN        auto
A-auto          auto                auto               auto
K-fi-fp8        auto                FLASHINFER         fp8
K-triton-fp8    auto                TRITON_ATTN        fp8
K-fa-fp8        auto                FLASH_ATTN         fp8
"}
while read -r NAME LB AB KV; do
  [ -n "${NAME:-}" ] || continue
  P=$OUT/$NAME.params.env
  cat > "$P" <<EOF
MIG_PROFILE=1g.16gb
CPU_LIMIT=7000
MEMORY_LIMIT=28000
GPU_MEMORY_UTILIZATION=0.90
KV_CACHE_DTYPE=$KV
MAX_NUM_SEQS=256
MAX_NUM_BATCHED_TOKENS=2048
LINEAR_BACKEND=$LB
ATTENTION_BACKEND=$AB
EOF
  echo "=== $NAME: linear=$LB attention=$AB kv=$KV ($(date -u +%T))"
  T0=$SECONDS
  RENDER_ALLOW_FA_FP8=1 PARAMS=$P RENDERED=$OUT/$NAME.sts.yaml REPLICAS_OVERRIDE=1 \
    bash "$STUDY_DIR/k8s/apply_config.sh" > "$OUT/$NAME.log" 2>&1
  RC=$?
  START_S=$((SECONDS - T0))
  if [ $RC -ne 0 ]; then
    printf '{"name":"%s","linear":"%s","attention":"%s","kv":"%s","started":false,"apply_exit":%d,"startup_s":%d}\n' \
      "$NAME" "$LB" "$AB" "$KV" "$RC" "$START_S" > "$OUT/$NAME.json"
    grep -E 'Error|Traceback|not supported|ValueError' "$OUT/$NAME.log" | tail -5
    continue
  fi
  R=$(kubectl -n $NS exec -i vllm-0 -c vllm -- python3 - < "$HERE/bench_in_pod.py" 2>>"$OUT/$NAME.log" \
      | grep '^BENCH_RESULT ' | cut -d' ' -f2-)
  python3 - "$NAME" "$LB" "$AB" "$KV" "$START_S" "${R:-null}" > "$OUT/$NAME.json" <<'PY'
import json, sys
name, lb, ab, kv, start, res = sys.argv[1:7]
bench = json.loads(res)
print(json.dumps({"name": name, "linear": lb, "attention": ab, "kv": kv, "started": bench is not None,
                  "startup_s": int(start), "bench": bench}))
PY
  grep -E 'Selected .*Kernel|Using .*[Bb]ackend|attention backend' "$OUT/$NAME.log" | head -5 > "$OUT/$NAME.kernels.txt"
  cat "$OUT/$NAME.kernels.txt"
done <<< "$COMBOS"
kubectl -n $NS scale sts vllm --replicas=0
python3 "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
```

- [ ] **Step 7: Write `kernel-probe/README.md`**

```markdown
# kernel-probe/ — study 28

Which `linear_backend` / `attention_backend` / KV dtype combinations start and how fast
they are for Qwen3-8B-FP8 on one `1g.16gb` slice of the RTX PRO 4500 (SM 12.0), vLLM
0.29.0. Decides the two kernel domains of the study (README "Kernel probe").

- `probe.sh` (toolbox): per combination, `../k8s/apply_config.sh` with one replica, then
  `bench_in_pod.py` in `vllm-0`; ends with vLLM at 0 replicas.
- `bench_in_pod.py`: prefill step (~2048-token prompt, mean of 4) and decode TPOT at 30
  concurrent short requests (the study's regime). Client-side wall times.
- `summarize.py results/`: the table in `results/summary.txt`; `within 15 %` = keep in the
  domain.
- `results/` (written by the run): per combination `.json`, `.log` (full apply log),
  `.kernels.txt` (the kernel vLLM reports it selected), `.params.env`, `.sts.yaml`.

Run it with the node to itself: `setsid nohup bash kernel-probe/probe.sh > kernel-probe/probe.log 2>&1 &`
(~1-2 h; the first combination also downloads the model onto the node).
```

- [ ] **Step 8: Syntax checks**

Run: `bash kernel-probe/tests/test_summarize.sh && shellcheck -x -S warning kernel-probe/probe.sh && python3 -m py_compile kernel-probe/bench_in_pod.py kernel-probe/summarize.py && echo compiled`
Expected: `0 failure(s)`, shellcheck silent, `compiled`.

- [ ] **Step 9: Commit (after the user agrees)**

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/kernel-probe
git commit -m "Study 28: kernel probe (linear/attention backends and KV dtype on a 1g.16gb slice)"
```

---

### Task 7: Akamas resources (with the `akamas-study-manager` plugin)

**Files:**
- Create (by the plugin): `akamas/system.yaml`, `akamas/components/{vllm,vllm_r0,vllm_r1,gpu0,container,cluster,cluster_loadtest,container_loadtest}.yaml`, `akamas/telemetry/prometheus.yaml`, `akamas/28-G7-4500-MIG-Min-Cost-Workflow.yaml`, `akamas/28-G7-4500-MIG-Min-Cost-Calibration-Workflow.yaml`, `akamas/28-G7-4500-MIG-Min-Cost.yaml`, `akamas/28-G7-4500-MIG-Min-Cost-Calibration.yaml`, `akamas/README.md`
- Create: `akamas/check_offline.py`

**Interfaces:**
- Consumes: the tokens of `k8s/params.env.template` (Task 2), the scripts' toolbox paths (Tasks 3, 5), study 26's telemetry catalog.
- Produces: the resources Tasks 10-11 create on Akamas; `check_offline.py` (re-run in Task 9 after the domains change).

- [ ] **Step 1: Generate the resources with the plugin**

Invoke `/akamas-study-manager:build` (never hand-write the YAML) with this brief, and let the plugin's schema reference decide field shapes:

- **Base:** study 26's `akamas/` (same system shape, components, telemetry, workflow); every file carries `kind:` (and `system:` where system-scoped) for `akamas create -f`.
- **System** `vLLM_Benchmark_28_G7_4500_MIG_Min_Cost` (description: the README's Objective in two sentences, packs GPU 1.4.0 / vLLM 1.12.0 / Kubernetes).
- **Components** (`prometheus` properties):
  - `vllm` — type `vLLM`, `pod: ^vllm-[0-9]+$`, `model: qwen3-8b-mig`; carries every vLLM parameter.
  - `vllm_r0` — type `vLLM`, `pod: ^vllm-0$`, `model: qwen3-8b-mig`; goal, constraints, windowing.
  - `vllm_r1` — type `vLLM`, `pod: ^vllm-1$`, `model: qwen3-8b-mig`; neighbour KPI only, empty with `none`.
  - `gpu0` — type `GPU`, `gpu: "0"`, `gpumodel: .*RTX PRO 4500.*`; carries `mig_profile`.
  - `container` — type `Kubernetes Container`, `pod: ^vllm-0$`; carries `cpu_limit`, `memory_limit`; cost metrics.
  - `cluster` — type `Kubernetes Cluster`, `noderole: llm-serving-g7-4500`.
  - `cluster_loadtest` — type `Kubernetes Cluster`, `noderole: system-m8a`.
  - `container_loadtest` — type `Kubernetes Container`, `pod: ^aiperf-mig-r[01]-.*`.
- **Telemetry** `Prometheus_28_G7_4500_MIG_Min_Cost`: study 26's catalog unchanged, except `time_to_first_token_p95` and `inter_token_latency_p95`, whose rate window becomes `[150s]` instead of `[$DURATION$]`; the header comment states why (README "Goal": the `_150s` metrics exist only on `vLLM_PD_Topology` in vLLM pack 1.12.0).
- **Workflow** `28-G7-4500-MIG-Min-Cost-Workflow`: `Write config` (FileConfigurator, toolbox, `ignoreUnsubstitutedTokens: false`, source `/work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load/k8s/params.env.template`, target `.../k8s/params.env`, key `/home/akamas/.ssh/id_rsa`), `Apply config` (Executor, `retries: 0`, `timeout: 60m`, `bash .../k8s/apply_config.sh`), `RunTest` (Executor, `retries: 0`, `timeout: 45m`, `bash .../k8s/run_test.sh`).
- **Workflow** `28-G7-4500-MIG-Min-Cost-Calibration-Workflow`: the same three tasks; `RunTest` command `RT_MODE=ramp bash .../k8s/run_test.sh`, `timeout: 65m`.
- **parametersSelection** (both studies, identical):
  - `gpu0.mig_profile` categories `[none, 1g.16gb]`
  - `container.cpu_limit` domain `[2000, 7000]`
  - `container.memory_limit` domain `[6000, 28000]`
  - `vllm.gpu_memory_utilization` domain `[0.80, 0.95]`
  - `vllm.kv_cache_dtype` categories `[auto, fp8]`
  - `vllm.max_num_seqs` domain `[16, 256]`
  - `vllm.max_num_batched_tokens` domain `[1024, 8192]`
  - `vllm.linear_backend` categories `[auto, cutlass, flashinfer_cutlass, deep_gemm, marlin, humming, triton]` (provisional: Task 9 narrows it)
  - `vllm.attention_backend` categories `[FLASHINFER, FLASH_ATTN, TRITON_ATTN]` (provisional: Task 9 narrows it)
- **parameterConstraints** (both): `vllm.attention_backend != "FLASH_ATTN" || vllm.kv_cache_dtype == "auto"` (quoted values: evaluated as a boolean).
- **Study** `28-G7-4500-MIG-Min-Cost`: goal `minimize`, formula `2.0683 * vllm_r0.active_gpus + 0.04522 * container.container_cpu_limit / 1000 + 0.0043325 * container.container_memory_limit / 1073741824`, unit `USD per hour`; constraints `vllm_r0.time_to_first_token_p95:max <= 1500`, `vllm_r0.inter_token_latency_p95:max <= 300`, `vllm_r0.request_success_rate:avg >= 3.135`; windowing `stability`, metric `vllm_r0.request_success_rate`, `width: 24`, `maxStdDev: 300000000`, `when: {metric: vllm_r0.request_success_rate, is: max}`; `numberOfTrials: 1`; KPIs (8, Italian names) `Costo orario` (the goal formula, minimize), `TTFT P95 150s` (`vllm_r0.time_to_first_token_p95`, minimize, aggregation max), `ITL P95 150s` (`vllm_r0.inter_token_latency_p95`, minimize, aggregation max), `Richieste completate` (`vllm_r0.request_success_rate`, maximize), `KV cache in uso` (`vllm_r0.kv_cache_usage_avg`, minimize), `CPU usata` (`container.container_cpu_used`, minimize), `Richieste vicino` (`vllm_r1.request_success_rate`, maximize), `Temperatura GPU` (`gpu0.gpu_temp`, minimize); steps:
  - `baseline` (baseline): none, 7000, 28000, 0.90, auto, 256, 2048, auto, FLASHINFER
  - `half GPU bf16` (preset): 1g.16gb, 7000, 28000, 0.90, auto, 256, 2048, auto, FLASHINFER
  - `half GPU fp8` (preset): 1g.16gb, 7000, 28000, 0.90, fp8, 256, 2048, auto, FLASHINFER
  - `half GPU fp8 lean` (preset): 1g.16gb, 2000, 8000, 0.90, fp8, 256, 2048, auto, FLASHINFER
  - `optimize` (optimize): `numberOfExperiments: 40`
  (value order: mig_profile, cpu_limit, memory_limit, gpu_memory_utilization, kv_cache_dtype, max_num_seqs, max_num_batched_tokens, linear_backend, attention_backend — every parameter in every preset.)
- **Study** `28-G7-4500-MIG-Min-Cost-Calibration`, workflow `28-G7-4500-MIG-Min-Cost-Calibration-Workflow`: goal `maximize` `vllm_r0.request_success_rate` (unit `requests per second`); constraints the two latency ones only; windowing `stability`, metric `vllm_r0.request_success_rate`, `width: 6`, `maxStdDev: 300000000`, `when` max; the same KPIs; steps `baseline` (as above) and `half GPU bf16` (preset, as above); no optimize step.
- **`akamas/README.md`**: study 26's shape — resources table, steps, versions, validation section (filled in Steps 2-3 and Task 10), create commands for both studies from the toolbox (calibration first, then `akamas delete study 28-G7-4500-MIG-Min-Cost-Calibration` only after its data is exported).

- [ ] **Step 2: Write `akamas/check_offline.py`**

```python
"""Offline checks of study 28's Akamas YAML against the repo rules and the local pack checkouts.

Usage: python3 akamas/check_offline.py [--packs ~/akamas/offline/optimization-packs]
Exit 0 if every check passes; prints one line per failure otherwise.
"""
import glob
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
PACKS = os.path.expanduser(sys.argv[sys.argv.index('--packs') + 1] if '--packs' in sys.argv
                           else '~/akamas/offline/optimization-packs')
fails = []


def load(p):
    with open(p) as f:
        return yaml.safe_load(f)


components = {d['name']: d for d in map(load, glob.glob(os.path.join(HERE, 'components', '*.yaml')))}
for name in components:
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        fails.append('component name %s' % name)

# Pack component types: name -> {parameter: domain}
ctypes = {}
for p in glob.glob(os.path.join(PACKS, '*', 'component-types', '*.yaml')):
    d = load(p)
    ctypes[d['name']] = {x['name']: x.get('domain', {}) for x in d.get('parameters', [])}

tokens = set(re.findall(r'\$\{([a-z0-9_]+\.[a-z0-9_]+)\}',
                        open(os.path.join(STUDY, 'k8s', 'params.env.template')).read()))
for sp in glob.glob(os.path.join(HERE, '28-*.yaml')):
    s = load(sp)
    if s.get('kind') != 'study':
        continue
    sel = {x['name']: x for x in s['parametersSelection']}
    for t in sorted(tokens - set(sel)):
        fails.append('%s: template token %s not in parametersSelection' % (s['name'], t))
    for pname, x in sel.items():
        comp, par = pname.split('.')
        ctype = components[comp]['componentType']
        dom = ctypes.get(ctype, {}).get(par)
        if dom is None:
            fails.append('%s: %s not a parameter of %s' % (s['name'], pname, ctype))
            continue
        if 'categories' in x:
            extra = set(map(str, x['categories'])) - set(map(str, dom.get('categories', [])))
            if extra:
                fails.append('%s: %s categories %s not in the pack' % (s['name'], pname, sorted(extra)))
        elif 'domain' in x:
            lo, hi = dom['domain']
            if not (lo <= x['domain'][0] <= x['domain'][1] <= hi):
                fails.append('%s: %s domain %s outside the pack %s' % (s['name'], pname, x['domain'], dom['domain']))
    for st in s.get('steps', []):
        if not re.match(r'^[a-zA-Z\s][a-zA-Z0-9_\s]*$', st['name']):
            fails.append('%s: step name %r' % (s['name'], st['name']))
        if st.get('type') in ('baseline', 'preset') and set(st.get('values', {})) != set(sel):
            fails.append('%s: step %r does not render every parameter' % (s['name'], st['name']))
    if len(s.get('kpis', [])) > 8:
        fails.append('%s: %d KPIs (max 8)' % (s['name'], len(s['kpis'])))
    fa = 'FLASH_ATTN' in map(str, sel.get('vllm.attention_backend', {}).get('categories', []))
    has_c = any('FLASH_ATTN' in c['formula'] for c in s.get('parameterConstraints', []))
    if fa != has_c:
        fails.append('%s: FLASH_ATTN in the domain (%s) but FLASH_ATTN/fp8 constraint present (%s)' % (s['name'], fa, has_c))

tel = open(os.path.join(HERE, 'telemetry', 'prometheus.yaml')).read()
keys = set(re.findall(r'\$([A-Za-z_]+)\$', tel)) - {'DURATION'}
for k in sorted(keys):
    if '_' in k:
        fails.append('telemetry placeholder $%s$ has an underscore' % k)
for line in fails:
    print('FAIL ' + line)
print('%d failure(s)' % len(fails))
sys.exit(1 if fails else 0)
```

- [ ] **Step 3: Run the offline checks**

Run: `python3 akamas/check_offline.py && grep -c 'p95' akamas/telemetry/prometheus.yaml && grep -n '\[150s\]' akamas/telemetry/prometheus.yaml`
Expected: `0 failure(s)`; the `[150s]` grep shows exactly the `time_to_first_token_p95` and `inter_token_latency_p95` queries. If `check_offline.py` flags `container.cpu_limit` / `memory_limit` as missing, the installed Kubernetes pack is older than the local checkout: STOP and tell the user (README "Risks").

- [ ] **Step 4: Commit (after the user agrees)**

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/akamas
git commit -m "Study 28: Akamas system, components, telemetry, workflows and studies (main and calibration)"
```

---

### Task 7b: Local AIPerf dry run of the load flags

Studies 26 and 27 never ran `--request-rate` + `--arrival-pattern gamma` + `--random-seed` with an `inputs_json` dataset in one invocation (26: concurrency + inputs_json; 27: request rate + synthetic data). Check the combination locally before any node time is spent. (Run once while writing this plan, 2026-10-02: `Creating interval generator: pattern=gamma, rate=3.3, smoothness=4.0`, 198 requests in 60 s, first request 0.29 s after the start, interval CV 0.47.)

**Files:**
- Create: `k8s/tests/mock_openai.py`, `k8s/tests/dry_run_aiperf.sh`

**Interfaces:**
- Consumes: `render_job.sh` (Task 4): the dry run extracts the load arguments from the rendered Job, so it tests what the study will run.

- [ ] **Step 1: Write `k8s/tests/mock_openai.py`**

```python
"""Minimal OpenAI-compatible chat endpoint for a local AIPerf dry run (study 28).

Streams 8 content chunks per request with 20 ms between them, answers /health and
/v1/models. Counts requests in /tmp/mock_openai.count (one line per request, with its time).
Usage: python3 mock_openai.py [port]   (default 18000)
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18000
COUNT = '/tmp/mock_openai.count'


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/health'):
            self._send(200, b'ok', 'text/plain')
        elif self.path.startswith('/v1/models'):
            self._send(200, json.dumps({'object': 'list', 'data': [{'id': 'qwen3-8b-mig', 'object': 'model'}]}).encode())
        else:
            self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n) or b'{}')
        with open(COUNT, 'a') as f:
            f.write('%.3f\n' % time.time())
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()

        def chunk(data):
            b = ('data: %s\n\n' % data).encode()
            self.wfile.write(b'%x\r\n%s\r\n' % (len(b), b))
            self.wfile.flush()

        base = {'id': 'mock', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': req.get('model')}
        for _ in range(8):
            chunk(json.dumps(dict(base, choices=[{'index': 0, 'delta': {'content': 'tok '}, 'finish_reason': None}])))
            time.sleep(0.02)
        chunk(json.dumps(dict(base, choices=[{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                              usage={'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18})))
        chunk('[DONE]')
        self.wfile.write(b'0\r\n\r\n')
        self.wfile.flush()


ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
```

- [ ] **Step 2: Write `k8s/tests/dry_run_aiperf.sh`**

```bash
#!/bin/bash
# Local dry run of the study's AIPerf load (fixed mode) against mock_openai.py: AIPerf 0.11.0
# must accept --request-rate + gamma + seed + grace period with an inputs_json dataset, and
# send at R from the first second (no ramp, so no dead time).
# Usage: bash k8s/tests/dry_run_aiperf.sh [duration_s]   (network: pip and the tokenizer)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
D=${1:-60}
W=$(mktemp -d /tmp/aiperf-dry-XXXXXX)
python3 -m venv "$W/venv"; "$W/venv/bin/pip" install --quiet aiperf==0.11.0
rm -f /tmp/mock_openai.count
python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
A="$W/venv/bin/aiperf"; export AIPERF_TIMING_RATE_RAMP_UPDATE_INTERVAL=10
COMMON=(--model qwen3-8b-mig --tokenizer Qwen/Qwen3-8B-FP8 --url http://127.0.0.1:18000 --endpoint-type chat --streaming --ui simple)
# An inputs.json in AIPerf's own format (the Job uses the ShareGPT one, same format).
"$A" profile "${COMMON[@]}" --synthetic-input-tokens-mean 100 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 8 --num-dataset-entries 50 --concurrency 1 --request-count 1 \
  --output-artifact-dir "$W/prep" > "$W/prep.log" 2>&1
rm -f /tmp/mock_openai.count
# The load arguments exactly as render_job.sh writes them, with a shorter duration.
bash "$K8S/render_job.sh" 0 fixed 3.3 0 "$W/job.yaml"
LOAD=$(grep -m1 'MEASURED RUN START: ' "$W/job.yaml" | sed -e 's/.*MEASURED RUN START: //' -e 's/"$//' -e "s/--benchmark-duration 780/--benchmark-duration $D/")
echo "load args: $LOAD"
# shellcheck disable=SC2086  # LOAD is a list of flags
"$A" profile "${COMMON[@]}" --input-file "$W/prep/inputs.json" --custom-dataset-type inputs_json $LOAD \
  --output-artifact-dir "$W/run" > "$W/run.log" 2>&1
grep -h 'Creating interval generator' "$W/run/logs/aiperf.log" | sed 's/.* - INFO - //'
python3 - "$D" <<'PY'
import statistics as st, sys
d = float(sys.argv[1])
t = [float(x) for x in open('/tmp/mock_openai.count')]
iv = [b - a for a, b in zip(t, t[1:])]
print('requests %d in %.0f s = %.2f req/s; interval cv %.2f (gamma k=4: 0.50)' % (len(t), d, len(t) / d, st.pstdev(iv) / st.mean(iv)))
PY
```

- [ ] **Step 3: Run it**

Run: `bash k8s/tests/dry_run_aiperf.sh 60 && shellcheck -x -S warning k8s/tests/dry_run_aiperf.sh`
Expected (~2-3 min, mostly pip): `Creating interval generator: pattern=gamma, rate=3.3, smoothness=4.0` and `requests ~198 in 60 s = ~3.3 req/s; interval cv ~0.5`. A count far below 198 means a dead time or a rejected flag: read `$W/run.log` (keep the temp dir by removing the `rm -rf` from the trap) before going further.

- [ ] **Step 4: Commit (after the user agrees)**

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/k8s/tests/{mock_openai.py,dry_run_aiperf.sh}
git commit -m "Study 28: local AIPerf dry run of the fixed-rate gamma load against a mock endpoint"
```

---

### Task 8: Sync and bring the node up — STOP: ask the user first

**Files:** none in the repo (cluster and toolbox state).

- [ ] **Step 1: Ask the user** for the go-ahead to push, pull on the toolbox, and scale the GPU node group (cost starts: ~3.3 USD/h). Do nothing below without it.

- [ ] **Step 2: Push and pull** (the user pushes, or confirms that you push):

```bash
git push origin master
kubectl -n akamas exec deploy/toolbox -c toolbox -- sh -c 'cd /work/vllm-benchmark && git pull --ff-only && git log --oneline -1'
```
Expected: the toolbox's last commit equals the local `git log --oneline -1`.

- [ ] **Step 3: Scale up and tag**

```bash
AWS_PROFILE=lab aws eks update-nodegroup-config --cluster-name vllm-bench --region us-east-2 \
  --nodegroup-name llm-serving-g7-4500 --scaling-config minSize=0,maxSize=1,desiredSize=1
kubectl wait --for=condition=Ready node -l node-role=llm-serving-g7-4500 --timeout=15m
cd studies/28-g7-4500-mig-min-cost-fixed-load/infra && AWS_PROFILE=lab ./eks/gpu-nodegroup.sh --always-on && ./gpu-sharing/install.sh
```
Expected: node Ready, `AlwaysOn=true on <asg> and its running instance(s)`, install.sh ends with `config=exclusive nvidia.com/gpu=1`. On `InsufficientInstanceCapacity` stop and report to the user (study 17's `infra/eks/gpu-capacity-fallback.sh probe` checks the other AZs and sizes in seconds with a capacity reservation; this study's infra has no copy of it).

- [ ] **Step 4: Node facts the study depends on**

```bash
kubectl get node -l node-role=llm-serving-g7-4500 -o jsonpath='{.items[0].status.allocatable.cpu} {.items[0].status.allocatable.memory}{"\n"}'
kubectl describe node -l node-role=llm-serving-g7-4500 | sed -n '/Allocated resources/,/Events/p'
kubectl -n monitoring get pods -l app.kubernetes.io/name=dcgm-exporter -o wide | grep -c g7 || true
```
Expected: allocatable CPU minus the DaemonSets' requests >= 14 cores (two replicas at 7000 m). If not, lower `container.cpu_limit`'s maximum in both study YAMLs and the README to what fits, then re-run `check_offline.py`. dcgm-exporter has a pod on the g7 node (helm revision 22, since 2026-09-29).

---

### Task 9: Run the kernel probe and fix the kernel domains — STOP: ask the user first

**Files:**
- Modify: `README.md` ("Parameters tuned", "Kernel probe"), both study YAMLs' `vllm.linear_backend` / `vllm.attention_backend` categories and, if needed, `parameterConstraints`
- Create (by the run): `kernel-probe/results/*`

- [ ] **Step 1: Ask the user** for the go-ahead (~1-2 h of node time, no Akamas study involved).

- [ ] **Step 2: Run it on the toolbox**

```bash
kubectl -n akamas exec deploy/toolbox -c toolbox -- sh -c 'mkdir -p /tmp/kp28 && cd /work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load && KP_OUT=/tmp/kp28/results setsid nohup bash kernel-probe/probe.sh > /tmp/kp28/probe.log 2>&1 &'
```
Outside the git checkout on purpose: Step 5 commits these files locally, and the next `git pull` on the toolbox would refuse to overwrite untracked files at the same paths. Note the UTC start time. Follow it: `kubectl -n akamas exec deploy/toolbox -c toolbox -- tail -5 /tmp/kp28/probe.log` every ~10 min. Expected end: the summary table and vLLM at 0 replicas.

- [ ] **Step 3: Bring the results back**

```bash
kubectl -n akamas exec deploy/toolbox -c toolbox -- tar -C /tmp/kp28 -czf - results probe.log | tar -C studies/28-g7-4500-mig-min-cost-fixed-load/kernel-probe -xzf -
cat studies/28-g7-4500-mig-min-cost-fixed-load/kernel-probe/results/summary.txt
```

- [ ] **Step 4: Decide the domains**

- `vllm.linear_backend`: `auto` plus every backend whose `L-*` row says `yes` (at least two categories; if only `auto` qualifies, keep `auto` and the best other one and note it).
- `vllm.attention_backend`: `FLASHINFER` plus every `A-*` backend that says `yes`.
- `K-fa-fp8`: if it started, FLASH_ATTN accepts fp8 on SM 12.0: remove the guard from `render_statefulset.sh` (and its test case), the constraint from both studies and the README sentence; if it did not start, keep all three.
- `.kernels.txt`: if a backend's kernel line shows that vLLM fell back to another kernel, drop it (it would duplicate a point).

- **Memory and CPU floors, measured.** Every probe combination ran at 28000 MB / 7000 m, so the probe window gives vLLM's real footprint for free. With Prometheus port-forwarded (`kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 19090:9090`), query over the probe window (start time from Step 2 to the end): `max_over_time(container_memory_working_set_bytes{namespace="gpu-sharing",pod="vllm-0",container="vllm"}[3h])` and `max_over_time(rate(container_cpu_usage_seconds_total{namespace="gpu-sharing",pod="vllm-0",container="vllm"}[2m])[3h:1m])`. Set `container.memory_limit`'s lower bound to ~1.3x the peak working set plus 1074 MB (the 1 GiB `/dev/shm`, which counts against the limit), rounded up to 500 MB, in both study YAMLs, the README and the `half GPU fp8 lean` preset if it falls below; check that 2000 m is above the peak CPU of the decode test (if not, raise the CPU lower bound to the peak rounded up to 500 m). The optimizer is pushed towards the memory floor by the cost: a guessed floor would make it spend ~8 min per OOMKill.

Write the table and the decisions into the README's "Kernel probe" section, update both study YAMLs, then run `python3 akamas/check_offline.py` and `bash k8s/tests/test_render_statefulset.sh`. Expected: `0 failure(s)` for both.

- [ ] **Step 5: Commit (after the user agrees)**, then push and pull on the toolbox again (Task 8, Step 2):

```bash
git add studies/28-g7-4500-mig-min-cost-fixed-load/{README.md,kernel-probe/results,kernel-probe/probe.log,akamas/28-G7-4500-MIG-Min-Cost.yaml,akamas/28-G7-4500-MIG-Min-Cost-Calibration.yaml,k8s/render_statefulset.sh,k8s/tests/test_render_statefulset.sh}
git commit -m "Study 28: kernel probe results and kernel domains"
```

---

### Task 10: Calibration study and the target rate — STOP: ask the user first

**Files:**
- Modify: `README.md` (calibration results, R), `k8s/run_test.sh` default `RT_RATE` and the main study's success-rate constraint if R changes, `akamas/README.md` (validation)

- [ ] **Step 1: Ask the user** for the go-ahead to `akamas create` and start the calibration study (~2 h).

- [ ] **Step 2: Create and start** (from the toolbox): run the calibration part of `akamas/README.md`'s "Setup & run" block, in its order (`akamas create -f <file>` per file, as `.claude/rules/akamas-yaml.md` requires for files with `kind:`: system, the eight components, telemetry instance, both workflows, the calibration study, `sleep 120`, start), e.g. `kubectl -n akamas exec -it deploy/toolbox -c toolbox -- bash` then paste the block.
Expected: every `create` succeeds. A KPI, domain or step-name rejection is fixed in the YAML, re-checked with `check_offline.py`, committed and pulled before retrying. Wait 2 minutes, then `akamas start study 28-G7-4500-MIG-Min-Cost-Calibration`; if it stays RUNNING with no experiment after 5 min (study 24/27's Airflow DAG issue), delete it, create it again, wait a minute and start again.

- [ ] **Step 3: Check the first trial end to end**

`akamas list experiment 28-G7-4500-MIG-Min-Cost-Calibration` and the RunTest log (`akamas log --dump -s 28-G7-4500-MIG-Min-Cost-Calibration -e 1 -l ERROR,WARN`, never `-d`). Expected: Apply config ends with the warm-up line; RunTest shows `MEASURED RUN START`, the watchdog armed, then either the watchdog or the Job completing; the experiment is FINISHED with a score and every KPI. Also check, on Prometheus over experiment 1: `sum by(finished_reason)(increase(vllm:request_success_total{model_name="qwen3-8b-mig",pod="vllm-0"}[1h]))` — if `abort` or `error` show up, filter the telemetry's `request_success_rate` query to `finished_reason=~"stop|length"` before the main study (final review, Declined-to-judge ruling); and the warm-up TTFT in the Apply config log (should be well under 1.5 s once JIT is done).

- [ ] **Step 4: Decide R with the user**

Read each step's score (max req/s within the SLO on `vllm-0`). Apply the README's decision rule (R between the half and whole capacities; well below; above) and tell the user the two capacities and the proposed R. If R changes from 3.3: set `RT_RATE`'s default in `run_test.sh`, the constraint to `0.95 x R` in `28-G7-4500-MIG-Min-Cost.yaml`, the README; re-run `bash k8s/tests/test_run_test.sh` and `check_offline.py`; commit; push and pull.

- [ ] **Step 5: Record and clean up** — calibration results in the README, `akamas export study 28-G7-4500-MIG-Min-Cost-Calibration studies/28-g7-4500-mig-min-cost-fixed-load/results/calibration-export.tar.gz` (from the toolbox, then copy back), commit after the user agrees.

---

### Task 11: Create and start the study — STOP: ask the user first

**Files:**
- Modify: `README.md` (status RUNNING, start time), `studies/README.md` row, `ROADMAP.md` section B row, `akamas/README.md` (created ids)

- [ ] **Step 1: Ask the user** for the go-ahead (~22 h, ~75 USD of node time).

- [ ] **Step 2: Create and start**

```bash
kubectl -n akamas exec deploy/toolbox -c toolbox -- sh -c '
cd /work/vllm-benchmark/studies/28-g7-4500-mig-min-cost-fixed-load/akamas &&
akamas create -f 28-G7-4500-MIG-Min-Cost.yaml && sleep 120 && akamas start study 28-G7-4500-MIG-Min-Cost'
```
Expected: the study RUNNING with experiment 1 (`baseline`) in progress within 5 min (same DAG workaround as Task 10).

- [ ] **Step 3: First experiment check** — as Task 10, Step 3: the baseline must be VALID; if it is INVALID the target is not reachable and the study is stopped (`akamas finish study 28-G7-4500-MIG-Min-Cost --workspace default`) and reported to the user.

- [ ] **Step 4: Update the records** — README status `RUNNING (started <UTC time>)`, `studies/README.md` row status, ROADMAP section B status; commit after the user agrees. At the end of the study, the `study-recap` skill closes it.
