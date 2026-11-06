# Study 49 — Smallest MIG slice serving 128 users — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build study 49 **locally**, in `studies/49-g7e-dra-mig-smallest-slice/`. The scope is
the scripts, manifests, Akamas YAML and docs, all tested offline. There is no cluster, toolbox
or AWS action until the user approves the g7e capacity probe.

**Architecture:**
- The folder is a copy of study 46, reshaped:
  - the **customer** (StatefulSet `vllm-customer`, 1 replica, tuned flags) runs on a slice of
    profile P;
  - **neighbours** (StatefulSet `vllm-neighbour`, 0/1/3 replicas, vLLM defaults) run on the
    remaining slices of the same profile.
- One Load task drives everything at once: the customer at 128 closed-loop users, each
  neighbour at 64 users per quarter of GPU.
- A tiny `slice-roles` exporter (Python, on the load node) publishes
  `study49_slice_role{GPU_I_ID, role}`. Joined to DCGM, it gives per-slice GPU metrics and the
  customer slice's framebuffer, which is the goal.

**Tech Stack:** bash, kubectl, Python 3 (exporter and parsers), AIPerf 0.11.0, vLLM 0.29.0, DRA
`dra-driver-nvidia-gpu` (DynamicMIG), dcgm-exporter, Akamas 4.1, vLLM pack 1.15.0, GPU pack
1.4.0, Kubernetes pack 1.9.0.

**Spec:** `docs/superpowers/specs/2026-11-06-study-49-g7e-mig-smallest-slice-design.md`

## Global Constraints

- Layouts:

  | Profile | Neighbours | Neighbour users each | Customer slice |
  |---|---|---|---|
  | `none` | 0 | – | whole GPU (claim `gpu-full`) |
  | `2g.48gb` | 1 | 128 | claim `mig-2g48gb` |
  | `1g.24gb` | 3 | 64 | claim `mig-1g24gb` |

- Customer load: 128 users, closed loop, ShareGPT. 20 s warm-up, then `LOAD_SECONDS=600`.
- SLA on the customer:
  - TTFT p95 ≤ 1500 ms and ITL p95 ≤ 300 ms are Akamas constraints;
  - the client-side completed ≥ 95 % check fails the trial (exit 6).
- Neighbours must be busy. The trial fails (exit 6) if any neighbour client has errors ≥ 1 %,
  completed < 95 %, or no result. Any vLLM restart exits 5.
- Goal: minimize `gpu_customer.gpu_fb_total / 1024 + vllm.time_to_first_token_p95 / 1000000`.
  - `gpu_fb_total` is in MiB, so the first term is the GiB of GPU memory in the customer's
    slice.
  - The tie-break is ≤ 0.0015 under the SLA.
- Neighbours run fixed flags: `--gpu-memory-utilization=0.90` plus the fixed model flags.
- Node `llm-serving-g7e-4xl` (g7e.4xlarge), load node `m8a-2xlarge`, namespace `dra-mig-49`,
  model served as `qwen3-8b-dra`.
- Placeholder keys in Akamas component properties have no underscore (memory).
- No branch: commit on master. No push, no toolbox sync, no `akamas create`, no AWS calls.

## Review Focus

1. **The slice-to-GPU_I_ID mapping is wrong or empty.** Expected: apply fails (exit 7) when a
   vLLM pod's GPU instance id cannot be read, rather than publishing a mapping that silently
   scores the wrong slice. Tested in Task 2 (`roles: missing id -> fail`).
2. **`none` layout (no MIG instance).** Expected: the exporter publishes the customer role
   without a `GPU_I_ID` label, so the join matches the whole-GPU series. Tested in Task 2
   (`roles: none layout`).
3. **A neighbour client dies or falls behind.** Expected: exit 6. Tested in Task 3.
4. **Customer misses the SLA.** Expected: not a script failure. Akamas marks
   CONSTRAINTS_VIOLATED, and the scores of the other slices are unaffected. Tested in Task 3
   (`customer slow -> rc 0`).
5. **Watermark rendered without its exponent** (study 46). Expected: values > 0.10 become 0 with
   a WARNING, carried over from study 46. Tested in Task 2.

---

### Task 1: Scaffold from study 46

- Copy `infra/`, `k8s/00-claims.yaml` and `k8s/monitoring/dcgm-exporter-values.yaml`. Change
  the namespace to `dra-mig-49` and the hostPath to `hf-cache-study49`.
- Write `k8s/01-vllm_template.yaml`, adapted from 46's StatefulSet:
  - two StatefulSets, `vllm-customer` (`@TUNED_ARGS@`, 1 replica) and `vllm-neighbour`
    (`@NEIGHBOURS@` replicas, fixed `--gpu-memory-utilization=0.90`);
  - both carry the label `app: vllm` and a `role` label;
  - headless Services `vllm-customer` and `vllm-neighbour`.
- Write `k8s/02-aiperf-clients_template.yaml`: `@CLIENTS@` replicas. Client 0 drives the
  customer; client i drives `vllm-neighbour-(i-1)`.
- Write `k8s/03-slice-roles.yaml`:
  - a ConfigMap `slice-roles` with a `metrics` key;
  - a Deployment on the load node running a Python HTTP server that serves the ConfigMap file
    at `/metrics`;
  - a Service with the label `app: slice-roles`.
- Write `k8s/monitoring/servicemonitor.yaml`: vLLM (`app: vllm`) and slice-roles, 5 s.
- Commit.

### Task 2: apply_config.sh and the slice-roles mapping (TDD)

**Interfaces:**
- `apply_config.sh`:
  - `RENDER_ONLY=1` prints the customer flags, then `layout=<P> neighbours=<n>
    neighbour_users=<u>`.
  - Exit codes: 2 bad params; 3 node or apply failure; 4 crash loop; 5 unschedulable;
    6 AIPerf clients not ready; 7 GPU instance mapping failed.
- `lib_roles.sh`: `roles_metrics LAYOUT CUSTOMER_ID NEIGHBOUR_IDS...` prints the exporter
  file:
  - `study49_slice_role{GPU_I_ID="<id>",role="customer|neighbour"} 1` per slice;
  - for `none`, `study49_slice_role{role="customer"} 1`;
  - it fails (rc 1) when a MIG layout has an empty id or the wrong number of neighbour ids.
- `gpu_instance_id POD`, also in `lib_roles.sh`:
  - runs `nvidia-smi -q` in the pod and prints the `GPU Instance ID` value;
  - prints nothing for a whole GPU.

- [ ] Write `test_apply_config.sh` from 46's rendering tests, plus layout lines:
  - 1g → 3 neighbours / 64;
  - 2g → 1 / 128;
  - none → 0;
  - the watermark guard;
  - bad profile → rc 2.

  Write `test_roles.sh`:
  - 1g layout with 4 ids;
  - 2g layout;
  - none layout without a label;
  - missing id → rc 1;
  - wrong neighbour count → rc 1.

  Run them and watch them FAIL.
- [ ] Implement, then run. Expected: all pass. Commit.

### Task 3: Load task (TDD)

**Interfaces:**
- `lib_load.sh`:
  - `run_all` starts every client at once and prints one line per client:
    `C<i> role=<customer|neighbour> users=<u> ttft_p95=<ms> itl_p95=<ms> requests=<n> errors=<n>`;
  - it reuses 46's CSV parser.
- `lib_checks.sh` provides `check_clients SUMMARY`, which returns rc 1 with a reason when:
  - a neighbour has errors ≥ 1 % of its total, completed < 95 % or NA values;
  - the customer has completed < 95 % or no result.

  The customer's TTFT and ITL are never checked here.
- `run_load.sh`: restart check (exit 5), `run_all`, `check_clients` (exit 6), full AIPerf
  logs, and `RESULT customer ...` / `RESULT neighbours ...` lines.

- [ ] Write `test_checks.sh` cases, then watch them fail:
  - all healthy → 0;
  - customer slow (TTFT 9000) → 0;
  - neighbour errors 2 % → 1;
  - neighbour NA → 1;
  - customer completed 90 % → 1;
  - no customer line → 1.
- [ ] Implement, run all tests and shellcheck. Commit.

### Task 4: Akamas resources (akamas-study-manager plugin)

- **System** `vLLM_Benchmark_49_G7E_MIG_Smallest_Slice`.
- **Components:**

  | Component | Type | Properties |
  |---|---|---|
  | `vllm` | vLLM | pod `^vllm-customer-0$`, all tuned parameters |
  | `vllm_neighbours` | vLLM | pod `^vllm-neighbour-.*` |
  | `container` | Kubernetes Container | customer pod |
  | `gpu0` | GPU | `mig_profile`, whole GPU, slice `.*` |
  | `gpu_customer` | GPU | slice `customer` |
  | `gpu_neighbours` | GPU | slice `neighbour` |

- **Telemetry** `Prometheus_49_G7E_MIG_Smallest_Slice`:
  - 46's vLLM, container and DCGM queries;
  - every DCGM query is joined with
    `* on(GPU_I_ID) group_left max by(GPU_I_ID)(study49_slice_role{role=~"$SLICE$"})`.
- **Workflow** `49-G7E-MIG-Smallest-Slice-Workflow`:

  | Task | Operator | Timeout |
  |---|---|---|
  | Write config | FileConfigurator | – |
  | Apply config | Executor | 60m |
  | Load | Executor | 20m |

- **Study** `49-G7E-MIG-Smallest-Slice`:
  - goal and constraints as in Global Constraints;
  - trim `[90s, 45s]` on Load;
  - 13 parameters (design table);
  - steps: baseline (none, gmu 0.90, defaults), presets 2g and 1g at the same defaults,
    baseline repeat;
  - optimize: 15 experiments, 4 init, 6 max failed;
  - 8 KPIs from the design.
- Not created on Akamas yet. Validation with `akamas create` is pending the user's go.

### Task 5: Docs

- `README.md`, `akamas/README.md` and `infra/README.md` for the study.
- A row in `studies/README.md` (TODO, built locally) and a row in `ROADMAP.md`.
- Commit.
