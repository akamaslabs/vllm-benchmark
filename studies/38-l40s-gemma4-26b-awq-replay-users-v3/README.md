# 38-l40s-gemma4-26b-awq-replay-users-v3

**Status:** DESIGNED (2026-10-15).
**Dates:** scaffolded 2026-10-15 from study 35's folder (Gatling plumbing) with study 36's script
fixes and study 37's study manifest.

## Objective

**The customer PoC study (study 37) run as is on our L40S, with the load's periodic valleys
removed**, to see what it finds when the score is not driven by the load's shape. Studies 35 and 36
were stopped on 2026-10-15 (35 after 30 experiments, 36 after 20): their scores were inflated by
deterministic valleys in the offered load (study 35's running notes, 2026-10-15) and a single run's
spread (+-7 % on repeated configurations) covered the differences between the good
configurations.

Everything that defines the study is study 37's, unchanged on purpose (the customer will run that
one and editing it on site is impractical): the customer's compose as baseline (cyankiwi AWQ W4A16
rev 0ef577a, vLLM 0.29.0, thinking off by default), **score = users**
(`aiperf.aiperf_active_sessions`), **constraint = client-side e2e median over 150 s <= 3 s**,
`stability` windowing `when: max` on the users (6 samples), the 17 parameters with study 36's
domains (including `wna16_activation_dtype` {auto, fp8}; `long_prefill_token_threshold` not
tuned, fixed off), steps `baseline`, `kv fp8`, `kv fp8 mtp3`, `study 35 best`, then `optimize`
(AKAMAS, 60 experiments, 20 failures). The domains are left wide on purpose (2026-10-15, user):
the optimizer explores freely instead of being handed the region studies 35/36 found.

## What changes from studies 35/36

| Item | Studies 35/36 | Study 38 |
|---|---|---|
| Study manifest | their own (10 / 9 steps) | **study 37's** (`../37-customer-onsite-docker/akamas/study.yaml`): name, system, workflow, description changed only |
| Ramp | 1 -> 200 users over 6000 s (one every 30 s) | **1 -> 200 over 3000 s (one every 15 s)** |
| Cluster | 35: vllm-bench-gatling; 36: vllm-bench | vllm-bench-gatling (study 35's `llm-replay` namespace, node `gpu=l40s`, ServiceMonitor `vllm-replay`) |
| Scripts | 35: as built; 36: + fp8 activations, scientific-notation fix | study 36's (`wna16_activation_dtype` rendered as environment, tiny values in scientific notation accepted) on study 35's Gatling plumbing |
| Dataset | the 10-user template, user k = session k mod 10 | **unchanged** (see "Load") |

## Load

The template holds 10 sessions; rendered user k replays session k mod 10 with its own texts. Under
a ramp of one user every 30 s the same session came back every 300 s, and the offered load had a
sawtooth with that period (-5 to -10 % input tokens at ~102 and ~117 users in every run of studies
35/36): the windowing scored configurations already over 3 s at those valleys.

**Fix: one user every 15 s**, so that the 10 sessions enter once every 150 s, the length of the
window of the e2e median: each window holds one full cycle and the valleys average out. Simulated
from the template (zero queueing, 2.5 s per request; deviation of each 150 s window's prompt
characters/s from the 10-minute trend):

| Ramp | Dataset | sd | worst valley |
|---|---|---|---|
| 6000 s (studies 35/36) | k mod 10 | 5.5 % | -12.8 % |
| 6000 s | desynchronized (shuffled order per block of 10, random start in the first half of the session), 3 seeds | 4.0-6.1 % | -9 to -17 % |
| **3000 s (this study)** | **k mod 10** | **1.1 %** | **-1.7 %** |
| 3000 s | desynchronized, 3 seeds | 2.4-6.5 % | -6 to -12 % |

Desynchronizing the users was the first idea and is worse: random start points bunch heavy
sessions together, and with a fixed seed those bunches are as deterministic as the period was. With
10 sessions the ordered round robin is the smoothest load, provided the window holds a whole cycle.
With the customer's 100 sessions the question does not arise.

Side effects of the faster ramp: the 150 s median follows the load with half the users per
window behind it (10 instead of 5), so scores come out a little higher than under study 35's ramp
(the same for every configuration); each experiment takes ~40-55 min instead of ~75-90 (64
experiments in ~2.5 days). The knees of studies 35/36 (60-122 users) fall inside the ramp. 240
users rendered (+20 %): the simulation uses 200 sessions over 3000 s.

## Risks

- Study 37's windowing (`when: max`) still scores the best valid window anywhere in the ramp; with
  the valleys gone that matters less, and study 37 keeps it (the customer's 100 sessions have no
  period). Re-score offline on the first breach as for studies 35/36.
- fp8 KV and fp8 activations change the numerics: accuracy check before proposing them.
- The vllm-bench-gatling nodes carry `AlwaysOn=true` (the GPU node group `llm-serving-l40s-ab` and
  `system`): remove the tags and scale the GPU group to 0 when the study ends.

## Runbook

1. `KUBE_CONTEXT=lab-vllm-bench-gatling bash infra/provision.sh` (idempotent; study 35's
   namespace, PVCs, Services and ServiceMonitor); GPU node `gpu=l40s` Ready; no vLLM running in
   `llm-replay`.
2. The template on the toolbox: `/work/customer-data/38-l40s-gemma4-26b-awq-replay-users-v3/`
   (`load/README.md`).
3. Commit, push, `git pull` on the toolbox; create and start (`akamas/README.md`).

## Offline checks (2026-10-15)

`python3 akamas/check_offline.py` 0 failures; `bash k8s/tests/test_*.sh` 7 suites 0 failures.

## Running notes

## Results

<Filled in by the study-recap skill once the study finishes.>

## Conclusions

<Filled in by the study-recap skill once the study finishes.>
