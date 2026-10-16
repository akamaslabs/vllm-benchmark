# 38-l40s-gemma4-26b-awq-replay-users-v3

**Status:** STOPPED (2026-10-16 ~10:45 UTC, by the user, after 26 experiments: 4 presets + 22 optimizer; exp. 27 aborted). Gatling GPU node group scaled to 0, AlwaysOn tags removed.
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

Raw data: `results/export.tar.gz`, `results/trials.csv` (Akamas score, recomputed score,
first-breach score: recomputed = Akamas on all 26 runs). Started 2026-10-15 09:34 UTC; ~45-60 min
per experiment (35: 75-90).

**Akamas score = first-breach score in all 26 runs**: with one user every 15 s the e2e median
never came back under 3 s once over it, so the valleys of studies 35/36 are gone.

| Experiment | Users | vs baseline | Configuration |
|---|---|---|---|
| baseline (1) | 82.8 | | the compose |
| `kv fp8` (2) | 107.8 | +30 % | fp8 KV only |
| `kv fp8 mtp3` (3) | 109.2 | +32 % | + MTP K=3 |
| **`study 35 best` (4)** | **126.0** | **+52 %** | fp8 KV, MTP K=1, 179 seqs, 2496 batched, gmu 0.90, block 32, Humming linear, async off, O2, stream 11 |
| **9 (optimizer)** | **126.0** | **+52 %** | as 4 with 242 seqs, Humming MoE, fp8 activations, stream 13 |
| 5 | 122.8 | +48 % | fp8 KV, MTP K=4, 163 seqs, 6763 batched, block 48, Humming, throughput, fp8 activations |
| 18 | 121.0 | +46 % | as 4 with 187 seqs, block 64, async on, fp8 activations |
| 15 | 120.0 | +45 % | fp8 KV, MTP K=3, 219 seqs, 11541 batched, block 96, kernels auto, throughput |
| 25 | 115.0 | +39 % | **near-copy of 4** (208 seqs, capture 292, stream 12) |
| 21 / 23 | 111.0 / 112.2 | | as 4 with block 112 + fp8 act. / kernels swapped |
| 10 | 58.5 | -29 % | `max_num_seqs` 16 |

- Optimizer experiments 5-26: 58.5-126, 19 of 22 between 101 and 123; none above the
  `study 35 best` preset in 22 experiments. No configuration without MTP above 108 (exp. 19: 101).

## Conclusions

- **The 15 s/user ramp removed the load's valleys** (Akamas = first breach in 26 of 26) and halved
  the time per experiment. The scores run higher than under the 30 s/user ramp (baseline 82.8
  against ~72): the 150 s median trails the load by more users and more sessions are young. Compare
  configurations within one ramp, not across.
- **It did not remove the run-to-run noise:** a near-copy of the best configuration (exp. 25) scored
  115 against 126; with exps. 21/23 (small changes) at 111-112 the spread of one run is ~+-6 %.
  The good region (fp8 KV + MTP, ~150-250 seqs, small batched tokens) holds 115-126 users and the
  optimizer cannot rank inside it with one run each.
- **Same levers as studies 35/36:** fp8 KV +30 %; MTP plus the batch/kernel settings of
  `study 35 best` to +52 %; fp8 activations with no visible gain (exp. 9 = exp. 4).
- **Next:** repeat the baseline and the top 3-4 configurations (4, 9, 5, 15) twice each to measure
  the noise and pick the winner (proposed, not run); for the customer study (37), plan repeats of
  its top configurations at the end. Specific to: AWQ Gemma 4 26B-A4B, one L40S, the customer's
  10-session template, AIPerf 0.11.0 closed-loop replay.
