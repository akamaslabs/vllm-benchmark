# Study 30 accuracy check: comparison

## Anchor (baseline-a, card protocol, vs RedHat's FP8 column)

| Task | Accuracy | Card | Tolerance | Within | Truncated |
|---|---|---|---|---|---|
| gsm8k_platinum_cot_llama | 95.37 | 95.37 | +-1.4 | yes | 1 / 1209 |
| ifeval | 89.28 | 89.34 | +-3.1 | yes | 2 / 541 |

## Accuracy (greedy)

| Task | Run | Primary | Secondary | Truncated |
|---|---|---|---|---|
| gsm8k_platinum_cot_llama | baseline-a | exact_match 95.20 | exact_match,flexible-extract 95.20 | 0 / 1209 |
| gsm8k_platinum_cot_llama | baseline-b | exact_match 95.37 | exact_match,flexible-extract 95.37 | 0 / 1209 |
| gsm8k_platinum_cot_llama | best | exact_match 95.86 | exact_match,flexible-extract 95.86 | 0 / 1209 |
| gsm8k_platinum_cot_llama | best-kv-auto | exact_match 95.20 | exact_match,flexible-extract 95.20 | 1 / 1209 |
| gsm8k_platinum_cot_llama | best-no-mtp | exact_match 95.62 | exact_match,flexible-extract 95.62 | 0 / 1209 |
| ifeval | baseline-a | prompt_level_strict_acc 88.72 | inst_level_strict_acc,none 92.33 | 3 / 541 |
| ifeval | baseline-b | prompt_level_strict_acc 88.54 | inst_level_strict_acc,none 92.21 | 3 / 541 |
| ifeval | best | prompt_level_strict_acc 89.65 | inst_level_strict_acc,none 92.81 | 3 / 541 |
| ifeval | best-kv-auto | prompt_level_strict_acc 88.91 | inst_level_strict_acc,none 92.57 | 3 / 541 |
| ifeval | best-no-mtp | prompt_level_strict_acc 89.46 | inst_level_strict_acc,none 93.05 | 5 / 541 |

## Paired comparisons (greedy, primary metric, B - A)

| Task | B vs A | n | A | B | Delta | 95 % CI | Lost | Gained | McNemar p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k_platinum_cot_llama | best vs baseline-a | 1209 | 95.20 | 95.86 | +0.66 | [+0.17, +1.24] | 2 | 10 | 0.0386 | no degradation |
| gsm8k_platinum_cot_llama | best-kv-auto vs baseline-a | 1209 | 95.20 | 95.20 | +0.00 | [-0.50, +0.41] | 4 | 4 | 1 | no degradation |
| gsm8k_platinum_cot_llama | best-no-mtp vs baseline-a | 1209 | 95.20 | 95.62 | +0.41 | [-0.17, +0.99] | 4 | 9 | 0.267 | no degradation |
| gsm8k_platinum_cot_llama | baseline-b vs baseline-a | 1209 | 95.20 | 95.37 | +0.17 | [-0.25, +0.66] | 3 | 5 | 0.727 | no degradation |
| ifeval | best vs baseline-a | 541 | 88.72 | 89.65 | +0.92 | [-0.55, +2.40] | 6 | 11 | 0.332 | no degradation |
| ifeval | best-kv-auto vs baseline-a | 541 | 88.72 | 88.91 | +0.18 | [-1.48, +1.85] | 11 | 12 | 1 | no degradation |
| ifeval | best-no-mtp vs baseline-a | 541 | 88.72 | 89.46 | +0.74 | [-1.11, +2.59] | 11 | 15 | 0.557 | no degradation |
| ifeval | baseline-b vs baseline-a | 541 | 88.72 | 88.54 | -0.18 | [-0.55, +0.00] | 1 | 0 | 1 | no degradation |

## Flips (doc_id)

- gsm8k_platinum_cot_llama, best vs baseline-a: lost [688, 1153], gained [146, 230, 604, 690, 712, 725, 753, 829, 869, 916]
- gsm8k_platinum_cot_llama, best-kv-auto vs baseline-a: lost [195, 332, 521, 1153], gained [322, 604, 690, 869]
- gsm8k_platinum_cot_llama, best-no-mtp vs baseline-a: lost [69, 932, 950, 1153], gained [146, 230, 322, 712, 753, 761, 829, 916, 1101]
- gsm8k_platinum_cot_llama, baseline-b vs baseline-a: lost [332, 932, 1153], gained [146, 322, 604, 712, 916]
- ifeval, best vs baseline-a: lost [23, 77, 157, 235, 236, 395], gained [89, 102, 146, 167, 277, 319, 369, 445, 467, 468, 535]
- ifeval, best-kv-auto vs baseline-a: lost [23, 61, 77, 157, 162, 185, 235, 295, 310, 444, 496], gained [146, 167, 212, 270, 277, 317, 319, 354, 369, 401, 445, 467]
- ifeval, best-no-mtp vs baseline-a: lost [13, 23, 61, 74, 120, 152, 157, 162, 235, 310, 372], gained [89, 102, 106, 167, 213, 270, 277, 317, 319, 354, 369, 401, 467, 519, 535]
- ifeval, baseline-b vs baseline-a: lost [157], gained []
