#!/bin/bash
# compare.py on synthetic lm-eval 0.4.13 outputs with known deltas and verdicts.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); E=$(dirname "$HERE")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
# mk root run protocol task n wrong_ranges [version] [truncated]
#   wrong_ranges: "a:b,c:d" = docs a..b-1 and c..d-1 answered wrong ("" = none)
mk() {
  python3 -I - "$@" <<'EOF'
import json, os, sys
root, run, protocol, task, n, wrong = sys.argv[1:7]
version = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
trunc = int(sys.argv[8]) if len(sys.argv) > 8 else 0
n = int(n)
bad = set()
for r in filter(None, wrong.split(",")):
    a, b = map(int, r.split(":")); bad.update(range(a, b))
td = os.path.join(root, run, "lmeval", protocol, task)
d = os.path.join(td, "gemma4-26b-l40s")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, f"samples_{task}_2026-10-08T00-00-00.jsonl"), "w") as f:
    for i in range(n):
        ok = i not in bad
        if task == "ifeval":
            f.write(json.dumps({"doc_id": i, "filter": "none", "prompt_level_strict_acc": ok,
                                "inst_level_strict_acc": [ok, True]}) + "\n")
        else:
            for flt in ("strict-match", "flexible-extract"):
                f.write(json.dumps({"doc_id": i, "filter": flt, "exact_match": 1.0 if ok else 0.0}) + "\n")
with open(os.path.join(d, "results_2026-10-08T00-00-00.json"), "w") as f:
    json.dump({"versions": {task: version}}, f)
with open(os.path.join(td, "finished_before.json"), "w") as f:
    json.dump({"stop": 100.0, "length": 5.0, "abort": 0.0}, f)
with open(os.path.join(td, "finished_after.json"), "w") as f:
    json.dump({"stop": 100.0 + n - trunc, "length": 5.0 + trunc, "abort": 0.0}, f)
EOF
}
G=gsm8k_platinum_cot_llama
q() { python3 -I -c "import json,sys; r=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$1/comparison.json" "$2"; }
pair() { echo "[p for p in r['pairs'] if p['task']=='$1' and p['b']=='$2'][0]"; }
cmp() { python3 -I "$E/compare.py" "$1" > "$1/stdout" 2>&1; }

# S1: same accuracy, 5 lost + 5 gained on GSM8K; identical IFEval -> no degradation.
S=$TMP/s1
mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:45,950:955
mk $S baseline-a greedy ifeval 500 0:50; mk $S best greedy ifeval 500 0:50
cmp $S && ok "s1: exit 0" || ko "s1: exit 0 ($(tail -2 $S/stdout))"
[ "$(q $S "$(pair $G best)['verdict']")" = "no degradation" ] && ok "s1: gsm8k no degradation" || ko "s1: gsm8k verdict"
[ "$(q $S "($(pair $G best)['delta'], len($(pair $G best)['lost']), len($(pair $G best)['gained']))")" = "(0.0, 5, 5)" ] \
  && ok "s1: delta 0, 5 lost, 5 gained" || ko "s1: delta/flips"
[ "$(q $S "$(pair ifeval best)['verdict']")" = "no degradation" ] && ok "s1: ifeval no degradation" || ko "s1: ifeval verdict"
[ "$(q $S "round(r['accuracy']['$G']['best']['exact_match'], 2)")" = 95.0 ] && ok "s1: accuracy 95.0" || ko "s1: accuracy"
[ "$(q $S "round(r['accuracy']['ifeval']['best']['inst_level_strict_acc,none'], 2)")" = 95.0 ] && ok "s1: inst-level secondary" || ko "s1: inst-level secondary"
grep -q 'no degradation' "$S/comparison.md" && ok "s1: markdown written" || ko "s1: markdown written"
[ "$(q $S "r['warnings']")" = "[]" ] && ok "s1: no warning for runs not in the results (skipped ablations)" || ko "s1: no warnings: $(q $S "r['warnings']")"

# S2: 60 lost of 1000 -> degradation, McNemar tiny.
S=$TMP/s2
mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50,900:960
cmp $S
[ "$(q $S "$(pair $G best)['verdict']")" = degradation ] && ok "s2: degradation" || ko "s2: verdict"
[ "$(q $S "round($(pair $G best)['delta'], 2)")" = -6.0 ] && ok "s2: delta -6.0" || ko "s2: delta"
[ "$(q $S "$(pair $G best)['mcnemar_p'] < 1e-6")" = True ] && ok "s2: McNemar p tiny" || ko "s2: McNemar"

# S3: 5 lost of 2000 -> measurable drop within the margin.
S=$TMP/s3
mk $S baseline-a greedy $G 2000 0:100; mk $S best greedy $G 2000 0:100,1990:1995
cmp $S
[ "$(q $S "$(pair $G best)['verdict']")" = "measurable drop within the margin" ] && ok "s3: drop within margin" || ko "s3: verdict $(q $S "$(pair $G best)['ci']")"

# S4: IFEval, 100 prompts, 4 lost 2 gained -> inconclusive.
S=$TMP/s4
mk $S baseline-a greedy ifeval 100 0:10; mk $S best greedy ifeval 100 0:8,90:94
cmp $S
[ "$(q $S "$(pair ifeval best)['verdict']")" = inconclusive ] && ok "s4: inconclusive" || ko "s4: verdict $(q $S "$(pair ifeval best)['ci']")"

# S5: noise floor and ablations paired against baseline-a; anchor; truncation warning.
S=$TMP/s5
mk $S baseline-a greedy $G 1000 0:50; mk $S baseline-b greedy $G 1000 0:49,999:1000
mk $S best-kv-auto greedy $G 1000 0:50 1.0 20
mk $S baseline-a card $G 1000 0:47; mk $S baseline-a card ifeval 500 0:150
cmp $S
[ "$(q $S "$(pair $G baseline-b)['a']")" = baseline-a ] && ok "s5: noise floor pair" || ko "s5: noise floor pair"
[ "$(q $S "$(pair $G best-kv-auto)['a']")" = baseline-a ] && ok "s5: ablation pair" || ko "s5: ablation pair"
[ "$(q $S "r['anchor']['$G']['within']")" = True ] && ok "s5: gsm8k anchor within" || ko "s5: gsm8k anchor"
[ "$(q $S "r['anchor']['ifeval']['within']")" = False ] && ok "s5: ifeval anchor outside" || ko "s5: ifeval anchor"
[ "$(q $S "r['truncations']['$G']['best-kv-auto']['length']")" = 20.0 ] && ok "s5: 20 truncations counted" || ko "s5: truncations"
[ "$(q $S "any('best-kv-auto' in w and 'truncated' in w for w in r['warnings'])")" = True ] && ok "s5: truncation warning" || ko "s5: truncation warning"
[ "$(q $S "any('anchor' in w and 'ifeval' in w for w in r['warnings'])")" = True ] && ok "s5: anchor warning" || ko "s5: anchor warning"

# S6: the anchor run truncated (5 % of the card answers on "length") -> warning, n shown.
S=$TMP/s6
mk $S baseline-a greedy $G 1209 0:50; mk $S baseline-a card $G 1209 0:56 1.0 60
cmp $S
[ "$(q $S "any('baseline-a card' in w and 'truncated' in w for w in r['warnings'])")" = True ] && ok "s6: card truncation warning" || ko "s6: card truncation warning"
grep -q '| 60 / 1209 |' "$S/comparison.md" && ok "s6: anchor truncations out of n" || ko "s6: anchor truncations out of n"

# S7: a run without one task's greedy results -> its comparisons are reported missing.
S=$TMP/s7
mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50; mk $S best greedy ifeval 500 0:50
cmp $S
[ "$(q $S "any('baseline-a' in w and 'ifeval' in w and 'missing' in w for w in r['warnings'])")" = True ] \
  && ok "s7: missing task warned" || ko "s7: missing task warned"
S=$TMP/s8
mk $S best greedy $G 1000 0:50
cmp $S
[ "$(q $S "any('best vs baseline-a' in w and 'missing' in w for w in r['warnings'])")" = True ] \
  && ok "s8: missing baseline-a warned" || ko "s8: missing baseline-a warned"

# Refusals.
S=$TMP/r1; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 999 0:50
cmp $S; [ $? = 2 ] && grep -q 'doc_id' "$S/stdout" && ok "refuse: different doc_id sets" || ko "refuse: different doc_id sets"
S=$TMP/r2; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50 2.0
cmp $S; [ $? = 2 ] && grep -q 'different task versions' "$S/stdout" && ok "refuse: different task versions" || ko "refuse: different task versions"
S=$TMP/r3; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50
cp $S/best/lmeval/greedy/$G/gemma4-26b-l40s/samples_${G}_2026-10-08T00-00-00.jsonl $S/best/lmeval/greedy/$G/gemma4-26b-l40s/samples_${G}_2026-10-09T00-00-00.jsonl
cmp $S; [ $? = 2 ] && grep -q 'found 2' "$S/stdout" && ok "refuse: two samples files" || ko "refuse: two samples files"
S=$TMP/r4; mk $S baseline-a greedy $G 1000 0:50; mk $S best greedy $G 1000 0:50
gzip $S/best/lmeval/greedy/$G/gemma4-26b-l40s/samples_*.jsonl
cmp $S && [ "$(q $S "$(pair $G best)['n']")" = 1000 ] && ok "reads gzipped samples" || ko "reads gzipped samples"

echo "$FAILS failure(s)"; [ $FAILS = 0 ]
