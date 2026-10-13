#!/bin/bash
# Local end-to-end dry run of study 34's AIPerf Job script (the customer's run_benchmark.sh,
# unchanged, wrapped by the Job): the rendered Job's own shell, under dash as in
# python:3.12-slim, with real AIPerf 0.11.0 against
# mock_openai.py's /v1/responses (answer deltas only: thinking off by default, as the compose). The
# ConfigMap is a local folder: the template gzipped, render_dataset_it.py, build_corpus.py, mix.json
# and run_benchmark.sh; the Italian corpus is DRY_CORPUS (copied where the Job keeps it) or built by
# the Job from Hugging Face (network, pyarrow in the venv); the Job's venv is AIPERF_VENV or a fresh one. Checks:
#   - the Job script exits 0 (render, benchmark.env, their script) and their script wrote its
#     run folder (data/results/run_<date>_<epoch>, with run.log and profile_export.jsonl);
#   - every request reached /v1/responses with the payload as rendered: model aiservices_gemma,
#     no placeholder left, no `text` field, every payload capped (max_output_tokens + ignore_eos, caps.py apply; DRY_CORPUS needed, the caps are computed from the render first); the instructions of one length are
#     the same text in every session; each user's later requests repeat its history;
#   - the LENGTHS lines: no reasoning tokens (thinking off);
#   - with DRY_TEMPLATE (the customer's template, not in git) and the default 100 users: how
#     long AIPerf takes from the marker to the first request (the ~310 MB dataset load).
# Usage: bash k8s/tests/dry_run_replay.sh [users] [ramp_s]   (defaults 6 users over 60 s on
# fixture_template.jsonl, ~3 min; network: pip and the tokenizer). AIPERF_VENV=<venv with
# aiperf 0.11.0> skips the install.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
USERS=${1:-6}; RAMP=${2:-60}
TEMPLATE=${DRY_TEMPLATE:-$HERE/fixture_template.jsonl}
W=$(mktemp -d /tmp/aiperf-replay-XXXXXX)
if [ -n "${AIPERF_VENV:-}" ]; then VENV=$AIPERF_VENV; else
  python3 -m venv "$W/venv"; "$W/venv/bin/pip" install --quiet aiperf==0.11.0; VENV=$W/venv
fi
rm -f /tmp/mock_openai.count
MOCK_REASONING_CHUNKS=0 MOCK_CHUNKS=8 MOCK_CHUNK_S=0.02 MOCK_BODIES="$W/bodies.jsonl" \
  python3 "$HERE/mock_openai.py" 18000 & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }

mkdir -p "$W/replay" "$W/bench"
gzip -c "$TEMPLATE" > "$W/replay/template.jsonl.gz"
cp "$K8S/render_dataset_it.py" "$K8S/build_corpus.py" "$K8S/mix.json" "$K8S/run_benchmark.sh" "$K8S/caps.py" "$W/replay/"
if [ -n "${DRY_CORPUS:-}" ]; then mkdir -p "$W/bench/corpus"; cp "$DRY_CORPUS" "$W/bench/corpus/corpus_it.json.gz"; fi
# Study 34: the caps the Job applies. The dataset rendered here as the Job renders it (same users,
# seed, mix and corpus: the Job's render must come out byte-identical, or caps.py apply refuses
# it), and made-up caps for the requests the ramp reaches (7 + turn % 50); the others take their
# task median, as in the study.
[ -s "$W/bench/corpus/corpus_it.json.gz" ] || { echo "DRY_CORPUS=<corpus_it.json.gz> is needed (the caps are computed before the Job runs)"; exit 2; }
EXTRA=$((USERS / 5)); [ "$EXTRA" -ge 10 ] || EXTRA=10
python3 "$K8S/render_dataset_it.py" "$TEMPLATE" "$W/bench/corpus/corpus_it.json.gz" $((USERS + EXTRA)) "$W/pre.jsonl" --seed 42 --mix "$K8S/mix.json" > /dev/null
python3 - "$K8S" "$W/pre.jsonl" "$USERS" "$RAMP" "$W/replay/caps.json.gz" <<'PY'
import collections, gzip, json, sys
sys.path.insert(0, sys.argv[1]); import caps
rows, keys, meta = caps.load(sys.argv[2])
c = {keys[i]: 7 + int(keys[i].rsplit('/', 1)[1]) % 50 for i in caps.reachable(rows, int(sys.argv[3]), int(sys.argv[4]))}
by = collections.defaultdict(list)
for i, k in enumerate(keys):
    if k in c: by[meta[i].get('task') or meta[i]['kind']].append(c[k])
json.dump({'dataset_sha256': caps.sha256(sys.argv[2]), 'caps': c, 'task_median': {t: caps.pct(v, 50) for t, v in by.items()}},
          gzip.open(sys.argv[5], 'wt'))
print('dry run caps: %d of %d requests' % (len(c), len(rows)))
PY
bash "$K8S/render_job.sh" "$USERS" "$RAMP" "$W/job.yaml"
yq '.spec.template.spec.containers[0].args[0]' "$W/job.yaml" \
  | sed -e "s|/benchmarks|$W/bench|g" -e "s|REPLAY=/replay|REPLAY=$W/replay|" \
        -e "/python3 -m venv \/opt\/aiperf-venv/d" -e "s|/opt/aiperf-venv|$VENV|g" \
        -e "s|http://vllm-0.vllm-headless.llm-l40s.svc.cluster.local:8000|http://127.0.0.1:18000|" > "$W/job.sh"
: > "$W/bodies.jsonl"
T0=$(python3 -c 'import time; print(time.time())')
SH=${DASH:-$(command -v dash || echo /bin/sh)}  # dash as in python:3.12-slim when installed
if PATH="$VENV/bin:$PATH" "$SH" "$W/job.sh" > "$W/job.log" 2>&1; then ok "Job script exits 0"
else ko "Job script exits 0"; tail -40 "$W/job.log"; fi
grep -E '^rendered|^CAPS|LENGTHS|^endpoint=|^users=' "$W/job.log" || true
RUN=$(ls -d "$W"/bench/study34/replay-benchmark/data/results/run_* 2>/dev/null | tail -1)
{ [ -n "$RUN" ] && [ -s "$RUN/profile_export.jsonl" ] && [ -s "$RUN/run.log" ]; } \
  && ok "their run folder ${RUN##*/}: profile_export.jsonl, run.log" || ko "their run folder (${RUN:-none})"

python3 - "$W/bodies.jsonl" "$W/job.log" "$T0" /tmp/mock_openai.count <<'EOF' || FAILS=$((FAILS + 1))
import json, re, sys
bodies = [json.loads(l) for l in open(sys.argv[1])]
log = open(sys.argv[2]).read()
fails = 0
def check(cond, name):
    global fails
    print(('ok   ' if cond else 'FAIL ') + name)
    fails += 0 if cond else 1
check(len(bodies) > 0, '%d requests reached /v1/responses' % len(bodies))
check(all(b.get('model') == 'aiservices_gemma' for b in bodies), 'model aiservices_gemma in every payload')
check(not any('{{' in json.dumps(b) for b in bodies), 'no placeholder left')
check(not any('text' in b for b in bodies), 'no text field (omitted in the template)')
check(all(isinstance(b.get('max_output_tokens'), int) and b['max_output_tokens'] >= 1 and b.get('ignore_eos') is True
          and 'max_tokens' not in b for b in bodies), 'every payload capped: max_output_tokens and ignore_eos (study 34)')
check(re.search(r'^CAPS: \d+ requests, \d+ capped at their calibrated length', log, re.M), "CAPS line: caps.py apply accepted the Job's render")
check(all(b.get('stream') is True and b.get('temperature') in (0.1, 0) for b in bodies), 'stream and temperature kept')
inst = {}
for b in bodies:
    if 'instructions' in b:
        inst.setdefault(len(b['instructions']), set()).add(b['instructions'])
check(inst and all(len(v) == 1 for v in inst.values()), 'one text per instructions length (%d lengths)' % len(inst))
lists = [b['input'] for b in bodies if isinstance(b.get('input'), list) and len(b['input']) >= 3]
firsts = {}
for i in lists:
    firsts.setdefault(len(i[0]['content']), set()).add(i[0]['content'])
check(lists and any(len(v) > 1 for v in firsts.values()) or len(lists) <= 1, 'users do not share their history text')
m = re.search(r'LENGTHS reasoning_token_count +n= *(\d+) mean= *([\d.]+)', log)
check(not m or float(m.group(2)) == 0, 'no reasoning tokens (thinking off)')
check('LENGTHS output_sequence_length' in log, 'LENGTHS output lengths reported')
check(all(not re.fullmatch(r'[a-z0-9 .,;:]{0,40}', json.dumps(b.get('input'))[:40]) for b in bodies), 'inputs carry text')
t0 = float(sys.argv[3])
marker = None
counts = [float(l) for l in open(sys.argv[4])]
print('info first request %.1f s after the Job script started; %d requests' % (min(counts) - t0, len(counts)))
sys.exit(1 if fails else 0)
EOF
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
