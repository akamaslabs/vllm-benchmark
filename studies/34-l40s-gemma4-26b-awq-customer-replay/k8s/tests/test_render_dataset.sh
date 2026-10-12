#!/bin/bash
# Tests for ../render_dataset.py on tests/fixture_template.jsonl (made up, the customer's shape)
# and AIPerf's shakespeare.txt (CORPUS=<path>, or found in a python3 that has aiperf installed).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE"); R=$K8S/render_dataset.py
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }
CORPUS=${CORPUS:-$(python3 -c "import aiperf, os; print(os.path.join(os.path.dirname(aiperf.__file__), 'dataset', 'generator', 'assets', 'shakespeare.txt'))" 2>/dev/null)}
[ -s "$CORPUS" ] || { echo "skip: no corpus (set CORPUS=<aiperf>/dataset/generator/assets/shakespeare.txt)"; exit 0; }
gzip -c "$HERE/fixture_template.jsonl" > "$TMP/t.jsonl.gz"
python3 "$R" "$TMP/t.jsonl.gz" "$CORPUS" 7 "$TMP/d.jsonl" > "$TMP/out.txt" 2>&1 && ok "renders 7 users" || ko "renders 7 users"
python3 - "$TMP/d.jsonl" "$HERE/fixture_template.jsonl" <<'PY' || FAILS=$((FAILS + 1))
import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
tpl = [json.loads(l) for l in open(sys.argv[2])]
fails = 0
def check(cond, name):
    global fails
    print(('ok   ' if cond else 'FAIL ') + name); fails += 0 if cond else 1
by = {}
for r in rows: by.setdefault(r['session_id'], []).append(r)
check(sorted(by) == sorted('user_%04d_c%d' % (u % 3, u // 3) for u in range(7)), '7 users named <session>_c<copy>')
check(all(len(v) == 25 for v in by.values()), "each user replays its template session's 25 requests")
check(not any('{{' in json.dumps(r) for r in rows), 'no placeholder left')
check(not any('text' in r['payload'] for r in rows), 'text field dropped')
ph = re.compile(r'\{\{\w+: (\d+) chars\}\}')
def lengths(v):
    if isinstance(v, str):
        m = ph.fullmatch(v); return [int(m.group(1))] if m else []
    if isinstance(v, list): return [x for e in v for x in lengths(e)]
    if isinstance(v, dict): return [x for k, e in v.items() for x in lengths(e)]
    return []
def actual(v):
    if isinstance(v, str): return [len(v)]
    if isinstance(v, list): return [x for e in v for x in actual(e)]
    if isinstance(v, dict): return [x for k, e in v.items() if k in ('input', 'instructions', 'content') for x in actual(e)]
    return []
src = {}
for r in tpl: src.setdefault(r['session_id'], []).append(r)
same = all(lengths(t['payload']) == actual(r['payload']) and t.get('delay') == r.get('delay')
           for sid, v in by.items() for t, r in zip(src[sid.rsplit('_c', 1)[0]], v))
check(same, 'every text has its placeholder length; delays kept')
u0, u3 = by['user_0000_c0'], by['user_0000_c1']
check(u0[1]['payload']['input'][0]['content'] == u0[0]['payload']['input'][0]['content'], 'history repeated within a user')
check(u0[0]['payload']['input'][0]['content'] != u3[0]['payload']['input'][0]['content'], 'two copies of a session differ')
check(u0[0]['payload']['instructions'] == by['user_0001_c0'][0]['payload']['instructions'], 'instructions of one length shared across users')
sys.exit(1 if fails else 0)
PY
python3 "$R" "$TMP/t.jsonl.gz" "$CORPUS" 7 "$TMP/d2.jsonl" >/dev/null 2>&1 && cmp -s "$TMP/d.jsonl" "$TMP/d2.jsonl" \
  && ok "deterministic (same seed, same file)" || ko "deterministic"
echo '{"session_id": "x", "payload": {"input": "{{image: 3 items}}"}}' > "$TMP/bad.jsonl"
python3 "$R" "$TMP/bad.jsonl" "$CORPUS" 1 "$TMP/b.jsonl" >/dev/null 2>&1; rc=$?
[ $rc = 2 ] && ok "unknown placeholder: exit 2" || ko "unknown placeholder: exit $rc"
python3 "$R" "$TMP/t.jsonl.gz" "$CORPUS" 0 "$TMP/z.jsonl" >/dev/null 2>&1; rc=$?
[ $rc = 2 ] && ok "users 0: exit 2" || ko "users 0: exit $rc"
echo "$FAILS failure(s)"; [ $FAILS = 0 ]
