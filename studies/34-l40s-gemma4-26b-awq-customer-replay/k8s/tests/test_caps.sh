#!/bin/bash
# caps.py (study 34's per-request output caps): measure against a local non-streaming mock of
# /v1/responses, then apply. A made-up rendered dataset (3 sessions, delays in ms, meta with
# kind/task), so no corpus is needed. Checks:
#   - measure sends only the requests the ramp can reach (user k starts at ramp * k / users, its
#     requests at their delays; a user past `users` at the shortest session's end), not streamed,
#     uncapped, and records each one's output_tokens under "<session>/<turn>";
#   - apply writes max_output_tokens (calibrated, or the task median when not measured) and
#     ignore_eos into every payload and prints the CAPS line;
#   - apply refuses (exit 2, file untouched) a dataset that is not the calibrated one.
# Usage: bash k8s/tests/test_caps.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); K8S=$(dirname "$HERE")
W=$(mktemp -d /tmp/test-caps-XXXXXX)
FAILS=0
ok() { echo "ok   $1"; }
ko() { echo "FAIL $1"; FAILS=$((FAILS + 1)); }

# Mock: output_tokens = 10 x the request's number (from its input text), records every body.
cat > "$W/mock.py" <<'EOF'
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with open(sys.argv[2], 'a') as f: f.write(json.dumps(body) + '\n')
        n = int(body['input'].split()[-1])
        out = json.dumps({'status': 'completed', 'usage': {'input_tokens': 5, 'output_tokens': 10 * n,
                          'output_tokens_details': {'reasoning_tokens': 0}}}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out))); self.end_headers(); self.wfile.write(out)
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
EOF
python3 "$W/mock.py" 18011 "$W/bodies.jsonl" & MOCK=$!
trap 'kill $MOCK 2>/dev/null; wait $MOCK 2>/dev/null; rm -rf "$W"' EXIT
sleep 1

# 3 sessions in file order; users=2, ramp=100 s: user 0 starts at 0, user 1 at 50, user 2 (past
# `users`) at the shortest session's think time (session b: 30 s).
#   a: delays 0, 40, 40, 40 s -> starts 0, 40, 80, 120: turns 0-2 reachable
#   b: delays 0, 30 s        -> starts 50, 80: both reachable
#   c: delays 0, 50, 50 s    -> starts 30, 80, 130: turns 0-1 reachable
python3 - "$W/d.jsonl" <<'EOF'
import json, sys
S = {'a_c0': [0, 40, 40, 40], 'b_c0': [0, 30], 'c_c0': [0, 50, 50]}
tasks = {'a_c0': ('chat', 'qa'), 'b_c0': ('task', 'title'), 'c_c0': ('task', 'translate')}
n = 0
with open(sys.argv[1], 'w') as f, open(sys.argv[1] + '.meta.jsonl', 'w') as m:
    for s, ds in S.items():
        for d in ds:
            n += 1
            r = {'session_id': s, 'payload': {'model': 'aiservices_gemma', 'stream': True, 'input': 'request %d' % n}}
            if d: r['delay'] = d * 1000
            f.write(json.dumps(r) + '\n')
            m.write(json.dumps({'session_id': s, 'kind': tasks[s][0], 'task': tasks[s][1]}) + '\n')
EOF

python3 "$K8S/caps.py" measure "$W/d.jsonl" http://127.0.0.1:18011 "$W/caps.json.gz" --users 2 --ramp-s 100 --concurrency 3 > "$W/m.log" 2>&1 \
  && ok "measure exits 0" || { ko "measure exits 0"; cat "$W/m.log"; }
python3 - "$W/caps.json.gz" "$W/bodies.jsonl" "$W/d.jsonl" <<'EOF' && ok "measure: reachable set, keys, lengths, no cap, not streamed, sha256" || ko "measure contents"
import gzip, hashlib, json, sys
c = json.load(gzip.open(sys.argv[1], 'rt'))
want = {'a_c0/0': 10, 'a_c0/1': 20, 'a_c0/2': 30, 'b_c0/0': 50, 'b_c0/1': 60, 'c_c0/0': 70, 'c_c0/1': 80}
assert c['caps'] == want, c['caps']
assert c['reachable'] == 7 and c['measured'] == 7 and c['requests'] == 9 and not c['errors'], c
assert c['dataset_sha256'] == hashlib.sha256(open(sys.argv[3], 'rb').read()).hexdigest()
assert c['task_median'] == {'qa': 20, 'title': 60, 'translate': 80}, c['task_median']
bodies = [json.loads(l) for l in open(sys.argv[2])]
assert len(bodies) == 7 and all(b['stream'] is False and 'max_output_tokens' not in b for b in bodies)
EOF
grep -q 'with reasoning tokens (must be 0)' "$W/m.log" && ok "measure prints its summary" || ko "measure summary"

cp "$W/d.jsonl" "$W/d.orig.jsonl"
python3 "$K8S/caps.py" apply "$W/d.jsonl" "$W/caps.json.gz" > "$W/a.log" 2>&1 && ok "apply exits 0" || { ko "apply exits 0"; cat "$W/a.log"; }
python3 - "$W/d.jsonl" <<'EOF' && ok "apply: calibrated caps, task medians for the rest, ignore_eos everywhere" || ko "apply contents"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
got = [r['payload']['max_output_tokens'] for r in rows]
assert got == [10, 20, 30, 20, 50, 60, 70, 80, 80], got   # a_c0/3 -> qa median 20, c_c0/2 -> translate 80
assert all(r['payload']['ignore_eos'] is True for r in rows)
assert all(r['payload']['stream'] is True and r['payload']['model'] == 'aiservices_gemma' for r in rows)
EOF
grep -q '^CAPS: 9 requests, 7 capped at their calibrated length, 2 at their task median' "$W/a.log" && ok "apply prints the CAPS line" || { ko "CAPS line"; cat "$W/a.log"; }

# A dataset that is not the calibrated one: refused, untouched.
cp "$W/d.orig.jsonl" "$W/e.jsonl"; cp "$W/d.jsonl.meta.jsonl" "$W/e.jsonl.meta.jsonl"
sed -i.bak 's/request 1"/request 1 "/' "$W/e.jsonl"; cp "$W/e.jsonl" "$W/e.before"
python3 "$K8S/caps.py" apply "$W/e.jsonl" "$W/caps.json.gz" > "$W/e.log" 2>&1; rc=$?
[ $rc = 2 ] && grep -q 'is not the calibrated one' "$W/e.log" && cmp -s "$W/e.jsonl" "$W/e.before" \
  && ok "apply refuses another dataset (exit 2, untouched)" || { ko "apply refuses another dataset (rc $rc)"; cat "$W/e.log"; }

echo "$FAILS failure(s)"
[ $FAILS = 0 ]
