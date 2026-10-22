"""Long-context smoke benchmark, run inside the vLLM pod (stdlib only).

For each target prompt length: a haystack of numbered filler lines with one hidden
"secret code" line, then a question asking for it. Greedy decoding, streaming, one
request at a time. Prints TTFT, TPOT, whether the answer contains the code, and the
first output tokens (to spot garbage output).
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "dsv2-lite"
LENGTHS = [4000, 16000, 30000]
REPEATS = 3
CODE = "7319"
CITIES = ["Turin", "Milan", "Rome", "Naples", "Genoa", "Bologna", "Florence", "Bari"]
WEATHER = ["sunny", "rainy", "cloudy", "windy", "foggy", "snowy"]


def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def ntokens(text):
    with post("/tokenize", {"model": MODEL, "prompt": text}) as r:
        return json.load(r)["count"]


def haystack(target):
    # ~22 tokens per line (measured); the needle sits at 40 % of the text
    n = max(10, target // 22)
    lines = [f"Line {i}: in {CITIES[i % 8]} the weather was {WEATHER[i % 6]} "
             f"and the temperature was {10 + i % 23} degrees." for i in range(n)]
    lines.insert(int(n * 0.4), f"Line X: the secret code is {CODE}.")
    q = ("\n\nQuestion: what is the secret code written in the text above? "
         "Answer with the number only.\nAnswer:")
    return "\n".join(lines) + q


def run(prompt, long_out=False):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 128 if long_out else 64,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if long_out:
        body["ignore_eos"] = True
    t0 = time.perf_counter()
    first = last = None
    text, chunks = "", 0
    with post("/v1/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and d["choices"][0].get("text"):
                now = time.perf_counter()
                first = first or now
                last = now
                text += d["choices"][0]["text"]
                chunks += 1
            if d.get("usage"):
                out_tokens = d["usage"]["completion_tokens"]
    ttft = (first - t0) * 1000 if first else None
    tpot = ((last - first) * 1000 / max(1, out_tokens - 1)) if first else None
    return ttft, tpot, out_tokens, text


results = []
for L in LENGTHS:
    p = haystack(L)
    n = ntokens(p)
    for rep in range(REPEATS):
        ttft, tpot, k, text = run(p)
        ok = CODE in text
        _, tpot128, k128, _ = run(p, long_out=True)
        results.append({"target": L, "prompt_tokens": n, "rep": rep, "ttft_ms": round(ttft, 1),
                        "tpot_ms": round(tpot128, 2), "out_tokens": k128, "correct": ok,
                        "head": text.strip()[:60]})
        print(json.dumps(results[-1]), flush=True)
print("SUMMARY " + json.dumps(results))
