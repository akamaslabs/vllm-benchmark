"""Stub OpenAI chat server for the TraceLab replay spike (no GPU).

Streams max_tokens tokens after a fixed TTFT and logs one JSON line per request: arrival time,
session (X-Correlation-ID / X-Session-Affinity header), prompt length in characters, and the
share of this session's previous prompt that is an exact prefix of this one (what a prefix
cache could reuse). Usage: python3 -I stub_trace.py PORT LOGFILE.
"""
import asyncio
import json
import os
import sys
import time

from aiohttp import web

TTFT_S = float(os.environ.get("STUB_TTFT_S", "0.05"))
ITL_S = float(os.environ.get("STUB_ITL_S", "0.001"))
MAX_OUT = int(os.environ.get("STUB_MAX_OUT", "50"))


def prompt_text(body):
    parts = []
    for m in body.get("messages", []):
        c = m.get("content")
        parts.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(parts) if parts else str(body.get("prompt", ""))


def common_prefix(a, b):
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:  # binary search on equal prefixes (fast for long prompts)
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


async def models(_request):
    return web.json_response({"object": "list", "data": [{"id": "stub", "object": "model"}]})


async def chat(request):
    body = await request.json()
    sid = request.headers.get("X-Correlation-ID") or request.headers.get("X-Session-Affinity") or "?"
    text = prompt_text(body)
    prev = request.app["last"].get(sid)
    reuse = common_prefix(prev, text) / len(prev) if prev else None
    request.app["last"][sid] = text
    t0 = time.time()
    n_req = int(body.get("max_completion_tokens") or body.get("max_tokens") or 10)
    request.app["log"].write(json.dumps({"t": t0, "sid": sid, "chars": len(text), "reuse": reuse,
                                         "max_tokens": n_req,
                                         "affinity": request.headers.get("X-Session-Affinity")}) + "\n")
    request.app["log"].flush()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(TTFT_S)
    for _ in range(min(n_req, MAX_OUT)):
        chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(t0), "model": "stub",
                 "choices": [{"index": 0, "delta": {"content": " tok"}, "finish_reason": None}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(ITL_S)
    last = {"id": "x", "object": "chat.completion.chunk", "created": int(t0), "model": "stub",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    await resp.write(f"data: {json.dumps(last)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


def main():
    port, logfile = int(sys.argv[1]), sys.argv[2]
    app = web.Application(client_max_size=1024 ** 3)
    app["log"] = open(logfile, "a")
    app["last"] = {}
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host="127.0.0.1", port=port, print=None)


if __name__ == "__main__":
    main()
