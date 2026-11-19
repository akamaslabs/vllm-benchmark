"""Stub OpenAI chat server for the study 54 ramp spike (no GPU).

Streams a fixed reply after a fixed TTFT and logs one JSON line per request: arrival time, a session
key (hash of the first user message, which AIPerf keeps in the history of every later turn) and the
number of user messages (the turn index + 1). Usage: python3 -I stub_server.py PORT LOGFILE.
"""
import asyncio
import hashlib
import json
import os
import sys
import time

from aiohttp import web

TTFT_S = float(os.environ.get("STUB_TTFT_S", "0.05"))
ITL_S = float(os.environ.get("STUB_ITL_S", "0.005"))


def session_key(messages):
    users = [m for m in messages if m.get("role") == "user"]
    first = users[0]["content"] if users else ""
    if isinstance(first, list):
        first = json.dumps(first)
    return hashlib.sha1(first.encode()).hexdigest()[:12], len(users)


async def models(_request):
    return web.json_response({"object": "list", "data": [{"id": "stub", "object": "model"}]})


async def chat(request):
    body = await request.json()
    sid, n_user = session_key(body.get("messages", []))
    t0 = time.time()
    request.app["log"].write(json.dumps({"t": t0, "sid": sid, "turn": n_user - 1}) + "\n")
    request.app["log"].flush()
    n_out = int(body.get("max_completion_tokens") or body.get("max_tokens") or 10)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(TTFT_S)
    for i in range(n_out):
        chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(t0), "model": "stub",
                 "choices": [{"index": 0, "delta": {"content": " tok"}, "finish_reason": None}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(ITL_S)
    last = {"id": "x", "object": "chat.completion.chunk", "created": int(t0), "model": "stub",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
    await resp.write(f"data: {json.dumps(last)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


def main():
    port, logfile = int(sys.argv[1]), sys.argv[2]
    app = web.Application()
    app["log"] = open(logfile, "a")
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host="127.0.0.1", port=port, print=None)


if __name__ == "__main__":
    main()
