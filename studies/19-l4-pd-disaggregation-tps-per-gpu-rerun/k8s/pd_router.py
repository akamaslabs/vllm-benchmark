"""Study 18 front door: prefill/decode router + client-side latency exporter.

Every request of every preset goes through this process, aggregated ones included, so the
routing hop costs the same in every trial and the latency it measures is comparable.

  aggregated (PD_PREFILL_INSTANCES=0): round-robin over the decode instances, which serve
      whole requests.
  disaggregated (PD_PREFILL_INSTANCES>0): the NixlConnector protocol of vLLM 0.29.0's own
      tests/v1/kv_connector/nixl_integration/toy_proxy_server.py. First a copy of the
      request goes to a prefill instance with max_tokens=1, stream=false and
      kv_transfer_params.do_remote_decode=true. Its response's kv_transfer_params are then
      attached to the original request, which streams from a decode instance. Both legs
      carry the same X-Request-Id.

Why its own metrics: vllm:time_to_first_token_seconds on a decode instance starts when THAT
instance receives the request, i.e. after the remote prefill, so it under-reports TTFT. This
router measures from the moment it receives the request. It publishes vLLM's own series
names (vllm:time_to_first_token_seconds, vllm:inter_token_latency_seconds,
vllm:e2e_request_latency_seconds, vllm:request_success, vllm:prompt_tokens,
vllm:generation_tokens) with model_name=<PD_ROUTER_MODEL_LABEL>. The study telemetry's
existing vLLM queries therefore work unchanged for the vLLM_PD_Topology component, whose
`model` property selects this label. Token counters come from the `usage` vLLM returns, so
each prompt and output token is counted once. On the decode instance a transferred prompt
is counted again, but that is not counted here.

Study 19 change (vs study 18): tokens are counted AS THEY STREAM, not when a request ends.
The router always asks the decode instance for stream_options.continuous_usage_stats, so
every chunk carries cumulative usage. Prompt tokens count once at the first chunk, and
output tokens count by the delta of completion_tokens on every chunk. Study 18 added all
4096 + 256 tokens of a request at its end. In 30 s telemetry windows that turned
throughput into a sawtooth: 0 in a window with no completion, a spike in the next. The
means were right, but Akamas scores the best window, and the noise leaked into the score.
The usage field is stripped again from chunks the client did not ask it on.

Topology comes from /pd-config/topology.env (rendered per trial). Instance i of a role listens
on <base>+i: prefill 8100+, decode 8200+. The launcher uses the same convention.
Dependencies are only what vllm/vllm-openai:v0.29.0 already ships: fastapi, uvicorn, httpx,
prometheus_client.
"""
import itertools
import json
import os
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


def read_topology(path=os.environ.get("PD_TOPOLOGY_FILE", "/pd-config/topology.env")):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


TOPO = read_topology()
P = int(TOPO.get("PD_PREFILL_INSTANCES") or 0)
D = int(TOPO.get("PD_DECODE_INSTANCES") or 1)
PREFILL_BASE = int(os.environ.get("PD_PREFILL_PORT_BASE", "8100"))
DECODE_BASE = int(os.environ.get("PD_DECODE_PORT_BASE", "8200"))
LABEL = os.environ.get("PD_ROUTER_MODEL_LABEL", "qwen3-8b-router")
PORT = int(os.environ.get("PD_ROUTER_PORT", "8000"))

PREFILL_URLS = [f"http://127.0.0.1:{PREFILL_BASE + i}" for i in range(P)]
DECODE_URLS = [f"http://127.0.0.1:{DECODE_BASE + i}" for i in range(D)]
_prefill_rr = itertools.cycle(range(P)) if P else None
_decode_rr = itertools.cycle(range(D))

# Buckets cover this study's range: 4096-token prompts on L4 (TTFT ~0.5 s idle, seconds
# under load) and single-token gaps (tens of ms, up to a whole prefill chunk when a
# prefill interleaves with decode).
TTFT = Histogram("vllm:time_to_first_token_seconds", "Router-side TTFT: request received to first token forwarded.",
                 ["model_name"], buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 60, 120])
ITL = Histogram("vllm:inter_token_latency_seconds", "Router-side gap between consecutive forwarded token chunks.",
                ["model_name"], buckets=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 0.125,
                                         0.15, 0.2, 0.3, 0.5, 0.75, 1, 2.5, 5])
E2E = Histogram("vllm:e2e_request_latency_seconds", "Router-side end-to-end latency of successful requests.",
                ["model_name"], buckets=[0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600])
PREFILL_LEG = Histogram("pd_router_prefill_leg_seconds", "Duration of the remote-prefill leg (disaggregated only).",
                        ["model_name"], buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 60])
SUCCESS = Counter("vllm:request_success", "Requests the router completed successfully.", ["model_name"])
FAILED = Counter("pd_router_request_failures", "Requests that failed at the router.", ["model_name", "stage"])
PROMPT_TOKENS = Counter("vllm:prompt_tokens", "Prompt tokens of completed requests (usage.prompt_tokens).", ["model_name"])
GEN_TOKENS = Counter("vllm:generation_tokens", "Output tokens of completed requests (usage.completion_tokens).", ["model_name"])

client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))
app = FastAPI()


def _headers(rid):
    return {"X-Request-Id": rid, "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}


def _record_usage(usage):
    if not usage:
        return
    PROMPT_TOKENS.labels(LABEL).inc(usage.get("prompt_tokens") or 0)
    GEN_TOKENS.labels(LABEL).inc(usage.get("completion_tokens") or 0)


def _has_token(obj):
    for c in obj.get("choices") or []:
        delta = c.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or c.get("text"):
            return True
    return False


async def _remote_prefill(api, body, rid):
    req = dict(body)
    req["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
                                 "remote_block_ids": None, "remote_host": None, "remote_port": None}
    req["stream"] = False
    req["max_tokens"] = 1
    if "max_completion_tokens" in req:
        req["max_completion_tokens"] = 1
    for k in ("stream_options", "min_tokens", "min_completion_tokens"):
        req.pop(k, None)
    r = await client.post(PREFILL_URLS[next(_prefill_rr)] + "/v1" + api, json=req, headers=_headers(rid))
    r.raise_for_status()
    return r.json().get("kv_transfer_params")


async def _handle(api, request: Request):
    t0 = time.perf_counter()
    body = await request.json()
    rid = str(uuid.uuid4())
    if P:
        try:
            kvp = await _remote_prefill(api, body, rid)
        except Exception as e:  # a failed prefill leg is a failed request, never a silent local prefill
            FAILED.labels(LABEL, "prefill").inc()
            return JSONResponse({"error": f"remote prefill failed: {e!r}"}, status_code=502)
        PREFILL_LEG.labels(LABEL).observe(time.perf_counter() - t0)
        if kvp:
            body["kv_transfer_params"] = kvp
    url = DECODE_URLS[next(_decode_rr)] + "/v1" + api

    if not body.get("stream"):
        r = await client.post(url, json=body, headers=_headers(rid))
        if r.status_code == 200:
            elapsed = time.perf_counter() - t0
            TTFT.labels(LABEL).observe(elapsed)
            E2E.labels(LABEL).observe(elapsed)
            SUCCESS.labels(LABEL).inc()
            _record_usage(r.json().get("usage"))
        else:
            FAILED.labels(LABEL, "decode").inc()
        return Response(content=r.content, status_code=r.status_code, media_type="application/json")

    # Streaming: always ask the decode instance for per-chunk cumulative usage (token
    # counters as the tokens stream), and strip it again where the client did not ask.
    client_opts = body.get("stream_options") or {}
    client_wants_usage = bool(client_opts.get("include_usage"))
    client_wants_continuous = client_wants_usage and bool(client_opts.get("continuous_usage_stats"))
    body["stream_options"] = {**client_opts, "include_usage": True, "continuous_usage_stats": True}
    upstream = await client.send(client.build_request("POST", url, json=body, headers=_headers(rid)), stream=True)
    if upstream.status_code != 200:
        content = await upstream.aread()
        await upstream.aclose()
        FAILED.labels(LABEL, "decode").inc()
        return Response(content=content, status_code=upstream.status_code, media_type="application/json")

    async def relay():
        first = last = None
        done = False
        prompt_counted = False
        completion_seen = 0
        buf = b""
        try:
            async for chunk in upstream.aiter_bytes():
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    forward = True
                    line = event.strip()
                    if line.startswith(b"data:"):
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            done = True
                        else:
                            try:
                                obj = json.loads(data)
                            except ValueError:
                                obj = None
                            if obj:
                                if _has_token(obj):
                                    now = time.perf_counter()
                                    if first is None:
                                        first = now
                                        TTFT.labels(LABEL).observe(now - t0)
                                    else:
                                        ITL.labels(LABEL).observe(now - last)
                                    last = now
                                usage = obj.get("usage")
                                if usage:
                                    if not prompt_counted and usage.get("prompt_tokens"):
                                        PROMPT_TOKENS.labels(LABEL).inc(usage["prompt_tokens"])
                                        prompt_counted = True
                                    ct = usage.get("completion_tokens") or 0
                                    if ct > completion_seen:
                                        GEN_TOKENS.labels(LABEL).inc(ct - completion_seen)
                                        completion_seen = ct
                                    if not obj.get("choices"):
                                        forward = client_wants_usage  # final usage-only chunk
                                    elif not client_wants_continuous:
                                        obj.pop("usage", None)
                                        event = b"data: " + json.dumps(obj, separators=(",", ":")).encode()
                    if forward:
                        yield event + b"\n\n"
            if buf:
                yield buf
        finally:
            await upstream.aclose()
            if done:
                SUCCESS.labels(LABEL).inc()
                E2E.labels(LABEL).observe(time.perf_counter() - t0)
            else:
                FAILED.labels(LABEL, "stream").inc()

    return StreamingResponse(relay(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/completions", request)


@app.get("/v1/models")
async def models():
    r = await client.get(DECODE_URLS[0] + "/v1/models")
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.get("/health")
async def health():
    """200 only when every backend of the rendered topology answers /health."""
    for u in PREFILL_URLS + DECODE_URLS:
        try:
            r = await client.get(u + "/health", timeout=5)
            if r.status_code != 200:
                return JSONResponse({"status": "starting", "backend": u, "code": r.status_code}, status_code=503)
        except httpx.HTTPError as e:
            return JSONResponse({"status": "starting", "backend": u, "error": type(e).__name__}, status_code=503)
    return {"status": "ok", "prefill_instances": P, "decode_instances": D}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    print(f"pd_router: prefill={PREFILL_URLS} decode={DECODE_URLS} label={LABEL} port={PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
