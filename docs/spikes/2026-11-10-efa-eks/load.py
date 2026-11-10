# Tiny closed-loop load for the spike: CONC workers send /v1/completions (fixed prompt, ignore_eos,
# MAX_TOKENS output) for SECONDS, streaming, and print one RESULT line (TTFT, inter-token latency,
# output tokens/s). Not a benchmark tool: it only makes two transports comparable on one layout.
# Usage: python3 load.py URL CONC SECONDS [MAX_TOKENS] [PROMPT_WORDS]
import asyncio, json, statistics, sys, time, aiohttp

url, conc, seconds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 200
words = int(sys.argv[5]) if len(sys.argv) > 5 else 450
ttfts, itls, tokens, errors = [], [], 0, 0

async def worker(session, wid, stop_at):
    global tokens, errors
    n = 0
    while time.perf_counter() < stop_at:
        n += 1
        prompt = "Request %d-%d. " % (wid, n) + "The quick brown fox jumps over the lazy dog. " * (words // 9)
        body = {"model": "spike", "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True,
                "temperature": 0.0, "stream": True}
        t0 = time.perf_counter(); last = None; got = 0
        try:
            async with session.post(url + "/v1/completions", json=body) as r:
                if r.status != 200:
                    errors += 1; await r.read(); continue
                async for line in r.content:
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
                    if not json.loads(line[6:])["choices"][0].get("text"): continue
                    now = time.perf_counter(); got += 1
                    if last is None: ttfts.append(now - t0)
                    else: itls.append(now - last)
                    last = now
            tokens += got
        except Exception:
            errors += 1

async def main():
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as s:
        await asyncio.gather(*[worker(s, i, t0 + seconds) for i in range(conc)])
    dt = time.perf_counter() - t0
    p = lambda xs, q: sorted(xs)[int(len(xs) * q)] * 1e3 if xs else float("nan")
    print("RESULT conc=%d requests=%d errors=%d ttft_p50=%.0fms ttft_p95=%.0fms itl_p50=%.1fms itl_p95=%.1fms chunks_per_s=%.0f"
          % (conc, len(ttfts), errors, p(ttfts, .5), p(ttfts, .95), p(itls, .5), p(itls, .95), tokens / dt), flush=True)

asyncio.run(main())
