"""
aimodel proxy — per-app Ollama routing with stats logging.

Listens on two ports (one per app), forwards all requests to a single Ollama
instance, and prints one stats line per completed /api/chat or /api/generate
response:

  2026-06-10 14:23:01  [generative-radio]  in:  847 tok @ 38.5 tok/s   out: 312 tok @ 44.2 tok/s

When inference slots are full, incoming requests queue and the proxy prints
status lines so you can see concurrency at a glance:

  2026-06-10 14:23:01  [logger          ]  queued   (active=4/4  waiting=1)
  2026-06-10 14:23:24  [logger          ]  started  (waited 23.1s)
  2026-06-10 14:23:26  [logger          ]  in:  523 tok @ 41.2 tok/s   out:  89 tok @ 48.1 tok/s

Run:
  python server.py
  AIMODEL_OLLAMA_URL=http://localhost:11434 \
  AIMODEL_PROXY_PORT_RADIO=11430 \
  AIMODEL_PROXY_PORT_LOGGER=11431 \
  python server.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stderr,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("AIMODEL_OLLAMA_URL", "http://127.0.0.1:11434")

APP_PORTS: dict[int, str] = {
    int(os.getenv("AIMODEL_PROXY_PORT_RADIO", "11430")): "generative-radio",
    int(os.getenv("AIMODEL_PROXY_PORT_LOGGER", "11431")): "logger",
}

# Match the Ollama setting so the proxy queues at the same limit.
_OLLAMA_NUM_PARALLEL = int(os.getenv("OLLAMA_NUM_PARALLEL", "4"))

# Initialized in lifespan (requires event loop).
_client: httpx.AsyncClient
_inference_sem: asyncio.Semaphore

# Active / waiting counters (asyncio is single-threaded — no locks needed).
_active = 0
_queued = 0

# ---------------------------------------------------------------------------
# Lifespan — create / close shared resources
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _client, _inference_sem
    _inference_sem = asyncio.Semaphore(_OLLAMA_NUM_PARALLEL)
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    yield
    await _client.aclose()

# ---------------------------------------------------------------------------
# Stats formatting
# ---------------------------------------------------------------------------

_COL_WIDTH = max(len(n) for n in APP_PORTS.values())


def _fmt_stats(app: str, body: dict) -> str | None:
    """Return a formatted stats line from an Ollama response body, or None."""
    prompt_count: int | None = body.get("prompt_eval_count")
    prompt_ns: int | None = body.get("prompt_eval_duration")
    eval_count: int | None = body.get("eval_count")
    eval_ns: int | None = body.get("eval_duration")

    if not (prompt_count and prompt_ns and eval_count and eval_ns):
        return None

    prompt_tps = prompt_count / (prompt_ns / 1e9)
    eval_tps = eval_count / (eval_ns / 1e9)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = f"[{app:{_COL_WIDTH}}]"
    return (
        f"{ts}  {label}"
        f"  in: {prompt_count:>5} tok @ {prompt_tps:>5.1f} tok/s"
        f"   out: {eval_count:>4} tok @ {eval_tps:>5.1f} tok/s"
    )


# ---------------------------------------------------------------------------
# Inference slot — queuing + monitoring
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _inference_slot(app: str):
    """Acquire one parallel inference slot; log if the request had to queue."""
    global _active, _queued

    label = f"[{app:{_COL_WIDTH}}]"
    waited = False
    t0: float = 0.0

    if _inference_sem.locked():
        _queued += 1
        waited = True
        t0 = asyncio.get_event_loop().time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{ts}  {label}  queued   (active={_active}/{_OLLAMA_NUM_PARALLEL}  waiting={_queued})",
            flush=True,
        )

    await _inference_sem.acquire()

    if waited:
        _queued -= 1
        elapsed = asyncio.get_event_loop().time() - t0
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts}  {label}  started  (waited {elapsed:.1f}s)", flush=True)

    _active += 1
    try:
        yield
    finally:
        _active -= 1
        _inference_sem.release()


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

def _app_name(request: Request) -> str:
    port: int = request.scope["server"][1]
    return APP_PORTS.get(port, f"unknown:{port}")


def _strip_keep_alive(body: dict) -> dict:
    """Remove keep_alive from the request so OLLAMA_KEEP_ALIVE governs."""
    if "keep_alive" in body:
        body = dict(body)
        del body["keep_alive"]
    return body


def _disable_thinking(body: dict) -> dict:
    """Inject think=False so reasoning models (e.g. Qwen3) skip extended thinking."""
    if body.get("think") is not False:
        body = dict(body)
        body["think"] = False
    return body


_STATS_PATHS = {"/api/chat", "/api/generate"}


async def _proxy_non_streaming(
    app: str,
    method: str,
    url: str,
    headers: dict,
    content: bytes,
) -> Response:
    async with _inference_slot(app):
        try:
            r = await _client.request(method, url, content=content, headers=headers)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
            return Response(content=f"Proxy error: {exc}", status_code=504)

    try:
        body = r.json()
        line = _fmt_stats(app, body)
        if line:
            print(line, flush=True)
    except Exception:
        pass

    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=dict(r.headers),
    )


async def _proxy_streaming(
    app: str,
    method: str,
    url: str,
    headers: dict,
    content: bytes,
) -> StreamingResponse:
    """Stream response chunks; hold the inference slot for the full generation."""

    async def _generate() -> AsyncIterator[bytes]:
        async with _inference_slot(app):
            try:
                async with _client.stream(method, url, content=content, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
                        try:
                            obj = json.loads(chunk.decode())
                            if obj.get("done"):
                                line = _fmt_stats(app, obj)
                                if line:
                                    print(line, flush=True)
                        except Exception:
                            pass
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
                yield json.dumps({"error": f"Proxy error: {exc}", "done": True}).encode()

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/health")
async def health(request: Request) -> dict:
    port: int = request.scope["server"][1]
    app_name = APP_PORTS.get(port, f"unknown:{port}")
    return {
        "status": "ok",
        "app": app_name,
        "ollama": OLLAMA_URL,
        "slots": {"active": _active, "queued": _queued, "limit": _OLLAMA_NUM_PARALLEL},
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def proxy(path: str, request: Request) -> Response:
    app_name = _app_name(request)
    raw_body = await request.body()

    # Normalize keep_alive on inference paths.
    forward_body = raw_body
    is_inference = f"/{path}" in _STATS_PATHS
    if is_inference and raw_body:
        try:
            parsed = json.loads(raw_body)
            cleaned = _strip_keep_alive(parsed)
            cleaned = _disable_thinking(cleaned)
            # Detect whether the client requested streaming.
            streaming = cleaned.get("stream", True)
            forward_body = json.dumps(cleaned).encode()
        except Exception:
            streaming = False
    else:
        streaming = False

    # Build forwarded headers (drop host so httpx sets the correct one).
    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    target_url = f"{OLLAMA_URL}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    if is_inference and streaming:
        return await _proxy_streaming(app_name, request.method, target_url, fwd_headers, forward_body)
    elif is_inference:
        return await _proxy_non_streaming(app_name, request.method, target_url, fwd_headers, forward_body)
    else:
        # Non-inference pass-through (tags, blobs, version, etc.) — no slot needed.
        try:
            r = await _client.request(
                request.method, target_url, content=raw_body, headers=fwd_headers,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
            return Response(content=f"Proxy error: {exc}", status_code=504)
        return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))


# ---------------------------------------------------------------------------
# Multi-port server entry point
# ---------------------------------------------------------------------------

async def _serve_all() -> None:
    servers = []
    for port, name in APP_PORTS.items():
        cfg = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(cfg)
        servers.append((name, port, server))

    print(f"aimodel proxy ready — forwarding to {OLLAMA_URL}", flush=True)
    for name, port, _ in servers:
        print(f"  {name:<20} → http://127.0.0.1:{port}", flush=True)
    print(f"  inference slots: {_OLLAMA_NUM_PARALLEL}", flush=True)
    print(flush=True)

    await asyncio.gather(*[s.serve() for _, _, s in servers])


if __name__ == "__main__":
    try:
        asyncio.run(_serve_all())
    except KeyboardInterrupt:
        pass
