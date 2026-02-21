"""FastAPI app exposing streaming decode and TTS endpoints."""

from __future__ import annotations

import json
import os
import traceback

from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

from server.megakernel_backend import MegakernelDecodeBackend
from server.qwen3_tts_bridge import Qwen3TTSBridge
from server.schemas import DecodeRequest, TTSRequest

app = FastAPI(title="Qwen Megakernel TTS Server", version="0.1.0")

_decode_backend = MegakernelDecodeBackend(
    model_name=os.getenv("QWEN3_DECODER_MODEL", "Qwen/Qwen3-0.6B")
)
_tts_bridge = Qwen3TTSBridge()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.on_event("startup")
def startup_warmup():
    # Force full model/runtime initialization at boot (no lazy first-request load).
    _tts_bridge.preload()


@app.post("/v1/decode/stream")
def decode_stream(req: DecodeRequest):
    def event_stream():
        for event in _decode_backend.stream_decode(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            stop_token_id=req.stop_token_id,
        ):
            yield json.dumps(event) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.websocket("/v1/decode/ws")
async def decode_ws(ws: WebSocket):
    await ws.accept()
    payload = await ws.receive_json()
    req = DecodeRequest(**payload)
    for event in _decode_backend.stream_decode(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        stop_token_id=req.stop_token_id,
    ):
        await ws.send_json(event)
    await ws.close()


@app.websocket("/v1/tts/ws")
async def tts_ws(ws: WebSocket):
    await ws.accept()
    payload = await ws.receive_json()
    req = TTSRequest(**payload)

    try:
        async for event in _tts_bridge.stream_audio(
            text=req.text,
            max_tokens=req.max_tokens,
            chunk_tokens=req.chunk_tokens,
            sample_rate=req.sample_rate,
            strict_streaming=req.strict_streaming,
        ):
            await ws.send_json(event)
    except Exception as exc:
        await ws.send_json(
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        await ws.close()
