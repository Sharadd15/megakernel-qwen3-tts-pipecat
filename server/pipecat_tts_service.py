"""Pipecat TTS service that consumes the websocket streaming endpoint."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator

import websockets

try:
    from pipecat.frames.frames import TTSAudioRawFrame
    from pipecat.services.tts_service import TTSService
except Exception:  # pragma: no cover - optional dependency during static checks
    TTSAudioRawFrame = object  # type: ignore

    class TTSService:  # type: ignore
        pass


class MegakernelQwenTTSService(TTSService):
    def __init__(
        self,
        ws_url: str = "ws://127.0.0.1:8000/v1/tts/ws",
        sample_rate: int = 24000,
        chunk_tokens: int = 8,
        strict_streaming: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ws_url = ws_url
        self._sample_rate = sample_rate
        self._chunk_tokens = chunk_tokens
        self._strict_streaming = strict_streaming

    async def run_tts(self, text: str) -> AsyncGenerator[TTSAudioRawFrame, None]:
        payload = {
            "text": text,
            "sample_rate": self._sample_rate,
            "chunk_tokens": self._chunk_tokens,
            "strict_streaming": self._strict_streaming,
        }

        async with websockets.connect(self._ws_url, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps(payload))
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    break
                except websockets.ConnectionClosed:
                    break

                event = json.loads(msg)
                if "error" in event:
                    raise RuntimeError(event["error"])

                if "summary" in event:
                    break

                pcm = base64.b64decode(event["pcm16_b64"])
                yield TTSAudioRawFrame(
                    audio=pcm,
                    sample_rate=event["sample_rate"],
                    num_channels=event.get("channels", 1),
                )
