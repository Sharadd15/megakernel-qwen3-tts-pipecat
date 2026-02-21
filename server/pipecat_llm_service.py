"""Pipecat LLM stage backed by the local megakernel decode websocket endpoint."""

from __future__ import annotations

import asyncio
import json

import websockets

try:
    from pipecat.frames.frames import TextFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
except Exception:  # pragma: no cover - optional dependency during static checks
    TextFrame = object  # type: ignore

    class FrameProcessor:  # type: ignore
        async def process_frame(self, frame, direction):
            return

        async def push_frame(self, frame, direction):
            return

    class FrameDirection:  # type: ignore
        DOWNSTREAM = None


class MegakernelLLMService(FrameProcessor):
    def __init__(
        self,
        ws_url: str = "ws://127.0.0.1:8000/v1/decode/ws",
        max_tokens: int = 256,
        prompt_prefix: str = "",
    ):
        super().__init__()
        self.ws_url = ws_url
        self.max_tokens = max_tokens
        self.prompt_prefix = prompt_prefix

    async def _generate_text(self, prompt: str) -> str:
        payload = {"prompt": prompt, "max_tokens": self.max_tokens}
        pieces: list[str] = []

        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
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
                pieces.append(event.get("text", ""))

        return "".join(pieces).strip()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            prompt = frame.text.strip()
            if prompt:
                if self.prompt_prefix:
                    prompt = f"{self.prompt_prefix}\n\nUser: {prompt}\nAssistant:"
                text = await self._generate_text(prompt)
                if text:
                    await self.push_frame(TextFrame(text=text), FrameDirection.DOWNSTREAM)
            return

        await self.push_frame(frame, direction)
