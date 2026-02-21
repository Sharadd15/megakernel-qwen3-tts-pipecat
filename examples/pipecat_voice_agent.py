"""Pipecat demo aligned with assignment scope: STT -> LLM -> Megakernel-backed Qwen3-TTS."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.pipecat_llm_service import MegakernelLLMService
from server.pipecat_tts_service import MegakernelQwenTTSService


def _require(name: str):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


async def main():
    try:
        from pipecat.frames.frames import InterimTranscriptionFrame, TextFrame, TranscriptionFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.transports.local import audio as local_audio_mod
    except Exception as exc:
        raise RuntimeError(
            "Pipecat dependencies are missing. Install pipecat + deepgram extras."
        ) from exc

    class FinalTranscriptToText(FrameProcessor):
        """Convert final STT transcript frames into text prompts for LLM."""

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            if isinstance(frame, InterimTranscriptionFrame):
                return
            if isinstance(frame, TranscriptionFrame):
                text = frame.text.strip()
                if text:
                    await self.push_frame(TextFrame(text=text), FrameDirection.DOWNSTREAM)
                return

            await self.push_frame(frame, direction)

    deepgram_api_key = _require("DEEPGRAM_API_KEY")

    LocalAudioTransport = local_audio_mod.LocalAudioTransport
    transport_sig = inspect.signature(LocalAudioTransport.__init__)

    if "params" in transport_sig.parameters:
        params_cls = getattr(local_audio_mod, "LocalAudioTransportParams", None)
        if params_cls is None:
            raise RuntimeError(
                "LocalAudioTransport requires `params`, but LocalAudioTransportParams is unavailable in this Pipecat build."
            )
        params_sig = inspect.signature(params_cls.__init__)
        params_kwargs = {}
        if "sample_rate" in params_sig.parameters:
            params_kwargs["sample_rate"] = 16000
        params = params_cls(**params_kwargs)
        transport = LocalAudioTransport(params)
    else:
        transport_kwargs = {}
        if "sample_rate" in transport_sig.parameters:
            transport_kwargs["sample_rate"] = 16000
        transport = LocalAudioTransport(**transport_kwargs)
    stt = DeepgramSTTService(api_key=deepgram_api_key)
    stt_to_text = FinalTranscriptToText()
    llm = MegakernelLLMService(
        ws_url=os.getenv("MEGAKERNEL_LLM_WS", "ws://127.0.0.1:8000/v1/decode/ws"),
        max_tokens=int(os.getenv("MEGAKERNEL_LLM_MAX_TOKENS", "128")),
        prompt_prefix=os.getenv("MEGAKERNEL_LLM_PREFIX", "Reply briefly and clearly."),
    )
    tts = MegakernelQwenTTSService(
        ws_url=os.getenv("MEGAKERNEL_TTS_WS", "ws://127.0.0.1:8000/v1/tts/ws")
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            stt_to_text,
            llm,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
