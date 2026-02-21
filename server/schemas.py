"""Typed request/response models for megakernel streaming services."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DecodeRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(256, ge=1, le=4096)
    stop_token_id: Optional[int] = Field(None, description="Optional forced stop token")


class DecodeTokenEvent(BaseModel):
    token_id: int
    text: str
    step: int
    elapsed_ms: float


class DecodeSummary(BaseModel):
    tokens_generated: int
    elapsed_ms: float
    tokens_per_second: float
    ttft_ms: float


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    max_tokens: int = Field(512, ge=1, le=8192)
    chunk_tokens: int = Field(8, ge=1, le=256)
    sample_rate: int = Field(24000, ge=8000, le=96000)
    strict_streaming: bool = Field(
        True,
        description="If true, fail when the TTS model does not expose native streaming",
    )


class TTSAudioChunkEvent(BaseModel):
    pcm16_b64: str
    sample_rate: int
    channels: int = 1
    samples: int
    elapsed_ms: float


class TTSSummary(BaseModel):
    audio_seconds: float
    elapsed_ms: float
    ttfc_ms: float
    rtf: float
    decode_tokens_per_second: float
