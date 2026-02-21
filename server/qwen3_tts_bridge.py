"""Qwen3-TTS bridge that turns decoded talker tokens into PCM chunks.

This module supports multiple qwen-tts package APIs:
- legacy `QwenTTS`
- current `Qwen3TTSModel`
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import types
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from qwen_megakernel.model import MAX_SEQ_LEN, VOCAB_SIZE
from server.megakernel_backend import MegakernelDecodeBackend

logger = logging.getLogger(__name__)


class _Qwen3ModelAdapter:
    """Adapter for qwen_tts.Qwen3TTSModel style APIs."""

    def __init__(self, model):
        self.model = model
        self._custom_generate_fn = None
        self._merge_kwargs_patched = False

    def _common_kwargs(self) -> dict:
        kwargs = {
            "language": os.getenv("QWEN3_TTS_LANGUAGE", "Auto"),
        }
        speaker = os.getenv("QWEN3_TTS_SPEAKER", "Vivian")
        instruct = os.getenv("QWEN3_TTS_INSTRUCT", "")
        if speaker:
            kwargs["speaker"] = speaker
        if instruct:
            kwargs["instruct"] = instruct
        return kwargs

    def _pick_generate_method(self):
        for name in (
            "generate_custom_voice",
            "generate_voice_design",
            "generate",
            "infer",
        ):
            fn = getattr(self.model, name, None)
            if callable(fn):
                return name, fn
        return None, None

    def set_talker_decode_fn(self, fn):
        # Best-effort attach points for future/variant runtimes.
        for obj in (self.model, getattr(self.model, "talker", None)):
            if obj is None:
                continue
            for name in (
                "set_talker_decode_fn",
                "set_talker_decoder",
                "set_decoder_backend",
                "set_decode_backend",
                "set_decoder",
                "set_decode_fn",
                "set_backend",
            ):
                hook = getattr(obj, name, None)
                if callable(hook):
                    try:
                        hook(fn)
                        return True
                    except TypeError:
                        try:
                            hook("megakernel", fn)
                            return True
                        except Exception:
                            continue
                    except Exception:
                        continue
        return False

    def install_internal_custom_generate(self, custom_generate_fn) -> bool:
        """Install internal hook so Qwen3TTSForConditionalGeneration passes custom_generate to talker.generate."""
        inner = getattr(self.model, "model", None)
        talker = getattr(inner, "talker", None) if inner is not None else None
        if inner is None or talker is None:
            return False

        cls = type(inner)
        if not getattr(cls, "_megakernel_generate_patched", False):
            original_generate = cls.generate

            def patched_generate(self_obj, *args, **kwargs):
                talker_obj = getattr(self_obj, "talker", None)
                custom_gen = getattr(talker_obj, "_megakernel_custom_generate", None)
                if custom_gen is not None and "custom_generate" not in kwargs:
                    kwargs["custom_generate"] = custom_gen
                return original_generate(self_obj, *args, **kwargs)

            cls.generate = patched_generate
            cls._megakernel_generate_patched = True

        setattr(talker, "_megakernel_custom_generate", custom_generate_fn)
        self._custom_generate_fn = custom_generate_fn

        # Patch talker.generate directly to guarantee custom_generate injection.
        if not getattr(talker, "_megakernel_talker_generate_patched", False):
            original_talker_generate = talker.generate

            def patched_talker_generate(this, *args, **kwargs):
                if "custom_generate" not in kwargs and self._custom_generate_fn is not None:
                    kwargs["custom_generate"] = self._custom_generate_fn
                return original_talker_generate(*args, **kwargs)

            talker.generate = types.MethodType(patched_talker_generate, talker)
            setattr(talker, "_megakernel_talker_generate_patched", True)

        # Ensure generate_* wrappers in qwen_tts forward custom_generate through
        # _merge_generate_kwargs -> self.model.generate(**gen_kwargs).
        merge_fn = getattr(self.model, "_merge_generate_kwargs", None)
        if callable(merge_fn) and not self._merge_kwargs_patched:
            def patched_merge_generate_kwargs(**kwargs):
                out = merge_fn(**kwargs)
                if self._custom_generate_fn is not None:
                    out["custom_generate"] = self._custom_generate_fn
                return out

            setattr(self.model, "_merge_generate_kwargs", patched_merge_generate_kwargs)
            self._merge_kwargs_patched = True
        return True

    def generate(self, text: str):
        name, generate_fn = self._pick_generate_method()
        if generate_fn is None:
            raise RuntimeError("No generate method found on Qwen3TTSModel")

        kwargs = self._common_kwargs()
        kwargs.update({"text": text})
        if self._custom_generate_fn is not None:
            # Forward explicitly via generate_* -> self.model.generate(**gen_kwargs)
            kwargs["custom_generate"] = self._custom_generate_fn

        # Adjust kwargs for methods that do not support all fields.
        if name == "generate_voice_design":
            kwargs.pop("speaker", None)
        if name in ("generate", "infer"):
            kwargs.pop("speaker", None)
            kwargs.pop("instruct", None)

        out = generate_fn(**kwargs)

        # Common return: (wavs, sr)
        if isinstance(out, tuple) and len(out) >= 1:
            wavs = out[0]
            if isinstance(wavs, (list, tuple)):
                if len(wavs) == 0:
                    raise RuntimeError(
                        "Qwen3-TTS generate returned an empty wav list (no audio produced)"
                    )
                return wavs[0]
            return wavs

        # Fallback: direct waveform
        return out


class Qwen3TTSBridge:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        decoder_model_name: str = "Qwen/Qwen3-0.6B",
        verbose: bool = True,
    ):
        self.model_name = os.getenv("QWEN3_TTS_MODEL", model_name)
        decoder_model_name = os.getenv("QWEN3_DECODER_MODEL", decoder_model_name)
        self.decode_backend = MegakernelDecodeBackend(
            model_name=decoder_model_name,
            verbose=verbose,
        )
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._tts = None
        self._backend_connected = False
        self._backend_mode = "none"
        self._active_request_id = ""
        self._active_text = ""
        self._last_decode_summary = {
            "tokens_per_second": 0.0,
            "tokens_generated": 0,
            "hook_executions": 0,
            "request_id": "",
        }

    def preload(self) -> None:
        """Eagerly load qwen_tts runtime and attach backend hooks."""
        self._ensure_tts()

    def _ensure_tts(self):
        if self._tts is not None:
            return

        load_errors = []

        try:
            from qwen_tts import QwenTTS  # type: ignore

            self._tts = QwenTTS(model_name=self.model_name)
            self._backend_connected = self._attach_megakernel_backend()
            return
        except Exception as exc:
            load_errors.append(f"QwenTTS path: {exc!r}")

        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore

            model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )
            self._tts = _Qwen3ModelAdapter(model)
            self._backend_connected = self._attach_megakernel_backend()
            return
        except Exception as exc:
            load_errors.append(f"Qwen3TTSModel path: {exc!r}")

        raise RuntimeError(
            "Unable to initialize qwen-tts runtime. "
            + " | ".join(load_errors)
        )

    def _decode_with_megakernel(
        self, prompt: str, max_tokens: int = 512, stop_token_id: int | None = None
    ) -> list[int]:
        token_ids: list[int] = []
        for event in self.decode_backend.stream_decode(
            prompt=prompt, max_tokens=max_tokens, stop_token_id=stop_token_id
        ):
            if "summary" in event:
                self._last_decode_summary = event["summary"]
                break
            token_ids.append(event["token_id"])
        return token_ids

    def _talker_custom_generate_proxy(
        self,
        model,
        input_ids,
        logits_processor=None,
        stopping_criteria=None,
        generation_config=None,
        **model_kwargs,
    ):
        """Custom generate hook using a real megakernel step loop.

        This path currently supports batch size 1 and returns token ids as
        `torch.LongTensor`, which is an allowed `GenerationMixin.generate`
        return type and is generally more stable across qwen_tts variants than
        synthetic hidden-state objects.
        """
        del logits_processor
        del stopping_criteria
        del model_kwargs

        if input_ids is None:
            raise RuntimeError("custom_generate requires input_ids")
        if input_ids.ndim != 2:
            raise RuntimeError(f"Expected 2D input_ids, got shape={tuple(input_ids.shape)}")
        if input_ids.shape[0] != 1:
            raise RuntimeError("Megakernel custom_generate currently supports batch size 1 only")

        self._last_decode_summary["hook_executions"] = int(
            self._last_decode_summary.get("hook_executions", 0)
        ) + 1
        self._last_decode_summary["request_id"] = self._active_request_id

        if generation_config is None:
            generation_config = model.generation_config
        cfg_max_new_tokens = int(getattr(generation_config, "max_new_tokens", 256))
        min_new_tokens = int(getattr(generation_config, "min_new_tokens", 2))
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        eos_ids = []
        if isinstance(eos_token_id, list):
            eos_ids.extend(int(x) for x in eos_token_id)
        elif eos_token_id is not None:
            eos_ids.append(int(eos_token_id))
        if not eos_ids:
            model_eos = getattr(model.config, "eos_token_id", None)
            if model_eos is not None:
                eos_ids.append(int(model_eos))
        talker_cfg = getattr(model.config, "talker_config", object())
        codec_eos_id = getattr(talker_cfg, "codec_eos_token_id", None)
        if codec_eos_id is not None:
            eos_ids.append(int(codec_eos_id))
        codec_bos_id = int(getattr(talker_cfg, "codec_bos_id", 0))
        eos_ids = list(dict.fromkeys(eos_ids))
        eos_token_id = eos_ids[0] if eos_ids else None

        # Prevent runaway audio on strict mode by bounding talker decode length.
        hard_cap = int(os.getenv("QWEN3_TTS_STRICT_MAX_NEW_TOKENS", "192"))
        word_count = max(1, len([w for w in self._active_text.strip().split() if w]))
        char_count = len(self._active_text.strip())
        heuristic_budget = 12 + (word_count * 10) + min(char_count, 120) // 6
        if self._active_text.rstrip().endswith((".", "!", "?")):
            heuristic_budget += 8
        heuristic_budget = max(24, min(heuristic_budget, hard_cap))
        max_new_tokens = max(1, min(cfg_max_new_tokens, hard_cap, heuristic_budget))
        min_new_tokens = max(1, min(min_new_tokens, max_new_tokens))
        num_code_layers = int(
            getattr(
                talker_cfg,
                "num_code_layers",
                getattr(talker_cfg, "num_codebooks", 16),
            )
        )
        codebook_size = int(
            getattr(
                talker_cfg,
                "codebook_size",
                getattr(talker_cfg, "codec_size", 1024),
            )
        )

        codec_vocab_size = int(getattr(talker_cfg, "vocab_size", VOCAB_SIZE))
        dec = self.decode_backend.decoder
        hidden_size = int(getattr(model.config, "hidden_size", 1024))
        generated: list[int] = []
        hidden_states = []
        device = input_ids.device

        t0 = time.perf_counter()
        first = None
        with self.decode_backend.decode_lock:
            dec.reset()

            prompt_ids = [int(x) for x in input_ids[0].tolist()]
            if len(prompt_ids) >= MAX_SEQ_LEN:
                # Keep the latest context to stay within fixed KV cache size.
                prompt_ids = prompt_ids[-(MAX_SEQ_LEN - 1) :]
            if len(prompt_ids) == 0:
                # Some generation paths can provide empty bootstrap ids when using inputs_embeds.
                current = codec_bos_id
            else:
                for tid in prompt_ids[:-1]:
                    safe_tid = int(tid) % max(VOCAB_SIZE, 1)
                    dec.step(safe_tid)
                current = int(prompt_ids[-1]) % max(VOCAB_SIZE, 1)

            # Reserve room for EOS to avoid position overflow.
            remaining_positions = max(0, MAX_SEQ_LEN - dec.position - 1)
            steps_to_run = max(0, min(max_new_tokens, remaining_positions))
            for _ in range(steps_to_run):
                tok = int(dec.step(current))
                if first is None:
                    first = time.perf_counter()

                # Keep token in talker vocab range.
                tok = tok % max(codec_vocab_size, 1)

                # Avoid immediate EOS-only outputs that can break downstream decode.
                if eos_token_id is not None and tok == int(eos_token_id) and len(generated) < min_new_tokens:
                    tok = codec_bos_id

                generated.append(tok)
                # Feed next-step token in decoder vocab range.
                current = int(tok) % max(VOCAB_SIZE, 1)

                # qwen_tts variants consume custom_generate hidden_states; keep a
                # compatible synthetic structure per generated step.
                code_id = int(tok) % max(codebook_size, 1)
                code_tensor = torch.full(
                    (1, max(num_code_layers, 1)),
                    code_id,
                    dtype=torch.long,
                    device=device,
                )
                fake_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32, device=device)
                hidden_states.append(((fake_hidden,), code_tensor))

                if eos_ids and tok in eos_ids and len(generated) >= min_new_tokens:
                    break

        if not generated:
            generated = [codec_bos_id]

        # Ensure a hard stop token so downstream talker code extraction can
        # reliably find segment boundaries.
        if eos_token_id is not None and generated[-1] != int(eos_token_id):
            generated.append(int(eos_token_id))

        # Keep memory bounded if runtime sets a small cap.
        if len(generated) > max_new_tokens:
            generated = generated[:max_new_tokens]

        elapsed = max(time.perf_counter() - t0, 1e-9)
        self._last_decode_summary = {
            "tokens_generated": len(generated),
            "elapsed_ms": elapsed * 1000.0,
            "tokens_per_second": len(generated) / elapsed,
            "ttft_ms": ((first or time.perf_counter()) - t0) * 1000.0,
            "max_new_tokens": max_new_tokens,
            "hook_executions": int(self._last_decode_summary.get("hook_executions", 1)),
            "request_id": self._active_request_id,
        }
        logger.info(
            "Megakernel talker custom_generate executed: request_id=%s tokens=%d tps=%.2f",
            self._active_request_id,
            len(generated),
            self._last_decode_summary["tokens_per_second"],
        )

        if generated:
            generated_tensor = torch.tensor(
                [generated], dtype=input_ids.dtype, device=device
            )
            sequences = torch.cat([input_ids, generated_tensor], dim=1)
        else:
            sequences = input_ids

        return GenerateDecoderOnlyOutput(
            sequences=sequences,
            hidden_states=tuple(hidden_states),
        )

    def _attach_megakernel_backend(self) -> bool:
        if self._tts is None:
            return False

        # Prefer internal patch path for Qwen3TTSModel variants where public hooks
        # are present but do not actually drive talker decode in generate().
        install = getattr(self._tts, "install_internal_custom_generate", None)
        if callable(install):
            try:
                ok = bool(install(self._talker_custom_generate_proxy))
            except Exception:
                ok = False
            if ok:
                self._backend_mode = "internal_custom_generate_megakernel"
                return True

        hook = getattr(self._tts, "set_talker_decode_fn", None)
        if callable(hook):
            try:
                ok = bool(hook(self._decode_with_megakernel))
                if ok:
                    self._backend_mode = "public_hook"
                    return True
            except Exception:
                pass

        for obj in (self._tts, getattr(self._tts, "talker", None)):
            if obj is None:
                continue
            for name in (
                "set_talker_decode_fn",
                "set_talker_decoder",
                "set_decoder_backend",
                "set_decode_backend",
                "set_decoder",
                "set_decode_fn",
                "set_backend",
            ):
                h = getattr(obj, name, None)
                if callable(h):
                    try:
                        h(self._decode_with_megakernel)
                        self._backend_mode = "public_hook"
                        return True
                    except TypeError:
                        try:
                            h("megakernel", self._decode_with_megakernel)
                            self._backend_mode = "public_hook"
                            return True
                        except Exception:
                            continue
                    except Exception:
                        continue

        return False

    @staticmethod
    def _float_to_pcm16_bytes(wave: np.ndarray) -> bytes:
        wave = np.asarray(wave)
        if wave.dtype != np.float32:
            wave = wave.astype(np.float32, copy=False)
        wave = np.clip(wave, -1.0, 1.0)
        pcm16 = (wave * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    async def _run_in_pool(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, lambda: fn(*args, **kwargs))

    def _reset_pool(self) -> None:
        """Reset executor after timed-out/poisoned generation jobs."""
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._pool = ThreadPoolExecutor(max_workers=1)

    async def stream_audio(
        self,
        text: str,
        max_tokens: int,
        chunk_tokens: int,
        sample_rate: int,
        strict_streaming: bool,
    ) -> AsyncIterator[dict]:
        del max_tokens
        del chunk_tokens

        self._ensure_tts()
        assert self._tts is not None
        self._active_request_id = uuid4().hex[:12]
        self._active_text = text
        self._last_decode_summary = {
            "tokens_per_second": 0.0,
            "tokens_generated": 0,
            "hook_executions": 0,
            "request_id": self._active_request_id,
        }

        t0 = time.perf_counter()
        first_chunk_at = None
        audio_samples = 0

        if strict_streaming and not self._backend_connected:
            raise RuntimeError(
                "Megakernel talker backend is not attached to this Qwen3-TTS runtime"
            )

        internal_megakernel_mode = self._backend_mode == "internal_custom_generate_megakernel"
        if strict_streaming and not internal_megakernel_mode:
            raise RuntimeError(
                "Strict mode requires internal_custom_generate_megakernel backend mode"
            )

        synth = getattr(self._tts, "generate", None)
        if not callable(synth):
            raise RuntimeError("Qwen3-TTS runtime does not expose generate()")
        gen_timeout_s = float(os.getenv("QWEN3_TTS_GENERATE_TIMEOUT_SEC", "120"))
        try:
            wave = await asyncio.wait_for(
                self._run_in_pool(synth, text),
                timeout=max(gen_timeout_s, 1.0),
            )
        except TimeoutError as exc:
            # run_in_executor timeout does not kill the underlying worker thread.
            # Reset the pool so subsequent requests aren't blocked behind a hung job.
            self._reset_pool()
            raise RuntimeError(
                f"TTS generation timed out after {gen_timeout_s:.1f}s"
            ) from exc
        except Exception:
            # Defensive pool reset after hard runtime failures (CUDA/runtime errors).
            self._reset_pool()
            raise
        wave = np.asarray(wave, dtype=np.float32)

        frame = max(1, int(sample_rate * 0.04))
        for i in range(0, wave.shape[0], frame):
            part = wave[i : i + frame]
            pcm = self._float_to_pcm16_bytes(part)
            now = time.perf_counter()
            if first_chunk_at is None:
                first_chunk_at = now
            samples = len(pcm) // 2
            audio_samples += samples
            yield {
                "pcm16_b64": base64.b64encode(pcm).decode("ascii"),
                "sample_rate": sample_rate,
                "channels": 1,
                "samples": samples,
                "elapsed_ms": (now - t0) * 1000.0,
            }

        decode_tps = float(self._last_decode_summary.get("tokens_per_second", 0.0))
        tokens_generated = int(self._last_decode_summary.get("tokens_generated", 0))
        hook_executions = int(self._last_decode_summary.get("hook_executions", 0))
        if strict_streaming and hook_executions <= 0:
            raise RuntimeError(
                "Strict mode requires talker custom_generate hook execution, "
                f"but hook_executions={hook_executions}"
            )
        if strict_streaming and tokens_generated <= 0:
            raise RuntimeError(
                "Strict mode requires megakernel talker tokens, "
                f"but tokens_generated={tokens_generated}"
            )
        if strict_streaming and decode_tps <= 0.0:
            raise RuntimeError(
                "Strict mode requires megakernel-driven talker decode, "
                f"but decode_tokens_per_second={decode_tps:.2f}"
            )

        total_elapsed = max(time.perf_counter() - t0, 1e-9)
        audio_seconds = audio_samples / float(sample_rate)
        ttfc_ms = ((first_chunk_at or time.perf_counter()) - t0) * 1000.0
        yield {
            "summary": {
                "audio_seconds": audio_seconds,
                "elapsed_ms": total_elapsed * 1000.0,
                "ttfc_ms": ttfc_ms,
                "rtf": total_elapsed / max(audio_seconds, 1e-6),
                "backend_mode": self._backend_mode,
                "request_id": self._active_request_id,
                "hook_executions": hook_executions,
                "tokens_generated": tokens_generated,
                "decode_tokens_per_second": decode_tps,
            }
        }
