"""Megakernel decode backend with incremental streaming helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from threading import Lock

from qwen_megakernel.model import Decoder


class MegakernelDecodeBackend:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", verbose: bool = True):
        self.decoder = Decoder(model_name=model_name, verbose=verbose)
        self.tokenizer = self.decoder.tokenizer
        # Decoder owns mutable GPU buffers/KV cache; never step it concurrently.
        self.decode_lock = Lock()

    def stream_decode(
        self,
        prompt: str,
        max_tokens: int,
        stop_token_id: int | None = None,
    ) -> Iterator[dict]:
        with self.decode_lock:
            self.decoder.reset()

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            if not prompt_ids:
                return

            for tid in prompt_ids[:-1]:
                self.decoder.step(tid)

            eos_id = self.tokenizer.eos_token_id
            tok = prompt_ids[-1]

            start = time.perf_counter()
            first_token_at = None

            for step in range(max_tokens):
                tok = self.decoder.step(tok)
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now

                piece = self.tokenizer.decode([tok], skip_special_tokens=False)
                yield {
                    "token_id": tok,
                    "text": piece,
                    "step": step,
                    "elapsed_ms": (now - start) * 1000.0,
                }

                if tok == eos_id:
                    break
                if stop_token_id is not None and tok == stop_token_id:
                    break

            end = time.perf_counter()
            tokens_generated = step + 1
            elapsed_s = max(end - start, 1e-9)

            yield {
                "summary": {
                    "tokens_generated": tokens_generated,
                    "elapsed_ms": elapsed_s * 1000.0,
                    "tokens_per_second": tokens_generated / elapsed_s,
                    "ttft_ms": ((first_token_at or end) - start) * 1000.0,
                }
            }
