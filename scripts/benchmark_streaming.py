"""Benchmark streaming token and audio endpoints.

Measures:
- decode token/s
- decode TTFT
- TTS TTFC
- TTS RTF
- end-to-end latency

Includes evaluator-friendly diagnostics for buffering behavior and target miss.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

import websockets

TTFC_TARGET_MS = 90.0
RTF_TARGET = 0.3


def _format_server_error(event: dict) -> str:
    details = [f"error={event.get('error', 'unknown')}"]
    if "error_type" in event:
        details.append(f"type={event['error_type']}")
    if "traceback" in event:
        details.append(f"traceback=\n{event['traceback']}")
    return "\n".join(details)


async def bench_decode(
    ws_url: str,
    prompt: str,
    max_tokens: int,
    recv_timeout_s: float,
):
    t0 = time.perf_counter()
    first = None
    n = 0

    async with websockets.connect(
        ws_url,
        ping_interval=None,
        close_timeout=2,
    ) as ws:
        await ws.send(json.dumps({"prompt": prompt, "max_tokens": max_tokens}))
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=recv_timeout_s)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"decode ws timed out waiting for next event (>{recv_timeout_s:.1f}s)"
                ) from exc
            event = json.loads(msg)
            if "error" in event:
                raise RuntimeError(_format_server_error(event))
            if "summary" in event:
                summary = event["summary"]
                break
            n += 1
            if first is None:
                first = time.perf_counter()

    return {
        "tokens": n,
        "ttft_ms": ((first or time.perf_counter()) - t0) * 1000.0,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "tokens_per_second": float(summary.get("tokens_per_second", 0.0)),
    }


async def bench_tts(
    ws_url: str,
    text: str,
    strict_streaming: bool,
    sample_rate: int,
    recv_timeout_s: float,
):
    t0 = time.perf_counter()
    first = None
    audio_samples = 0
    async with websockets.connect(
        ws_url,
        ping_interval=None,
        close_timeout=2,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "text": text,
                    "sample_rate": sample_rate,
                    "strict_streaming": strict_streaming,
                }
            )
        )
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=recv_timeout_s)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"tts ws timed out waiting for next event (>{recv_timeout_s:.1f}s)"
                ) from exc
            event = json.loads(msg)
            if "error" in event:
                raise RuntimeError(_format_server_error(event))
            if "summary" in event:
                summary = event["summary"]
                break
            if first is None:
                first = time.perf_counter()
            audio_samples += int(event.get("samples", 0))

    elapsed = time.perf_counter() - t0
    audio_seconds = audio_samples / float(sample_rate)
    ttfc_ms = ((first or time.perf_counter()) - t0) * 1000.0
    return {
        "ttfc_ms": ttfc_ms,
        "elapsed_ms": elapsed * 1000.0,
        "audio_seconds": audio_seconds,
        "rtf": elapsed / max(audio_seconds, 1e-6),
        "reported": summary,
        "likely_buffered": ttfc_ms >= (elapsed * 1000.0 * 0.9),
    }


def _mean(items: list[float]) -> float:
    return statistics.fmean(items) if items else 0.0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--text", default="Hello. This is a streaming TTS benchmark.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Requested TTS sample rate sent to /v1/tts/ws.",
    )
    parser.add_argument(
        "--strict-streaming",
        action="store_true",
        help="Enforce strict megakernel talker decode validation checks.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured runs (warmup run is done automatically).",
    )
    parser.add_argument(
        "--fail-on-target-miss",
        action="store_true",
        help="Exit non-zero if TTFC>90ms or RTF>0.3.",
    )
    parser.add_argument(
        "--decode-timeout-sec",
        type=float,
        default=120.0,
        help="Per-message recv timeout for decode websocket.",
    )
    parser.add_argument(
        "--tts-timeout-sec",
        type=float,
        default=180.0,
        help="Per-message recv timeout for tts websocket.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup requests and immediately run measured iterations.",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.sample_rate < 8000:
        raise ValueError("--sample-rate must be >= 8000")
    if args.decode_timeout_sec <= 0 or args.tts_timeout_sec <= 0:
        raise ValueError("timeouts must be > 0")

    decode_url = f"ws://{args.host}:{args.port}/v1/decode/ws"
    tts_url = f"ws://{args.host}:{args.port}/v1/tts/ws"

    if not args.skip_warmup:
        await bench_decode(
            decode_url,
            args.prompt,
            args.max_tokens,
            args.decode_timeout_sec,
        )
        await bench_tts(
            tts_url,
            args.text,
            args.strict_streaming,
            args.sample_rate,
            args.tts_timeout_sec,
        )

    decode_runs = []
    tts_runs = []
    for _ in range(args.runs):
        decode_runs.append(
            await bench_decode(
                decode_url,
                args.prompt,
                args.max_tokens,
                args.decode_timeout_sec,
            )
        )
        tts_runs.append(
            await bench_tts(
                tts_url,
                args.text,
                args.strict_streaming,
                args.sample_rate,
                args.tts_timeout_sec,
            )
        )

    decode_tps = [r["tokens_per_second"] for r in decode_runs]
    decode_ttft = [r["ttft_ms"] for r in decode_runs]
    ttfc = [r["ttfc_ms"] for r in tts_runs]
    rtf = [r["rtf"] for r in tts_runs]
    e2e = [r["elapsed_ms"] for r in tts_runs]

    latest_summary = tts_runs[-1]["reported"]
    likely_buffered = any(r["likely_buffered"] for r in tts_runs)

    print("=" * 72)
    print("Megakernel + Qwen3-TTS benchmark")
    print("=" * 72)
    print(f"Runs: {args.runs} ({'no warmup' if args.skip_warmup else 'after 1 warmup'})")
    print(f"strict_streaming: {args.strict_streaming}")
    print(f"sample_rate: {args.sample_rate}")
    print(f"decode_timeout_sec: {args.decode_timeout_sec:.1f}")
    print(f"tts_timeout_sec: {args.tts_timeout_sec:.1f}")
    print("Targets:")
    print(f"  TTFC < {TTFC_TARGET_MS:.1f} ms")
    print(f"  RTF  < {RTF_TARGET:.2f}")
    print()
    print("Averages:")
    print(f"  Decode tok/s: {_mean(decode_tps):.2f}")
    print(f"  Decode TTFT: {_mean(decode_ttft):.2f} ms")
    print(f"  TTS TTFC:    {_mean(ttfc):.2f} ms")
    print(f"  TTS RTF:     {_mean(rtf):.4f}")
    print(f"  End-to-end:  {_mean(e2e):.2f} ms")
    print()
    print("Best:")
    print(f"  Decode tok/s: {max(decode_tps):.2f}")
    print(f"  Decode TTFT: {min(decode_ttft):.2f} ms")
    print(f"  TTS TTFC:    {min(ttfc):.2f} ms")
    print(f"  TTS RTF:     {min(rtf):.4f}")
    print(f"  End-to-end:  {min(e2e):.2f} ms")
    print()
    print("Diagnostics:")
    print(f"  Likely buffered TTS path: {'yes' if likely_buffered else 'no'}")
    print(
        "  Megakernel talker decode token rate (reported): "
        f"{float(latest_summary.get('decode_tokens_per_second', 0.0)):.2f}"
    )
    print(f"  Talker hook executions: {int(latest_summary.get('hook_executions', 0))}")
    print(f"  Talker tokens generated: {int(latest_summary.get('tokens_generated', 0))}")
    print(f"  Reported backend mode: {latest_summary.get('backend_mode', 'unknown')}")
    print("Reported summary (last run):")
    print(json.dumps(latest_summary, indent=2))

    ttfc_ok = _mean(ttfc) < TTFC_TARGET_MS
    rtf_ok = _mean(rtf) < RTF_TARGET
    if not ttfc_ok or not rtf_ok:
        print()
        print("Target status: MISS")
        if not ttfc_ok:
            print(f"  - TTFC average {_mean(ttfc):.2f} ms is above {TTFC_TARGET_MS:.1f} ms")
        if not rtf_ok:
            print(f"  - RTF average {_mean(rtf):.4f} is above {RTF_TARGET:.2f}")
    else:
        print()
        print("Target status: PASS")

    if args.fail_on_target_miss and (not ttfc_ok or not rtf_ok):
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
