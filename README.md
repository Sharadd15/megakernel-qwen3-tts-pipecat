## RTX 5090 Megakernel -> Qwen3-TTS -> Pipecat

Take-home submission for integrating AlpinDale's RTX 5090 decode megakernel with Qwen3-TTS in a Pipecat pipeline.

## What This Repo Implements

- Megakernel decode service
  - `POST /v1/decode/stream` (NDJSON)
  - `WS /v1/decode/ws`
- Qwen3-TTS service
  - `WS /v1/tts/ws` (PCM16 audio chunks + summary)
- Strict talker backend enforcement
  - strict mode fails unless talker decode is megakernel-driven and non-zero
- Pipecat integration
  - `STT -> LLM -> TTS -> audio` adapter services
- Benchmarking
  - decode tok/s, decode TTFT, TTFC, RTF, end-to-end latency

## Architecture (Current)

1. `server/megakernel_backend.py` runs Qwen3-0.6B decode via CUDA megakernel.
2. `server/qwen3_tts_bridge.py` patches Qwen3-TTS talker generation to call a custom megakernel decode loop.
3. `/v1/tts/ws` emits PCM chunks and ends with a summary frame.
4. Strict validation checks:
   - `backend_mode == internal_custom_generate_megakernel`
   - `hook_executions > 0`
   - `tokens_generated > 0`
   - `decode_tokens_per_second > 0`

## Assignment Status

### Step 1: Adapt megakernel for Qwen3-TTS talker
Status: **Done (functional integration)**

- Talker decode path is attached to megakernel custom generate hook.
- Runtime proof fields are present in TTS summary.

### Step 2: Build streaming inference server
Status: **Done**

- Prompt-to-token streaming decode endpoint implemented.
- TTS websocket endpoint implemented with chunked PCM emission.

### Step 3: Integrate with Pipecat
Status: **Done (integration code complete)**

- Pipecat LLM/TTS service adapters implemented.
- Voice agent example added.

### Step 4: Validate E2E and hit performance targets
Status: **Partial**

- Megakernel decode throughput is strong.
- Strict talker integration proof is present.
- TTFC/RTF targets are not consistently met on strict path.

## Latest Strict Benchmark (Reported)

From strict run output shared during validation:

- Decode tok/s: `895.72`
- Decode TTFT: `33.15 ms`
- TTS TTFC: `845.53 ms`
- TTS RTF: `0.3556`
- End-to-end: `853.44 ms`

Diagnostics:

- `Likely buffered TTS path: yes`
- `backend_mode: internal_custom_generate_megakernel`
- `hook_executions: 1`
- `tokens_generated: 30`
- `decode_tokens_per_second: 993.70`

Target status:

- TTFC target (`< 90 ms`): **MISS**
- RTF target (`< 0.3`): **MISS**

Interpretation:

- Megakernel talker hookup is working and measurable.
- Latency remains too high in current strict runtime path.
- Emission is chunked but still behaves as buffered-at-source in measured strict runs.

## Why Targets Are Missed

Primary bottlenecks observed:

1. Qwen3-TTS runtime behavior in this environment is not consistently incremental under strict path.
2. Generative path can block before first chunk, inflating TTFC.
3. Repeated runs required defensive stabilization (decoder lock, executor reset on timeout) to avoid stalls.

## Stabilization Changes Applied

- Serialized megakernel decoder access with a lock across both decode endpoints and strict talker custom-generate path.
- Reset thread pool after TTS generation timeout/runtime exceptions to avoid stale worker blocking subsequent requests.
- Strict validation fails fast when megakernel talker proof fields are missing/zero.

## Reproducible Commands

Start server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --ws websockets
```

Strict benchmark:

```bash
python scripts/benchmark_streaming.py \
  --host 127.0.0.1 --port 8000 \
  --runs 1 --skip-warmup \
  --strict-streaming \
  --text "Hi." \
  --decode-timeout-sec 60 \
  --tts-timeout-sec 90
```

Non-strict benchmark:

```bash
python scripts/benchmark_streaming.py \
  --host 127.0.0.1 --port 8000 \
  --runs 1 --skip-warmup \
  --text "Hi." \
  --decode-timeout-sec 60 \
  --tts-timeout-sec 90
```

## Environment Notes

- GPU target: RTX 5090 (`sm_120`)
- Headless cloud VMs may not support local audio devices; run Pipecat local transport on a machine with microphone/speaker.

