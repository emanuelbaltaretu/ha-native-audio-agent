# TTS Server Deployment

## Architecture

```
┌─────────────────┐         HTTP chunked          ┌──────────────────────┐
│  RPi4 (rpi166)  │  ◄────────────────────────►  │  vm101 (prod-gpu)    │
│                 │   POST /tts/stream-pcm        │                      │
│  Wake word      │         PCM/WAV               │  Supertonic ONNX     │
│  VAD            │                               │  Docker :8020        │
│  Mic/playback   │                               │  10 cores, 31GB RAM  │
│  Barge-in       │                               │  Docker :8020        │
└─────────────────┘                               └──────────────────────┘
```

## vm101 Deployment

**Server:** `192.168.0.55` (prod-gpu-ubuntu)

The server code is shared package code:

```text
src/ha_native_audio_agent/tts/supertonic_server.py
src/ha_native_audio_agent/tts/protocol.py
src/ha_native_audio_agent/tts/chunking.py
```

`docker/vm101/tts_server.py` is only a thin Docker entrypoint.

### Local model configuration

Model weights are not committed and are not copied into the image. Create a local,
uncommitted env file:

```bash
cd ~/ha-native-audio-agent/docker/vm101
cp .env.example .env
```

Set:

```dotenv
SUPERTONIC_MODEL_DIR_HOST=/absolute/path/to/supertonic3_final
```

### Start

```bash
cd ~/ha-native-audio-agent/docker/vm101
docker compose up -d
```

### Verify

```bash
curl http://192.168.0.55:8020/health
# → {"status": "ok", "model": "supertonic3_final"}
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/tts` | POST | Generate complete WAV (non-streaming) |
| `/tts/stream` | POST | Deprecated legacy WAV stream; use `/tts/stream-pcm` |
| `/tts/stream-pcm` | POST | Raw PCM framed streaming (recommended) |
| `/health` | GET | Health check |
| `/voices` | GET | List available voices |

### POST /tts/stream-pcm Request

```json
{
  "text": "Text de sintetizat.",
  "steps": 5,
  "speed": 1.5,
  "lang": "ro",
  "voice": "F1"
}
```

Response: raw PCM frames. Each frame starts with a 6-byte little-endian header:
`sample_rate_div100` (`uint16`) and `byte_count` (`uint32`), followed by signed 16-bit
mono PCM. A frame with `byte_count=0` is an explicit EOF marker.

Optional latency controls:

| Field | Default | Purpose |
|---|---:|---|
| `first_steps` | same as `steps` | Diffusion steps only for the first chunk; can reduce quality |
| `first_max` | `60` | Maximum characters in the first TTS chunk |
| `next_max` | `240` | Maximum characters in later chunks |
| `pcm_frame_bytes` | `8192` | PCM frame size sent to the client after each generated chunk |

## Benchmarks (vm101, i5-12400, 10 cores)

### Text: 507 chars Romanian (with English words + numbers)

| Config | TTFA | Total gen | Audio dur | RTF | vs RPi4 |
|--------|:---:|:---:|:---:|:---:|:---:|
| steps=2 | **0.25s** | 3.21s | ~30s | ~0.1 | 18x faster |
| steps=3 | **0.34s** | 4.22s | ~30s | ~0.14 | — |
| steps=5 | **0.49s** | 6.12s | ~30s | ~0.2 | — |

**Quality-safe default:** use `steps=5` without aggressive `first_steps` or `first_max`
until a specific text/voice profile has been listened to and accepted.

### RPi client first-frame benchmark (June 13, 2026)

Measured from `rpi166` to vm101 using:

```bash
python3 scripts/benchmark_tts_ttfa.py \
  --url http://192.168.0.55:8020/tts/stream-pcm \
  --text "Am aprins lumina din dormitor."
```

| Config | First PCM frame p50 | Server TTFA p50 | Full stream p50 |
|---|---:|---:|---:|
| `steps=5` | 0.897s | 0.892s | 0.901s |
| `steps=3` | 0.529s | 0.523s | 0.532s |
| `steps=5, first_steps=3` | 0.608s | 0.603s | 0.611s |
| `steps=5, first_steps=3, first_max=18` | **0.419s** | **0.414s** | 1.096s |
| `steps=5, first_steps=3, first_max=14` | 0.430s | 0.424s | 1.032s |

These numbers are latency measurements, not quality approvals. In live listening,
aggressive `first_steps`/`first_max` can damage the first phrase. Treat them as
experimental knobs and validate by ear before making them defaults.

## Chunking Strategy

The streaming server splits text for optimal TTFA:
- First chunk: ~60 chars (generates in 0.25s at steps=2)
- Subsequent chunks: ~240 chars (generated in parallel)
- Gaps between chunks are minimal on vm101 (RTF << 1.0)
