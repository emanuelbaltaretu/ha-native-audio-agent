# TTS Server Deployment

## Architecture

```
┌─────────────────┐         HTTP chunked          ┌──────────────────────┐
│  RPi4 (rpi166)  │  ◄────────────────────────►  │  vm101               │
│                 │   POST /tts/stream-pcm        │                      │
│  Wake word      │         PCM/WAV               │  Supertonic ONNX     │
│  VAD            │                               │  Docker :8020        │
│  Mic/playback   │                               │  10 cores, 31GB RAM  │
│  Barge-in       │                               │  Docker :8020        │
└─────────────────┘                               └──────────────────────┘
```

## vm101 Deployment

**Server:** `192.168.0.55`

The default vm101 deployment uses CPU ONNX inference. OpenVINO/GPU remains
available as an opt-in experiment, but measured TTFA is slower for the current
Supertonic model on this host.

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

**Quality-safe default:** use `steps=5` without aggressive `first_steps` or `first_max`
until a specific text/voice profile has been listened to and accepted.

### RPi client to vm101 streaming benchmark (June 13, 2026)

Measured from `rpi166` to vm101 using:

```bash
python3 scripts/benchmark_tts_ttfa.py \
  --url http://192.168.0.55:8020/tts/stream-pcm \
  --text "Am aprins lumina din dormitor."
```

| Config | First PCM frame p50 | Server TTFA p50 | Full stream p50 |
|---|---:|---:|---:|
| vm101 CPU live, `steps=5` | 0.922s | 0.912s | 0.923s |
| vm101 CPU live, `steps=5, first_steps=4` | 0.758s | 0.753s | 0.760s |
| vm101 CPU live, `steps=5, first_steps=3` | 0.675s | 0.671s | 0.677s |
| vm101 CPU test, `steps=5, first_steps=2` | **0.416s** | **0.412s** | 0.417s |
| vm101 OpenVINO/GPU, `steps=5` | 1.114s | 1.110s | 1.116s |
| vm101 OpenVINO/GPU, `steps=5, first_steps=3` | 0.692s | 0.688s | 0.694s |

These numbers are latency measurements, not quality approvals. In live listening,
aggressive `first_steps`/`first_max` can damage the first phrase. Treat them as
experimental knobs and validate by ear before making them defaults.

### RPi local vs vm101 generation benchmark (June 13, 2026)

Measured from `rpi166` without audio playback against non-streaming `/tts`, using
`"Am aprins lumina din dormitor."`:

| Host | Config | HTTP elapsed p50 | Generation p50 |
|---|---|---:|---:|
| RPi local CPU | `steps=5` | 3.897s | 3.890s |
| RPi local CPU | `steps=3` | 2.665s | 2.660s |
| RPi local CPU | `steps=2` | 2.022s | 2.010s |
| vm101 CPU | `steps=5` | 0.758s | 0.750s |
| vm101 CPU | `steps=3` | 0.592s | 0.580s |
| vm101 CPU | `steps=2` | 0.427s | 0.420s |
| vm101 OpenVINO/GPU | `steps=5` | 1.099s | 1.090s |
| vm101 OpenVINO/GPU | `steps=3` | 0.693s | 0.690s |
| vm101 OpenVINO/GPU | `steps=2` | 0.484s | 0.480s |

Conclusion: keep TTS inference on vm101 CPU for now. RPi should stay focused on
microphone capture, wake word, VAD, playback, and barge-in.

## Chunking Strategy

The streaming server splits text for optimal TTFA:
- First chunk: ~60 chars (generates in 0.25s at steps=2)
- Subsequent chunks: ~240 chars (generated in parallel)
- Gaps between chunks are minimal on vm101 (RTF << 1.0)
