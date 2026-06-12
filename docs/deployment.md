# TTS Server Deployment

## Architecture

```
┌─────────────────┐         HTTP chunked          ┌──────────────────────┐
│  RPi4 (rpi166)  │  ◄────────────────────────►  │  vm101 (prod-gpu)    │
│                 │     POST /tts/stream          │                      │
│  Wake word      │         PCM/WAV               │  Supertonic ONNX     │
│  VAD            │                               │  Docker :8020        │
│  Mic/playback   │                               │  10 cores, 31GB RAM  │
│  Barge-in       │                               │  TTFA ~0.3-0.5s     │
└─────────────────┘                               └──────────────────────┘
```

## vm101 Deployment

**Server:** `192.168.0.55` (prod-gpu-ubuntu)

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
| `/tts/stream` | POST | Chunked WAV streaming (optimized TTFA) |
| `/health` | GET | Health check |
| `/voices` | GET | List available voices |

### POST /tts/stream Request

```json
{
  "text": "Text de sintetizat.",
  "steps": 5,
  "speed": 1.5,
  "lang": "ro",
  "voice": "F1"
}
```

Response: `Transfer-Encoding: chunked` with WAV chunks. First chunk arrives in ~0.3-0.5s.

## Benchmarks (vm101, i5-12400, 10 cores)

### Text: 507 chars Romanian (with English words + numbers)

| Config | TTFA | Total gen | Audio dur | RTF | vs RPi4 |
|--------|:---:|:---:|:---:|:---:|:---:|
| steps=2 | **0.25s** | 3.21s | ~30s | ~0.1 | 18x faster |
| steps=3 | **0.34s** | 4.22s | ~30s | ~0.14 | — |
| steps=5 | **0.49s** | 6.12s | ~30s | ~0.2 | — |

**Recommendation:** Use `steps=5` for best quality. Even at steps=5, TTFA < 0.5s.

## Chunking Strategy

The streaming server splits text for optimal TTFA:
- First chunk: ~60 chars (generates in 0.25s at steps=2)
- Subsequent chunks: ~240 chars (generated in parallel)
- Gaps between chunks are minimal on vm101 (RTF << 1.0)
