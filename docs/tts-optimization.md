# TTS Optimization & Streaming Research

> **Purpose:** Document measured TTS performance on x86_64 (Intel i7-12700KF), options for speed/quality tuning, and the end-to-end streaming pattern.
> **Date:** June 2026

## Test Methodology

- **Hardware:** Intel Core i7-12700KF (12C/20T), CPU-only ONNX inference
- **Text:** 762-character Romanian sentence with numbers, English loanwords, and technical terms
- **Voice:** Supertonic 3, F1 (female), lang=ro
- **Metric:** RTF (Real-Time Factor) — lower is better. RTF < 1 means faster than real-time.

## Results: Supertonic 3

| Config | Audio duration | Generation time | RTF | File size | Notes |
|---|---|---|---|---|---|
| **steps=5, speed=1.0** | 77.4s | 29.6s | **0.38x** | 6.6 MB WAV | Fastest, lowest quality. Robotic on some phonemes |
| **steps=8, speed=1.0** | 77.4s | 45.3s | **0.59x** | 6.6 MB WAV | Default. Balanced quality/speed |
| **steps=12, speed=1.0** | 77.4s | 65.7s | **0.85x** | 6.6 MB WAV | Highest quality, smoothest |
| **steps=8, speed=1.5** | 51.9s | 30.8s | **0.60x** | 4.4 MB WAV | Faster playback, shorter output. Natural at 1.5x |

### Key observations

- On x86_64, even steps=12 is faster than real-time (0.85x RTF).
- All step values produce the same audio duration at the same speed setting.
- `speed` parameter compresses/expands audio duration proportionally.
- Audio samples are in `/tmp/tts-samples/` for subjective quality comparison.

## Estimated RPi4 Performance

Supertonic 3 runs on RPi4 but expect **3-5x slower** than x86_64 based on community reports:

| Config | x86_64 RTF | Estimated RPi4 RTF | Real-world feel |
|---|---|---|---|
| steps=5 | 0.38x | ~1.1-1.9x | Near realtime for short replies |
| steps=8 | 0.59x | ~1.8-3.0x | Slight delay before playback |
| steps=12 | 0.85x | ~2.6-4.3x | Noticeable delay |

For RPi4, **steps=5** is the practical choice. For lower latency and smaller image size, consider selective INT8 quantization.

## INT8 Quantization — tested ✅ (with caveats)

**Concluzie finală: doar vector_estimator.onnx poate fi cuantizat INT8 fără pierdere de calitate.**

Supertonic 3 are 4 modele ONNX. Am testat toate combinațiile:

| Model ONNX | Dimensiune FP32 | INT8 funcțional? |
|---|---|---|
| `duration_predictor.onnx` | 3.7 MB | ✅ dar irelevant (prea mic) |
| `text_encoder.onnx` | 36 MB | ✅ dar beneficiu minor |
| `vector_estimator.onnx` | 257 MB | **✅ funcționează perfect** ← cel mai important |
| `vocoder.onnx` | 101 MB | **❌ distorsionează complet audio** |

**Motivul:** Vocoder-ul generează forma de undă finală la 44.1kHz. Dynamic quantization distruge precizia greutăților și produce artefacte audio neinteligibile. Vector_estimator (model de difuzie) tolerează INT8 pentru că lucrează în spațiul latent, nu direct pe waveform.

**Rezultate finale (speed=1.5, text 7.8s RO, i7-12700KF):**

| Config | Generare | Audio | RTF | Dimensiune | vs FP32 s8 |
|---|---|---|---|---|---|
| **FP32 step=8** | 5.27s | 7.77s | 0.678 | 398 MB | 1.00x |
| **FP32 step=5** | 3.31s | 7.77s | 0.426 | 398 MB | 1.59x |
| **FINAL step=8** (ve INT8) | 4.34s | 7.77s | 0.559 | **207 MB** | 1.21x |
| **FINAL step=5** (ve INT8) | **2.97s** | 7.77s | **0.382** | **207 MB** | **1.77x** |

**Recomandare: FINAL step=5** — doar vector_estimator INT8, restul FP32. Calitate identică cu FP32 (confirmat subiectiv), viteză de 2.6x peste timp real, imagine Docker de 207 MB (-48%).

**Cum se face:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("vector_estimator.onnx", "vector_estimator_int8.onnx", weight_type=QuantType.QInt8)
```
Apoi se înlocuiește doar acest fișier în cache. Restul rămân FP32. Funcționează în Docker standard, fără seccomp.

**NU există "Supertonic MNN"** — e o confuzie. Supertonic 3 e ONNX-only.

## TTS Options Summary

| TTS Engine | RTF (x86_64) | RTF (RPi4 est.) | Romanian | API Key | Type |
|---|---|---|---|---|---|
| **Supertonic 3** steps=5 | 0.38x | ~1.5x | ✅ | ❌ | Local ONNX |
| **Supertonic 3** steps=8 | 0.59x | ~2.5x | ✅ | ❌ | Local ONNX |
| **Supertonic 3** steps=12 | 0.85x | ~3.5x | ✅ | ❌ | Local ONNX |
| **Supertonic 3 INT8** | 0.64x | ~1.5-2.5x | ✅ | ❌ | Local ONNX, 102 MB |
| **Supertonic 3 HYBRID** | 0.50x | ~1.2-2.0x | ✅ | ❌ | Local ONNX, 173 MB |
| **Piper** (excluded) | ~0.10x | ~0.20x | ✅ | ❌ | Local ONNX |
| **Edge TTS** | N/A | N/A | ✅ | ❌ | Cloud API |
| **Cloud TTS** | N/A | N/A | varies | ✅ | Cloud API |

## Streaming End-to-End (LLM writes → TTS speaks)

This is not natively supported by any local TTS engine. Implementation pattern:

```
LLM streaming output (token-by-token)
    ↓
Buffer text until sentence boundary (., !, ?, \n, ,)
    ↓
Completed sentence → Supertonic TTS API → audio chunk
    ↓ (in parallel)
Play audio while LLM continues generating next sentences
    ↓
Repeat until LLM finishes
```

### Requirements

1. **LLM must support streaming responses** (e.g. PydanticAI with `stream=True`).
2. **Orchestrator** buffers text and splits at sentence boundaries.
3. **TTS must accept partial text** and stream audio back (Supertonic supports this via OpenAI-compatible `/v1/audio/speech` with `response_format=opus`).
4. **Audio playback must chunk** — start playing before full response is generated.

### Priority for v1

**Not a v1 requirement.** Implement the simple listen→STT→LLM→TTS→listen loop first. Add streaming TTS as an optimization post-Milestone 1, only if measured latency is a problem.

---

## vm101 Deployment Results (June 2026)

**Hardware:** Intel i5-12400 (6C/12T), 31GB RAM, CPU-only ONNX inference
**Server:** Docker container, port 8020, streaming via HTTP chunked

### Streaming TTFA Benchmark (507 chars Romanian text with code-switching)

| Config | TTFA | Total gen | RTF | vs RPi4 speedup |
|--------|:---:|:---:|:---:|:---:|
| **steps=2, speed=1.5** | **0.25s** | 3.21s | ~0.10 | 18x |
| **steps=3, speed=1.5** | **0.34s** | 4.22s | ~0.14 | — |
| **steps=5, speed=1.5** | **0.49s** | 6.12s | ~0.20 | — |

**Verdict:** On vm101, even steps=5 achieves TTFA < 0.5s. The i5-12400's AVX2 support makes ONNX inference ~18x faster than Cortex-A72 on RPi4.

**Recommended config for vm101:** steps=5, speed=1.5 — best quality, TTFA still under 0.5s.

### Final Architecture

```
RPi4 (rpi166)                    vm101 (192.168.0.55)
┌─────────────────┐  HTTP chunked  ┌──────────────────────┐
│ Wake word + VAD  │  ◄─────────►  │  Supertonic TTS       │
│ Microfon         │  POST/tts     │  (Docker :8020)        │
│ Playback + barge │  stream       │  steps=5, speed=1.5    │
└─────────────────┘               │  TTFA ~0.5s            │
                                   └──────────────────────┘
```

### RPi4 Benchmark Reference

| Config | Non-streaming RTF | Streaming TTFA | Streaming RTF | Max gap |
|--------|:---:|:---:|:---:|:---:|
| steps=2, speed=1.5 | 1.041 | **4.71s** | **0.953** ✅ | 4.5s |
| steps=3, speed=1.5 | 1.339 | 6.32s | 1.309 ❌ | 8.0s |
| steps=5, speed=1.5 | ~1.9 | ~9.6s | ~1.9 | ~15s |

### Excluded Alternatives

- **mms-tts-ron** (Xenova/facebook): ONNX quantized tested. Quality worse than Supertonic, sample rate 16kHz only, poor code-switching (English chars → unknown tokens). Rejected.
- **Piper**: Rejected earlier (quality/language).
- **Edge TTS**: Rejected earlier (cloud, code-switching issues).
