"""TTS HTTP client for the Supertonic streaming server.

Handles:
- /tts/stream-pcm (raw PCM streaming) — primary, low overhead
- /tts/stream (chunked WAV) — legacy compat
- /tts (single WAV) — fallback
- Profiling timestamps (T5-T7) and TTFA metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from .emitter import AudioChunk
from .protocol import PCM_FRAME_HEADER_SIZE, decode_pcm_frame_header

logger = logging.getLogger(__name__)

DEFAULT_TTS_URL = "http://localhost:8020"

@dataclass
class TTFAStats:
    """TTFA measurement across multiple requests."""
    values: list[float] = field(default_factory=list)

    def record(self, ttfa: float) -> None:
        self.values.append(ttfa)

    @property
    def p50(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        return s[len(s) // 2]

    @property
    def p95(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        idx = min(int(len(s) * 0.95), len(s) - 1)
        return s[idx]

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def __str__(self) -> str:
        if not self.values:
            return "no data"
        return f"p50={self.p50:.3f}s p95={self.p95:.3f}s mean={self.mean:.3f}s n={len(self.values)}"


@dataclass
class ProfilePoint:
    label: str
    time: float
    rel: float = 0.0

    def __repr__(self) -> str:
        return f"{self.label}={self.rel*1000:.1f}ms"


class Profile:
    """Timing profile for a single TTS request."""
    def __init__(self) -> None:
        self._points: list[ProfilePoint] = []
        self._t0: float | None = None

    def mark(self, label: str) -> None:
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        self._points.append(ProfilePoint(label=label, time=now, rel=now - self._t0))

    def __str__(self) -> str:
        if not self._points:
            return "no profile data"
        parts = [str(p) for p in self._points]
        return " | ".join(parts)

    @property
    def ttfa(self) -> float:
        """Time to first audio (T0 → T6)."""
        if not self._points:
            return 0.0
        t_start = next((p.time for p in self._points if p.label == "T0"), self._points[0].time)
        t_first_audio = next((p.time for p in self._points if p.label == "T6"), t_start)
        return t_first_audio - t_start

    @property
    def server_inference(self) -> float:
        """Server-side inference time (T1 → T3)."""
        t1 = next((p.time for p in self._points if p.label == "T1"), None)
        t3 = next((p.time for p in self._points if p.label == "T3"), None)
        if t1 and t3:
            return t3 - t1
        return 0.0


@dataclass
class TTSClientConfig:
    url: str = DEFAULT_TTS_URL
    voice: str = "F1"
    lang: str = "ro"
    steps: int = 5
    first_steps: int | None = None
    speed: float = 1.5
    first_max: int | None = None
    next_max: int | None = None
    pcm_frame_bytes: int | None = None
    timeout: float = 30.0
    max_retries: int = 2


def _parse_wav_header(data: bytes) -> tuple[int, int, int]:
    """Parse a WAV header, return (sample_rate, num_channels, data_offset)."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV header")
    fmt_size = struct.unpack_from("<I", data, 16)[0]
    audio_format = struct.unpack_from("<H", data, 20)[0]
    num_channels = struct.unpack_from("<H", data, 22)[0]
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    if audio_format != 1:
        raise ValueError(f"Unsupported WAV format: {audio_format}")
    offset = 20 + fmt_size
    while offset + 8 <= len(data):
        cid = data[offset:offset + 4]
        csz = struct.unpack_from("<I", data, offset + 4)[0]
        if cid == b"data":
            return sample_rate, num_channels, offset + 8
        offset += 8 + csz
    raise ValueError("No data chunk found")


def _pcm_from_wav(wav_bytes: bytes) -> tuple[bytes, int, int]:
    sr, ch, offset = _parse_wav_header(wav_bytes)
    return wav_bytes[offset:], sr, ch


class TTSClient:
    """HTTP client for the Supertonic TTS server."""

    def __init__(self, config: TTSClientConfig | None = None) -> None:
        self._config = config or TTSClientConfig()
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            follow_redirects=True,
        )
        self.ttfa_stats = TTFAStats()

    @property
    def config(self) -> TTSClientConfig:
        return self._config

    def _payload(
        self,
        text: str,
        *,
        voice: str | None = None,
        lang: str | None = None,
        steps: int | None = None,
        first_steps: int | None = None,
        speed: float | None = None,
        first_max: int | None = None,
        next_max: int | None = None,
        pcm_frame_bytes: int | None = None,
    ) -> dict:
        payload = {
            "text": text,
            "voice": voice or self._config.voice,
            "lang": lang or self._config.lang,
            "steps": steps if steps is not None else self._config.steps,
            "speed": speed if speed is not None else self._config.speed,
        }
        resolved_first_steps = (
            first_steps if first_steps is not None else self._config.first_steps
        )
        resolved_first_max = first_max if first_max is not None else self._config.first_max
        resolved_next_max = next_max if next_max is not None else self._config.next_max
        resolved_pcm_frame_bytes = (
            pcm_frame_bytes if pcm_frame_bytes is not None else self._config.pcm_frame_bytes
        )
        if resolved_first_steps is not None:
            payload["first_steps"] = resolved_first_steps
        if resolved_first_max is not None:
            payload["first_max"] = resolved_first_max
        if resolved_next_max is not None:
            payload["next_max"] = resolved_next_max
        if resolved_pcm_frame_bytes is not None:
            payload["pcm_frame_bytes"] = resolved_pcm_frame_bytes
        return payload

    async def health(self) -> dict:
        resp = await self._http.get(f"{self._config.url}/health")
        resp.raise_for_status()
        return resp.json()

    async def voices(self) -> list[str]:
        resp = await self._http.get(f"{self._config.url}/voices")
        resp.raise_for_status()
        return resp.json().get("voices", [])

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        lang: str | None = None,
        steps: int | None = None,
        first_steps: int | None = None,
        speed: float | None = None,
        first_max: int | None = None,
        next_max: int | None = None,
        pcm_frame_bytes: int | None = None,
    ) -> tuple[bytes, float]:
        """Non-streaming synthesize via /tts. Returns (wav_bytes, duration_seconds)."""
        payload = self._payload(
            text,
            voice=voice,
            lang=lang,
            steps=steps,
            first_steps=first_steps,
            speed=speed,
            first_max=first_max,
            next_max=next_max,
            pcm_frame_bytes=pcm_frame_bytes,
        )
        resp = await self._http.post(f"{self._config.url}/tts", json=payload)
        resp.raise_for_status()
        duration = float(resp.headers.get("X-Audio-Duration", 0))
        return resp.content, duration

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        lang: str | None = None,
        steps: int | None = None,
        first_steps: int | None = None,
        speed: float | None = None,
        first_max: int | None = None,
        next_max: int | None = None,
        pcm_frame_bytes: int | None = None,
        on_chunk: Callable[[AudioChunk], None] | None = None,
    ) -> list[AudioChunk]:
        """Streaming via /tts/stream (chunked WAV, legacy)."""
        payload = self._payload(
            text,
            voice=voice,
            lang=lang,
            steps=steps,
            first_steps=first_steps,
            speed=speed,
            first_max=first_max,
            next_max=next_max,
            pcm_frame_bytes=pcm_frame_bytes,
        )

        async with self._http.stream("POST", f"{self._config.url}/tts/stream", json=payload) as resp:
            resp.raise_for_status()
            chunks: list[AudioChunk] = []
            buffer = b""
            async for raw in resp.aiter_bytes():
                buffer += raw
                while True:
                    riff = buffer.find(b"RIFF")
                    if riff < 0 or riff + 44 > len(buffer):
                        break
                    file_size = struct.unpack_from("<I", buffer, riff + 4)[0] + 8
                    chunk_end = riff + file_size
                    if chunk_end > len(buffer):
                        break
                    wav_data = buffer[riff:chunk_end]
                    buffer = buffer[chunk_end:]
                    try:
                        pcm_data, pcm_sr, pcm_ch = _pcm_from_wav(wav_data)
                    except ValueError:
                        logger.warning("failed to parse WAV, skipping")
                        continue
                    dur = len(pcm_data) / (pcm_ch * 2) / pcm_sr
                    ac = AudioChunk(data=pcm_data, sample_rate=pcm_sr,
                                    num_channels=pcm_ch, duration=dur, is_final=False)
                    chunks.append(ac)
                    if on_chunk:
                        on_chunk(ac)

            if chunks:
                chunks[-1].is_final = True
                if on_chunk:
                    on_chunk(chunks[-1])
        return chunks

    async def synthesize_stream_pcm(
        self,
        text: str,
        *,
        voice: str | None = None,
        lang: str | None = None,
        steps: int | None = None,
        first_steps: int | None = None,
        speed: float | None = None,
        first_max: int | None = None,
        next_max: int | None = None,
        pcm_frame_bytes: int | None = None,
        on_chunk: Callable[[AudioChunk], None] | None = None,
    ) -> tuple[list[AudioChunk], Profile]:
        """Streaming via /tts/stream-pcm (raw PCM, low overhead).

        Returns (chunks, profile) with timing breakdown across client and server.
        T0 = request created (client)
        T5 = first byte received (client)
        T6 = first PCM chunk decoded (client)
        T7 = first chunk delivered to callback (client)

        Server-side times (T1-T4) are parsed from response headers if available.
        """
        profile = Profile()
        profile.mark("T0")  # request created

        payload = self._payload(
            text,
            voice=voice,
            lang=lang,
            steps=steps,
            first_steps=first_steps,
            speed=speed,
            first_max=first_max,
            next_max=next_max,
            pcm_frame_bytes=pcm_frame_bytes,
        )

        async with self._http.stream(
            "POST", f"{self._config.url}/tts/stream-pcm", json=payload
        ) as resp:
            resp.raise_for_status()
            # T5: first byte received (headers are in, body streaming starts)
            profile.mark("T5")

            sr = int(resp.headers.get("X-Sample-Rate", 44100))
            server_ttfa = float(resp.headers.get("X-TTFA", 0))
            server_gen = float(resp.headers.get("X-Gen-Time", 0))
            total_chunks = int(resp.headers.get("X-Total-Chunks", "0"))

            chunks: list[AudioChunk] = []
            buf = b""
            chunk_count = 0

            async for raw in resp.aiter_bytes():
                buf += raw
                while len(buf) >= PCM_FRAME_HEADER_SIZE:
                    frame_header = decode_pcm_frame_header(buf[:PCM_FRAME_HEADER_SIZE])
                    if frame_header.is_eof:
                        buf = buf[PCM_FRAME_HEADER_SIZE:]
                        break
                    actual_sr = frame_header.sample_rate
                    byte_count = frame_header.byte_count
                    total_size = PCM_FRAME_HEADER_SIZE + byte_count
                    if len(buf) < total_size:
                        break  # incomplete chunk

                    pcm_data = buf[PCM_FRAME_HEADER_SIZE:total_size]
                    buf = buf[total_size:]
                    chunk_count += 1

                    dur = len(pcm_data) / 2 / actual_sr  # mono, 16-bit
                    ac = AudioChunk(
                        data=pcm_data, sample_rate=actual_sr,
                        num_channels=1, duration=dur,
                        is_final=(total_chunks > 0 and chunk_count >= total_chunks),
                    )
                    chunks.append(ac)

                    if chunk_count == 1:
                        # T6: first PCM decoded
                        profile.mark("T6")
                        if on_chunk:
                            on_chunk(ac)
                            profile.mark("T7")  # delivered to callback
                    else:
                        if on_chunk:
                            on_chunk(ac)

            if chunks and not chunks[-1].is_final:
                chunks[-1].is_final = True

        # Compute TTFA
        ttfa = profile.ttfa
        self.ttfa_stats.record(ttfa)

        if server_ttfa > 0:
            network_overhead = ttfa - server_ttfa
            logger.info(
                f"TTS PCM: {len(text)}ch -> {len(chunks)} chunks "
                f"TTFA={ttfa:.3f}s (server={server_ttfa:.3f}s, net={network_overhead:.3f}s) "
                f"profile=[{profile}] | {self.ttfa_stats}"
            )

        return chunks, profile

    async def close(self) -> None:
        await self._http.aclose()
