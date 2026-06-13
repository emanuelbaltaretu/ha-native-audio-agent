"""Persistent audio player — keeps aplay pipe open across TTS responses.

Avoids the ~200ms overhead of spawning aplay per response.
Supports ALSA plughw directly. Handles sample rate changes.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Callable

from .client import AudioChunk

logger = logging.getLogger(__name__)

# Default ALSA device for Jabra
ALSA_DEVICE = "plughw:0,0"


class PersistentPlayer:
    """Manages a persistent aplay process for low-latency playback.

    Keeps aplay stdin pipe open across multiple TTS responses.
    Automatically restarts aplay if sample rate changes.
    """

    def __init__(self, device: str = ALSA_DEVICE) -> None:
        self._device = device
        self._proc: subprocess.Popen | None = None
        self._current_sr: int = 0
        self._current_ch: int = 1
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.stdin is not None

    def _start_aplay(self, sample_rate: int, channels: int = 1) -> None:
        """Start aplay process for the given audio parameters."""
        self._stop_aplay()
        self._proc = subprocess.Popen(
            ["aplay", "-D", self._device, "-r", str(sample_rate),
             "-c", str(channels), "-f", "S16_LE", "-t", "raw"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._current_sr = sample_rate
        self._current_ch = channels
        logger.debug(f"aplay started: {sample_rate}Hz/{channels}ch")

    def _stop_aplay(self) -> None:
        """Stop current aplay process if running."""
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            self._proc = None
        self._current_sr = 0

    async def play(
        self,
        chunk: AudioChunk,
    ) -> None:
        """Play a single audio chunk. Restarts aplay if sample rate changes."""
        async with self._lock:
            if not self.is_open or chunk.sample_rate != self._current_sr:
                self._start_aplay(chunk.sample_rate, chunk.num_channels)

            if self._proc and self._proc.stdin:
                try:
                    self._proc.stdin.write(chunk.data)
                    self._proc.stdin.flush()
                except BrokenPipeError:
                    logger.warning("aplay pipe broken, restarting")
                    self._start_aplay(chunk.sample_rate, chunk.num_channels)
                    if self._proc and self._proc.stdin:
                        self._proc.stdin.write(chunk.data)
                        self._proc.stdin.flush()

    async def play_chunks(self, chunks: list[AudioChunk]) -> None:
        """Play a list of chunks sequentially."""
        if not chunks:
            return
        for chunk in chunks:
            await self.play(chunk)

    async def silence(self, duration_ms: int = 50) -> None:
        """Play a short silence to keep aplay alive during gaps."""
        if not self.is_open:
            return
        sr = self._current_sr or 44100
        samples = sr * duration_ms // 1000
        silence = b"\x00\x00" * samples
        async with self._lock:
            if self._proc and self._proc.stdin:
                try:
                    self._proc.stdin.write(silence)
                    self._proc.stdin.flush()
                except BrokenPipeError:
                    pass

    async def close(self) -> None:
        """Stop aplay and clean up."""
        self._stop_aplay()

    def __del__(self) -> None:
        self._stop_aplay()
