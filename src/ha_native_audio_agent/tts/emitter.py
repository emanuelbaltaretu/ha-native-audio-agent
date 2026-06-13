"""AudioEmitter — adapted from LiveKit Agents (Apache 2.0).

Manages a PCM audio buffer with is_final tracking, segment management,
and slow-generation flush detection.

Changes from LiveKit original:
- No rtc.AudioFrame dependency — works with raw PCM bytes
- Duration is calculated from byte count + sample rate + channels
- Simplified: no WAV/OGG decoder, no complex codec pipelines
- Callback-based output instead of channel-based
"""

from __future__ import annotations

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Number of samples held back to mark the last audio as is_final (10ms)
_TAIL_SAMPLES_FACTOR = 10  # ms


@dataclass
class AudioChunk:
    """A chunk of synthesized audio ready for playback."""
    data: bytes
    sample_rate: int
    num_channels: int
    duration: float  # seconds
    segment_id: str = ""
    is_final: bool = False
    delta_text: str = ""


@dataclass
class _SegmentCtx:
    segment_id: str = ""
    audio_duration: float = 0.0


class AudioEmitter:
    """Manages audio output from TTS synthesis.

    Accepts raw PCM bytes, tracks per-segment duration,
    holds back a small tail for is_final marking, and
    notifies via callback when audio chunks are ready.
    """

    def __init__(
        self,
        *,
        label: str = "tts",
        on_audio: Callable[[AudioChunk], None] | None = None,
    ) -> None:
        self._label = label
        self._on_audio = on_audio

        # State set by initialize()
        self._request_id: str = ""
        self._sample_rate: int = 0
        self._num_channels: int = 1
        self._mime_type: str = "audio/pcm"
        self._streaming: bool = False
        self._initialized = False
        self._started = False
        self._input_ended = False

        # Segment tracking
        self._segment: _SegmentCtx | None = None
        self._segment_durations: list[float] = []

        # Hold-back for is_final
        self._last_frame: bytes | None = None
        self._tail_samples: int = 0

        # Slow-generation detection
        self._sent_start: float | None = None
        self._sent_duration: float = 0.0
        self._flush_timer: asyncio.TimerHandle | None = None

        # Timing
        self._start_time: float | None = None

    @property
    def num_segments(self) -> int:
        return len(self._segment_durations)

    def pushed_duration(self, idx: int | None = None) -> float:
        """Total audio duration pushed so far, or for a specific segment index."""
        if not self._segment_durations:
            return 0.0
        if idx is not None:
            return self._segment_durations[idx] if 0 <= idx < len(self._segment_durations) else 0.0
        return sum(self._segment_durations)

    def initialize(
        self,
        *,
        request_id: str,
        sample_rate: int,
        num_channels: int = 1,
        mime_type: str = "audio/pcm",
        stream: bool = False,
    ) -> None:
        self._request_id = request_id
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._mime_type = mime_type
        self._streaming = stream
        self._initialized = True
        self._started = True
        self._tail_samples = sample_rate * _TAIL_SAMPLES_FACTOR // 1000

    def start_segment(self, segment_id: str) -> None:
        if self._segment is not None:
            raise RuntimeError("start_segment called before previous ended")
        self._segment_durations.append(0.0)
        self._segment = _SegmentCtx(segment_id=segment_id)

    def _emit_frame(self, data: bytes, *, is_final: bool = False) -> None:
        """Send a frame downstream via callback."""
        if self._segment is None:
            return
        duration = self._bytes_to_duration(len(data))
        self._segment.audio_duration += duration
        self._segment_durations[-1] += duration

        chunk = AudioChunk(
            data=data,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            duration=duration,
            segment_id=self._segment.segment_id,
            is_final=is_final,
        )
        if self._on_audio:
            self._on_audio(chunk)

    def push(self, data: bytes) -> None:
        """Push a chunk of raw PCM bytes into the emitter.

        Holds back the last _tail_samples to allow is_final marking.
        """
        if not data or not self._segment:
            return

        # Combine with any held-back tail
        if self._last_frame:
            combined = self._last_frame + data
        else:
            combined = data

        head, tail = self._split_tail(combined)
        if head:
            self._emit_frame(head, is_final=False)
        self._last_frame = tail

    def flush(self) -> None:
        """Flush the held-back tail without marking segment as ended."""
        if self._last_frame is None or self._segment is None:
            return
        self._emit_frame(self._last_frame, is_final=False)
        self._last_frame = None
        self._sent_start = None
        self._sent_duration = 0.0
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None

    def end_input(self) -> None:
        """Mark end of input — no more data will be pushed."""
        self._input_ended = True

    def end_segment(self) -> None:
        """End the current segment with is_final marker."""
        if self._segment is None:
            return

        if self._last_frame:
            self._emit_frame(self._last_frame, is_final=True)
            self._last_frame = None
        elif self._segment.audio_duration > 0:
            # Send a tiny empty marker for timing
            empty = b"\x00\x00" * (self._sample_rate // 100 * self._num_channels)
            chunk = AudioChunk(
                data=empty,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
                duration=0.01,
                segment_id=self._segment.segment_id,
                is_final=True,
            )
            if self._on_audio:
                self._on_audio(chunk)

        self._segment = None

    def _split_tail(self, data: bytes) -> tuple[bytes | None, bytes]:
        """Split PCM data into (head, tail) where tail is _tail_samples.

        If data is too small, returns (None, data).
        """
        samples = self._bytes_to_samples(len(data))
        if samples <= self._tail_samples:
            return None, data

        tail_samples = self._tail_samples
        head_samples = samples - tail_samples
        head_bytes = head_samples * self._num_channels * 2  # 16-bit
        return data[:head_bytes], data[head_bytes:]

    def _bytes_to_duration(self, nbytes: int) -> float:
        if self._sample_rate == 0:
            return 0.0
        samples = self._bytes_to_samples(nbytes)
        return samples / self._sample_rate

    def _bytes_to_samples(self, nbytes: int) -> int:
        if self._num_channels == 0:
            return 0
        return nbytes // (self._num_channels * 2)  # 16-bit samples

    async def join(self) -> None:
        """Wait for all pending audio to be processed (no-op in callback mode)."""
        pass

    async def aclose(self) -> None:
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None
        self._segment = None
        self._last_frame = None
