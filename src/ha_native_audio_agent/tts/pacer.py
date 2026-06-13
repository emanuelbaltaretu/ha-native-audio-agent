"""SentenceStreamPacer — adapted from LiveKit Agents (Apache 2.0).

Controls the pacing of text sent to TTS. Buffers sentences and decides when
to flush based on remaining audio duration. Reduces waste from interruptions
and improves speech quality by sending larger chunks.

Changes from LiveKit original:
- No rtc.AudioFrame dependency — works with raw PCM duration tracking
- StreamPacerWrapper uses a callback-based buffer_level hint instead
  of direct AudioEmitter reference
- Removed LiveKit-specific telemetry/logging cruft
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass


@dataclass
class StreamPacerOptions:
    min_remaining_audio: float  # seconds — send next batch when buffer drops below this
    max_text_length: int  # chars — max text to send in one TTS call


@dataclass
class TokenData:
    """A chunk of text ready for synthesis."""
    token: str
    segment_id: str = ""


class SentenceStream(ABC):
    """Abstract incremental sentence stream — simplified from LiveKit's SentenceStream."""

    def __init__(self) -> None:
        self._event_ch: asyncio.Queue[TokenData] = asyncio.Queue()
        self._closed = False

    @abstractmethod
    def push_text(self, text: str) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def end_input(self) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    @property
    def closed(self) -> bool:
        return self._closed

    def _do_close(self) -> None:
        self._closed = True

    async def __anext__(self) -> TokenData:
        if self._closed and self._event_ch.empty():
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._event_ch.get(), timeout=30)
        except asyncio.TimeoutError:
            raise StopAsyncIteration

    def __aiter__(self) -> AsyncIterator[TokenData]:
        return self


class SentenceStreamPacer:
    """Controls pacing of text-to-speech synthesis.

    Buffers incoming sentences and decides when to send the next batch
    based on how much audio is still queued for playback.
    """

    def __init__(self, *, min_remaining_audio: float = 5.0, max_text_length: int = 300) -> None:
        self._options = StreamPacerOptions(
            min_remaining_audio=min_remaining_audio,
            max_text_length=max_text_length,
        )

    def wrap(
        self,
        sent_stream: SentenceStream,
        *,
        buffer_level_cb: Callable[[], float] | None = None,
    ) -> PacedStream:
        """Wrap a SentenceStream with pacing logic.

        Args:
            sent_stream: The underlying sentence stream.
            buffer_level_cb: Callback returning current audio buffer duration in seconds.
                If not provided, pacing sends text as soon as sentences arrive.

        Returns:
            A PacedStream that yields TokenData chunks at the right pace.
        """
        return PacedStream(
            options=self._options,
            sent_stream=sent_stream,
            buffer_level_cb=buffer_level_cb,
        )


class PacedStream(SentenceStream):
    """Buffers sentences and yields them paced by audio buffer level."""

    def __init__(
        self,
        sent_stream: SentenceStream,
        *,
        options: StreamPacerOptions,
        buffer_level_cb: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self._sent_stream = sent_stream
        self._options = options
        self._buffer_level_cb = buffer_level_cb or (lambda: 0.0)
        self._audio_start_time = 0.0

        self._closing = False
        self._input_ended = False
        self._sentences: list[str] = []
        self._wakeup_event = asyncio.Event()
        self._wakeup_timer: asyncio.TimerHandle | None = None

        self._recv_task = asyncio.create_task(self._recv_task())
        self._send_task = asyncio.create_task(self._send_task())

    def push_text(self, text: str) -> None:
        self._sent_stream.push_text(text)

    def flush(self) -> None:
        self._sent_stream.flush()

    def end_input(self) -> None:
        self._sent_stream.end_input()
        self._input_ended = True

    async def aclose(self) -> None:
        await self._sent_stream.aclose()
        self._closing = True
        if self._wakeup_timer:
            self._wakeup_timer.cancel()
            self._wakeup_timer = None
        self._wakeup_event.set()
        await self._cancel_tasks()

    async def _cancel_tasks(self) -> None:
        for t in (self._recv_task, self._send_task):
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass

    async def _recv_task(self) -> None:
        try:
            async for ev in self._sent_stream:
                self._sentences.append(ev.token)
                self._wakeup_event.set()
        except (StopAsyncIteration, asyncio.CancelledError):
            pass
        finally:
            self._input_ended = True
            self._wakeup_event.set()

    async def _send_task(self) -> None:
        first_sentence = True
        prev_buffer = 0.0
        prev_check_time = 0.0
        generation_started = False
        generation_stopped = False

        while not self._closing:
            await self._wakeup_event.wait()
            self._wakeup_event.clear()
            if self._wakeup_timer:
                self._wakeup_timer.cancel()
                self._wakeup_timer = None

            if self._closing or (self._input_ended and not self._sentences):
                break

            buffer = self._buffer_level_cb()
            now = time.time()

            # detect if audio generation started/stopped
            if now - prev_check_time >= 0.1:
                if prev_buffer < buffer:
                    generation_started = True
                elif generation_started:
                    generation_stopped = True
                prev_buffer = buffer
                prev_check_time = now

            remaining = buffer - (now - self._audio_start_time) if self._audio_start_time > 0 else 0.0
            remaining = max(0.0, remaining)

            if first_sentence or (
                generation_stopped and remaining <= self._options.min_remaining_audio
            ):
                batch: list[str] = []
                while self._sentences:
                    batch.append(self._sentences.pop(0))
                    total = sum(len(s) for s in batch)
                    if first_sentence and batch:
                        break  # send first sentence immediately
                    if total >= self._options.max_text_length:
                        break

                if batch:
                    text = " ".join(batch)
                    self._event_ch.put_nowait(TokenData(token=text))
                    generation_started = False
                    generation_stopped = False
                    first_sentence = False
                    self._audio_start_time = time.time()

            # schedule next check
            wait_time = 0.2 if generation_started and not generation_stopped else max(0.5, remaining - self._options.min_remaining_audio)
            self._wakeup_timer = asyncio.get_event_loop().call_later(
                wait_time, self._wakeup_event.set
            )
