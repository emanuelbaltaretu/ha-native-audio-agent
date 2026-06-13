"""SynthesizeStream — high-level streaming TTS abstraction.

Combines LiveKit patterns:
- Text input via push_text/flush/end_input (from SynthesizeStream)
- Sentence-level chunking with optional pacing (from SentenceStreamPacer)
- Audio output via AudioEmitter with is_final tracking
- HTTP/WS transport for the actual TTS backend

Usage:
    async with synthesizer.stream() as stream:
        stream.push_text("Bună ziua, aceasta este o ")
        stream.push_text("testare a sistemului.")
        stream.end_input()
        async for chunk in stream:
            play(chunk.data, chunk.sample_rate)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from .client import TTSClient, TTSClientConfig
from .emitter import AudioChunk, AudioEmitter
from .pacer import PacedStream, SentenceStream, SentenceStreamPacer, TokenData

logger = logging.getLogger(__name__)


@dataclass
class _SimpleSentenceStream(SentenceStream):
    """Minimal sentence stream that buffers text and yields on flush/end_input."""
    _sentences: asyncio.Queue[TokenData] = field(default_factory=asyncio.Queue)
    _flush_queued: bool = False

    def push_text(self, text: str) -> None:
        self._sentences.put_nowait(TokenData(token=text))

    def flush(self) -> None:
        self._flush_queued = True

    def end_input(self) -> None:
        self._closed = True

    async def aclose(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __aiter__(self) -> AsyncIterator[TokenData]:
        return self._iterator()

    async def _iterator(self) -> AsyncIterator[TokenData]:
        """Accumulate until flush or end_input, then yield."""
        buffer = ""
        while not (self._closed and self._sentences.empty()):
            try:
                tok = await asyncio.wait_for(self._sentences.get(), timeout=0.1)
                buffer += tok.token
            except asyncio.TimeoutError:
                pass

            if self._flush_queued or self._closed:
                self._flush_queued = False
                if buffer.strip():
                    yield TokenData(token=buffer)
                    buffer = ""


class SynthesizeStream:
    """Streaming text-to-speech synthesizer.

    Push text incrementally, get AudioChunks asynchronously.
    Optionally uses SentenceStreamPacer to pace TTS requests
    based on playback buffer level.
    """

    class _FlushSentinel:
        pass

    def __init__(
        self,
        *,
        client: TTSClient | None = None,
        client_config: TTSClientConfig | None = None,
        pacer: SentenceStreamPacer | bool = False,
        playback_buffer_cb: Callable[[], float] | None = None,
        on_audio: Callable[[AudioChunk], None] | None = None,
    ) -> None:
        self._client = client or TTSClient(config=client_config)
        self._on_audio = on_audio

        # Text input channel
        self._input_ch: asyncio.Queue[str | SynthesizeStream._FlushSentinel] = asyncio.Queue()
        self._input_ended = False

        # Output channel for audio chunks
        self._output_ch: asyncio.Queue[AudioChunk | None] = asyncio.Queue()

        # Pacing
        self._use_pacer = pacer if isinstance(pacer, bool) else True
        self._sentence_pacer: SentenceStreamPacer | None = pacer if isinstance(pacer, SentenceStreamPacer) else (
            SentenceStreamPacer() if pacer else None
        )
        self._playback_buffer_cb = playback_buffer_cb

        # Internal state
        self._task: asyncio.Task | None = None
        self._started = False

    def push_text(self, text: str) -> None:
        """Push text incrementally for synthesis."""
        if not text:
            return
        self._input_ch.put_nowait(text)

    def flush(self) -> None:
        """Flush current text buffer — end current segment."""
        self._input_ch.put_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        """Mark end of input — no more text will be pushed."""
        self.flush()
        self._input_ended = True

    async def _run(self) -> None:
        """Main loop: read text → send to TTS → emit audio chunks."""
        # Build the sentence stream
        if self._use_pacer and self._sentence_pacer:
            sent_stream = _SimpleSentenceStream()
            paced = self._sentence_pacer.wrap(
                sent_stream=sent_stream,
                buffer_level_cb=self._playback_buffer_cb,
            )
        else:
            sent_stream = _SimpleSentenceStream()
            paced = sent_stream  # no pacing

        # Input reader: feeds text into the sentence stream
        async def _input_reader() -> None:
            text_buffer = ""
            while not sent_stream.closed:
                try:
                    item = await asyncio.wait_for(self._input_ch.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    if self._input_ended:
                        break
                    continue

                if isinstance(item, SynthesizeStream._FlushSentinel):
                    if text_buffer.strip():
                        sent_stream.push_text(text_buffer)
                        text_buffer = ""
                    sent_stream.flush()
                else:
                    text_buffer += item

            # Flush remaining text
            if text_buffer.strip():
                sent_stream.push_text(text_buffer)
                sent_stream.flush()
            sent_stream.end_input()

        # Sentence processor: sends each chunk to TTS
        async def _sentence_processor() -> None:
            async for token in paced:
                text = token.token.strip()
                if not text:
                    continue

                # Use the raw PCM endpoint; the WAV endpoint is kept only for compatibility.
                chunks, _profile = await self._client.synthesize_stream_pcm(
                    text=text,
                    on_chunk=lambda c: self._output_ch.put_nowait(c),
                )

                # Mark end of this sentence's audio
                if chunks:
                    chunks[-1].is_final = True
                    if self._on_audio:
                        self._on_audio(chunks[-1])

        tasks = [
            asyncio.create_task(_input_reader()),
            asyncio.create_task(_sentence_processor()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            self._output_ch.put_nowait(None)  # signal end
            await sent_stream.aclose()

    async def __aenter__(self) -> SynthesizeStream:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *args) -> None:
        self.end_input()
        if self._task:
            await asyncio.wait_for(self._task, timeout=30)
        await self._client.close()

    async def __anext__(self) -> AudioChunk:
        chunk = await self._output_ch.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        return self
