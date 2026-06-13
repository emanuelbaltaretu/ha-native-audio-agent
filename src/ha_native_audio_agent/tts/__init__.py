"""TTS module for streaming text-to-speech and local playback."""

from __future__ import annotations

from typing import Any

__all__ = [
    "TTSClient",
    "TTSClientConfig",
    "Profile",
    "TTFAStats",
    "SynthesizeStream",
    "AudioChunk",
    "AudioEmitter",
    "PersistentPlayer",
    "RawAplayPlayer",
    "SentenceStreamPacer",
    "StreamPacerOptions",
    "PacedStream",
    "ConnectionPool",
    "PCM_EOF_BYTE_COUNT",
    "PCM_FRAME_HEADER",
    "PCM_FRAME_HEADER_SIZE",
    "decode_pcm_frame_header",
    "encode_pcm_frame_header",
]


def __getattr__(name: str) -> Any:
    if name in {"TTSClient", "TTSClientConfig", "Profile", "TTFAStats"}:
        from . import client

        return getattr(client, name)
    if name in {"AudioChunk", "AudioEmitter"}:
        from . import emitter

        return getattr(emitter, name)
    if name in {"PersistentPlayer", "RawAplayPlayer"}:
        from . import player

        return getattr(player, name)
    if name in {"SentenceStreamPacer", "StreamPacerOptions", "PacedStream"}:
        from . import pacer

        return getattr(pacer, name)
    if name == "ConnectionPool":
        from . import pool

        return getattr(pool, name)
    if name == "SynthesizeStream":
        from . import synthesizer

        return getattr(synthesizer, name)
    if name in {
        "PCM_EOF_BYTE_COUNT",
        "PCM_FRAME_HEADER",
        "PCM_FRAME_HEADER_SIZE",
        "decode_pcm_frame_header",
        "encode_pcm_frame_header",
    }:
        from . import protocol

        return getattr(protocol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
