"""TTS module — streaming text-to-speech adapted from LiveKit Agents patterns.

Provides:
- TTSClient — HTTP/WS client for the Supertonic TTS server
- SynthesizeStream — async streaming synthesizer with push_text API
- AudioEmitter — PCM buffer management with is_final tracking
- SentenceStreamPacer — pacing control based on playback buffer level
- ConnectionPool — persistent WebSocket connection manager

All patterns adapted from LiveKit Agents (Apache 2.0).
"""

from .client import TTSClient, TTSClientConfig, Profile, TTFAStats
from .emitter import AudioChunk, AudioEmitter
from .pacer import PacedStream, SentenceStreamPacer, StreamPacerOptions
from .player import PersistentPlayer
from .pool import ConnectionPool
from .synthesizer import SynthesizeStream

__all__ = [
    "TTSClient",
    "TTSClientConfig",
    "Profile",
    "TTFAStats",
    "SynthesizeStream",
    "AudioChunk",
    "AudioEmitter",
    "PersistentPlayer",
    "SentenceStreamPacer",
    "StreamPacerOptions",
    "PacedStream",
    "ConnectionPool",
]
