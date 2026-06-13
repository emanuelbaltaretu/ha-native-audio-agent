"""Wire protocol helpers for framed PCM TTS streams."""

from __future__ import annotations

import struct
from dataclasses import dataclass

PCM_FRAME_HEADER = "<HI"
PCM_FRAME_HEADER_SIZE = struct.calcsize(PCM_FRAME_HEADER)
PCM_EOF_BYTE_COUNT = 0


@dataclass(frozen=True)
class PcmFrameHeader:
    sample_rate: int
    byte_count: int

    @property
    def is_eof(self) -> bool:
        return self.byte_count == PCM_EOF_BYTE_COUNT


def encode_pcm_frame_header(sample_rate: int, byte_count: int) -> bytes:
    """Encode a PCM frame header.

    The current protocol is intentionally tiny:
    - uint16: sample_rate // 100
    - uint32: byte_count

    A zero byte_count is an explicit EOF frame.
    """
    return struct.pack(PCM_FRAME_HEADER, sample_rate // 100, byte_count)


def decode_pcm_frame_header(data: bytes) -> PcmFrameHeader:
    if len(data) != PCM_FRAME_HEADER_SIZE:
        raise ValueError(f"expected {PCM_FRAME_HEADER_SIZE} header bytes, got {len(data)}")
    sample_rate_div100, byte_count = struct.unpack(PCM_FRAME_HEADER, data)
    return PcmFrameHeader(sample_rate=sample_rate_div100 * 100, byte_count=byte_count)


def iter_pcm_frames(data: bytes, frame_bytes: int) -> list[bytes]:
    """Split PCM data into even-sized frames suitable for S16_LE audio."""
    frame_bytes = int(frame_bytes)
    if frame_bytes <= 0:
        return [data]
    if frame_bytes % 2:
        frame_bytes -= 1
    return [data[start:start + frame_bytes] for start in range(0, len(data), frame_bytes)]
