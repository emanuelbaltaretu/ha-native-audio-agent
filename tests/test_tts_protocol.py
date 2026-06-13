from ha_native_audio_agent.tts.chunking import chunk_text_stream
from ha_native_audio_agent.tts.protocol import (
    PCM_FRAME_HEADER_SIZE,
    decode_pcm_frame_header,
    encode_pcm_frame_header,
    iter_pcm_frames,
)


def test_pcm_frame_header_roundtrip() -> None:
    header = encode_pcm_frame_header(sample_rate=44100, byte_count=8192)

    decoded = decode_pcm_frame_header(header)

    assert len(header) == PCM_FRAME_HEADER_SIZE
    assert decoded.sample_rate == 44100
    assert decoded.byte_count == 8192
    assert not decoded.is_eof


def test_pcm_eof_frame() -> None:
    decoded = decode_pcm_frame_header(encode_pcm_frame_header(44100, 0))

    assert decoded.is_eof


def test_iter_pcm_frames_preserves_even_frame_size() -> None:
    frames = iter_pcm_frames(b"1234567890", frame_bytes=5)

    assert frames == [b"1234", b"5678", b"90"]


def test_chunk_text_stream_splits_long_first_sentence_at_word_boundary() -> None:
    chunks = chunk_text_stream(
        "Am aprins lumina din dormitor. Sistemul raspunde natural.",
        first_max=18,
        next_max=240,
    )

    assert chunks[0] == "Am aprins lumina"
    assert "din dormitor" in chunks[1]
