"""Text chunking helpers for streaming TTS."""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_at_word_boundary(text: str, max_len: int) -> tuple[str, str]:
    """Split text at a word boundary, keeping the first piece <= max_len when possible."""
    text = text.strip()
    if len(text) <= max_len:
        return text, ""
    split_at = text.rfind(" ", 0, max_len + 1)
    if split_at < 10:
        split_at = max_len
    return text[:split_at].strip(), text[split_at:].strip()


def chunk_text_stream(text: str, *, first_max: int = 60, next_max: int = 240) -> list[str]:
    """Chunk text for request-level streaming.

    This is not model-internal audio streaming. TTFA is still bounded by generating the
    first text chunk, so callers should choose `first_max` carefully and avoid cutting
    words unless explicitly optimizing latency.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""
    max_for_phase = first_max
    pending = list(sentences)

    while pending:
        sent = pending.pop(0)
        if not current and len(sent) > max_for_phase:
            head, tail = split_at_word_boundary(sent, max_for_phase)
            if head:
                chunks.append(head)
                max_for_phase = next_max
            if tail:
                pending.insert(0, tail)
            continue

        if len(current) + len(sent) + 1 <= max_for_phase:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
                max_for_phase = next_max
            current = sent

    if current:
        chunks.append(current)

    if len(chunks) == 1:
        return chunks
    if len(chunks[0]) < 10 and len(chunks) > 1:
        chunks[0] = chunks[0] + " " + chunks[1]
        chunks.pop(1)
    return chunks
