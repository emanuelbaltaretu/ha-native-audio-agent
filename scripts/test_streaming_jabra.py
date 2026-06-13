#!/usr/bin/env python3
"""Test TTS streaming: grab chunked WAV from Supertonic server and play on Jabra.

Uses httpx for proper chunked transfer-encoding handling.
"""

import subprocess
import sys
import json
import struct
import time
import urllib.request
import urllib.error

TTS_URL = "http://192.168.0.55:8020"
TEST_TEXT = "Bună ziua, sistemul voice home agent funcționează în streaming. Aceasta este o testare a modulului de sinteză vocală, care rulează pe server și trimite audio la difuzor."
VOICE = "F1"
LANG = "ro"
STEPS = 2
SPEED = 1.5


def parse_wav_header(data: bytes) -> tuple[int, int, int]:
    """Return (sample_rate, channels, data_offset) from WAV header."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV")
    offset = 20 + struct.unpack_from("<I", data, 16)[0]
    fmt = struct.unpack_from("<H", data, 20)[0]
    ch = struct.unpack_from("<H", data, 22)[0]
    sr = struct.unpack_from("<I", data, 24)[0]
    if fmt != 1:
        raise ValueError(f"Format {fmt} not PCM")
    while offset + 8 <= len(data):
        cid = data[offset:offset+4]
        csz = struct.unpack_from("<I", data, offset+4)[0]
        if cid == b"data":
            return sr, ch, offset + 8
        offset += 8 + csz
    raise ValueError("No data chunk")


def main():
    payload = json.dumps({
        "text": TEST_TEXT,
        "voice": VOICE, "lang": LANG,
        "steps": STEPS, "speed": SPEED,
    }).encode()

    print(f"⚡ Streaming {len(TEST_TEXT)} chars → Jabra...")
    t0 = time.time()

    req = urllib.request.Request(
        f"{TTS_URL}/tts/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    resp = urllib.request.urlopen(req, timeout=30)
    info = resp.info()

    sr = int(info.get("X-Sample-Rate", 24000))
    ttfa_str = info.get("X-TTFA", "0")
    print(f"  Sample rate: {sr} Hz, TTFA: {ttfa_str}s")

    # Read raw byte stream (HTTP chunked decoded by urllib)
    buffer = b""
    chunk_idx = 0
    first_chunk_time = None
    audio_proc = None
    chunk_start = 0

    while True:
        raw = resp.read(65536)
        if not raw:
            break
        buffer += raw

        # The server sends each WAV as a separate HTTP chunk.
        # But urllib already decodes chunked TE, so we get the raw
        # concatenated WAV data. Each WAV starts with RIFF header.
        # Split on RIFF boundaries.
        while True:
            riff_idx = buffer.find(b"RIFF", chunk_start if chunk_start > 0 else 0)
            if riff_idx < 0:
                break

            # Need at least 44 bytes for WAV header
            if riff_idx + 44 > len(buffer):
                break

            # Get total file size from RIFF header
            file_size = struct.unpack_from("<I", buffer, riff_idx + 4)[0] + 8
            chunk_end = riff_idx + file_size

            if chunk_end > len(buffer):
                break  # incomplete WAV, wait for more data

            wav_data = buffer[riff_idx:chunk_end]
            buffer = buffer[chunk_end:]
            chunk_start = 0

            chunk_idx += 1
            now = time.time()
            if first_chunk_time is None:
                first_chunk_time = now
                print(f"  Chunk 1 arrived at {now - t0:.2f}s ({len(wav_data)} bytes)")

            # Parse WAV → PCM
            try:
                pcm_sr, pcm_ch, offset = parse_wav_header(wav_data)
                pcm_data = wav_data[offset:]
                dur = len(pcm_data) / (pcm_ch * 2) / pcm_sr
                print(f"  Chunk {chunk_idx}: {len(wav_data)}B WAV → {len(pcm_data)}B PCM, {dur:.2f}s audio")
            except ValueError as e:
                print(f"  ⚠ Chunk {chunk_idx} parse error: {e}")
                continue

            # Play via aplay
            if audio_proc is None:
                print(f"  Starting aplay: {pcm_sr}Hz, {pcm_ch}ch, s16le...")
                audio_proc = subprocess.Popen(
                    ["aplay", "-D", "plughw:0,0", "-r", str(pcm_sr), "-c", str(pcm_ch),
                     "-f", "S16_LE", "-t", "raw"],
                    stdin=subprocess.PIPE,
                )

            if audio_proc and audio_proc.stdin:
                audio_proc.stdin.write(pcm_data)
                audio_proc.stdin.flush()

    # Close aplay
    if audio_proc:
        audio_proc.stdin.close()
        audio_proc.wait()

    total = time.time() - t0
    ttfa_val = first_chunk_time - t0 if first_chunk_time else 0
    print(f"✅ Done in {total:.2f}s total, {chunk_idx} chunks, TTFA={ttfa_val:.2f}s")
    resp.close()


if __name__ == "__main__":
    main()
