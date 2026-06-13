#!/usr/bin/env python3
"""Clean TTS streaming test — uses our tts module with proper AudioChunk playback."""

import asyncio
import sys
import time
import subprocess
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ha_native_audio_agent.tts import TTSClient, TTSClientConfig, AudioChunk


async def on_chunk(chunk: AudioChunk, *, aplay_proc: subprocess.Popen | None = None):
    """Callback for each audio chunk: pipe to aplay."""
    if aplay_proc and aplay_proc.stdin:
        aplay_proc.stdin.write(chunk.data)
        aplay_proc.stdin.flush()


async def main():
    config = TTSClientConfig(
        url="http://192.168.0.55:8020",
        voice="F1",
        lang="ro",
        steps=2,
        speed=1.5,
    )
    client = TTSClient(config)

    # Health check
    health = await client.health()
    print(f"✓ Server: {health}")

    text = sys.argv[1] if len(sys.argv) > 1 else (
        "Bună ziua, acesta este un test al sistemului de sinteză vocală "
        "în streaming. Funcționează direct pe Raspberry Pi patru, "
        "prin difuzorul Jabra."
    )

    print(f"⚡ {len(text)} chars → Jabra via ALSA plughw:0,0")
    t0 = time.time()

    # Collect chunks and play
    chunks = await client.synthesize_stream(text=text)

    if not chunks:
        print("✗ No audio received")
        return

    # Start aplay
    sr = chunks[0].sample_rate
    ch = chunks[0].num_channels
    proc = subprocess.Popen(
        ["aplay", "-D", "plughw:0,0", "-r", str(sr), "-c", str(ch),
         "-f", "S16_LE", "-t", "raw"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    total_audio = 0.0
    for i, chunk in enumerate(chunks):
        if proc.stdin:
            proc.stdin.write(chunk.data)
        total_audio += chunk.duration
        print(f"  Chunk {i+1}: {len(chunk.data)}B PCM, {chunk.duration:.2f}s"
              f"{' [FINAL]' if chunk.is_final else ''}")

    if proc.stdin:
        proc.stdin.close()
    proc.wait()

    elapsed = time.time() - t0
    print(f"✅ {elapsed:.2f}s total, {len(chunks)} chunks, {total_audio:.2f}s audio, RTF={elapsed/max(total_audio,0.01):.2f}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
