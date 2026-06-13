#!/usr/bin/env python3
"""Streaming TTS test — measure chunks, gaps, TTFA, play on Jabra."""

import asyncio
import time
import subprocess
import sys
sys.path.insert(0, '/app/src')

from ha_native_audio_agent.tts import TTSClient, TTSClientConfig

LONG_TEXT = (
    "Bună ziua și bine ați venit la testarea sistemului nostru de sinteză vocală "
    "în streaming. Astăzi vom evalua calitatea și performanța acestui sistem "
    "folosind un text mai lung pentru a vedea cum se comportă la mai multe "
    "chunk-uri. Sistemul folosește Supertonic trei pe un server cu placă "
    "grafică, iar redarea se face pe un Raspberry Pi patru prin difuzorul "
    "Jabra. Scopul acestui test este să măsurăm timpii de răspuns și să "
    "identificăm eventuale pauze între fragmentele audio. Vom testa cu "
    "cinci pași și o viteză de unu virgulă cinci, pentru un echilibru "
    "între calitate și rapiditate."
)

async def main():
    client = TTSClient(TTSClientConfig(url="http://192.168.0.55:8020"))

    t0 = time.time()
    chunk_log = []
    audio_buffers = []

    def on_chunk(chunk):
        now = time.time()
        gap = now - chunk_log[-1]["arrival"] if chunk_log else 0.0
        entry = {
            "idx": len(chunk_log) + 1,
            "arrival": now - t0,
            "gap": gap,
            "size": len(chunk.data),
            "dur": chunk.duration,
            "final": chunk.is_final,
        }
        chunk_log.append(entry)
        audio_buffers.append(chunk)

    chunks = await client.synthesize_stream(
        text=LONG_TEXT, steps=5, speed=1.5, on_chunk=on_chunk,
    )

    total_audio = sum(c.duration for c in chunks)
    total_time = time.time() - t0

    # Play via aplay
    if chunks:
        sr = chunks[0].sample_rate
        proc = subprocess.Popen(
            ["aplay", "-D", "plughw:0,0", "-r", str(sr), "-c", "1",
             "-f", "S16_LE", "-t", "raw"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        for c in chunks:
            if proc.stdin:
                proc.stdin.write(c.data)
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    # Report
    print(f"TTFA: {chunk_log[0]['arrival']:.3f}s" if chunk_log else "TTFA: N/A")
    print(f"Total: {total_time:.2f}s for {total_audio:.2f}s audio "
          f"(RTF {total_time/total_audio:.2f})")
    print(f"Chunks: {len(chunk_log)}")
    for c in chunk_log:
        g = f"gap +{c['gap']:.3f}s" if c["gap"] > 0.05 else "gap ~0"
        print(f"  #{c['idx']}: +{c['arrival']:.2f}s | {c['size']//1024}KB "
              f"| {c['dur']:.2f}s audio | {g}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
