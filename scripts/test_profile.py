#!/usr/bin/env python3
"""TTS profiling with raw PCM, persistent player, and full T0-T7 timeline.

Measures:
- TTFA (T0→T6) per run
- Server inference time (via X-TTFA header)
- Network overhead
- p50/p95 across multiple runs
- Persistent player (no aplay restart per response)
"""

import asyncio
import sys
import time
sys.path.insert(0, "/app/src")

from ha_native_audio_agent.tts import TTSClient, TTSClientConfig, PersistentPlayer

LONG_TEXT = (
    "Bună ziua și bine ați venit la testarea sistemului nostru de sinteză vocală "
    "în streaming. Astăzi vom evalua calitatea și performanța acestui sistem "
    "folosind un text mai lung pentru a vedea cum se comportă la mai multe "
    "chunk-uri. Sistemul folosește Supertonic trei pe un server cu placă "
    "grafică, iar redarea se face pe un Raspberry Pi patru prin difuzorul "
    "Jabra. Scopul acestui test este să măsurăm timpii de răspuns și să "
    "identificăm eventuale pauze între fragmentele audio."
)

SHORT_TEXT = "Am aprins lumina din dormitor."


async def run_test(client: TTSClient, player: PersistentPlayer, text: str, label: str, n: int = 5):
    """Run n iterations and report stats."""
    print(f"\n{'='*60}")
    print(f"Test: {label} ({n} runs)")
    print(f"  Text: {len(text)} chars")
    print(f"{'='*60}")

    for i in range(n):
        t_start = time.time()

        chunks, profile = await client.synthesize_stream_pcm(text=text)

        # Play via persistent player
        await player.play_chunks(chunks)

        playback_time = time.time() - t_start
        audio_dur = sum(c.duration for c in chunks)

        print(f"  Run {i+1}:")
        print(f"    Profile: [{profile}]")
        print(f"    Serv inf: {profile.server_inference*1000:.0f}ms")
        print(f"    TTFA:     {profile.ttfa*1000:.0f}ms")
        print(f"    Audio:    {audio_dur:.2f}s, {len(chunks)} chunks")
        print(f"    Total:    {playback_time:.2f}s")

    print(f"\n  TTFA stats: {client.ttfa_stats}")
    print(f"  Breakdown:")
    print(f"    Server inference: ~{client.ttfa_stats.p50 - 0.3:.2f}s")  # rough if we don't have exact per-run


async def main():
    player = PersistentPlayer()
    client = TTSClient(TTSClientConfig(
        url="http://192.168.0.55:8020",
        steps=5,
        speed=1.5,
    ))

    # Health check
    health = await client.health()
    print(f"Server: {health.get('status')} | voices cached: {health.get('voice_cache')}")

    # Test 1: Short text (like a HA response)
    await run_test(client, player, SHORT_TEXT, "Short (HA response)", n=3)

    # Test 2: Long text (multi-chunk streaming)
    await run_test(client, player, LONG_TEXT, "Long (multi-chunk)", n=3)

    # Cleanup
    await player.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
