"""Live playback helper for framed PCM TTS streams."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass

from .player import RawAplayPlayer
from .protocol import PCM_FRAME_HEADER_SIZE, decode_pcm_frame_header


@dataclass(frozen=True)
class LivePlaybackResult:
    first_frame_seconds: float
    server_ttfa_seconds: float
    total_seconds: float
    frames: int
    audio_bytes: int
    sample_rate: int


def play_streaming_pcm(
    *,
    url: str,
    text: str,
    device: str,
    steps: int = 5,
    first_steps: int | None = None,
    first_max: int | None = None,
    next_max: int | None = None,
    pcm_frame_bytes: int | None = None,
    speed: float = 1.5,
    lang: str = "ro",
    voice: str = "F1",
    timeout: float = 30.0,
) -> LivePlaybackResult:
    """Stream framed PCM from the TTS backend and write it to an ALSA device."""
    payload: dict[str, str | int | float] = {
        "text": text,
        "steps": steps,
        "speed": speed,
        "lang": lang,
        "voice": voice,
    }
    if first_steps is not None:
        payload["first_steps"] = first_steps
    if first_max is not None:
        payload["first_max"] = first_max
    if next_max is not None:
        payload["next_max"] = next_max
    if pcm_frame_bytes is not None:
        payload["pcm_frame_bytes"] = pcm_frame_bytes

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_frame_seconds: float | None = None
    frames = 0
    audio_bytes = 0

    with urllib.request.urlopen(req, timeout=timeout) as response:
        headers = dict(response.info())
        sample_rate = int(headers.get("X-Sample-Rate", "44100"))
        server_ttfa_seconds = float(headers.get("X-TTFA", "0"))

        with RawAplayPlayer(device=device) as player:
            while True:
                header = response.read(PCM_FRAME_HEADER_SIZE)
                if not header:
                    break
                if len(header) < PCM_FRAME_HEADER_SIZE:
                    raise RuntimeError(f"incomplete PCM frame header: {len(header)} bytes")
                frame_header = decode_pcm_frame_header(header)
                if frame_header.is_eof:
                    break
                byte_count = frame_header.byte_count
                data = response.read(byte_count)
                if len(data) < byte_count:
                    raise RuntimeError(f"incomplete PCM frame: {len(data)} < {byte_count}")
                if first_frame_seconds is None:
                    first_frame_seconds = time.perf_counter() - t0
                frames += 1
                audio_bytes += len(data)
                player.write(data, sample_rate=sample_rate, channels=1)

    total_seconds = time.perf_counter() - t0
    return LivePlaybackResult(
        first_frame_seconds=first_frame_seconds or 0.0,
        server_ttfa_seconds=server_ttfa_seconds,
        total_seconds=total_seconds,
        frames=frames,
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://192.168.0.55:8020/tts/stream-pcm")
    parser.add_argument(
        "--text",
        default="Am aprins lumina din dormitor. Sistemul răspunde acum cu latență redusă.",
    )
    parser.add_argument("--device", default="plughw:0,0")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--first-steps", type=int)
    parser.add_argument("--first-max", type=int)
    parser.add_argument("--next-max", type=int)
    parser.add_argument("--pcm-frame-bytes", type=int)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--lang", default="ro")
    parser.add_argument("--voice", default="F1")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    print(f"playing device={args.device} url={args.url}")
    result = play_streaming_pcm(
        url=args.url,
        text=args.text,
        device=args.device,
        steps=args.steps,
        first_steps=args.first_steps,
        first_max=args.first_max,
        next_max=args.next_max,
        pcm_frame_bytes=args.pcm_frame_bytes,
        speed=args.speed,
        lang=args.lang,
        voice=args.voice,
        timeout=args.timeout,
    )
    print(
        f"first_frame={result.first_frame_seconds:.3f}s "
        f"server_ttfa={result.server_ttfa_seconds:.3f}s "
        f"total={result.total_seconds:.3f}s frames={result.frames} "
        f"audio_bytes={result.audio_bytes} sample_rate={result.sample_rate}"
    )


if __name__ == "__main__":
    main()
