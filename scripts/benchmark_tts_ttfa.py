#!/usr/bin/env python3
"""Benchmark Supertonic streaming TTFA without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

from ha_native_audio_agent.tts.protocol import PCM_FRAME_HEADER_SIZE, decode_pcm_frame_header



def read_stream(url: str, payload: dict, timeout: float) -> tuple[float, float, float, int, int, dict]:
    """Return first-frame latency, total stream time, server TTFA, frame count, bytes, headers."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_frame_time: float | None = None
    frames = 0
    audio_bytes = 0
    with urllib.request.urlopen(req, timeout=timeout) as response:
        headers = dict(response.info())
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
            frames += 1
            audio_bytes += len(data)
            if first_frame_time is None:
                first_frame_time = time.perf_counter() - t0
    total_time = time.perf_counter() - t0
    server_ttfa = float(headers.get("X-TTFA", 0))
    return first_frame_time or 0.0, total_time, server_ttfa, frames, audio_bytes, headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://192.168.0.55:8020/tts/stream-pcm")
    parser.add_argument("--text", default="Am aprins lumina din dormitor.")
    parser.add_argument("--runs", type=int, default=5)
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

    payload = {
        "text": args.text,
        "steps": args.steps,
        "speed": args.speed,
        "lang": args.lang,
        "voice": args.voice,
    }
    if args.first_steps is not None:
        payload["first_steps"] = args.first_steps
    if args.first_max is not None:
        payload["first_max"] = args.first_max
    if args.next_max is not None:
        payload["next_max"] = args.next_max
    if args.pcm_frame_bytes is not None:
        payload["pcm_frame_bytes"] = args.pcm_frame_bytes

    results = []
    print(f"url={args.url}")
    print(f"text_len={len(args.text)} payload={payload}")
    for index in range(args.runs):
        result = read_stream(args.url, payload, args.timeout)
        results.append(result)
        first, total, server, frames, audio_bytes, headers = result
        print(
            f"run={index + 1} first_frame={first:.3f}s server={server:.3f}s "
            f"total={total:.3f}s frames={frames} bytes={audio_bytes} "
            f"first_steps={headers.get('X-First-Steps')} first_max={headers.get('X-First-Max')} "
            f"pcm_frame_bytes={headers.get('X-PCM-Frame-Bytes')}"
        )
        time.sleep(0.4)

    first_values = [r[0] for r in results]
    server_values = [r[2] for r in results]
    total_values = [r[1] for r in results]
    print(
        "summary "
        f"first_frame_p50={statistics.median(first_values):.3f}s "
        f"server_p50={statistics.median(server_values):.3f}s "
        f"total_p50={statistics.median(total_values):.3f}s"
    )


if __name__ == "__main__":
    main()
