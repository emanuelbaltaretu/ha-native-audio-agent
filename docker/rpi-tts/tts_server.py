"""Streaming TTS HTTP server — Supertonic ONNX chunked streaming.

Optimizations for Time To First Audio (TTFA):
- Small first chunk (40-80 chars) for instant playback
- HTTP Transfer-Encoding: chunked — send audio as it's generated
- Parallel generation of subsequent chunks
- ORT thread count optimized for x86_64 CPU
"""

import io
import json
import time
import logging
import threading
from queue import Queue, Empty
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from supertonic.loader import load_model, load_voice_style_from_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tts-server")

MODEL_DIR = Path("/root/.cache/supertonic3")
DEFAULT_VOICE = "F1"
DEFAULT_LANG = "ro"
DEFAULT_STEPS = 2
DEFAULT_SPEED = 1.5

# Load model at startup
logger.info("Loading Supertonic 3 model...")
t0 = time.time()
tts = load_model(MODEL_DIR, auto_download=False, intra_op_num_threads=4, inter_op_num_threads=1)
style = load_voice_style_from_name(MODEL_DIR, DEFAULT_VOICE)
logger.info(f"Model loaded in {time.time()-t0:.1f}s, sr={tts.sample_rate}, threads=4")


def split_sentences(text: str):
    import re
    parts = re.split(r"(?<=[.?!;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_stream(text: str, first_max: int = 60, next_max: int = 240):
    """Chunk text for streaming — small first chunk for fast TTFA."""
    sentences = split_sentences(text)
    if not sentences:
        return [text]
    chunks = []
    current = ""
    max_for_phase = first_max
    for sent in sentences:
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


def generate_chunk(chunk_text, steps, speed, lang, voice):
    """Generate audio for a single chunk. Thread-safe with new style each call."""
    s = load_voice_style_from_name(MODEL_DIR, voice)
    t0 = time.time()
    wav, dur = tts([chunk_text], s, total_step=steps, speed=speed, lang=lang)
    gen_time = time.time() - t0
    # Convert to int16 PCM bytes
    import struct
    audio_int16 = (wav[0] * 32767).astype(np.int16)
    return audio_int16.tobytes(), float(dur[0]), gen_time


class StreamingTTSHandler(BaseHTTPRequestHandler):
    """Handles POST /tts/stream with chunked WAV streaming response."""

    def do_POST(self):
        if self.path not in ("/tts", "/tts/stream"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        text = body.get("text", "Test audio.")
        lang = body.get("lang", DEFAULT_LANG)
        steps = body.get("steps", DEFAULT_STEPS)
        speed = body.get("speed", DEFAULT_SPEED)
        voice = body.get("voice", DEFAULT_VOICE)
        streaming = self.path == "/tts/stream"

        if streaming:
            self._handle_streaming(text, steps, speed, lang, voice)
        else:
            self._handle_single(text, steps, speed, lang, voice)

    def _handle_single(self, text, steps, speed, lang, voice):
        """Non-streaming: generate all, send complete WAV."""
        t0 = time.time()
        s = load_voice_style_from_name(MODEL_DIR, voice)
        wav, dur = tts([text], s, total_step=steps, speed=speed, lang=lang)
        gen_time = time.time() - t0

        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, wav[0], tts.sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("X-Audio-Duration", f"{dur[0]:.2f}")
        self.send_header("X-Gen-Time", f"{gen_time:.2f}")
        self.send_header("X-RTF", f"{gen_time / dur[0]:.3f}" if dur[0] > 0 else "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(wav_bytes)

    def _handle_streaming(self, text, steps, speed, lang, voice):
        """
        Chunked streaming: generate first chunk, send immediately,
        then generate remaining chunks in parallel and send as ready.
        Uses Transfer-Encoding: chunked.
        Each chunk = WAV header + PCM data (header has chunk size).
        """
        chunks = chunk_text_stream(text)
        n_chunks = len(chunks)
        logger.info(f"Streaming: {len(text)}ch -> {n_chunks} chunks, "
                     f"sizes={[len(c) for c in chunks]}")

        # Start timing
        t_global_start = time.time()

        # Generate first chunk synchronously (for fastest TTFA)
        logger.info(f"Generating chunk 0 ({len(chunks[0])}ch)...")
        chunk0_data, chunk0_dur, chunk0_gen = generate_chunk(
            chunks[0], steps, speed, lang, voice)
        ttfa = time.time() - t_global_start
        logger.info(f"Chunk 0 ready in {chunk0_gen:.2f}s (TTFA={ttfa:.2f}s)")

        # Start parallel generation of remaining chunks
        remaining_results = [None] * (n_chunks - 1)

        def gen_remaining():
            with ThreadPoolExecutor(max_workers=min(3, n_chunks - 1)) as pool:
                futures = {}
                for i in range(1, n_chunks):
                    fut = pool.submit(generate_chunk, chunks[i], steps, speed, lang, voice)
                    futures[fut] = i - 1
                from concurrent.futures import as_completed
                for fut in as_completed(futures):
                    idx = futures[fut]
                    data, dur, gen_t = fut.result()
                    remaining_results[idx] = (data, dur, gen_t)
                    logger.info(f"Chunk {idx+1} ready in {gen_t:.2f}s")

        bg_thread = threading.Thread(target=gen_remaining, daemon=True)
        bg_thread.start()

        # Send HTTP chunked response
        sr = tts.sample_rate
        
        # Helper to build WAV header for given PCM data size
        def build_wav_header(pcm_data_size, sample_rate=sr, channels=1, bits=16):
            import struct
            data_size = pcm_data_size
            header_size = 44
            total_size = header_size + data_size
            
            header = b"RIFF"
            header += struct.pack("<I", total_size - 8)
            header += b"WAVE"
            header += b"fmt "
            header += struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                  sample_rate * channels * bits // 8,
                                  channels * bits // 8, bits)
            header += b"data"
            header += struct.pack("<I", data_size)
            return header

        self.send_response(200)
        self.send_header("Content-Type", "audio/x-wav-chunked")
        self.send_header("X-TTFA", f"{ttfa:.3f}")
        self.send_header("X-Total-Chunks", str(n_chunks))
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send_chunk(data):
            """Send one HTTP chunk: size_hex + data + CRLF."""
            chunk_size = len(data)
            self.wfile.write(f"{chunk_size:x}\r\n".encode())
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        # Send chunk 0 (the first audio)
        chunk0_header = build_wav_header(len(chunk0_data))
        send_chunk(chunk0_header + chunk0_data)

        # Send remaining chunks as they become ready
        for i in range(1, n_chunks):
            idx = i - 1
            while remaining_results[idx] is None:
                time.sleep(0.01)
            data, dur, gen_t = remaining_results[idx]
            chunk_header = build_wav_header(len(data))
            send_chunk(chunk_header + data)

        # End chunked response
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

        total = time.time() - t_global_start
        logger.info(f"Streaming done: {total:.2f}s total, TTFA={ttfa:.2f}s")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "model": "supertonic3_final"}).encode())
        elif self.path == "/voices":
            voices = sorted(p.stem for p in (MODEL_DIR / "voice_styles").glob("*.json"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"voices": voices}).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), StreamingTTSHandler)
    logger.info(f"TTS server listening on {args.host}:{args.port}")
    logger.info(f"  Streaming: chunked WAV via /tts/stream")
    logger.info(f"  Single:    complete WAV via /tts")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
