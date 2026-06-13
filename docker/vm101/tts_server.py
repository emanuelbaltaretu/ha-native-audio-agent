"""Streaming TTS HTTP server — Supertonic ONNX chunked streaming.

Optimizations:
- Voice style cached in RAM (load once at startup)
- Raw PCM endpoint (no WAV overhead)
- Profiling timestamps (T0-T4 logged)
- Model pre-loaded, persistent
"""

import io
import json
import time
import struct
import logging
import threading
import os
from queue import Queue, Empty
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from supertonic.loader import load_model, load_voice_style_from_name
import supertonic.loader as st_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tts-server")

MODEL_DIR = Path("/root/.cache/supertonic3")
DEFAULT_VOICE = "F1"
DEFAULT_LANG = "ro"
DEFAULT_STEPS = 5
DEFAULT_SPEED = 1.5

# === OpenVINO GPU acceleration ===
_openvino_available = False
_ov_gpu_devices = []

class OpenVINOModel:
    """Wraps an OpenVINO compiled model to match ONNX Runtime's run() interface."""
    def __init__(self, compiled_model, input_names, output_name):
        self._compiled = compiled_model
        self._infer_request = compiled_model.create_infer_request()
        self._input_names = input_names
        self._output_name = output_name
    
    def run(self, output_names, input_feed):
        """Mimics ort.InferenceSession.run(output_names, input_feed)."""
        for name in self._input_names:
            if name in input_feed:
                self._infer_request.set_tensor(name, input_feed[name])
        self._infer_request.infer()
        result = self._infer_request.get_output_tensor(0).data
        return (result,)

try:
    import openvino as ov
    _openvino_available = True
    _ov_core = ov.Core()
    _ov_gpu_devices = [d for d in _ov_core.available_devices if "GPU" in d.upper()]
    if _ov_gpu_devices:
        logger.info(f"OpenVINO GPU devices: {_ov_gpu_devices}")
        for d in _ov_gpu_devices:
            name = _ov_core.get_property(d, "FULL_DEVICE_NAME")
            logger.info(f"  {d}: {name}")
    else:
        logger.info(f"OpenVINO available (CPU only): {_ov_core.available_devices}")
except ImportError:
    logger.info("OpenVINO not installed, using CPU only")
except Exception as e:
    logger.warning(f"OpenVINO init failed: {e}")

# === Persistent model + voice style cache ===
logger.info("Loading Supertonic 3 model...")
t0_load = time.time()
tts = load_model(MODEL_DIR, auto_download=False, intra_op_num_threads=4, inter_op_num_threads=1)
logger.info(f"Model loaded in {time.time()-t0_load:.1f}s, sr={tts.sample_rate}, threads=4")

# Voice style cache — load once, reuse forever
_voice_cache: dict[str, tuple] = {}

def get_voice_style(voice_name: str):
    """Cached voice style loader — loaded once, kept in RAM."""
    if voice_name not in _voice_cache:
        _voice_cache[voice_name] = load_voice_style_from_name(MODEL_DIR, voice_name)
    return _voice_cache[voice_name]

# Pre-warm default voice
logger.info(f"Pre-warming voice '{DEFAULT_VOICE}'...")
get_voice_style(DEFAULT_VOICE)
logger.info("Voice style cached")

# === Replace bottleneck models with OpenVINO GPU ===
_ov_ve_model = None
_ov_voc_model = None

if _ov_gpu_devices:
    logger.info("Compiling vector estimator and vocoder on GPU with OpenVINO...")
    try:
        onnx_dir = MODEL_DIR / "onnx"
        
        # VE (bottleneck #1)
        t0_ov = time.time()
        ve_onnx_path = str(onnx_dir / "vector_estimator.onnx")
        ve_model = _ov_core.read_model(ve_onnx_path)
        # Set optimal config for GPU
        _ov_core.set_property("GPU", {"GPU_ENABLE_LOOP_UNROLLING": "YES"})
        ve_compiled = _ov_core.compile_model(ve_model, "GPU")
        ve_inputs = [i.any_name for i in ve_model.inputs]
        ve_output = ve_model.output(0).any_name
        _ov_ve_model = OpenVINOModel(ve_compiled, ve_inputs, ve_output)
        t_ve_ov = time.time() - t0_ov
        logger.info(f"  VE compiled on GPU in {t_ve_ov:.1f}s (inputs={ve_inputs})")
        
        # Vocoder (bottleneck #2)
        t0_ov = time.time()
        voc_onnx_path = str(onnx_dir / "vocoder.onnx")
        voc_model = _ov_core.read_model(voc_onnx_path)
        voc_compiled = _ov_core.compile_model(voc_model, "GPU")
        voc_inputs = [i.any_name for i in voc_model.inputs]
        voc_output = voc_model.output(0).any_name
        _ov_voc_model = OpenVINOModel(voc_compiled, voc_inputs, voc_output)
        t_voc_ov = time.time() - t0_ov
        logger.info(f"  Vocoder compiled on GPU in {t_voc_ov:.1f}s (inputs={voc_inputs})")
        
        # Replace ORT sessions with OpenVINO GPU versions
        tts.vector_est_ort = _ov_ve_model
        tts.vocoder_ort = _ov_voc_model
        logger.info("VE and Vocoder now running on GPU via OpenVINO")
    except Exception as e:
        logger.warning(f"OpenVINO GPU compile failed: {e}")
        logger.warning("Falling back to CPU for all models")
        _ov_ve_model = None
        _ov_voc_model = None
else:
    logger.info("No OpenVINO GPU devices, using CPU only")


def split_sentences(text: str):
    import re
    parts = re.split(r"(?<=[.?!;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_stream(text: str, first_max: int = 60, next_max: int = 240):
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
    """Generate audio for a single chunk. Uses cached voice style."""
    s = get_voice_style(voice)
    t0 = time.time()
    wav, dur = tts([chunk_text], s, total_step=steps, speed=speed, lang=lang)
    gen_time = time.time() - t0
    audio_int16 = (wav[0] * 32767).astype(np.int16)
    return audio_int16.tobytes(), float(dur[0]), gen_time


class StreamingTTSHandler(BaseHTTPRequestHandler):
    """Handles TTS requests with streaming support."""

    # Pre-allocate thread pool for parallel chunk generation
    _executor = ThreadPoolExecutor(max_workers=4)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        text = body.get("text", "Test audio.")
        lang = body.get("lang", DEFAULT_LANG)
        steps = body.get("steps", DEFAULT_STEPS)
        speed = body.get("speed", DEFAULT_SPEED)
        voice = body.get("voice", DEFAULT_VOICE)

        if self.path == "/tts/stream":
            self._handle_streaming(text, steps, speed, lang, voice)
        elif self.path == "/tts/stream-pcm":
            self._handle_streaming_pcm(text, steps, speed, lang, voice)
        elif self.path == "/tts":
            self._handle_single(text, steps, speed, lang, voice)
        else:
            self.send_error(404)

    # ── T0: request received ──

    def _handle_single(self, text, steps, speed, lang, voice):
        """Non-streaming: generate all, send complete WAV."""
        T0 = time.time()  # noqa
        s = get_voice_style(voice)
        wav, dur = tts([text], s, total_step=steps, speed=speed, lang=lang)
        gen_time = time.time() - T0

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
        """Chunked streaming via WAV chunks (legacy — kept for compat)."""
        T0 = time.time()  # noqa: request received (T0)
        chunks = chunk_text_stream(text)
        n_chunks = len(chunks)
        logger.info(f"Streaming WAV: {len(text)}ch -> {n_chunks} chunks")

        # T1 = start generating chunk 0
        chunk0_data, chunk0_dur, chunk0_gen = generate_chunk(
            chunks[0], steps, speed, lang, voice)
        # T3 = chunk 0 inference done
        T3 = time.time()

        # Generate remaining in parallel
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

        bg_thread = threading.Thread(target=gen_remaining, daemon=True)
        bg_thread.start()

        sr = tts.sample_rate

        def build_wav_header(pcm_data_size, sample_rate=sr, channels=1, bits=16):
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
        self.send_header("X-TTFA", f"{T3 - T0:.3f}")
        self.send_header("X-Total-Chunks", str(n_chunks))
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Transfer-Encoding", "chunked")
        # T4: first byte sent
        self.end_headers()

        def send_chunk(data):
            chunk_size = len(data)
            self.wfile.write(f"{chunk_size:x}\r\n".encode())
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        chunk0_header = build_wav_header(len(chunk0_data))
        send_chunk(chunk0_header + chunk0_data)

        for i in range(1, n_chunks):
            idx = i - 1
            while remaining_results[idx] is None:
                time.sleep(0.01)
            data, dur, gen_t = remaining_results[idx]
            chunk_header = build_wav_header(len(data))
            send_chunk(chunk_header + data)

        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

        total = time.time() - T0
        logger.info(f"WAV streaming done: {total:.2f}s total, TTFA={T3 - T0:.3f}s")

    def _handle_streaming_pcm(self, text, steps, speed, lang, voice):
        """Streaming via raw PCM — no WAV overhead.
        
        Sends raw PCM chunks with a tiny binary header per chunk:
          struct: <H (sample_rate // 100), I (byte_count)
          followed by raw PCM int16 data.
        """
        import struct as st

        T0 = time.time()  # noqa: request received
        chunks = chunk_text_stream(text)
        n_chunks = len(chunks)
        logger.info(f"Streaming PCM: {len(text)}ch -> {n_chunks} chunks, text='{chunks[0][:40]}...'")

        # T1 = start inference
        chunk0_data, chunk0_dur, chunk0_gen = generate_chunk(chunks[0], steps, speed, lang, voice)
        T3 = time.time()  # T3: inference done

        sr = tts.sample_rate
        # T4 estimate: time to send first response bytes
        # (will be set to T3 + header serialization time)

        # Generate remaining in parallel
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

        bg_thread = threading.Thread(target=gen_remaining, daemon=True)
        bg_thread.start()

        # Tiny per-chunk header: sample_rate_div100 (H), byte_count (I)
        def pcm_chunk_header(data_len: int) -> bytes:
            return st.pack("<HI", sr // 100, data_len)

        self.send_response(200)
        self.send_header("Content-Type", "audio/pcm; rate=44100; channels=1")
        self.send_header("X-TTFA", f"{T3 - T0:.3f}")
        self.send_header("X-Gen-Time", f"{chunk0_gen:.3f}")
        self.send_header("X-Total-Chunks", str(n_chunks))
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "keep-alive")
        # T4: first byte sent (headers written)
        self.end_headers()

        # Send chunk 0 raw PCM with tiny header
        ch0_hdr = pcm_chunk_header(len(chunk0_data))
        self.wfile.write(ch0_hdr + chunk0_data)
        self.wfile.flush()

        # Send remaining chunks
        for i in range(1, n_chunks):
            idx = i - 1
            while remaining_results[idx] is None:
                time.sleep(0.005)
            data, dur, gen_t = remaining_results[idx]
            hdr = pcm_chunk_header(len(data))
            self.wfile.write(hdr + data)
            self.wfile.flush()

        self.wfile.flush()
        total = time.time() - T0
        logger.info(
            f"PCM streaming done: {total:.2f}s total, "
            f"TTFA={T3 - T0:.3f}s (inference={chunk0_gen:.3f}s), "
            f"text_len={len(text)}"
        )

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            status = {
                "status": "ok",
                "model": "supertonic3_final",
                "sample_rate": tts.sample_rate,
                "voice_cache": list(_voice_cache.keys()),
                "steps_default": DEFAULT_STEPS,
                "providers": ["CPUExecutionProvider"],
                "openvino_available": _openvino_available,
                "gpu_devices": _ov_gpu_devices,
            }
            self.wfile.write(json.dumps(status).encode())
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
    # Enable HTTP keep-alive
    server.allow_reuse_address = True
    server.timeout = 30

    logger.info(f"TTS server listening on {args.host}:{args.port}")
    logger.info(f"  Endpoints:")
    logger.info(f"    WAV streaming:  POST /tts/stream")
    logger.info(f"    Raw PCM stream: POST /tts/stream-pcm  (NEW — recommended)")
    logger.info(f"    Single WAV:     POST /tts")
    logger.info(f"  Defaults: steps={DEFAULT_STEPS}, speed={DEFAULT_SPEED}, voice={DEFAULT_VOICE}")
    logger.info(f"  Optimizations: voice cache={list(_voice_cache.keys())}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
