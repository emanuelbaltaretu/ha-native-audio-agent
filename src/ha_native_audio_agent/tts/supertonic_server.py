"""Supertonic HTTP TTS server used by the Docker backends."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import chunk_text_stream
from .protocol import encode_pcm_frame_header, iter_pcm_frames

logger = logging.getLogger("tts-server")


@dataclass(frozen=True)
class ServerConfig:
    model_dir: Path = Path(os.getenv("SUPERTONIC_MODEL_DIR", "/root/.cache/supertonic3"))
    voice: str = os.getenv("TTS_DEFAULT_VOICE", "F1")
    lang: str = os.getenv("TTS_DEFAULT_LANG", "ro")
    steps: int = int(os.getenv("TTS_DEFAULT_STEPS", "5"))
    speed: float = float(os.getenv("TTS_DEFAULT_SPEED", "1.5"))
    first_chunk_max: int = int(os.getenv("TTS_FIRST_CHUNK_MAX", "60"))
    next_chunk_max: int = int(os.getenv("TTS_NEXT_CHUNK_MAX", "240"))
    pcm_frame_bytes: int = int(os.getenv("TTS_PCM_FRAME_BYTES", "8192"))
    first_steps: int | None = (
        int(os.environ["TTS_FIRST_STEPS"]) if os.getenv("TTS_FIRST_STEPS") else None
    )
    openvino_enabled: bool = os.getenv("TTS_OPENVINO_ENABLED", "auto").lower() != "false"
    threads: int = int(os.getenv("TTS_INTRA_OP_THREADS", "4"))
    inter_op_threads: int = int(os.getenv("TTS_INTER_OP_THREADS", "1"))
    parallel_chunks: int = int(os.getenv("TTS_PARALLEL_CHUNKS", "3"))


class OpenVINOModel:
    """Wrap an OpenVINO compiled model to match ONNX Runtime's run() interface."""

    def __init__(self, compiled_model: Any, input_names: list[str]) -> None:
        self._infer_request = compiled_model.create_infer_request()
        self._input_names = input_names

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> tuple[Any]:
        for name in self._input_names:
            if name in input_feed:
                self._infer_request.set_tensor(name, input_feed[name])
        self._infer_request.infer()
        return (self._infer_request.get_output_tensor(0).data,)


class SupertonicRuntime:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.openvino_available = False
        self.openvino_gpu_devices: list[str] = []
        self.voice_cache: dict[str, tuple] = {}
        self._request_lock = threading.Lock()

        from supertonic.loader import load_model

        logger.info("Loading Supertonic model from %s", config.model_dir)
        t0_load = time.time()
        self.tts = load_model(
            config.model_dir,
            auto_download=False,
            intra_op_num_threads=config.threads,
            inter_op_num_threads=config.inter_op_threads,
        )
        logger.info(
            "Model loaded in %.1fs, sr=%s, threads=%s",
            time.time() - t0_load,
            self.tts.sample_rate,
            config.threads,
        )

        self.get_voice_style(config.voice)
        self._try_enable_openvino()

    def get_voice_style(self, voice_name: str) -> tuple:
        if voice_name not in self.voice_cache:
            from supertonic.loader import load_voice_style_from_name

            self.voice_cache[voice_name] = load_voice_style_from_name(
                self.config.model_dir, voice_name
            )
        return self.voice_cache[voice_name]

    def _try_enable_openvino(self) -> None:
        if not self.config.openvino_enabled:
            logger.info("OpenVINO disabled by TTS_OPENVINO_ENABLED=false")
            return

        try:
            import openvino as ov
        except ImportError:
            logger.info("OpenVINO not installed, using CPU only")
            return
        except Exception as exc:
            logger.warning("OpenVINO init failed: %s", exc)
            return

        try:
            core = ov.Core()
            self.openvino_available = True
            self.openvino_gpu_devices = [d for d in core.available_devices if "GPU" in d.upper()]
            if not self.openvino_gpu_devices:
                logger.info("OpenVINO available but no GPU devices: %s", core.available_devices)
                return

            logger.info("OpenVINO GPU devices: %s", self.openvino_gpu_devices)
            onnx_dir = self.config.model_dir / "onnx"
            core.set_property("GPU", {"GPU_ENABLE_LOOP_UNROLLING": "YES"})

            ve_model = core.read_model(str(onnx_dir / "vector_estimator.onnx"))
            ve_compiled = core.compile_model(ve_model, "GPU")
            self.tts.vector_est_ort = OpenVINOModel(
                ve_compiled, [i.any_name for i in ve_model.inputs]
            )

            voc_model = core.read_model(str(onnx_dir / "vocoder.onnx"))
            voc_compiled = core.compile_model(voc_model, "GPU")
            self.tts.vocoder_ort = OpenVINOModel(
                voc_compiled, [i.any_name for i in voc_model.inputs]
            )
            logger.info("Vector estimator and vocoder are running via OpenVINO GPU")
        except Exception as exc:
            logger.warning("OpenVINO GPU compile failed: %s", exc)
            self.openvino_gpu_devices = []

    def generate_chunk(
        self, chunk_text: str, *, steps: int, speed: float, lang: str, voice: str
    ) -> tuple[bytes, float, float]:
        style = self.get_voice_style(voice)
        t0 = time.time()
        wav, dur = self.tts([chunk_text], style, total_step=steps, speed=speed, lang=lang)
        gen_time = time.time() - t0
        audio_int16 = (wav[0] * 32767).astype(np.int16)
        return audio_int16.tobytes(), float(dur[0]), gen_time


class StreamingTTSHandler(BaseHTTPRequestHandler):
    runtime: SupertonicRuntime

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/tts/stream-pcm":
            self._handle_streaming_pcm(body)
        elif self.path == "/tts":
            self._handle_single(body)
        elif self.path == "/tts/stream":
            self._handle_streaming_wav(body)
        else:
            self.send_error(404)

    def _request_params(self, body: dict[str, Any]) -> dict[str, Any]:
        cfg = self.runtime.config
        return {
            "text": body.get("text", "Test audio."),
            "lang": body.get("lang", cfg.lang),
            "steps": int(body.get("steps", cfg.steps)),
            "first_steps": body.get("first_steps", cfg.first_steps),
            "speed": float(body.get("speed", cfg.speed)),
            "voice": body.get("voice", cfg.voice),
            "first_max": int(body.get("first_max", cfg.first_chunk_max)),
            "next_max": int(body.get("next_max", cfg.next_chunk_max)),
            "pcm_frame_bytes": int(body.get("pcm_frame_bytes", cfg.pcm_frame_bytes)),
        }

    def _handle_single(self, body: dict[str, Any]) -> None:
        params = self._request_params(body)
        t0 = time.time()
        with self.runtime._request_lock:
            style = self.runtime.get_voice_style(params["voice"])
            wav, dur = self.runtime.tts(
                [params["text"]],
                style,
                total_step=params["steps"],
                speed=params["speed"],
                lang=params["lang"],
            )
        gen_time = time.time() - t0

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, wav[0], self.runtime.tts.sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-Audio-Duration", f"{dur[0]:.2f}")
        self.send_header("X-Gen-Time", f"{gen_time:.2f}")
        self.send_header("X-RTF", f"{gen_time / dur[0]:.3f}" if dur[0] > 0 else "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(wav_bytes)

    def _generate_remaining(
        self,
        chunks: list[str],
        *,
        steps: int,
        speed: float,
        lang: str,
        voice: str,
    ) -> list[tuple[bytes, float, float] | None]:
        remaining_results: list[tuple[bytes, float, float] | None] = [None] * (len(chunks) - 1)
        if len(chunks) <= 1:
            return remaining_results

        workers = max(1, min(self.runtime.config.parallel_chunks, len(chunks) - 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self.runtime.generate_chunk,
                    chunks[i],
                    steps=steps,
                    speed=speed,
                    lang=lang,
                    voice=voice,
                ): i - 1
                for i in range(1, len(chunks))
            }
            for fut in as_completed(futures):
                remaining_results[futures[fut]] = fut.result()
        return remaining_results

    def _handle_streaming_pcm(self, body: dict[str, Any]) -> None:
        params = self._request_params(body)
        t0 = time.time()
        first_steps = params["steps"] if params["first_steps"] is None else int(params["first_steps"])
        chunks = chunk_text_stream(
            params["text"], first_max=params["first_max"], next_max=params["next_max"]
        )
        logger.info("Streaming PCM: %sch -> %s chunks", len(params["text"]), len(chunks))

        chunk0_data, _chunk0_dur, chunk0_gen = self.runtime.generate_chunk(
            chunks[0],
            steps=first_steps,
            speed=params["speed"],
            lang=params["lang"],
            voice=params["voice"],
        )
        t3 = time.time()

        sr = self.runtime.tts.sample_rate
        frame_bytes = params["pcm_frame_bytes"]

        def write_pcm(data: bytes) -> None:
            for frame in iter_pcm_frames(data, frame_bytes):
                self.wfile.write(encode_pcm_frame_header(sr, len(frame)) + frame)
                self.wfile.flush()

        def write_eof() -> None:
            self.wfile.write(encode_pcm_frame_header(sr, 0))
            self.wfile.flush()

        self.send_response(200)
        self.send_header("Content-Type", f"audio/pcm; rate={sr}; channels=1")
        self.send_header("X-TTFA", f"{t3 - t0:.3f}")
        self.send_header("X-Gen-Time", f"{chunk0_gen:.3f}")
        self.send_header("X-Sample-Rate", str(sr))
        self.send_header("X-Steps", str(params["steps"]))
        self.send_header("X-First-Steps", str(first_steps))
        self.send_header("X-First-Max", str(params["first_max"]))
        self.send_header("X-PCM-Frame-Bytes", str(frame_bytes))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()

        write_pcm(chunk0_data)
        remaining_results = self._generate_remaining(
            chunks,
            steps=params["steps"],
            speed=params["speed"],
            lang=params["lang"],
            voice=params["voice"],
        )
        for result in remaining_results:
            if result is not None:
                write_pcm(result[0])
        write_eof()

        total = time.time() - t0
        logger.info(
            "PCM streaming done: %.2fs total, TTFA=%.3fs, text_len=%s",
            total,
            t3 - t0,
            len(params["text"]),
        )

    def _handle_streaming_wav(self, body: dict[str, Any]) -> None:
        self.send_error(410, "WAV chunk streaming is deprecated; use /tts/stream-pcm")

    def do_GET(self) -> None:
        if self.path == "/health":
            cfg = self.runtime.config
            status = {
                "status": "ok",
                "model": "supertonic3_final",
                "sample_rate": self.runtime.tts.sample_rate,
                "voice_cache": list(self.runtime.voice_cache.keys()),
                "steps_default": cfg.steps,
                "first_steps_default": cfg.first_steps,
                "first_chunk_max_default": cfg.first_chunk_max,
                "next_chunk_max_default": cfg.next_chunk_max,
                "pcm_frame_bytes_default": cfg.pcm_frame_bytes,
                "openvino_available": self.runtime.openvino_available,
                "gpu_devices": self.runtime.openvino_gpu_devices,
            }
            data = json.dumps(status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/voices":
            voices = sorted(p.stem for p in (self.runtime.config.model_dir / "voice_styles").glob("*.json"))
            data = json.dumps({"voices": voices}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], fmt % args)


def build_handler(runtime: SupertonicRuntime) -> type[StreamingTTSHandler]:
    class Handler(StreamingTTSHandler):
        pass

    Handler.runtime = runtime
    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("TTS_PORT", "8020")))
    parser.add_argument("--host", default=os.getenv("TTS_HOST", "0.0.0.0"))
    args = parser.parse_args()

    runtime = SupertonicRuntime(ServerConfig())
    server = ThreadingHTTPServer((args.host, args.port), build_handler(runtime))
    server.daemon_threads = True
    server.allow_reuse_address = True

    logger.info("TTS server listening on %s:%s", args.host, args.port)
    logger.info("Raw PCM stream: POST /tts/stream-pcm")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
