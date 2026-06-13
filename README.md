# HA Native Audio Agent

HA Native Audio Agent is an experimental low-latency native audio agent for Home Assistant. It runs a local audio frontend and an agent runtime on edge hardware, while staying independent of the Home Assistant Assist voice pipeline as the core runtime.

## Repository Layout

```text
config/                 Example configuration
docs/                   Research notes and milestone plans
docker/rpi-tts/         RPi4 Docker setup (benchmarks, TTS server source)
docker/vm101/           vm101 Docker setup (production TTS server)
src/ha_native_audio_agent/   Python package
tests/                  Unit and offline integration tests
```

## TTS Architecture

The text-to-speech runs as a **dedicated Docker container on vm101** (192.168.0.55), not on the RPi4.

| Component | Host | Role |
|-----------|------|------|
| RPi4 (rpi166) | Local network | Wake word, VAD, audio I/O, barge-in |
| vm101 (prod-gpu) | 192.168.0.55 | Supertonic ONNX TTS server |
| TTS model | Supertonic 3 FINAL | Vector estimator INT8, rest FP32 |

The Supertonic server implementation lives in the Python package
(`ha_native_audio_agent.tts.supertonic_server`) and the Docker directories are runtime
packaging only. Model weights are mounted from local paths and are never committed.

See [`docs/deployment.md`](docs/deployment.md) for setup and [`docs/tts-optimization.md`](docs/tts-optimization.md) for benchmarks.

## Current Status

This repository is being scaffolded for Milestone 0, a technical spike. See [`docs/research-findings.md`](docs/research-findings.md) and [`docs/milestone-0-plan.md`](docs/milestone-0-plan.md) for details.

## Development

```bash
python -m compileall src tests
python -m pytest
docker compose --profile tools config
docker compose -f docker/vm101/docker-compose.yml config
docker compose -f docker/rpi-tts/docker-compose.yml config
```

Hardware and provider diagnostics are available as explicit commands so the default test suite remains offline and safe.
