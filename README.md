# HA Native Audio Agent

HA Native Audio Agent is an experimental low-latency native audio agent for Home Assistant. It runs a local audio frontend and an agent runtime on edge hardware, while staying independent of the Home Assistant Assist voice pipeline as the core runtime.

## Repository Layout

```text
config/                 Example configuration
docs/                   Research notes and milestone plans
src/ha_native_audio_agent/   Python package
tests/                  Unit and offline integration tests
```

## Current Status

This repository is being scaffolded for Milestone 0, a technical spike. See [`docs/research-findings.md`](docs/research-findings.md) and [`docs/milestone-0-plan.md`](docs/milestone-0-plan.md) for details.

## Development

```bash
python -m compileall src tests
python -m pytest
```

Hardware and provider diagnostics are available as explicit commands so the default test suite remains offline and safe.
