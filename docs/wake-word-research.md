# Wake Word Research

This document captures current facts and open questions for local wake-word engines. It is not an implementation decision record.

## Common Integration Requirements

- Most local wake-word engines consume mono 16 kHz, 16-bit PCM audio.
- Wake detection should be implemented behind a replaceable interface so Porcupine, openWakeWord, and other detectors can be swapped without changing the agent runtime.
- Wake-word detection must stay local. No continuous microphone upload should be required.
- Sensitivity, cooldown, duplicate-trigger suppression, and diagnostic counters should be runtime configuration, not hardcoded behavior.

## Picovoice Porcupine

### Confirmed Facts

- Current Porcupine Python package versions require a Picovoice `AccessKey` at initialization.
- Audio processing is local/on-device; the AccessKey is for authentication, authorization, and usage accounting.
- Porcupine publicly documents Raspberry Pi support, including Raspberry Pi 4.
- Custom wake words are generated as platform-specific `.ppn` files through Picovoice tooling.
- Public docs describe usage in terms of monthly active users, but do not publish the exact device identity mechanism used in Docker.
- Public docs do not clearly document whether mounting a host home directory or a Picovoice cache directory into Docker changes activation/device behavior.

### Open Questions

- Whether any old `pvporcupine` version can be used safely and legally without an AccessKey in a modern Python/aarch64 Docker stack remains unproven.
- Docker behavior around activation limits and identity must be tested cautiously and without attempting to bypass Picovoice licensing or usage controls.
- The project should not rely on undocumented cache or home-directory behavior for production operation.

## openWakeWord

### Confirmed Facts

- openWakeWord is open-source and does not require a cloud account or runtime API key.
- It can run offline after models are available locally.
- It supports Linux ARM64/Raspberry Pi class devices according to its published docs and ecosystem usage.
- It uses local model files, commonly TensorFlow Lite or ONNX depending on backend.
- It supports configurable thresholds and can be combined with VAD/noise suppression features.
- Custom wake-word training is documented through project tooling and notebooks.

### Open Questions

- False-positive behavior in the target room must be measured with real TV/Sonos background audio.
- Custom model quality for Romanian household usage must be evaluated with local fixtures.

## Other Local Options

- Mycroft Precise and Snowboy are historically important but appear dormant or archived and should be treated as compatibility references rather than primary candidates until proven otherwise.
- PocketSphinx and Vosk can support keyword spotting patterns, but they are broader speech recognition systems rather than modern wake-word engines.
- Sherpa-ONNX has active local/offline speech tooling, including keyword spotting support, and is worth tracking as a possible adapter target.
- Wyoming wake components are useful references, but this project should not depend on Home Assistant's voice pipeline as the core runtime.

## Initial Engineering Implication

Milestone 0 should keep wake-word code modular and avoid binding the architecture to a single vendor. The spike can start with a manual trigger or simple placeholder while audio capture, agent runtime, tools, and clarification flow are being proven.
