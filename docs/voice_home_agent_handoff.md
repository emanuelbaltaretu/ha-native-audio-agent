# Technical Handoff: Local Voice Home Agent

## 1. Executive summary

Build a Dockerized local voice appliance running on a Raspberry Pi 4, using a Jabra Speak 510 as the USB audio device.

The system must avoid the standard Home Assistant Assist voice pipeline. It must:

- detect a wake phrase locally;
- capture a complete spoken turn using VAD;
- send the captured audio directly to an audio-capable multimodal model through an existing agent framework;
- preserve multi-turn conversation state;
- allow the assistant to ask clarifying questions and continue naturally after the user answers;
- support reusable skills or capabilities;
- call Home Assistant through MCP or a narrowly scoped API fallback;
- speak the response through the Jabra device.

The project is not a generic voice framework and not a realtime WebRTC product. The goal is a reliable household appliance with minimal moving parts, explicit safety rules, and provider flexibility.

---

## 2. Problem statement

Home Assistant Voice / Assist is currently unreliable in this environment, especially regarding wake-word false positives, VAD behavior, and conversational quality.

The device will sit near a Sonos speaker and may operate while the TV is on. False activation, self-triggering, and background speech are first-class engineering problems.

The assistant must support normal multi-turn interaction:

```text
User: Stinge lumina.
Assistant: Care lumină?
User: Cea din dormitor.
Assistant: Am stins-o.
```

The next user reply must be appended to the same conversation. The system must not require a new wake phrase for every clarification.

---

## 3. Target environment

| Component | Requirement |
|---|---|
| Compute | Raspberry Pi 4, aarch64 Linux |
| Deployment | Docker Compose |
| Audio device | Jabra Speak 510 over USB |
| Environment | Near a Sonos speaker; TV may be on |
| Home automation | Existing Home Assistant instance |
| Primary model strategy | Any framework-supported provider/model with direct audio input |
| Fallback | STT + text model only when direct audio input is unavailable or fails |

---

## 4. Functional goals

1. Local always-on wake-word detection.
2. Local VAD and utterance capture with pre-roll and reliable end-of-turn detection.
3. Send the captured audio turn directly to a multimodal model that accepts audio input.
4. Preserve one active conversation in RAM for 60 minutes after the most recent activity.
5. Reuse the same conversation when the user speaks again within that 60-minute window.
6. Allow the assistant to ask a short clarification question and continue naturally after the user answers.
7. Support repeated tool calls in one turn.
8. Control Home Assistant through its official MCP server where practical.
9. Support reusable skills or capability bundles.
10. Speak responses through the Jabra device initially.
11. Expose metrics, logs, health checks, and a diagnostic mode.
12. Keep the architecture model-provider agnostic by using the selected framework's existing provider integrations.

---

## 5. Explicit non-goals for v1

- No long-term memory system.
- No vector database.
- No automatic fact or preference extraction.
- No persistent personal memory across restarts.
- No multi-agent system.
- No autonomous background actions.
- No browser automation, shell access, or arbitrary code execution.
- No mandatory realtime speech-to-speech API.
- No mandatory Gemini dependency.
- No LiveKit, Pipecat, OpenVoiceOS, Rhasspy, Willow, or Home Assistant Assist pipeline as the core runtime.
- No Sonos output in v1. Use the Jabra speaker first to reduce echo and self-trigger problems.
- No custom model-provider abstraction unless the selected framework genuinely lacks required support.

---

## 6. Proposed architecture

```text
Jabra Speak 510
    |
    v
Local audio frontend
    - ALSA / sounddevice
    - ring buffer / pre-roll
    - wake word
    - VAD / endpointing
    - conversation session controller
    |
    v
Agent runtime
    - existing multimodal provider support
    - in-RAM conversation history
    - tool loop
    - clarification handling
    - skills/capabilities
    |
    +--> Home Assistant MCP
    |        |
    |        +--> exposed entities, scripts, scenes
    |
    +--> restricted direct HA API fallback
    |     only if MCP is insufficient
    |
    v
TTS
    |
    v
Jabra speaker
```

---

## 7. Agent framework selection

Start with a technical spike using PydanticAI.

Do not assume PydanticAI is final until the spike proves these requirements:

- direct audio messages work with at least two framework-supported providers or models;
- multi-step tool calls work;
- MCP client integration works against Home Assistant;
- conversation history can be passed back into later runs;
- structured output or an equivalent mechanism can represent a clarification question cleanly;
- capabilities or tool bundles can provide a lightweight skills mechanism;
- the runtime is stable in an aarch64 Docker container.

If PydanticAI fails a core requirement, compare Agno, Google ADK, OpenAI Agents SDK, and LangGraph only against the failed requirement. Do not restart broad framework research.

---

## 8. Audio frontend requirements

### 8.1 Audio device

- Enumerate audio devices at startup.
- Fail with a clear diagnostic if the Jabra device is missing.
- Allow input and output device names or IDs through configuration.
- Normalize capture to mono PCM using the model/provider's recommended sample rate.
- Use a rolling buffer so the first phonemes after the wake phrase are not clipped.
- Provide a command that records and plays back a short diagnostic sample.

### 8.2 Wake word

- The wake word engine must be swappable via configuration (`wake_word.provider`).
- Two primary options:
  - **Porcupine (Picovoice)**: 16 built-in keywords (alexa, americano, bumblebee, computer, grapefruit, grasshopper, hey barista, hey google, hey siri, jarvis, ok google, picovoice, porcupine, terminator, etc.). Custom keywords require a Picovoice AccessKey (free tier: 3 users) and platform-specific `.ppn` files. All modern versions (v2.0.0+) require the AccessKey even for built-in keywords. v1.9.5 works without a key but is unmaintained since 2020.
  - **OpenWakeWord**: Fully open-source (Apache 2.0), no API key or account needed. Supports custom keywords via fine-tuning on synthetic speech. Runs on RPi4 aarch64. Slightly lower accuracy than Porcupine but no external dependencies. Recommended for custom wake words.
- Recommendation: OpenWakeWord for custom wake words (e.g. "hey hermes"); Porcupine only if built-in keywords suffice and an AccessKey is acceptable.
- Both must implement the same interface: `start_listening()`, `stop_listening()`, `on_wake(callback)`.
- Wake sensitivity must be configurable.
- Use cooldown and duplicate-trigger suppression.
- Keep wake detection local.
- No continuous microphone upload.
- The wake engine must be replaceable without changing the agent runtime.

### 8.3 VAD and utterance capture

- Use Silero VAD unless testing finds a clearly better alternative.
- Separate speech-start and speech-end thresholds.
- Include configurable pre-roll and post-roll.
- Set minimum speech duration.
- Set silence-to-finish duration.
- Set maximum utterance duration.
- Record diagnostic metadata:
  - wake confidence;
  - speech duration;
  - VAD summary;
  - model latency;
  - tool latency;
  - TTS latency.

### 8.4 TV and background speech

The TV may be on while the device listens.

The design must include:

- conservative wake sensitivity;
- local wake verification where useful;
- configurable speech-start threshold;
- cooldown after rejected or false wake events;
- test fixtures containing TV dialogue;
- counters for false wakes and abandoned captures.

VAD must not be treated as a wake-word detector.

### 8.5 Speech-to-text

The system must support multiple speech-to-text strategies, with the primary path being direct audio input to a multimodal model:

- **Direct audio to multimodal LLM** (primary path): Send captured audio directly to an audio-capable model (e.g. Gemini, GPT-4o-audio). Preserves tone, emotion, and avoids STT transcription errors. Requires a provider that supports both audio input and tool calling simultaneously.
- **Parakeet TDT 0.6B** (STT fallback): ONNX CPU-optimized, ~300ms latency on short utterances, 25 languages, ~600MB footprint. Ideal for RPi4 since it runs entirely on CPU. OpenAI-compatible API on port :5093.
- **Whisper** (STT fallback): Higher accuracy than Parakeet, but requires more resources. Can run on CPU (slower) or GPU. Available in multiple sizes (tiny, base, small, medium, large).
- The fallback chain must be configurable: `direct → parakeet → whisper` or any subset.
- Provider and model names must be runtime configuration, not hardcoded.

---

## 9. Conversation session management

### 9.1 Session lifetime

Maintain one active conversation per device in RAM.

The session remains active for:

```text
60 minutes after the latest activity
```

Activity includes:

- a new user audio turn;
- an assistant response;
- a tool call;
- a tool result;
- a clarification question.

A new conversation is created only when:

- the existing conversation has been inactive for at least 60 minutes;
- the user explicitly asks to start over;
- the session is explicitly reset;
- the process restarts.

A short follow-up listening window and conversation lifetime are different concepts.

The assistant may stop actively listening after a response, but the conversation history remains available in RAM for the full 60-minute inactivity window.

### 9.2 Conversation state

Suggested minimal state:

```python
@dataclass
class ConversationSession:
    id: str
    messages: list[ModelMessage]
    created_at: datetime
    last_activity_at: datetime
    awaiting_user_turn: bool
    turn_count: int
```

Do not create a hardcoded intent-specific pending-action engine for v1.

The previous assistant question and tool context should remain in the conversation history so the model can continue naturally from the next user turn.

### 9.3 Audio retention

Do not retain one hour of raw audio.

For each user turn:

1. capture the audio;
2. send it to the model;
3. retain it only while needed for the request, retries, and diagnostics;
4. delete it after processing unless diagnostic capture is explicitly enabled.

Prefer storing a transcript or concise textual representation in conversation history when the provider returns one.

If the provider does not return a transcript, investigate the cleanest framework-supported method. Do not add permanent STT only for history until the limitation is proven.

### 9.4 Context growth

The session may last up to an hour, but model context must remain bounded.

Investigate the best framework-supported strategy for:

- retaining recent turns;
- compacting old tool outputs;
- summarizing older messages only when required;
- preserving unresolved references and decisions.

Do not implement long-term memory as part of this work.

---

## 10. Clarification and asking the user a question

The assistant must be able to ask a short question, end the current model turn, play the question through TTS, and start listening for the next user turn.

Example:

```text
User: Stinge lumina.
Assistant: Care lumină?
User: Cea din dormitor.
Assistant: Am stins lumina din dormitor.
```

### 10.1 Preferred semantic model

A clarification question is not an external side effect like turning on a light. It is a terminal conversational outcome for the current turn.

A likely implementation is a structured terminal output such as:

```python
class Respond(BaseModel):
    type: Literal["respond"]
    text: str

class AskUser(BaseModel):
    type: Literal["ask_user"]
    question: str

AgentOutput = Respond | AskUser
```

Example:

```json
{
  "type": "ask_user",
  "question": "Care lumină?"
}
```

The orchestrator then:

1. ends the current agent run;
2. sends the question to TTS;
3. waits until playback finishes;
4. rearms VAD;
5. captures the next utterance;
6. appends it as a new user turn to the same conversation;
7. invokes the agent again with the same conversation history.

### 10.2 Do not hardcode the implementation prematurely

The coding agent must verify the best supported method in the chosen framework.

Evaluate at least:

- structured terminal output;
- framework-native deferred input or user-input request mechanism;
- a terminal `ask_user` tool if the framework handles terminal tools cleanly.

Use the method that:

- reliably ends the current agent run;
- cannot accidentally generate a second assistant response;
- preserves conversation history;
- is easy to test;
- works across the selected providers;
- does not require intent-specific pending-action state.

### 10.3 If a tool is used

If the best framework-compatible method is a tool, it must be terminal.

Conceptually:

```python
ask_user(question: str)
```

Its semantics must be:

- speak exactly one concise question;
- stop the current agent turn;
- do not return to the model for further generation in the same run;
- rearm VAD after TTS;
- append the next audio as a new user message;
- continue the same conversation.

Do not implement `ask_user` as an ordinary tool that returns a result and allows the model to continue generating.

### 10.4 No clarification-specific timeout

Do not hardcode a special timeout for answering a clarification.

VAD determines when a spoken reply starts and ends.

The conversation itself remains available in RAM for 60 minutes of inactivity.

The active listening state may be cancelled explicitly or managed by the general device listening policy, but it must not destroy the conversation history.

---

## 11. Tool loop

The agent runtime must:

- allow multiple tool calls in one user turn;
- cap the number of tool steps;
- return structured tool errors to the model;
- avoid retrying destructive actions automatically;
- log every requested and executed tool call;
- preserve relevant tool results in conversation history;
- trim oversized tool outputs before storing them.

Suggested initial maximum:

```yaml
max_tool_steps: 6
```

---

## 12. Home Assistant integration

Use the official Home Assistant MCP server first.

Requirements:

- expose only entities, scripts, and scenes intended for voice control;
- exclude dangerous domains by default;
- maintain an allowlist for service calls;
- require explicit confirmation for sensitive operations;
- log every Home Assistant action;
- set timeouts for MCP calls;
- recover from MCP disconnects.

If MCP lacks a required operation, add a narrow direct Home Assistant WebSocket or REST tool for that exact operation.

Do not build a second complete Home Assistant client layer.

### 12.1 Suggested initial tool surface

- Get state by entity, area, or domain.
- Turn an entity or area on or off.
- Set light brightness.
- Set climate temperature or mode.
- Run an exposed scene.
- Run an exposed script.
- Get a compact room or home overview.

---

## 13. Skills and capabilities

Implement a lightweight skill system only if the selected agent framework does not already offer a suitable equivalent.

Suggested layout:

```text
skills/
  diana-sleep/
    SKILL.md
    manifest.yaml
    tools.py
  climate-diagnostics/
    SKILL.md
    manifest.yaml
    tools.py
  media/
    SKILL.md
    manifest.yaml
    tools.py
```

Possible semantics:

- `SKILL.md` contains task-specific instructions and boundaries.
- `manifest.yaml` declares:
  - name;
  - description;
  - required tools;
  - permissions;
  - optional configuration.
- `tools.py` is optional and may expose only registered typed tools.

Requirements:

- skills must not execute arbitrary shell commands;
- loading errors must not crash the service;
- skill selection should be lazy or request-relevant;
- the core runtime must remain usable with zero custom skills.

Do not implement a skill marketplace, autonomous skill creation, or self-modifying skills in v1.

---

## 14. Model and provider strategy

Use the chosen framework's native provider integrations.

Requirements:

- test at least two audio-capable providers or models;
- primary path is direct audio input;
- fallback path is STT (Parakeet or Whisper) followed by a text model;
- STT engine selection must be configurable (`stt.mode: direct | parakeet | whisper`);
- provider and model names must be configuration;
- do not hardcode one vendor;
- do not create a custom provider abstraction unless a concrete incompatibility requires it;
- record latency, errors, and cost metadata where available.

The coding agent must document which framework providers genuinely support audio input and tool calling together.

Do not assume that generic multimodal support automatically means audio tool-calling support.

---

## 15. TTS and playback

- Use the Jabra speaker in v1.
- The TTS engine must be swappable via configuration (`tts.provider`).
- Supported TTS engines:
  - **Supertonic 3**: 99M parameters, ONNX CPU, 31 languages including Romanian (`ro`). Local, open-weight, no API key. ~1.7s per short reply on x86_64, slower on RPi4. Quality is good with natural number handling. Supports streaming and OpenAI-compatible API. Tested with Romanian — generates natural speech with numbers and English loanwords.
  - **Piper**: Lightweight (~10MB models), very fast on CPU, supports Romanian. More robotic voice but ideal for latency-critical responses.
  - **Edge TTS**: Free, uses Microsoft Edge's speech synthesis API. Excellent Romanian voice quality. Can run locally (does not require a cloud account). Use as high-quality fallback.
  - **Cloud TTS provider**: Optional, configurable (e.g. Google Cloud TTS, OpenAI TTS). Use only when local TTS quality is insufficient and connectivity is available.
- Recommendation: Supertonic 3 for primary local TTS, Piper for low-latency fallback, Edge TTS or cloud provider for highest quality when online.
- All TTS engines must implement the same interface: `speak(text) -> play audio`.
- Romanian voice quality matters.
- Wait until playback completes before rearming VAD for a clarification reply.
- Prevent the assistant from recursively triggering itself.
- Prefer using the Jabra for both capture and playback because its own speakerphone DSP may help more than playing responses through Sonos.
- Sonos output is explicitly deferred.

Barge-in is optional for v1.

Do not block the entire project on perfect barge-in or software AEC.

---

## 16. Safety and reliability

- No arbitrary shell tools.
- No Docker, Proxmox, filesystem, or infrastructure administration tools in v1.
- All Home Assistant actions must be logged.
- All external calls need timeouts.
- The service must return safely to idle mode after failures.
- Restarting the container must not corrupt configuration.
- Conversation history may be lost on restart in v1.
- Secrets must come from Docker secrets or an environment file excluded from source control.
- Add health checks for:
  - audio device;
  - model provider;
  - MCP connection;
  - TTS provider;
  - process state.

---

## 17. Suggested repository structure

```text
voice-home-agent/
  docker-compose.yml
  Dockerfile
  pyproject.toml

  config/
    config.example.yaml

  src/
    main.py

    audio/
      devices.py
      capture.py
      ring_buffer.py
      wakeword.py
      vad.py
      playback.py
      state_machine.py

    agent/
      runtime.py
      sessions.py
      outputs.py
      prompts.py
      policies.py

    providers/
      fallback_stt.py
      tts.py

    home_assistant/
      mcp.py
      restricted_api.py

    skills/
      loader.py
      registry.py

    api/
      health.py
      diagnostics.py

  skills/

  tests/
    unit/
    integration/
    audio-fixtures/
```

Do not add a persistent memory package or database schema in v1.

---

## 18. Configuration sketch

```yaml
audio:
  input_device: "Jabra SPEAK 510 USB"
  output_device: "Jabra SPEAK 510 USB"
  sample_rate: 16000
  channels: 1

wake_word:
  provider: openwakeword  # porcupine | openwakeword
  phrase: "hey hermes"
  sensitivity: 0.35
  cooldown_ms: 4000

vad:
  provider: silero
  start_threshold: 0.70
  end_threshold: 0.40
  min_speech_ms: 250
  end_silence_ms: 700
  pre_roll_ms: 350
  post_roll_ms: 200
  max_utterance_seconds: 15

stt:
  mode: direct             # direct | parakeet | whisper
  provider: configurable
  model: configurable
  fallback_mode: parakeet  # parakeet | whisper
  fallback_stt_enabled: true

conversation:
  ram_session_ttl_minutes: 60
  max_turns_before_compaction: 20
  max_tool_steps: 6

agent:
  framework: pydantic_ai
  provider: configurable
  model: configurable

home_assistant:
  mcp_url: "http://homeassistant:8123/api/mcp"
  direct_api_fallback: false

tts:
  provider: supertonic      # supertonic | piper | edge-tts | cloud
  voice: "F1"               # Supertonic: M1-M5, F1-F5
  quality_steps: 8           # Supertonic: 5 (fast) - 12 (high), default 8
  speed: 1.0
  output: local_device
  cloud:
    provider: ""             # e.g. google, openai
    api_key: ""              # from env, not hardcoded
```

---

## 19. Implementation milestones

### Milestone 0 — Technical spike

Do only this milestone first.

- Run on aarch64 Docker.
- Record and play audio through Jabra.
- Send one captured audio turn directly through PydanticAI.
- Test two supported audio-capable providers or models.
- Receive a text response.
- Connect to a harmless test MCP server.
- Connect to Home Assistant MCP.
- Execute one harmless Home Assistant action.
- Preserve conversation history over at least three turns.
- Test one clarification flow.
- Compare structured output, framework-native user-input requests, and terminal-tool approaches for asking a clarification question.
- Produce a written go/no-go report.

### Milestone 1 — Local voice loop

- Wake word.
- VAD capture.
- Direct audio model request.
- TTS through Jabra.
- Return safely to idle mode.

### Milestone 2 — Conversation

- In-RAM session reuse.
- 60-minute inactivity TTL.
- Multi-turn history.
- Clarification question.
- Same-conversation continuation after the answer.
- Explicit reset or start-over command.
- Context compaction test.

### Milestone 3 — Home Assistant

- MCP integration.
- Allowlisted entities and services.
- Multi-step tool calls.
- Confirmation policy.
- Audit logging.
- Error recovery.

### Milestone 4 — Skills

- Lightweight skill loader or framework-native equivalent.
- At least two sample skills.
- Failure isolation.
- Permission declaration.

### Milestone 5 — Hardening

- TV-noise tests.
- False-wake measurements.
- Provider failure tests.
- MCP reconnect.
- Audio-device reconnect.
- Container restart behavior.
- Metrics and health endpoints.

Long-term memory is explicitly deferred beyond v1.

---

## 20. Acceptance criteria

- The service runs in Docker on RPi4.
- The Jabra Speak 510 is used for input and output.
- No audio is uploaded before a local wake trigger.
- A normal command completes without Home Assistant Assist.
- The assistant can ask one clarification question and continue correctly after the user answers.
- The clarification implementation ends the current agent turn cleanly.
- The next answer is appended to the same conversation.
- Conversation history remains in RAM for 60 minutes after latest activity.
- The same conversation can be resumed after the device returns to wake mode.
- At least one skill can be added without modifying the core runtime.
- Home Assistant operations are restricted and auditable.
- The direct-audio path works with at least two providers or models.
- Fallback STT is demonstrated by intentionally disabling the direct-audio path.
- The service recovers after provider or MCP failures.
- The assistant does not recursively trigger itself during normal Jabra playback.
- No long-term memory system is included in v1.

---

## 21. Required test scenarios

- Wake phrase in a quiet room.
- Wake phrase with TV dialogue playing.
- Random TV speech without wake phrase.
- Short command.
- Long command.
- Mid-sentence pause.
- Ambiguous entity requiring clarification.
- Clarification answer appended to the same conversation.
- Follow-up after returning to wake mode but within the 60-minute session TTL.
- Follow-up after more than 60 minutes, creating a new conversation.
- Two consecutive follow-up turns without repeating context manually.
- Tool failure followed by a concise spoken error.
- Provider timeout and fallback STT.
- Jabra disconnect and reconnect.
- Assistant playback must not trigger a new request.
- Session context compaction after many turns.
- Explicit conversation reset.

---

## 22. Instructions for the coding agent

- Do not implement the complete product immediately.
- Begin with Milestone 0.
- Before adding a dependency, explain which requirement it solves.
- Prefer framework-native support for providers, multimodal messages, MCP, structured outputs, history, and tools.
- Do not build a custom provider layer preemptively.
- Do not add long-term memory.
- Do not add a database unless required for non-memory operational state.
- Do not add vector databases.
- Do not add multi-agent orchestration.
- Do not add realtime media servers.
- Do not modify Home Assistant beyond enabling MCP and exposing selected entities.
- Keep the system independent of Home Assistant Assist.
- Preserve debuggability through structured logs and explicit state transitions.
- Add replayable audio fixtures for tests.
- At the end of every milestone provide:
  - changed files;
  - architecture impact;
  - known limitations;
  - commands to run;
  - test evidence.

---

## 23. First task to execute

Create the repository skeleton and complete Milestone 0 only.

Do not implement production wake word, persistent memory, full skills, or production TTS yet.

### Required outputs

1. Verify Jabra capture and playback inside an aarch64-compatible Docker container.
2. Verify direct audio input through PydanticAI with two different supported providers or models.
3. Verify a multi-step tool call against a harmless test MCP server.
4. Verify one harmless call through Home Assistant MCP.
5. Verify message history over three turns.
6. Verify one clarification flow:
   - assistant asks a concise question;
   - current agent run ends;
   - VAD is rearmed after playback;
   - the next audio turn is appended to the same conversation;
   - the agent continues naturally.
7. Compare the following clarification mechanisms:
   - structured terminal output;
   - framework-native request for additional user input;
   - terminal `ask_user(question)` tool.
8. Recommend the simplest reliable mechanism supported across the selected providers.
9. Produce a findings report containing:
   - exact code paths;
   - provider limitations;
   - audio and tool-call support;
   - ARM compatibility issues;
   - measured latency;
   - MCP findings;
   - clarification mechanism findings;
   - go/no-go recommendation for PydanticAI.
