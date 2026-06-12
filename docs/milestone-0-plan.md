# Milestone 0 Plan

> Milestone 0 is a **technical spike**. Its purpose is to validate the framework, provider, audio, and Home Assistant control paths before committing to an architecture for v1.

## Objectives

1. **Audio container on RPi:** Prove the Jabra Speak 510 works for both capture and playback inside an aarch64 Docker container on Raspberry Pi 4.
2. **Direct audio input via PydanticAI:** Deliver at least one end-to-end flow where spoken audio is sent to a model through PydanticAI and a text response is received.
3. **Multi-step tool calls against an MCP server:** Demonstrate the agent can call tools on an MCP server (e.g., `@modelcontextprotocol/server-everything`) across multiple steps.
4. **Message history across three turns:** Verify the agent loop preserves conversation context across at least three user turns (ask → respond → follow-up).
5. **Clarification flow:** Prove the agent can stop after one clarifying question, wait for the next user turn, and continue in the same conversation.
6. **Home Assistant control surface scoping:** Determine whether HA offers a standard MCP server endpoint, or whether the fallback path (Assist API / REST) must be used.

## Non-Goals (explicitly excluded from M0)

- Production wake-word detection. A placeholder trigger (e.g., keyboard press or simple energy threshold) is acceptable for the spike.
- Full VAD integration. VAD may be simulated with a fixed-duration recording for early testing.
- Long-term memory, vector search, or persistent state.
- Multi-agent orchestration.
- Any Home Assistant tool that performs destructive or state-changing operations on production HA instances.
- SSH, shell, Docker, Proxmox, or filesystem tools exposed to the model.
- Security hardening, secret rotation, or credential management.
- Deployment automation, CI/CD pipelines, or releases.
- Performance benchmarking, stress testing, or high-frequency audio loops.

## Test Tracks

Each track lists the specific question to answer and how it will be validated.

### Track A: Docker Audio on RPi

| Item | Detail |
|---|---|
| **Question** | Can the Jabra Speak 510 capture and play audio inside a Docker container on aarch64 Linux? |
| **Validation** | Run a short diagnostic container that records a few seconds from the Jabra mic and plays it back. Verify audible playback through the Jabra speaker. |
| **Pass criteria** | Audio recorded inside the container is recognisable on playback. ALSA device numbering inside and outside the container is consistent. |
| **Stop condition** | The Jabra device is not enumerable inside Docker after trying `/dev/snd` mount + `audio` group and PulseAudio forwarding. |

### Track B: PydanticAI Audio Input

| Item | Detail |
|---|---|
| **Question** | Can PydanticAI accept audio input via `AudioUrl` / `BinaryContent` and return a useful text response? |
| **Validation** | Send a short WAV file (sine tone or speech) through a PydanticAI agent that echoes back the content description. |
| **Pass criteria** | The model returns a reasonable text response describing the audio. At least one provider (OpenAI or Gemini) works. |
| **Stop condition** | Neither OpenAI audio models nor Gemini can produce a coherent response from audio input through PydanticAI. |

### Track C: Multi-Step Tool Calls via MCP

| Item | Detail |
|---|---|
| **Question** | Can the agent call MCP tools in sequence across multiple steps? |
| **Validation** | Start `@modelcontextprotocol/server-everything` locally. Ask the agent to echo a message, then call `add` (assuming available tools). |
| **Pass criteria** | The agent makes at least two distinct tool calls in a single turn, each returning correct results. |
| **Stop condition** | PydanticAI cannot complete a second tool call after the first returns. |

### Track D: Message History

| Item | Detail |
|---|---|
| **Question** | Can the agent preserve conversation context across multiple turns? |
| **Validation** | Three turns: user asks "What's the temperature?" → agent responds → user asks "How about in Celsius?" (referencing previous answer). |
| **Pass criteria** | The second response correctly converts the earlier value. |
| **Stop condition** | `all_messages` / `new_messages` does not retain context across turns, or the provider truncates history in a way that breaks context. |

### Track E: Clarification Flow

| Item | Detail |
|---|---|
| **Question** | Can the agent ask a clarification question, end the run, and continue in the same conversation after the next user turn? |
| **Validation** | User asks a vague command. Agent asks one clarifying question. User provides clarification. Agent completes the original intent. |
| **Pass criteria** | The clarification and follow-up are in the same conversation context. |
| **Stop condition** | The agent loop / usage-limit model cannot support early termination and resumption. |

### Track F: Home Assistant Control Surface

| Item | Detail |
|---|---|
| **Question** | What is the standard way for an external agent to call HA entities and services? |
| **Validation** | Survey HA documentation and community resources for: (a) HA-as-MCP-server, (b) HA Assist API via WebSocket, (c) HA REST API. |
| **Pass criteria** | A clear recommendation emerges for one of the three approaches. If HA-as-MCP-server is viable, a connectivity validation can be attempted against a test HA instance. |
| **Stop condition** | No safe, documented, auditable HA control surface is available. |

## Acceptance Evidence

Milestone 0 is complete when the following deliverables exist:

| # | Deliverable | How to verify |
|---|---|---|
| 1 | Diagnostic Docker audio snapshot (script or compose override) | Running it on the Pi produces a playable audio file. |
| 2 | PydanticAI agent script with audio input | The script accepts a file path and prints a text response. |
| 3 | PydanticAI agent script with MCP tool calls | The script calls two tools in sequence and prints results. |
| 4 | Script demonstrating three-turn history | Run three exchanges; the third turn references data from the first. |
| 5 | Script demonstrating clarification flow | The agent asks one question, waits for input, then proceeds. |
| 6 | HA control surface recommendation document | A short doc in `docs/` or inline notes in `research-findings.md` summarising the recommended approach. |
| 7 | Written report covering all six tracks | The report states pass / fail / partial for each track and lists any blocking unknowns. |

## Stop Conditions

The spike should be halted and the project re-assessed if any of the following occurs:

- **No viable provider path:** Neither OpenAI audio models nor Gemini can process audio input through PydanticAI well enough to return a coherent response.
- **No tool-compatible provider:** No provider that supports audio input also supports tool / function calling.
- **No MCP connectivity:** PydanticAI cannot connect to an MCP server or complete tool calls.
- **No HA control surface:** No safe, auditable, documented way to control Home Assistant from an external agent exists.
- **Audio Docker blocker:** The Jabra device cannot be accessed from within a Docker container after reasonable effort with both ALSA and PulseAudio approaches.

## Dependencies (to be added to `pyproject.toml` only as needed for a track)

- `pydantic-ai` — core framework
- `sounddevice` and `soundfile` — local audio capture / playback (Track A, Track B)
- `pyaudio` or equivalent — alternative audio I/O if needed
- `onnxruntime` or `pytorch` — Silero VAD (later, not required for initial spike)
- `pvporcupine` — optional wake-word proof (not required for initial spike)

Each dependency addition must reference the test track it supports.
