# Research Findings

> **Purpose:** Capture what has been confirmed through documentation and local observation, and what remains unknown or needs spike verification.  
> **Scope:** Pre-Milestone 0 research. Findings are updated as the spike progresses.

## Confirmed Facts

### PydanticAI Framework

- **Audio input** is natively supported through `AudioUrl` and `BinaryContent` types. This confirms the framework can accept raw audio or a URL pointing to audio data.
- **Message history APIs** (`all_messages`, `new_messages`) exist and support multi-turn conversation state within the agent loop.
- **Tool loops** with configurable **usage limits** are available, allowing a bounded number of tool calls per turn.
- **MCP client and toolset support** is documented. Newer documentation references `MCPToolset` for attaching MCP servers to an agent.
- **Structured output modes** (result types, JSON schema) are documented as a first-class feature.

**Gap / Unknown:** No documentation was found demonstrating audio input, tool calls, and structured output used together in a single agent run. Whether all three compose cleanly needs spike verification.

### Provider: OpenAI

- OpenAI audio-capable models (e.g., `gpt-4o-audio-preview` or similar) accept audio input directly and support **function (tool) calling**.
- OpenAI documentation indicates that **structured outputs** are not supported for audio / realtime models. The implications for a combined audio→tool→structured flow need testing.

### Provider: Google Gemini

- Gemini documentation shows support for **audio input**, **function calling**, and **structured output** as individual capabilities.
- Whether all three operate simultaneously in a single turn (audio → function call → structured response) has not been verified. A spike is required to confirm this works end-to-end.

### Provider: Anthropic

- Anthropic does not currently support audio input. The PydanticAI provider compatibility matrix confirms this. Anthropic is not viable for the direct-audio-input approach.

### Home Assistant MCP

- The documented Home Assistant MCP integration describes HA acting as an **MCP client** that connects to external MCP servers (e.g., to expose HA entities to another agent framework).
- No clearly documented built-in endpoint exposing Home Assistant as an **MCP server** was found. This means HA Native Audio Agent cannot currently rely on a standard HA-as-MCP-server contract.
- A test MCP server exists at `@modelcontextprotocol/server-everything` and can be used for tool-call validation during the spike.
- PydanticAI's newer MCP support is oriented around `MCPToolset`. This is the layer to test for attaching MCP tools to a voice agent.

### Raspberry Pi / Audio Hardware

- **Docker audio access** on the Pi likely requires mounting `/dev/snd` and membership in the `audio` group. An alternative is forwarding audio through PulseAudio or PipeWire sockets. This must be confirmed during the spike.
- **sounddevice** on Linux depends on the system `libportaudio2` package. This is relevant for any onboard VAD or capture logic written in Python.
- **Silero VAD** is plausible on Raspberry Pi 4 (aarch64). It can run via ONNX Runtime or PyTorch. Performance characteristics under Docker need measurement.
- **Porcupine** (Picovoice) supports Raspberry Pi but requires a Picovoice AccessKey (API key). Custom wake-word models are platform-specific and must be compiled for aarch64.

### Local Observations (rpi166)

- `rpi166` is online running **aarch64 Linux**.
- The **Jabra Speak 510** appears as ALSA card 0 for both capture and playback.
- Docker is installed and operational.
- An existing container called `linux-voice-assistant` is present. It serves as a reference / fallback example, not the foundation of this project.

## Key Unknowns Requiring Spike Investigation

1. **PydanticAI audio + tools + structured output:** Can a single agent run accept audio input, invoke an MCP tool, and return a structured result? No documentation was found demonstrating this combination.
2. **Gemini all-three-at-once:** Does Gemini support audio input, function calling, and structured output in a single turn?
3. **HA-as-MCP-server:** Is there a standard way to expose Home Assistant as an MCP server that PydanticAI can connect to? Or does the agent need to use HA Assist API or direct REST calls as a fallback?
4. **Docker audio reliability:** What is the minimum working configuration for audio capture and playback inside an aarch64 Docker container on Raspberry Pi OS?
5. **VAD performance under Docker:** Can Silero VAD sustain real-time inference inside the container without excessive CPU?
6. **Porcupine setup effort:** How much platform-specific build work is required for a custom wake word on aarch64 Linux?
