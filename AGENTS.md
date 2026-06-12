# Agent behavior guidance for this repository

## Public / open-source quality

- Treat this as a public open-source project from the first commit.
- Write clear docs, safe defaults, tests, and avoid local-only assumptions in committed files.
- Use `.env.example` for credential templates; never commit secrets.
- Product plans, milestone details, and architecture decisions belong in `docs/`.

## Safety / secrets

- No secrets in source, tests, logs, examples, docs, or chat output.
- Use `.env` or Docker secrets for local credentials; commit only `.env.example`.
- The runtime must not expose Docker, Proxmox, SSH, filesystem, shell, or homelab maintenance tools to the model.
- Home Assistant actions must be allowlisted, auditable, timeout-bound, and non-destructive by default.
- Direct Home Assistant API fallback must be narrow and added only when MCP or Assist-facing integration is insufficient for a specific operation.

## Milestone discipline

- Follow the milestone plan in `docs/milestone-0-plan.md` as the current workflow guide.
- Do Milestone 0 first; it is a technical spike, not the production voice loop.
- Before adding a dependency, document which milestone requirement it proves.
- Prefer framework-native support for audio input, tools, MCP, structured outputs, and message history.
- Do not build a custom model-provider abstraction until a concrete framework limitation is proven.

## Raspberry Pi and audio testing

- Prefer local/unit tests before hardware tests.
- Hardware tests on the Raspberry Pi should be short, explicit, and diagnostic.
- Do not run long wake-word loops, stress tests, or high-frequency polling without an explicit test plan.
- Audio diagnostics must avoid retaining raw audio unless diagnostic capture is explicitly enabled.
- When raw audio is captured for diagnostics, document where it is stored and delete it when no longer needed.

## Pi delegation

Pi delegation is only for Codex coding agents. If you are the Pi coding agent, you are the implementer, so do not call `pi -p`.

Use Pi aggressively to reduce paid-context usage for:

- codebase search and exploration;
- library and web documentation scouting;
- log/build/test/docker output summarization;
- `rg`/`git diff` summarization;
- repetitive edits;
- small or medium bounded implementation tasks.

Pi is a junior executor. It can inspect code, summarize facts, and implement bounded code changes, but it has no right to opinions or decisions. Codex owns all decisions and final correctness.

Pi implementation work is allowed only when the instruction is exact:

- specify the files or folders it may edit;
- specify the behavior to implement;
- specify files or folders it must not touch;
- specify validation commands to run when practical;
- specify the required return format;
- require changed files and commands run.

After Pi implements code, Codex must always:

- inspect `git diff --stat`;
- inspect `git diff`;
- run validation;
- fix or revert bad changes manually.

Do not use Pi for:

- architecture decisions;
- auth or security decisions;
- deployment or secrets;
- complex business logic;
- destructive operations.

Pi prompts must always be explicit, concise, bounded, and clear about output format. Use a 600 second timeout by default for substantial Pi tasks. Pi sessions are reset at each run, so provide context each time instead of assuming context is saved.

## Validation expectations

- Keep CI offline by default; CI must not require RPi, Home Assistant, provider API keys, or network access.
- Add replayable tests for deterministic logic.
- Add hardware/provider diagnostics as explicit commands that can be run manually.
- Every milestone report should include changed files, architecture impact, known limitations, commands run, and test evidence.
