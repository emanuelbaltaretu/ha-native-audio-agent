"""Docker entrypoint for the shared Supertonic TTS server."""

from ha_native_audio_agent.tts.supertonic_server import main


if __name__ == "__main__":
    main()
