"""Unit tests for Multimodal Live Voice & Audio Streaming Engine (Wave 4 Milestone 4)."""

import base64
import json
import math
import struct

from cli_agent.voice import (
    AudioDeviceManager,
    AudioFormatConfig,
    GeminiLiveSession,
    VADConfig,
    VADEngine,
    VADState,
    VoiceAgentEngine,
)


def generate_synthetic_pcm(
    amplitude: int,
    frequency: float = 440.0,
    sample_rate: int = 16000,
    duration_ms: int = 20,
) -> bytes:
    """Generates synthetic 16-bit mono PCM sine wave samples."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    samples: list[int] = []
    for i in range(num_samples):
        t = i / sample_rate
        val = int(amplitude * math.sin(2 * math.pi * frequency * t))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{num_samples}h", *samples)


def test_audio_format_config() -> None:
    config = AudioFormatConfig(input_sample_rate=16000, chunk_duration_ms=20)
    assert config.input_sample_rate == 16000
    assert config.output_sample_rate == 24000
    # 16000 samples/sec * 0.02s = 320 samples * 2 bytes = 640 bytes
    assert config.bytes_per_input_frame == 640


def test_vad_engine_silence_vs_speech() -> None:
    vad = VADEngine(VADConfig(energy_threshold=300.0, min_speech_frames=2, min_silence_frames=3))

    silent_frame = b"\x00" * 640
    assert vad.calculate_rms(silent_frame) == 0.0
    assert vad.process_frame(silent_frame) == VADState.SILENCE

    # Generate loud speech frame (amplitude 5000 -> RMS ~3535)
    speech_frame = generate_synthetic_pcm(amplitude=5000)
    assert vad.calculate_rms(speech_frame) > 1000.0

    # First speech frame (consecutive_speech = 1 < 2)
    s1 = vad.process_frame(speech_frame)
    assert s1 == VADState.SILENCE

    # Second speech frame (consecutive_speech = 2 == min_speech_frames)
    s2 = vad.process_frame(speech_frame)
    assert s2 == VADState.SPEECH_START

    # Third speech frame (active speech)
    s3 = vad.process_frame(speech_frame)
    assert s3 == VADState.SPEECH_ACTIVE

    # Silence hangover
    assert vad.process_frame(silent_frame) == VADState.SPEECH_ACTIVE
    assert vad.process_frame(silent_frame) == VADState.SPEECH_ACTIVE
    # 3rd silent frame triggers SPEECH_END
    assert vad.process_frame(silent_frame) == VADState.SPEECH_END
    # Subsequent silent frame is SILENCE
    assert vad.process_frame(silent_frame) == VADState.SILENCE


def test_audio_device_manager_playback_and_purge() -> None:
    dev = AudioDeviceManager()
    assert dev.buffered_playback_bytes == 0

    chunk = b"\x01\x02" * 100
    dev.enqueue_playback_audio(chunk)
    assert dev.buffered_playback_bytes == 200

    dev.clear_playback_buffer()
    assert dev.buffered_playback_bytes == 0


def test_gemini_live_session_setup_and_audio_messages() -> None:
    session = GeminiLiveSession(
        project_id="test-project",
        location="us-central1",
        model_name="gemini-2.0-flash-exp",
    )
    assert "us-central1-aiplatform.googleapis.com" in session.websocket_url

    setup_msg = session.build_setup_message(system_instruction="You are assistant")
    assert "setup" in setup_msg
    assert (
        "projects/test-project/locations/us-central1/publishers/google/models/gemini-2.0-flash-exp"
        in setup_msg["setup"]["model"]
    )
    assert setup_msg["setup"]["systemInstruction"]["parts"][0]["text"] == "You are assistant"

    pcm_chunk = b"\x10\x20" * 160
    audio_msg = session.build_audio_frame_message(pcm_chunk)
    assert "realtimeInput" in audio_msg
    media = audio_msg["realtimeInput"]["mediaChunks"][0]
    assert media["mimeType"] == "audio/pcm;rate=16000"
    decoded = base64.b64decode(media["data"])
    assert decoded == pcm_chunk

    barge_in = session.build_barge_in_message()
    assert barge_in["clientContent"]["interrupted"] is True


def test_gemini_live_session_parse_server_message() -> None:
    session = GeminiLiveSession("proj", "us-central1")

    audio_bytes = b"sample_pcm_audio_bytes"
    b64_audio = base64.b64encode(audio_bytes).decode("ascii")

    server_json = json.dumps(
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {"text": "Hello developer! "},
                        {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": b64_audio}},
                    ]
                },
                "turnComplete": True,
            },
            "toolCall": {"functionCalls": [{"name": "git_status", "args": {"short": True}}]},
        }
    )

    event = session.parse_server_message(server_json)
    assert event.text_chunk == "Hello developer! "
    assert event.audio_pcm == audio_bytes
    assert event.turn_complete is True
    assert event.tool_call is not None
    assert event.tool_call["functionCalls"][0]["name"] == "git_status"


def test_voice_agent_engine_barge_in_interruption() -> None:
    session = GeminiLiveSession("proj", "us-central1")
    vad = VADEngine(VADConfig(energy_threshold=300.0, min_speech_frames=1))
    dev = AudioDeviceManager()

    # Pre-load playback audio buffer to simulate assistant speaking
    dev.enqueue_playback_audio(b"\xaa\xbb" * 500)
    assert dev.buffered_playback_bytes == 1000

    text_tokens: list[str] = []
    dispatched_tools: list[str] = []

    engine = VoiceAgentEngine(
        session=session,
        vad_engine=vad,
        device_manager=dev,
        on_text_token=lambda t: text_tokens.append(t),
        on_tool_dispatch=lambda fn, args: dispatched_tools.append(fn),
    )

    # User speaks: loud frame
    loud_frame = generate_synthetic_pcm(amplitude=6000)
    state, barge_in_msg = engine.feed_input_pcm_frame(loud_frame)

    assert state == VADState.SPEECH_START
    assert engine.is_user_speaking is True
    # Barge-in must have cleared audio buffer immediately!
    assert dev.buffered_playback_bytes == 0
    assert engine.interruption_count == 1
    assert barge_in_msg is not None
    assert barge_in_msg["clientContent"]["interrupted"] is True

    # Incoming server message handling
    incoming_raw = json.dumps(
        {
            "serverContent": {
                "modelTurn": {"parts": [{"text": "Found 3 files"}]},
            },
            "toolCall": {"functionCalls": [{"name": "read_file", "args": {"path": "src/main.py"}}]},
        }
    )
    engine.handle_incoming_server_message(incoming_raw)
    assert text_tokens == ["Found 3 files"]
    assert dispatched_tools == ["read_file"]
