# Wave 4: Autonomous Delivery & Production Workflows
## 04. Multi-Modal Live Voice Streaming & Full-Duplex Audio Agent (W4.4)

### 1. Problem Statement & Motivation

Through Waves 1 to 4.3, `cli-agent` evolved into a high-velocity, visually rich engineering runtime featuring Tree-Sitter AST symbol graphs (W1.1), Hierarchical Multi-Agent Swarms (W2.1), Self-Healing TDD Repair Loops (W2.2), Multi-Tier Quota Circuit Breakers (W4.1), Autonomous GitHub/GitLab PR Synthesis (W4.2), and a Fullscreen 4-Zone Split-Pane TUI (W4.3). However, even in Wave 3.1's preliminary push-to-talk voice integration, human-agent interaction remained bound by **half-duplex, turn-based batch transcription**:

1. **High Turn-Around Latency in Batch STT $\rightarrow$ LLM $\rightarrow$ TTS Pipelines**: Traditional voice wrappers record an entire audio file until silence, upload the full clip to a Speech-to-Text endpoint, wait for full text completion, send a standard HTTP REST/SSE prompt, and finally synthesize audio chunk-by-chunk. This introduces `2,500ms – 4,500ms` of dead air per turn, destroying the natural cadence of architectural pair-programming.
2. **Inability to Interrupt ("Barge-In" Blindness)**: When reviewing a complex multi-file refactor or debugging a failing `pytest` traceback hands-free, an engineer frequently needs to interrupt the assistant mid-sentence (e.g., *"Hold on—skip `auth.py` and check the circuit breaker state in `fallback_router.py` instead"*). Without sub-50ms local Voice Activity Detection (VAD) coupled to immediate playback buffer truncation and upstream WebSocket cancellation, the assistant continues speaking stale analysis over the user.
3. **Disconnected Voice & Visual Workspace State**: Spoken architectural dialogue must not operate in a vacuum separate from the active codebase. When the developer asks a question or issues a command over voice, the model must execute local tools (`read_file`, `git_status`, `tdd_heal`, `find_symbol_definition`) mid-stream while simultaneously updating the **W4.3 Split-Pane TUI** (`ChatPane` live transcript, `DiffViewerPane` AST/unified diffs, `ProcessLogPane` test runs, and `StatusHUD` VAD telemetry).

**W4.4 Multi-Modal Live Voice Streaming (`src/cli_agent/voice.py`)** upgrades `cli-agent` to a **Full-Duplex, Low-Latency Bi-Directional Audio Agent** powered by the **Vertex AI Gemini Live API (`BidiGenerateContent`)** over persistent WebSockets. It delivers `< 300ms` voice-to-voice response latency, hybrid Energy + WebRTC VAD with instantaneous `< 30ms` barge-in cancellation, seamless mid-turn tool execution bridged directly to the W4.3 Split-Pane TUI, and a 100% deterministic headless testing architecture requiring zero physical audio hardware.

---

### 2. Protocol Architecture & Full-Duplex WebSocket Streaming

Instead of stateless HTTP request/response cycles, W4.4 establishes a persistent, full-duplex WebSocket session against the **Vertex AI Gemini Live Bi-Directional Streaming API (`LlmBidiService/BidiGenerateContent`)**.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       Local Developer Workstation (cli-agent W4.4)                      │
│                                                                                         │
│  ┌──────────────────────────┐                      ┌─────────────────────────────────┐  │
│  │   AudioDeviceManager     │                      │        VADEngine (Local)        │  │
│  │  • Input: 16kHz PCM S16  │───── 20ms Frames ───►│  • RMS dBFS Energy Threshold    │  │
│  │  • Output: 24kHz PCM S16 │      (640 bytes)     │  • WebRTC Sub-Band Spectral VAD │  │
│  └────────────▲─────────────┘                      └────────┬───────────────┬────────┘  │
│               │                                             │               │           │
│               │ Play 24kHz PCM                    Speech    │               │ Barge-In! │
│               │ OR Truncate (<5ms)                Frames    │               ▼           │
│  ┌────────────┴─────────────┐                               │    ┌───────────────────┐  │
│  │  Downstream Ring Buffer  │◄──────────────────────────────┼────│ Clear Playback Q  │  │
│  │  (Cancelable Audio Sink) │                               │    │ & Emit Interrupt  │  │
│  └────────────▲─────────────┘                               │    └──────────┬────────┘  │
└───────────────┼─────────────────────────────────────────────┼───────────────┼───────────┘
                │                                             ▼               │
                │                             ┌───────────────────────────────▼────────┐
                │                             │       GeminiLiveWebSocketSession       │
                │                             │  • Setup: Model, Voice, Tool Schemas   │
                │                             │  • Upstream: `realtimeInput.mediaChunks`│
                │                             └───────────────┬───────────────▲────────┘
                │                                             │               │
                │    Downstream ServerContent:                │               │
                │    • 24kHz PCM `inlineData`                 │ WSS Full-Duplex Stream
                │    • Real-Time Text Transcript              │ (`BidiGenerateContent`)
                │    • `toolCall` (`functionCalls`)           ▼               │
┌───────────────┴─────────────────────────────────────────────┴───────────────┴───────────┐
│        Google Cloud Vertex AI — `LlmBidiService/BidiGenerateContent` WebSocket          │
│  wss://{region}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.           │
│        LlmBidiService/BidiGenerateContent                                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2.1 Vertex AI Gemini Live WebSocket Endpoint & Handshake
- **Endpoint URI Template**:
  ```text
  wss://{region}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent
  ```
  *(where `{region}` defaults to `us-central1` or fails over across regions via W4.1 `QuotaCircuitBreaker`)*.
- **Authentication**: Uses Wave 3's `GoogleAuthManager` (`google.auth.default()` / ADC) to inject a valid `Authorization: Bearer <access_token>` header during the WebSocket TLS upgrade handshake.
- **Initial `setup` Envelope**: Immediately upon connection open, `GeminiLiveWebSocketSession` transmits a deterministic configuration frame declaring the model resource path, output modalities (`["AUDIO"]` with synchronized text transcription), voice profile, system instructions, and local tool declarations:
  ```json
  {
    "setup": {
      "model": "projects/{project_id}/locations/{region}/publishers/google/models/gemini-2.5-flash-live-preview",
      "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
          "voiceConfig": {
            "prebuiltVoiceConfig": {
              "voiceName": "Aoede"
            }
          }
        }
      },
      "systemInstruction": {
        "parts": [
          {
            "text": "You are cli-agent in live full-duplex architectural voice mode..."
          }
        ]
      },
      "tools": [
        {
          "functionDeclarations": [
            { "name": "read_file", "description": "..." },
            { "name": "git_status", "description": "..." },
            { "name": "tdd_heal", "description": "..." }
          ]
        }
      ]
    }
  }
  ```

#### 2.2 Asymmetric Upstream & Downstream Audio Specifications

| Direction | Sample Rate | Bit Depth & Encoding | Channels | Frame Duration & Byte Size | JSON Wire Envelope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Upstream (Client $\rightarrow$ Vertex AI)** | **`16,000 Hz` (`16 kHz`)** | `16-bit` signed little-endian PCM (`PCM_S16LE`) | `1` (Mono) | `20 ms` (`320` samples = `640 bytes` raw PCM per frame) | `{"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm;rate=16000", "data": "<base64>"}]}}` |
| **Downstream (Vertex AI $\rightarrow$ Client)** | **`24,000 Hz` (`24 kHz`)** | `16-bit` signed little-endian PCM (`PCM_S16LE`) | `1` (Mono) | `20 ms` (`480` samples = `960 bytes` raw PCM per frame) | `{"serverContent": {"modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "<base64>"}}, {"text": "..."}]}}}` |

- **Synchronized Real-Time Text Transcription**: Alongside the `24 kHz` `inlineData` PCM audio chunks, `GeminiLiveWebSocketSession` extracts incremental transcript text parts (`serverContent.modelTurn.parts[].text` and `serverContent.outputTranscription.text`) and streams them directly into the W4.3 `ChatPane` via `TUIEventType.CHAT_TOKEN` so the engineer can read code identifiers, file paths, and symbol names while listening.

---

### 3. Voice Activity Detection (VAD) & Instant Barge-In Cancellation

Continuous full-duplex audio requires a highly responsive local **Voice Activity Detection (`VADEngine`)** pipeline to distinguish intentional human speech from mechanical keyboard clicks, cooling fan hum, and ambient room noise, and to trigger instantaneous **barge-in cancellation** whenever the user speaks while the assistant is talking.

```
                     20ms Upstream Frame (16kHz, 16-bit PCM Mono = 640 bytes)
                                                │
                                                ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │ Stage 1: Acoustic RMS & dBFS Energy Gate                  │
                  │   RMS = sqrt( (1/N) * sum(sample_i^2) )                   │
                  │   dBFS = 20 * log10( max(RMS, 1e-9) / 32768.0 )           │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │ Passes `energy_threshold_dbfs` (-42.0 dBFS)
                                                ▼
                  ┌───────────────────────────────────────────────────────────┐
                  │ Stage 2: WebRTC Spectral / Zero-Crossing Voice Classifier │
                  │   • Optional `webrtcvad` C-extension (aggressiveness 0–3) │
                  │   • Built-in Pure-Python Spectral Energy + ZCR Fallback   │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       │ Consecutive speech frames >= ` onset_frames` (2) │
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │ Assistant Currently Silent  │                   │ Assistant Currently Speaking│
        │  • Transition -> SPEECH     │                   │  • TRIGGER BARGE-IN (<5ms): │
        │  • Stream PCM to WebSocket  │                   │    1. Flush Playback Buffer │
        │  • Update HUD: [MIC SPEAK]  │                   │    2. Send Client Interrupt │
        └─────────────────────────────┘                   │    3. Mark Turn Interrupted │
                                                          └─────────────────────────────┘
```

#### 3.1 Dual-Stage Hybrid VAD Algorithm (`VADEngine`)
1. **Stage 1 — Short-Time Energy & Dynamic Noise Floor Calibration (`dBFS`)**:
   - For every `20ms` (`320`-sample) `PCM_S16LE` frame, `VADEngine` computes the Root Mean Square (RMS) amplitude and converts it to decibels relative to full scale (`dBFS`):
     $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1} s_i^2}, \quad \text{dBFS} = 20 \log_{10}\left(\frac{\max(\text{RMS}, 10^{-9})}{32768.0}\right)$$
   - Frames below `energy_threshold_dbfs` (default: `-42.0 dBFS`, dynamically above the rolling ambient noise floor) are immediately classified as `SILENCE` in `< 0.05ms`.
2. **Stage 2 — WebRTC VAD & Spectral Zero-Crossing Verification**:
   - When `webrtcvad` is installed on the host, the `20ms` `16kHz` frame is evaluated via `webrtcvad.Vad(aggressiveness=2).is_speech(pcm_bytes, 16000)`.
   - When running in minimal environments without compiled C audio extensions, `VADEngine` executes a deterministic pure-Python speech discriminator that combines speech-band energy ratio (`300 Hz – 3,400 Hz`) with Zero-Crossing Rate (`ZCR` bounded to `[0.02, 0.35]`) to reject broadband transient clicks (such as mechanical keyboard strokes).
3. **Hysteresis & Hangover State Machine (`SILENCE` $\rightarrow$ `SPEECH_ONSET` $\rightarrow$ `ACTIVE_SPEECH` $\rightarrow$ `HANGOVER`)**:
   - **Onset Confirmation**: Requires `speech_onset_frames` (default: `2` consecutive `20ms` frames = `40ms`) above threshold to trigger `VADState.SPEECH`, preventing single-frame pops from falsely interrupting TTS playback.
   - **Hangover Tail**: Holds `VADState.SPEECH` active for `silence_hangover_frames` (default: `15` frames = `300ms`) after energy drops below threshold so natural mid-sentence pauses do not prematurely close the user's turn.

#### 3.2 Instant Barge-In Cancellation Protocol
When `VADEngine` transitions into confirmed `VADState.SPEECH` while `AudioDeviceManager.is_playing` is `True` (or while a model audio turn is in flight):
1. **Immediate Local Audio Buffer Truncation (`< 5ms`)**:
   - `VoiceAgentEngine` invokes `audio_manager.clear_playback_buffer()`, atomically draining all queued `24 kHz` downstream PCM frames so speaker output stops instantaneously.
2. **Upstream WebSocket Client Interruption Signal**:
   - `GeminiLiveWebSocketSession.send_client_interrupt()` immediately transmits a `clientContent` turn-interruption frame (and begins streaming the new `16 kHz` user speech frames), instructing Vertex AI Gemini Live to halt ongoing token/audio generation for the superseded turn:
     ```json
     {
       "clientContent": {
         "turns": [],
         "turnComplete": false
       }
     }
     ```
3. **Server Acknowledgement & State Synchronization**:
   - Any residual in-flight downstream audio frames tagged with the superseded turn ID are discarded by `GeminiLiveWebSocketSession` until the server emits `{"serverContent": {"interrupted": true}}`.
   - The W4.3 `ChatPane` appends `" [Interrupted by user]"` to the truncated assistant transcript and switches the `StatusHUD` indicator to `VAD Mic: [SPEAKING ●]`.

---

### 4. Voice-to-Tool Dispatch Bridge & W4.3 TUI Synchronization

A core architectural differentiator of W4.4 is that live voice sessions have full access to `cli-agent`'s local engineering toolset and update the visual **W4.3 Split-Pane TUI** in real time while the conversation continues.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       Voice-to-Tool Dispatch & Live TUI Bridge                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
   User (Voice): "Check git status, read circuit_breaker.py, and heal the failing test."
         │
         ▼ (16kHz PCM over WSS)
┌────────────────────────────────────────────────────────┐
│ Vertex AI Gemini Live (`BidiGenerateContent`)          │
│   Emits: {"toolCall": {"functionCalls": [              │
│     {"id": "call_1", "name": "git_status", "args": {}},│
│     {"id": "call_2", "name": "read_file",  "args": ..},│
│     {"id": "call_3", "name": "tdd_heal",   "args": ..} │
│   ]}}                                                  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ VoiceAgentEngine._handle_tool_call()                   │
│   1. Executes registered async/sync tool handler       │
│   2. Emits live visual `TUIEvent`s to W4.3 `TUIApp`:   │
│      • `read_file` / `edit_file` ──► `DIFF_UPDATE`     │
│        (Updates Top-Right `DiffViewerPane` + AST Delta)│
│      • `git_status` / `tdd_heal` ──► `PROCESS_LOG`     │
│        (Streams pytest lines to Bottom-Right Pane)     │
│      • Tool summary badge        ──► `CHAT_MESSAGE`    │
│        (Appends tool node in Left `ChatPane`)          │
│   3. Sends `toolResponse` JSON back over WebSocket     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼ (Resumes 24kHz Audio Synthesis)
   Assistant (Voice + Transcript): "I fixed the off-by-one cooldown check in
   circuit_breaker.py; all 14 unit tests are now green in the diff viewer."
```

#### 4.1 Mid-Turn Function Call Lifecycle Over WebSockets
1. **Detection of `toolCall` Frames**:
   - During an active voice turn, Vertex AI Gemini Live emits a `toolCall` envelope containing one or more structured `functionCalls` (`id`, `name`, `args`), such as `read_file`, `edit_file`, `git_status`, `find_symbol_definition`, or `tdd_heal`.
2. **Non-Blocking Execution & W4.3 TUI Visual Broadcast**:
   - `VoiceAgentEngine` dispatches each function call through the injected `tool_executor` while keeping the WebSocket heartbeat and upstream VAD loop alive:
     - **`read_file` / `edit_file` / `write_file`**: Computes the unified diff and Wave 1 Tree-Sitter `SymbolDelta`, immediately posting `TUIEvent(TUIEventType.DIFF_UPDATE, ...)` so the **Top-Right `DiffViewerPane`** displays the exact code changes while the assistant explains them aloud.
     - **`git_status` / `run_command` / `tdd_heal`**: Streams `pytest` and Git output lines via `TUIEvent(TUIEventType.PROCESS_LOG, ...)` into the **Bottom-Right `ProcessLogPane`**.
     - **Voice Activity & Turn State**: Posts `TUIEvent(TUIEventType.HUD_UPDATE, ...)` to update the **Bottom `StatusHUD`** (`[MIC LISTENING]`, `[MIC SPEAKING]`, `[TTS PLAYING]`, `[TOOL EXEC]`).
3. **Returning `toolResponse` to the Live Session**:
   - Once the local tool completes, `GeminiLiveWebSocketSession.send_tool_response()` transmits the structured result back over the open WebSocket:
     ```json
     {
       "toolResponse": {
         "functionResponses": [
           {
             "id": "call_3",
             "name": "tdd_heal",
             "response": {
               "status": "SUCCESS",
               "passed": 14,
               "failed": 0,
               "modified_files": ["src/cli_agent/resilience/circuit_breaker.py"]
             }
           }
         ]
       }
     }
     ```
   - The Gemini Live model immediately incorporates the tool output and continues speaking its `24 kHz` audio explanation without losing conversational context.

---

### 5. Technical Interface: `src/cli_agent/voice.py`

All audio format configurations, hybrid VAD engines, hardware/virtual audio device managers, Gemini Live WebSocket session controllers, and the voice-to-tool orchestrator reside in `src/cli_agent/voice.py`.

```python
from __future__ import annotations

import asyncio
import base64
import math
import struct
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)


class VADState(str, Enum):
    """Deterministic Voice Activity Detection states for upstream audio frames."""

    SILENCE = "SILENCE"
    SPEECH_ONSET = "SPEECH_ONSET"
    SPEECH = "SPEECH"
    HANGOVER = "HANGOVER"


class VoiceSessionState(str, Enum):
    """High-level lifecycle states for the full-duplex VoiceAgentEngine."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    MODEL_SPEAKING = "MODEL_SPEAKING"
    TOOL_EXECUTING = "TOOL_EXECUTING"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class AudioFormatConfig:
    """
    Immutable acoustic specification for upstream (16kHz) and downstream (24kHz)
    16-bit signed little-endian mono PCM audio streams.
    """

    upstream_sample_rate_hz: int = 16000
    downstream_sample_rate_hz: int = 24000
    channels: int = 1
    sample_width_bytes: int = 2  # 16-bit signed PCM (PCM_S16LE)
    frame_duration_ms: int = 20

    @property
    def upstream_samples_per_frame(self) -> int:
        """Number of 16-bit samples per upstream 20ms frame (320 samples at 16kHz)."""
        return (self.upstream_sample_rate_hz * self.frame_duration_ms) // 1000

    @property
    def upstream_bytes_per_frame(self) -> int:
        """Byte length of a single upstream 20ms frame (640 bytes at 16kHz mono S16LE)."""
        return self.upstream_samples_per_frame * self.channels * self.sample_width_bytes

    @property
    def downstream_samples_per_frame(self) -> int:
        """Number of 16-bit samples per downstream 20ms frame (480 samples at 24kHz)."""
        return (self.downstream_sample_rate_hz * self.frame_duration_ms) // 1000

    @property
    def downstream_bytes_per_frame(self) -> int:
        """Byte length of a single downstream 20ms frame (960 bytes at 24kHz mono S16LE)."""
        return self.downstream_samples_per_frame * self.channels * self.sample_width_bytes

    @property
    def upstream_mime_type(self) -> str:
        return f"audio/pcm;rate={self.upstream_sample_rate_hz}"

    @property
    def downstream_mime_type(self) -> str:
        return f"audio/pcm;rate={self.downstream_sample_rate_hz}"


@dataclass(frozen=True)
class VADFrameMetrics:
    """Diagnostic telemetry computed for a single 20ms upstream PCM audio frame."""

    rms_amplitude: float
    dbfs: float
    zero_crossing_rate: float
    is_raw_speech: bool
    vad_state: VADState
    triggered_barge_in: bool = False


@dataclass(frozen=True)
class VoiceToolCall:
    """Structured function call emitted by Gemini Live during a voice turn."""

    call_id: str
    name: str
    args: Dict[str, Any]


@dataclass(frozen=True)
class DownstreamServerEvent:
    """Normalized event decoded from a Vertex AI Gemini Live `BidiGenerateContent` message."""

    pcm_audio_bytes: bytes = b""
    transcript_text: str = ""
    tool_calls: Tuple[VoiceToolCall, ...] = ()
    turn_complete: bool = False
    interrupted: bool = False
    setup_complete: bool = False


@runtime_checkable
class WebSocketTransport(Protocol):
    """
    Protocol abstracting an async WebSocket connection (`websockets` client or
    deterministic in-memory mock server for headless unit tests).
    """

    async def send(self, payload: str) -> None:
        """Sends a UTF-8 JSON text frame over the WebSocket."""
        ...

    async def recv(self) -> str:
        """Awaits and returns the next UTF-8 JSON text frame from the WebSocket."""
        ...

    async def close(self) -> None:
        """Gracefully closes the WebSocket transport."""
        ...


class VADEngine:
    """
    Hybrid Energy-Based (RMS dBFS) and WebRTC / Spectral Voice Activity Detector
    with onset confirmation hysteresis and instant barge-in detection.
    """

    def __init__(
        self,
        audio_config: Optional[AudioFormatConfig] = None,
        energy_threshold_dbfs: float = -42.0,
        speech_onset_frames: int = 2,
        silence_hangover_frames: int = 15,
        webrtc_aggressiveness: int = 2,
        use_webrtc_if_available: bool = True,
    ) -> None:
        self.audio_config = audio_config or AudioFormatConfig()
        self.energy_threshold_dbfs = energy_threshold_dbfs
        self.speech_onset_frames = speech_onset_frames
        self.silence_hangover_frames = silence_hangover_frames
        self.webrtc_aggressiveness = webrtc_aggressiveness
        self.use_webrtc_if_available = use_webrtc_if_available
        self.state: VADState = VADState.SILENCE
        self._consecutive_speech_frames: int = 0
        self._hangover_remaining: int = 0

    @staticmethod
    def compute_rms_and_dbfs(pcm_s16le_bytes: bytes) -> Tuple[float, float]:
        """Computes exact Root Mean Square (RMS) and dBFS level for a 16-bit PCM mono buffer."""
        ...

    @staticmethod
    def compute_zero_crossing_rate(pcm_s16le_bytes: bytes) -> float:
        """Computes normalized Zero-Crossing Rate (ZCR in [0.0, 1.0]) across 16-bit PCM samples."""
        ...

    def process_frame(
        self,
        pcm_s16le_bytes: bytes,
        assistant_is_speaking: bool = False,
    ) -> VADFrameMetrics:
        """
        Evaluates a 20ms 16kHz PCM frame, updates the hysteresis state machine
        (`SILENCE` -> `SPEECH_ONSET` -> `SPEECH` -> `HANGOVER`), and sets
        `triggered_barge_in=True` when confirmed user speech overlaps assistant playback.
        """
        ...

    def reset(self) -> None:
        """Resets VAD state machine and frame counters back to `VADState.SILENCE`."""
        ...


class AudioDeviceManager:
    """
    Manages asynchronous 16kHz microphone capture and cancelable 24kHz speaker playback.
    Supports hardware audio drivers (`sounddevice` / `pyaudio`) as well as deterministic
    in-memory virtual queues for headless CI testing with zero audio hardware.
    """

    def __init__(
        self,
        audio_config: Optional[AudioFormatConfig] = None,
        headless: bool = False,
        max_playback_queue_frames: int = 2000,
    ) -> None:
        self.audio_config = audio_config or AudioFormatConfig()
        self.headless = headless
        self._input_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._playback_buffer: Deque[bytes] = deque(maxlen=max_playback_queue_frames)
        self.is_capturing: bool = False
        self.is_playing: bool = False
        self.total_bytes_captured: int = 0
        self.total_bytes_played: int = 0
        self.barge_in_truncations: int = 0

    async def start(self) -> None:
        """Initializes audio input/output streams (or virtual buffers in headless mode)."""
        ...

    async def stop(self) -> None:
        """Stops capture/playback streams and flushes pending buffers."""
        ...

    def push_synthetic_input_frame(self, pcm_16k_bytes: bytes) -> None:
        """Enqueues a raw 16kHz 16-bit PCM frame into the capture stream (used by headless tests)."""
        ...

    async def read_input_frame(self) -> Optional[bytes]:
        """Reads the next 20ms (640-byte) 16kHz PCM frame from the microphone or virtual input queue."""
        ...

    def enqueue_playback_pcm(self, pcm_24k_bytes: bytes) -> int:
        """
        Slices incoming 24kHz 16-bit PCM audio into 20ms (960-byte) frames and appends
        them to the cancelable playback ring buffer. Returns number of frames queued.
        """
        ...

    def pop_playback_frame(self) -> Optional[bytes]:
        """Pops and plays the next 20ms 24kHz PCM frame from `_playback_buffer`."""
        ...

    def clear_playback_buffer(self) -> int:
        """
        Atomically flushes all queued 24kHz TTS playback frames upon barge-in detection.
        Returns the number of discarded audio frames and sets `is_playing = False`.
        """
        ...


class GeminiLiveWebSocketSession:
    """
    Full-Duplex WebSocket protocol client for Vertex AI Gemini Live (`BidiGenerateContent`).
    Handles session setup, base64 16kHz PCM upstream framing, 24kHz PCM + transcript
    downstream decoding, barge-in interruption signaling, and tool call/response framing.
    """

    ENDPOINT_TEMPLATE = (
        "wss://{region}-aiplatform.googleapis.com/ws/"
        "google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
    )

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        model_id: str = "gemini-2.5-flash-live-preview",
        voice_name: str = "Aoede",
        system_instruction: str = "",
        tool_declarations: Sequence[Dict[str, Any]] = (),
        audio_config: Optional[AudioFormatConfig] = None,
        transport_factory: Optional[
            Callable[[str, Dict[str, str]], Awaitable[WebSocketTransport]]
        ] = None,
        token_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.model_id = model_id
        self.voice_name = voice_name
        self.system_instruction = system_instruction
        self.tool_declarations = list(tool_declarations)
        self.audio_config = audio_config or AudioFormatConfig()
        self._transport_factory = transport_factory
        self._token_provider = token_provider
        self._transport: Optional[WebSocketTransport] = None
        self.is_connected: bool = False

    @property
    def websocket_url(self) -> str:
        """Returns the formatted regional `BidiGenerateContent` WSS endpoint URI."""
        return self.ENDPOINT_TEMPLATE.format(region=self.region)

    def build_setup_payload(self) -> Dict[str, Any]:
        """Constructs the initial `setup` JSON message sent upon WebSocket handshake completion."""
        ...

    async def connect(self) -> None:
        """Establishes the WSS connection, authenticates via Bearer token, and sends `setup`."""
        ...

    async def send_audio_chunk(self, pcm_16k_bytes: bytes) -> None:
        """Encodes 16kHz 16-bit mono PCM bytes as base64 and sends a `realtimeInput.mediaChunks` frame."""
        ...

    async def send_client_interrupt(self) -> None:
        """Sends an immediate client turn interruption signal to halt ongoing server TTS generation."""
        ...

    async def send_tool_response(
        self,
        call_id: str,
        name: str,
        response_payload: Dict[str, Any],
    ) -> None:
        """Sends a `toolResponse.functionResponses` frame back to the Gemini Live session."""
        ...

    async def receive_event(self) -> DownstreamServerEvent:
        """Receives and parses the next JSON frame from `BidiGenerateContent` into a `DownstreamServerEvent`."""
        ...

    async def close(self) -> None:
        """Closes the underlying WebSocket transport."""
        ...


class VoiceAgentEngine:
    """
    Top-level full-duplex voice orchestrator bridging `AudioDeviceManager`, `VADEngine`,
    `GeminiLiveWebSocketSession`, local tool execution (`read_file`, `git_status`, `tdd_heal`),
    and live visual updates to the W4.3 Split-Pane `TUIApp`.
    """

    def __init__(
        self,
        session: GeminiLiveWebSocketSession,
        vad_engine: Optional[VADEngine] = None,
        audio_manager: Optional[AudioDeviceManager] = None,
        tool_executor: Optional[
            Callable[[str, Dict[str, Any]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]
        ] = None,
        tui_event_callback: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.session = session
        self.vad_engine = vad_engine or VADEngine()
        self.audio_manager = audio_manager or AudioDeviceManager(headless=True)
        self.tool_executor = tool_executor
        self.tui_event_callback = tui_event_callback
        self.state: VoiceSessionState = VoiceSessionState.DISCONNECTED
        self.transcript_history: List[Tuple[str, str]] = []
        self.barge_in_count: int = 0
        self.executed_tool_calls: List[VoiceToolCall] = []

    async def process_upstream_frame(self, pcm_16k_bytes: bytes) -> VADFrameMetrics:
        """
        Evaluates a single 20ms 16kHz PCM frame through `VADEngine`.
        If barge-in is triggered while the assistant is speaking, immediately truncates
        the local 24kHz playback buffer and sends a client interrupt over WebSocket.
        If speech is active, streams the PCM frame upstream to Gemini Live.
        """
        ...

    async def handle_downstream_event(self, event: DownstreamServerEvent) -> None:
        """
        Processes a decoded `DownstreamServerEvent`:
        - Queues 24kHz PCM audio into `audio_manager` (unless interrupted).
        - Streams real-time text transcript chunks to the W4.3 `ChatPane`.
        - Dispatches any `VoiceToolCall` entries (`read_file`, `git_status`, `tdd_heal`),
          emits live `DIFF_UPDATE` / `PROCESS_LOG` events to the W4.3 TUI, and returns
          `toolResponse` payloads over the WebSocket.
        """
        ...

    async def run_session(self, max_turns: Optional[int] = None) -> None:
        """Runs concurrent upstream capture/VAD and downstream playback/tool loops until stopped."""
        ...

    async def stop(self) -> None:
        """Gracefully stops audio capture/playback and closes the Gemini Live WebSocket session."""
        ...
```

---

### 6. Exhaustive Unit & Headless Testing Strategy (`tests/unit/test_voice.py`)

All tests for W4.4 reside in `tests/unit/test_voice.py`. Because automated CI pipelines and developer containers lack physical microphones, speakers, and ALSA/PulseAudio daemons, the entire test suite runs **100% headlessly with zero audio hardware required** using synthetic PCM wave generators, deterministic VAD energy spikes, and an in-memory `MockGeminiLiveWebSocketServer`.

#### 6.1 Headless Fixture Architecture

1. **`SyntheticPCMGenerator`**:
   - **`generate_silence(sample_rate_hz=16000, duration_ms=20)`**: Produces exact zero-amplitude (or low `-65 dBFS` Gaussian noise floor) `PCM_S16LE` byte buffers (`640 bytes` at `16 kHz`, `960 bytes` at `24 kHz`).
   - **`generate_sine_speech(freq_hz=440.0, amplitude=12000, sample_rate_hz=16000, duration_ms=20)`**: Synthesizes high-energy (`~ -8.7 dBFS`) sine-wave PCM frames within the human vocal passband (`300 Hz – 3,400 Hz`) using `struct.pack("<320h", ...)` to deterministically trigger `VADEngine` speech onset and barge-in transitions.
2. **`MockGeminiLiveWebSocketServer` (Implements `WebSocketTransport`)**:
   - Captures all client JSON frames (`setup`, `realtimeInput`, `clientContent` interrupts, `toolResponse`) into an inspectable list and plays back scripted server frames (`setupComplete`, `24 kHz` `inlineData` audio chunks, real-time `text` transcripts, `toolCall` requests, and `interrupted: true` acknowledgements) over `asyncio.Queue`.

#### 6.2 Core Test Scenarios in `tests/unit/test_voice.py`

1. **Asymmetric Audio Format & PCM Byte Framing (`test_audio_format_config_16khz_and_24khz_geometry`)**
   - Verify `AudioFormatConfig()` computes exact frame metrics:
     - Upstream (`16 kHz`, `16-bit`, mono, `20 ms`): `320` samples, `640` bytes per frame, MIME `"audio/pcm;rate=16000"`.
     - Downstream (`24 kHz`, `16-bit`, mono, `20 ms`): `480` samples, `960` bytes per frame, MIME `"audio/pcm;rate=24000"`.
   - Verify `GeminiLiveWebSocketSession.build_setup_payload()` and `send_audio_chunk()` base64-encode `640`-byte `16 kHz` frames into valid `realtimeInput.mediaChunks` envelopes with zero sample truncation.

2. **VAD Energy Thresholding, Hysteresis & Hangover Transitions (`test_vad_engine_energy_spikes_and_hangover`)**
   - Feed `VADEngine(energy_threshold_dbfs=-42.0, speech_onset_frames=2, silence_hangover_frames=3)` a sequence of synthetic `-70 dBFS` silence frames and assert `vad_state == VADState.SILENCE`.
   - Feed 1 synthetic `-8.7 dBFS` vocal sine frame and assert transition to `VADState.SPEECH_ONSET` (not yet triggering full speech on a single transient spike).
   - Feed a 2nd consecutive vocal sine frame and assert transition to confirmed `VADState.SPEECH`.
   - Follow with 2 silence frames and assert `VADState.HANGOVER` retains active speech capture before returning to `VADState.SILENCE` on the 4th silence frame.

3. **Instant Barge-In Cancellation & Playback Buffer Truncation (`test_barge_in_truncates_24khz_playback_and_sends_interrupt`)**
   - Pre-load `AudioDeviceManager(headless=True)` with `20` frames (`19,200 bytes` = `400 ms`) of downstream `24 kHz` TTS audio so `audio_manager.is_playing == True` and `voice_engine.state == VoiceSessionState.MODEL_SPEAKING`.
   - Push 2 consecutive `16 kHz` high-energy user speech frames through `voice_engine.process_upstream_frame()`.
   - Assert:
     1. `VADFrameMetrics.triggered_barge_in` is `True`.
     2. `audio_manager._playback_buffer` is immediately emptied (`len == 0`) and `audio_manager.barge_in_truncations == 1`.
     3. `MockGeminiLiveWebSocketServer` receives the upstream `clientContent` interruption signal followed by the user's new `16 kHz` `realtimeInput` audio frames.
     4. Any residual downstream audio chunks arriving prior to `{"serverContent": {"interrupted": true}}` are suppressed from speaker playback.

4. **Full-Duplex WebSocket Handshake & Synchronized Audio/Transcript Streaming (`test_gemini_live_websocket_session_full_duplex_turn`)**
   - Connect `GeminiLiveWebSocketSession` against `MockGeminiLiveWebSocketServer` using a generic project placeholder (`project_id="test-gcp-project"`, `region="us-central1"`).
   - Assert the WebSocket URL matches `wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent` and the initial frame contains the `setup` configuration.
   - Stream synthetic `24 kHz` PCM audio + text transcript (`"Inspecting QuotaCircuitBreaker state..."`) from the mock server and verify `AudioDeviceManager` queues the exact `24 kHz` PCM bytes while `VoiceAgentEngine` appends the transcript and notifies the W4.3 `ChatPane`.

5. **Voice-to-Tool Dispatch Bridge & Live W4.3 TUI Diff Updates (`test_voice_to_tool_dispatch_updates_tui_live`)**
   - Simulate a mid-turn `toolCall` frame from `MockGeminiLiveWebSocketServer` invoking `read_file`, `git_status`, and `tdd_heal`.
   - Provide a mock `tool_executor` and capture all events emitted to `tui_event_callback`.
   - Assert:
     1. `tool_executor` is invoked with exact tool names and arguments.
     2. Visual `TUIEvent` notifications (`DIFF_UPDATE` for file diffs/AST deltas, `PROCESS_LOG` for `git_status` / `tdd_heal` output, and `HUD_UPDATE` for status badges) are emitted to the W4.3 TUI callback in real time.
     3. `GeminiLiveWebSocketSession` transmits a valid `toolResponse.functionResponses` frame with matching `call_id` values back over the WebSocket so the voice turn resumes seamlessly.
