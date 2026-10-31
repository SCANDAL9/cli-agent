# Wave 5: Unified Reactive Swarm & Semantic Runtime
## 02. High-Performance Asynchronous Pub/Sub EventBus (W5.2)

### 1. Problem Statement & Architectural Motivation

In Wave 4 (`src/cli_agent/tui.py`), the Fullscreen Split-Pane Terminal User Interface introduced four specialized viewports (`ChatPane`, `DiffViewerPane`, `ProcessLogPane`, `StatusHUD`) driven by an internal `asyncio.Queue[TUIEvent]`. However, across the broader runtime:
- `MultiAgentTeam` (`src/cli_agent/team.py`) emitted status strings via an optional synchronous callback `on_event: Callable[[str, str], None]`.
- `WorktreeSession` (`src/cli_agent/worktree.py`) computed diffs only on explicit pull calls (`get_diff()`), with no reactive notification when files changed inside a worker's sandbox.
- `QuotaCircuitBreaker` (`src/cli_agent/circuit_breaker.py`) mutated internal endpoint health states without broadcasting real-time transitions (`CLOSED` $\rightarrow$ `OPEN` $\rightarrow$ `HALF_OPEN`) to the `StatusHUD`.
- `TDDExecutionEngine` (`src/cli_agent/tdd.py`), `HeadlessLSPManager` (`src/cli_agent/lsp.py`), and `MCPClientManager` (`src/cli_agent/mcp.py`) had no unified channel to stream subprocess log lines into `ProcessLogPane`.

Furthermore, when 4 concurrent DAG workers stream LLM tokens simultaneously while `pytest` and `pyright-langserver` emit diagnostic bursts, naive per-event UI rendering generates **`> 500` events/second**, saturating the Python `asyncio` event loop and causing severe terminal flicker.

**Milestone W5.2 (`src/cli_agent/events.py`)** introduces a high-performance, lock-free **Asynchronous Pub/Sub EventBus** with:
1. **Five Canonical Telemetry Topics**: `token.stream`, `swarm.node_state`, `worktree.diff_updated`, `subproc.log`, and `circuit.health`.
2. **Zero-Blocking Ring-Buffer Subscriptions**: Bounded `collections.deque` ring buffers (`maxlen=4096`) with deterministic drop-oldest backpressure protection so slow consumers never stall inference or tool execution.
3. **60 FPS (`16.67ms`) Frame-Paced Coalescer**: Automatically merges high-frequency token chunks, intermediate diff snapshots, and HUD counters within each `16.67ms` frame window before dispatching a single batch update to `TUIApp`.
4. **Bi-Directional `TUIApp` Integration**: Wires all 4 `TUIApp` panes (`ChatPane`, `DiffViewerPane`, `ProcessLogPane`, `StatusHUD`) to live `EventBus` subscriptions and routes user control signals (`tui.command`, `swarm.cancel`) back to the runtime orchestrator.

---

### 2. Core EventBus Architecture & Topic Taxonomy (`src/cli_agent/events.py`)

```
+---------------------------------------------------------------------------------------------------+
|                               Runtime Telemetry Producers                                         |
|  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐ |
|  │ VertexAgentClient│ │ DAGTaskScheduler │ │ WorktreeSession  │ │ TDD / LSP /  │ │ QuotaCircuit │ |
|  │ (LLM Streaming)  │ │ (MultiAgentTeam) │ │ & tools.py edits │ │ MCP Runners  │ │ Breaker      │ |
|  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └──────┬───────┘ └──────┬───────┘ |
+-----------│--------------------│--------------------│------------------│----------------│---------+
            │                    │                    │                  │                │
            ▼                    ▼                    ▼                  ▼                ▼
     [token.stream]     [swarm.node_state]  [worktree.diff_updated] [subproc.log]  [circuit.health]
            │                    │                    │                  │                │
            └────────────────────┴────────────────────┼──────────────────┴────────────────┘
                                                      │ publish_nowait(EventEnvelope) [O(1) < 50µs]
                                                      ▼
+---------------------------------------------------------------------------------------------------+
|                            Async Pub/Sub EventBus (src/cli_agent/events.py)                       |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ Topic Router & Subscriber Registry (Exact Topic Match + Wildcard "*" Audit Subscribers)     │  |
|  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  |
|                                                 │ Append to bounded deque(maxlen=4096)            |
|                                                 ▼                                                 |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ FrameCoalescer (60 FPS / 16.67ms Window):                                                   │  |
|  │  - Concatenates contiguous `token.stream` deltas per `(agent_id, turn_id)`                  │  |
|  │  - Deduplicates `swarm.node_state` & `circuit.health` (latest-state-wins per key)           │  |
|  │  - Deduplicates `worktree.diff_updated` per `(worktree_id, file_path)`                      │  |
|  │  - Batches `subproc.log` lines into a single array flush                                    │  |
|  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  |
+-------------------------------------------------│-------------------------------------------------+
                                                  │ Single Coalesced Frame Batch every 16.67ms
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                               TUIApp Bi-Directional Bridge (src/cli_agent/tui.py)                 |
|        ┌────────────────────────┬───────────────────────┬───────────────────────┐                 |
|        ▼                        ▼                       ▼                       ▼                 |
|  ┌───────────────┐      ┌───────────────┐       ┌───────────────┐       ┌───────────────┐         |
|  │   ChatPane    │      │DiffViewerPane │       │ProcessLogPane │       │   StatusHUD   │         |
|  └───────────────┘      └───────────────┘       └───────────────┘       └───────────────┘         |
+---------------------------------------------------------------------------------------------------+
```

#### 2.1 Canonical Telemetry Topics (`EventTopic`)

| Topic Enum (`EventTopic`) | Topic String | Primary Publisher(s) | Primary Subscriber(s) | Coalescing Strategy (16.67ms Window) |
| :--- | :--- | :--- | :--- | :--- |
| `TOKEN_STREAM` | `"token.stream"` | `VertexAgentClient.chat_stream()`, `MultiAgentTeam` workers | `ChatPane`, `StatusHUD` (token counter) | **String Concatenation**: Contiguous text/thought tokens for the same `agent_id` are joined via `"".join(chunks)` into a single delta event. |
| `SWARM_NODE_STATE` | `"swarm.node_state"` | `DAGTaskScheduler`, `MultiAgentTeam.spawn_worker()` | `ChatPane` (Swarm Tree), `StatusHUD` (active worker count) | **Latest-State-Wins per `(plan_id, node_id)`**: Intermediate rapid state transitions collapse to the most recent `SwarmNodeStatePayload`. |
| `WORKTREE_DIFF_UPDATED` | `"worktree.diff_updated"` | `tools.write_file()`, `tools.edit_file()`, `WorktreeSession`, `ASTThreeWayMerger` | `DiffViewerPane` | **Latest-Snapshot-Wins per `(worktree_id, file_path)`**: Rapid sequential edits to the same file within 16.6ms emit only the latest unified diff & `SymbolDelta`. |
| `SUBPROC_LOG` | `"subproc.log"` | `TDDExecutionEngine`, `HeadlessLSPClient`, `MCPClientManager`, `CIWatcher` | `ProcessLogPane` | **Ordered Batch Array**: Preserves 100% of log lines in chronological order, delivering them as a single batch list per frame tick. |
| `CIRCUIT_HEALTH` | `"circuit.health"` | `QuotaCircuitBreaker`, `FallbackModelRouter`, `VoiceAgentEngine` | `StatusHUD`, `ProcessLogPane` (on failover alert) | **Latest-State-Wins per `(model_id, region)`**: Only the newest `CircuitHealthPayload` snapshot per endpoint is rendered per frame. |

---

### 3. Strictly Typed Event Payloads & Envelope Schema

To guarantee `mypy --strict` compliance and eliminate untyped dictionary guesswork, `src/cli_agent/events.py` defines immutable `@dataclass(frozen=True, slots=True)` schemas for every telemetry topic:

```python
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

from .circuit_breaker import CircuitState
from .pr import SymbolDelta


class EventTopic(StrEnum):
    """Canonical Pub/Sub telemetry topics in the Wave 5 reactive runtime."""

    TOKEN_STREAM = "token.stream"
    SWARM_NODE_STATE = "swarm.node_state"
    WORKTREE_DIFF_UPDATED = "worktree.diff_updated"
    SUBPROC_LOG = "subproc.log"
    CIRCUIT_HEALTH = "circuit.health"
    TUI_CONTROL = "tui.control"  # Bi-directional UI -> Runtime control signals


@dataclass(frozen=True, slots=True)
class TokenStreamPayload:
    """Payload for `token.stream` events."""

    agent_id: str
    delta: str
    is_thought: bool = False
    is_complete: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = ""


@dataclass(frozen=True, slots=True)
class SwarmNodeStatePayload:
    """Payload for `swarm.node_state` events."""

    plan_id: str
    node_id: str
    role: str  # "architect" | "worker" | "qa"
    status: str  # "PENDING" | "READY" | "RUNNING" | "COMPLETED" | "FAILED" | "SKIPPED"
    active_tool: str | None = None
    worktree_branch: str | None = None
    dependencies: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorktreeDiffPayload:
    """Payload for `worktree.diff_updated` events."""

    worktree_id: str
    branch_name: str
    file_path: str
    diff_text: str
    insertions: int = 0
    deletions: int = 0
    symbol_delta: SymbolDelta | None = None
    lsp_diagnostics_count: int = 0


@dataclass(frozen=True, slots=True)
class SubprocLogPayload:
    """Payload for `subproc.log` events."""

    source: str  # "TDD" | "LSP" | "MCP" | "CI" | "GIT"
    line: str
    level: str = "INFO"  # "DEBUG" | "INFO" | "WARN" | "ERROR"
    worker_id: str | None = None


@dataclass(frozen=True, slots=True)
class CircuitHealthPayload:
    """Payload for `circuit.health` events."""

    model_id: str
    region: str
    tier: int
    state: CircuitState
    failure_count: int = 0
    cooldown_remaining_s: float = 0.0
    failover_from: str | None = None
    vad_active: bool = False


@dataclass(frozen=True, slots=True)
class TUIControlPayload:
    """Payload for `tui.control` bi-directional user actions emitted from TUIApp."""

    action: str  # "SUBMIT_PROMPT" | "CANCEL_SWARM" | "SWITCH_WORKTREE" | "TOGGLE_VOICE"
    argument: str = ""


PayloadT = (
    TokenStreamPayload
    | SwarmNodeStatePayload
    | WorktreeDiffPayload
    | SubprocLogPayload
    | CircuitHealthPayload
    | TUIControlPayload
    | dict[str, Any]
)


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Immutable message envelope routed through the EventBus."""

    topic: EventTopic
    payload: PayloadT
    timestamp: float = field(default_factory=time.monotonic)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
```

---

### 4. Ring-Buffer Subscriptions & 60 FPS Coalescing Engine

#### 4.1 Bounded Ring-Buffer Subscriber (`RingBufferSubscription`)

Standard `asyncio.Queue` instances either block the producer when full (`QueueFull`) or grow without bound (`maxsize=0`), risking `MemoryError` if a TUI consumer pauses or disconnects.

`RingBufferSubscription` solves this using a `collections.deque(maxlen=capacity)` paired with a lightweight `asyncio.Event` wake-up signal:

1. **Non-Blocking Publish ($O(1)$)**:
   - When `bus.publish_nowait(topic, payload)` is called:
     - If `len(sub._ring) == sub._ring.maxlen`, `deque.append()` automatically evicts the oldest unconsumed event in $O(1)$ C-level memory time and increments `sub.dropped_count`.
     - Calls `sub._wake_event.set()` (non-blocking boolean flag flip).
2. **Batch Drain**:
   - The consumer awaits `await sub._wake_event.wait()`, clears `_wake_event`, and drains all currently buffered envelopes in a tight synchronous slice (`list(sub._ring); sub._ring.clear()`).

#### 4.2 60 FPS (`16.67ms`) Frame Coalescing Algorithm

When `FrameCoalescer.coalesce_batch(envelopes: Sequence[EventEnvelope]) -> list[EventEnvelope]` processes a burst of $N$ raw events accumulated during a `16.67ms` frame window:

```python
def coalesce_frame_events(envelopes: Sequence[EventEnvelope]) -> list[EventEnvelope]:
    """Coalesces high-frequency events within a 16.67ms (60 FPS) frame window."""
    if len(envelopes) <= 1:
        return list(envelopes)

    token_accum: dict[tuple[str, bool], TokenStreamPayload] = {}
    node_states: dict[tuple[str, str], EventEnvelope] = {}
    diff_updates: dict[tuple[str, str], EventEnvelope] = {}
    circuit_states: dict[tuple[str, str], EventEnvelope] = {}
    ordered_passthrough: list[EventEnvelope] = []

    for env in envelopes:
        if env.topic == EventTopic.TOKEN_STREAM and isinstance(env.payload, TokenStreamPayload):
            p = env.payload
            key = (p.agent_id, p.is_thought)
            if key in token_accum:
                prev = token_accum[key]
                token_accum[key] = TokenStreamPayload(
                    agent_id=p.agent_id,
                    delta=prev.delta + p.delta,
                    is_thought=p.is_thought,
                    is_complete=prev.is_complete or p.is_complete,
                    input_tokens=max(prev.input_tokens, p.input_tokens),
                    output_tokens=max(prev.output_tokens, p.output_tokens),
                    model_id=p.model_id or prev.model_id,
                )
            else:
                token_accum[key] = p

        elif env.topic == EventTopic.SWARM_NODE_STATE and isinstance(
            env.payload, SwarmNodeStatePayload
        ):
            node_states[(env.payload.plan_id, env.payload.node_id)] = env

        elif env.topic == EventTopic.WORKTREE_DIFF_UPDATED and isinstance(
            env.payload, WorktreeDiffPayload
        ):
            diff_updates[(env.payload.worktree_id, env.payload.file_path)] = env

        elif env.topic == EventTopic.CIRCUIT_HEALTH and isinstance(
            env.payload, CircuitHealthPayload
        ):
            circuit_states[(env.payload.model_id, env.payload.region)] = env

        else:
            # SUBPROC_LOG and TUI_CONTROL preserve exact chronological ordering
            ordered_passthrough.append(env)

    coalesced: list[EventEnvelope] = [
        EventEnvelope(topic=EventTopic.TOKEN_STREAM, payload=tp) for tp in token_accum.values()
    ]
    coalesced.extend(node_states.values())
    coalesced.extend(diff_updates.values())
    coalesced.extend(circuit_states.values())
    coalesced.extend(ordered_passthrough)
    return coalesced
```

#### 4.3 Frame Pacing & Latency Characteristics

- **Minimum Frame Interval**: $\Delta t_{\text{frame}} = \frac{1000\text{ ms}}{60} \approx 16.67\text{ ms}$ (configurable via `fps_limit: int = 60`).
- **Idle Zero-CPU Guarantee**: When no events are published, the coalescer sleeps on `await sub._wake_event.wait()` consuming **0.0% CPU** (zero busy-wait polling loops).
- **First-Event Immediate Edge Trigger**: When transitioning from idle to active, the first event batch can flush immediately, and subsequent bursts within the next `16.67ms` window are coalesced before the next frame tick.

---

### 5. Bi-Directional Integration with `TUIApp` (`src/cli_agent/tui.py`)

#### 5.1 Forward Telemetry Path (`EventBus` $\rightarrow$ `TUIApp` Panes)

`TUIApp` attaches a `TUIEventBusBridge` that subscribes to all five telemetry topics and updates the four viewports atomically per frame tick:

1. **`token.stream` $\rightarrow$ `ChatPane` & `StatusHUD`**:
   - Appends `payload.delta` to `ChatPane.streaming_tokens` (or per-worker collapsible stream buffer).
   - If `payload.is_complete` is `True`, commits the streaming turn into `ChatPane.messages`.
   - Updates `StatusHUD.input_tokens` and `StatusHUD.output_tokens` when non-zero.
2. **`swarm.node_state` $\rightarrow$ `ChatPane` & `StatusHUD`**:
   - Dynamically constructs/updates the `SwarmNode` hierarchy inside `ChatPane.swarm_root`, showing each DAG node's status badge (`[PENDING]`, `[READY]`, `[RUNNING ▶]`, `[COMPLETED ✓]`, `[FAILED ✗]`), active worktree branch, and current tool invocation.
   - Updates `StatusHUD.active_swarm_workers` count (`sum(1 for n in nodes if n.status == "RUNNING")`).
3. **`worktree.diff_updated` $\rightarrow$ `DiffViewerPane`**:
   - Calls `DiffViewerPane.update_file_diff(file_path=payload.file_path, diff_text=payload.diff_text, symbol_delta=payload.symbol_delta)`.
   - Annotates the header with the originating `worktree_id` / `branch_name` and live LSP diagnostic badge (`LSP: 0 errors`).
4. **`subproc.log` $\rightarrow$ `ProcessLogPane`**:
   - Maps `payload.source` (`"TDD"`, `"LSP"`, `"MCP"`, `"CI"`) to `LogSource` enum and appends lines into `ProcessLogPane`'s bounded ring buffer (`maxlen=2000`).
5. **`circuit.health` $\rightarrow$ `StatusHUD`**:
   - Updates `StatusHUD.active_model`, `StatusHUD.region`, `StatusHUD.tier`, `StatusHUD.circuit_state`, `StatusHUD.cooldown_seconds`, and `StatusHUD.vad_active`.

#### 5.2 Reverse Control Path (`TUIApp` $\rightarrow$ `EventBus` $\rightarrow$ Runtime)

`TUIApp` is not merely a passive dashboard; user interactions in `TUIApp` publish `EventTopic.TUI_CONTROL` events back onto the `EventBus`:

| User Keystroke / Command in `TUIApp` | Published `TUIControlPayload` | Subscribing Runtime Component & Effect |
| :--- | :--- | :--- |
| Typing prompt + `Enter` | `action="SUBMIT_PROMPT", argument="<text>"` | Main REPL / `LeadArchitect` initiates a new turn or DAG workflow. |
| `Ctrl+C` during active DAG swarm | `action="CANCEL_SWARM", argument="user_interrupt"` | `DAGTaskScheduler` cancels its `asyncio.TaskGroup` and triggers immediate worktree cleanup. |
| `[` or `]` in `DiffViewerPane` | `action="CYCLE_DIFF_FILE", argument="+1/-1"` | Cycles active file inspection across concurrent worker worktrees. |
| `Ctrl+D` in `DiffViewerPane` | `action="TOGGLE_DIFF_MODE"` | Switches between `UNIFIED_DIFF` and Tree-Sitter `AST_OUTLINE`. |
| `/voice` or `Ctrl+V` | `action="TOGGLE_VOICE"` | `VoiceAgentEngine` starts/stops full-duplex microphone VAD streaming. |

---

### 6. Module Public Interface (`src/cli_agent/events.py`)

```python
class RingBufferSubscription:
    """Bounded lock-free ring-buffer subscriber with drop-oldest backpressure semantics."""

    def __init__(
        self,
        subscription_id: str,
        topics: set[EventTopic] | None = None,
        maxlen: int = 4096,
    ) -> None: ...

    @property
    def dropped_count(self) -> int: ...

    @property
    def pending_count(self) -> int: ...

    def push_nowait(self, envelope: EventEnvelope) -> None: ...

    def drain_nowait(self) -> list[EventEnvelope]: ...

    async def wait_and_drain(self, timeout: float | None = None) -> list[EventEnvelope]: ...


class EventBus:
    """High-performance asynchronous Pub/Sub telemetry broker with 60 FPS coalescing."""

    def __init__(self, default_maxlen: int = 4096, fps_limit: int = 60) -> None: ...

    def subscribe(
        self,
        topics: Iterable[EventTopic] | None = None,
        maxlen: int | None = None,
    ) -> RingBufferSubscription: ...

    def unsubscribe(self, subscription: RingBufferSubscription) -> None: ...

    def publish_nowait(self, topic: EventTopic, payload: PayloadT) -> EventEnvelope: ...

    async def publish(self, topic: EventTopic, payload: PayloadT) -> EventEnvelope: ...

    async def iter_coalesced_frames(
        self,
        subscription: RingBufferSubscription,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[list[EventEnvelope]]: ...


# Process-wide singleton accessor with dependency-injection override support
def get_event_bus() -> EventBus: ...
def set_event_bus(bus: EventBus | None) -> None: ...
```
