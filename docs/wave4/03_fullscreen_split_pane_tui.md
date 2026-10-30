# Wave 4: Autonomous Delivery & Production Workflows
## 03. Fullscreen Split-Pane Terminal User Interface (W4.3)

### 1. Problem Statement & Motivation

Through Waves 1 to 4.2, `cli-agent` evolved from a sequential single-turn assistant into a highly concurrent autonomous engineering runtime. During a single task execution, the runtime simultaneously generates:
- **Hierarchical Multi-Agent Swarm Streams (W2.1)**: Lead Architect planning, parallel Feature Worker code edits inside isolated Git worktrees, and QA Verifier critiques.
- **Real-Time AST & File Diffs (W1.1 / W4.2)**: Tree-Sitter symbol graph mutations (`SymbolDelta`) and multi-file unified diffs (`git diff`).
- **Self-Healing TDD & Remote CI Output (W2.2 / W4.2)**: Streaming `pytest` subprocess stdout/stderr, coverage matrices, and Model Context Protocol (MCP) JSON-RPC server telemetry.
- **Resilience & Multi-Modal Telemetry (W3.1 / W4.1)**: Real-time `EndpointHealthSnapshot` circuit breaker state transitions (`CLOSED` $\leftrightarrow$ `OPEN` $\leftrightarrow$ `HALF_OPEN`), token consumption velocity, and Voice Activity Detection (VAD) microphone capture states.

In a traditional **single-stream scrolling REPL**, multiplexing these concurrent telemetry channels causes severe **cognitive and visual overload**:

1. **Interleaved Stream Corruption**: Asynchronous `pytest` tracebacks, MCP server diagnostics, and sub-agent tool logs interleave with streaming LLM markdown tokens, tearing ANSI escape sequences and destroying reading continuity.
2. **Spatial Context Loss (Scrollback Blindness)**: Inspecting a 120-line unified diff or an AST symbol outline pushes the active reasoning prompt and sub-agent progress tree hundreds of lines off-screen, forcing constant manual scrollback navigation.
3. **Invisible System Telemetry**: Critical operational signals—such as a Wave 4.1 quota circuit breaker tripping from `anthropic/claude-sonnet-4-6@us-east5` to Tier 2 `google/gemini-2.5-pro@us-central1`, active token budget burn, or an open Voice VAD microphone—remain buried until a turn completes.

**W4.3 Fullscreen Split-Pane Terminal User Interface (`src/cli_agent/tui.py`)** solves this by introducing an ergonomic, flicker-free, 4-zone spatial workspace that segregates conversational reasoning, live code/AST diffs, background process streams, and real-time health telemetry into dedicated, keyboard-navigable viewports while preserving instantaneous toggling (`--tui` and `/tui`) with the classic line-oriented REPL.

---

### 2. Layout Topology & Visual Architecture

The W4.3 Fullscreen TUI partitions the terminal alternate screen buffer (`DECSET 1049`) into a responsive 2-column split-pane grid anchored by a persistent 2-line **Bottom Status HUD** and an interactive multi-line input bar.

```
+---------------------------------------------------+---------------------------------------------------+
| [1] LEFT PANE (50% Width, 100% Main Height)       | [2] TOP-RIGHT PANE (50% Width, 55% Main Height)   |
| Streaming Chat & Multi-Agent Swarm Tree           | Real-Time AST & Unified Diff Viewer               |
|---------------------------------------------------|---------------------------------------------------|
| > User: Implement W4.1 Quota Circuit Breaker      | File: src/cli_agent/resilience/circuit_breaker.py |
|                                                   | AST Delta: +QuotaCircuitBreaker, ~allow_request   |
| [*] Lead Architect (claude-sonnet-4-6@us-east5)   |---------------------------------------------------|
|  ├─ [✓] Worker-1 (worktree: feat/cb-state)        | @@ -42,6 +42,19 @@ class QuotaCircuitBreaker:     |
|  │   └─ edit_file: circuit_breaker.py (+84 -4)    | +    def allow_request(self, ep: ModelEndpoint):  |
|  └─ [▶] QA-Verifier (worktree: feat/cb-state)     | +        state = self.get_state(ep)               |
|      └─ run_command: pytest tests/unit/resilience | +        if state == CircuitState.CLOSED:         |
|                                                   | +            return True                          |
| Assistant (Streaming):                            | -        return False                             |
| Verifying sliding-window 503 trip thresholds and  | +        return self._try_half_open_probe(ep)     |
| Retry-After header jitter across fallback tiers...|                                                   |
|                                                   +---------------------------------------------------+
|                                                   | [3] BOTTOM-RIGHT PANE (50% Width, 45% Main Height)|
|                                                   | Streaming TDD Runner & MCP Server Logs            |
|                                                   |---------------------------------------------------|
|                                                   | [TDD Cycle 1/3] $ pytest tests/unit/resilience -q |
|                                                   | .............                                     |
|                                                   | 13 passed in 0.14s (Coverage: 98.4%)              |
|                                                   | [MCP:git] JSON-RPC <- tools/call (git_diff_stat)  |
|                                                   | [CIWatcher] PR #42 checks: RUNNING (2/4 green)    |
+---------------------------------------------------+---------------------------------------------------+
| [Prompt] > /tui or type instruction... (Tab: Cycle Focus | Ctrl+D: Diff Mode | Ctrl+L: Clear Logs)    |
+-------------------------------------------------------------------------------------------------------+
| [4] STATUS HUD | Model: anthropic/claude-sonnet-4-6@us-east5 (Tier 1) | CB: [CLOSED ●] 0 fails        |
| Tokens: 14.2k in / 3.8k out ($0.09) | Swarm: 2 active | VAD Mic: [MUTED ○] (Space to talk) | 60 FPS   |
+-------------------------------------------------------------------------------------------------------+
```

#### 2.1 Viewport Specification Matrix

| Viewport ID | Component Class | Default Geometry | Primary Data Sources | Interactive Keybindings & Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **Left Pane** | `ChatPane` | `50%` width $\times$ `100%` body height | Streaming LLM markdown chunks, collapsible `SwarmNode` tree (`Lead` $\rightarrow$ `Worker` $\rightarrow$ `QA`), tool call summaries, and prompt composer. | `Up`/`Down`/`PgUp`/`PgDn` scrollback; `Enter` toggles sub-agent node expansion; auto-scrolls on new token unless user scrolled up (`scroll_lock`). |
| **Top-Right Pane** | `DiffViewerPane` | `50%` width $\times$ `55%` body height | Live unified diffs (`write_file` / `edit_file` patches) and Wave 1 Tree-Sitter `SymbolDelta` outlines (`+` Added, `~` Modified, `-` Removed symbols). | `Ctrl+D` toggles between **Unified Diff View** and **AST Symbol Outline View**; `[` and `]` cycle across modified files in the active worktree. |
| **Bottom-Right Pane** | `ProcessLogPane` | `50%` width $\times$ `45%` body height | Asynchronous ring-buffered (`maxlen=2000`) line streams from Wave 2 `TDDExecutionEngine` (`pytest`), Wave 4.2 `CIWatcher`, and MCP stdio/SSE servers. | `j`/`k` line scroll; `f` toggles filter between `ALL`, `TDD`, `MCP`, and `CI`; automatically highlights pytest `FAILED` / `ERROR` tracebacks. |
| **Bottom Status HUD** | `StatusHUD` | `100%` width $\times$ `2` fixed rows (docked bottom) | `EndpointHealthSnapshot` from W4.1 `QuotaCircuitBreaker`, active `(model_id, region, tier)`, session ITPM/OTPM token counters, and W3.1 Voice VAD state. | Color-coded circuit indicator (`GREEN` = `CLOSED`, `AMBER` = `HALF_OPEN` with probe spinner, `RED` = `OPEN` with live cooldown countdown). |

#### 2.2 Responsive Terminal Breakpoints

`TUILayoutManager` dynamically recalculates pane geometry on `SIGWINCH` terminal resize events:
- **Standard Split Mode (`columns >= 100`, `rows >= 24`)**: Full 3-pane asymmetric split + docked `StatusHUD`.
- **Compact Vertical Mode (`80 <= columns < 100`)**: Collapses the right column into a tabbed secondary pane (`Tab 1: Diff/AST`, `Tab 2: TDD/MCP Logs`) stacked below `ChatPane` (`60%` / `40%` vertical split).
- **Minimal Safe Mode (`columns < 80` or `rows < 18`)**: Displays `ChatPane` (`100%` main area) + `StatusHUD`, keeping `DiffViewerPane` and `ProcessLogPane` accessible via hotkey overlays (`Alt+1`, `Alt+2`, `Alt+3`).

---

### 3. Terminal Engine Selection & Rendering Pipeline

#### 3.1 Comparative Evaluation: `Textual` vs. `prompt_toolkit` vs. `Rich Live`

| Evaluation Criterion | `Rich.live.Live` + `Rich.layout.Layout` | `prompt_toolkit.Application` (Full-Screen) | `Textual` (Async DOM/CSS TUI Framework) |
| :--- | :--- | :--- | :--- |
| **Dependency Footprint** | **Zero new dependencies** (`rich` is already core to `cli-agent`). | **Zero new dependencies** (`prompt_toolkit` is already installed for REPL history/completion). | Requires heavy third-party dependency tree (`textual`, CSS parser, custom driver). |
| **Interactive Input & Key Dispatch** | **Weak**: `Rich.live` is output-only; it lacks native non-blocking keystroke dispatch, multi-line text editing, or per-pane scroll focus without raw `termios` hacks. | **Native**: Full-featured `BufferControl`, `Window`, `HSplit`/`VSplit` containers, mouse wheel support, and composable `KeyBindings`. | **Strong**: Widget-level focus and keybindings, but hijacks standard stdio and complicates instant drop-out to standard REPL. |
| **Rich Renderable Compatibility** | **Native**: Directly renders `Syntax`, `Markdown`, `Tree`, `Table`, and `Panel`. | **High via ANSI Adapter**: Renders `Rich` objects into ANSI segments via an in-memory `Console` buffer cached per dirty pane. | **Native**: Built atop `Rich` segment rendering. |
| **Seamless `/tui` Runtime Toggle** | Trivial start/stop, but no input loop. | **Seamless**: Shares the exact same `PromptSession` history and `asyncio` loop; enters/exits alternate screen buffer cleanly in `< 5ms`. | **Intrusive**: Takes ownership of the global event loop and signal handlers, making hot-swapping back to raw REPL fragile. |
| **Headless Unit Testability** | Easy string capture via `Console(record=True)`. | **Deterministic**: Supports `DummyInput` / `create_pipe_input()` and virtual screen buffer inspection (`Screen` / `Char` grid) with zero TTY required. | Requires `Pilot` async test harness and external `textual` runtime. |

#### 3.2 Architectural Decision: Hybrid `Rich.layout` Compositor + `prompt_toolkit` Non-Blocking Event Driver

To achieve **zero external package bloat**, **100% compatibility with existing `Rich` renderables**, and **hermetic headless unit testing**, W4.3 implements a **Hybrid Compositor Pipeline**:

1. **Declarative Viewport Rendering (`Rich.layout.Layout` + Dirty-Flag Caching)**:
   - Each pane (`ChatPane`, `DiffViewerPane`, `ProcessLogPane`, `StatusHUD`) encapsulates its own state model and produces a standard `Rich` renderable (`Panel`, `Syntax`, `Tree`, `Table`).
   - A pane only re-rasterizes its `Rich` renderable when its `_dirty` flag is set or when terminal dimensions `(width, height)` change, bounding CPU overhead during high-frequency token streaming.
2. **Alternate Screen & Input Dispatch (`prompt_toolkit` / Virtual Framebuffer Driver)**:
   - Keyboard navigation (`Tab` focus cycling, arrow scrollback, prompt editing, `/tui` toggle) and alternate screen buffer lifecycle (`\x1b[?1049h` / `\x1b[?1049l`) are driven through an async event loop integrated with `prompt_toolkit` key bindings and a deterministic `VirtualScreenBuffer` abstraction.
3. **60 FPS Frame-Paced Coalescing (`asyncio` Event Loop Integration)**:
   - Concurrent producers—LLM SSE token streams, `asyncio.create_subprocess_exec("pytest", ...)` stdout pipes, MCP JSON-RPC readers, and `QuotaCircuitBreaker` callbacks—push immutable `TUIEvent` messages into an unbounded `asyncio.Queue[TUIEvent]`.
   - Instead of repainting the terminal on every single token or log line (which would cause I/O backpressure at `> 500` events/sec), the `TUIApp` compositor drains all queued events within a **`16.6ms` (`60 Hz`) frame tick** and performs a single coalesced screen diff flush.

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ LLM Token Stream     │  │ Subprocess Pipes     │  │ AST & Worktree Diff  │  │ W4.1 Circuit Breaker │
│ & Swarm Coordinator  │  │ (Pytest TDD & MCP)   │  │ Watcher (TreeSitter) │  │ & W3.1 Voice VAD     │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │                         │
           └─────────────────────────┴────────────┬────────────┴─────────────────────────┘
                                                  │ Non-blocking `put_nowait(TUIEvent)`
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │   asyncio.Queue[TUIEvent]   │
                                   └──────────────┬──────────────┘
                                                  │ Drained in batches every 16.6ms (60 FPS)
                                                  ▼
                                   ┌─────────────────────────────┐
                                   │       TUIApp Event Bus      │
                                   └──┬───────┬───────┬───────┬──┘
                  ┌───────────────────┘       │       │       └───────────────────┐
                  ▼                           ▼       ▼                           ▼
         ┌─────────────────┐        ┌───────────────────┐        ┌─────────────────┐
         │    ChatPane     │        │  DiffViewerPane   │        │ ProcessLogPane  │
         │  (_dirty=True)  │        │   (_dirty=True)   │        │  (_dirty=True)  │
         └────────┬────────┘        └─────────┬─────────┘        └────────┬────────┘
                  └───────────────────────────┼───────────────────────────┘
                                              ▼
                              ┌───────────────────────────────┐
                              │       TUILayoutManager        │
                              │  (Rich Layout + StatusHUD)    │
                              └───────────────┬───────────────┘
                                              │ Renders to VirtualScreenBuffer (ANSI / Grid)
                                              ▼
                              ┌───────────────────────────────┐
                              │ Alternate Screen TTY / Test   │
                              └───────────────────────────────┘
```

---

### 4. Slash Command & CLI Flag Integration (`--tui` & `/tui`)

`cli-agent` supports both startup boot into the Fullscreen TUI and zero-loss runtime switching mid-session.

#### 4.1 Startup CLI Flag (`--tui`)
- Registered in the CLI argument parser (`src/cli_agent/main.py` / `cli.py`):
  - `cli-agent --tui`: Launches directly into `TUIApp` alternate screen mode upon session initialization.
  - Environment variable fallback: `CLI_AGENT_TUI=1` enables `--tui` by default when stdout is an interactive TTY (`sys.stdout.isatty() == True`).
  - **Non-TTY Safety Guard**: If `--tui` is passed in a non-interactive pipe or CI runner (`isatty() == False` and not in headless test mode), `cli-agent` logs a structured warning and gracefully falls back to standard streaming output so automated scripts never hang on terminal control codes.

#### 4.2 Runtime Slash Command Toggle (`/tui`)
- Executing `/tui` inside the standard scrolling REPL or inside the Fullscreen TUI prompt composer triggers an immediate mode transition while preserving 100% of session state:
  1. **Standard REPL $\rightarrow$ Fullscreen TUI (`/tui` or `/tui on`)**:
     - Hydrates `ChatPane` with the existing conversation history and active `SwarmNode` tree.
     - Hydrates `DiffViewerPane` with the current Git worktree `git diff` and Tree-Sitter `SymbolDelta` snapshot.
     - Hydrates `StatusHUD` with the active `QuotaCircuitBreaker.get_health_snapshot()` and cumulative session token counters.
     - Enters the alternate screen buffer (`\x1b[?1049h`), hides raw cursor flicker, and starts the `TUIApp` async compositor loop.
  2. **Fullscreen TUI $\rightarrow$ Standard REPL (`/tui`, `/tui off`, or `Ctrl+X`)**:
     - Gracefully cancels the `16.6ms` compositor tick task and unhooks TUI keybindings while keeping background subprocesses alive.
     - Exits the alternate screen buffer (`\x1b[?1049l`), restoring the user's prior terminal scrollback buffer intact.
     - Prints a concise transition summary banner (`[TUI Exited — Session context & active worktree preserved]`) and resumes the standard `prompt_toolkit` line REPL.

---

### 5. Technical Interface: `src/cli_agent/tui.py`

All layouts, pane state models, virtual screen buffers, and async event loop controllers reside in `src/cli_agent/tui.py`.

```python
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Sequence, Tuple

from rich.console import Console, RenderableType
from rich.layout import Layout

from cli_agent.pr import SymbolDelta
from cli_agent.resilience.circuit_breaker import CircuitState, EndpointHealthSnapshot, ModelEndpoint


class FocusPane(str, Enum):
    """Identifies the currently focused interactive pane for keyboard navigation."""

    CHAT = "CHAT"
    DIFF = "DIFF"
    LOGS = "LOGS"


class DiffViewMode(str, Enum):
    """Display mode for the Top-Right DiffViewerPane."""

    UNIFIED_DIFF = "UNIFIED_DIFF"
    AST_OUTLINE = "AST_OUTLINE"


class LogSource(str, Enum):
    """Origin subsystem tag for lines streamed into ProcessLogPane."""

    ALL = "ALL"
    TDD = "TDD"
    MCP = "MCP"
    CI = "CI"
    SYSTEM = "SYSTEM"


class LayoutMode(str, Enum):
    """Responsive layout mode computed from terminal (columns, rows) dimensions."""

    STANDARD_SPLIT = "STANDARD_SPLIT"  # cols >= 100, rows >= 24
    COMPACT_VERTICAL = "COMPACT_VERTICAL"  # 80 <= cols < 100
    MINIMAL_SAFE = "MINIMAL_SAFE"  # cols < 80 or rows < 18


@dataclass(frozen=True)
class TerminalDimensions:
    """Immutable terminal viewport dimensions in character cells."""

    columns: int = 120
    rows: int = 40

    @property
    def layout_mode(self) -> LayoutMode:
        if self.columns >= 100 and self.rows >= 24:
            return LayoutMode.STANDARD_SPLIT
        if self.columns >= 80 and self.rows >= 18:
            return LayoutMode.COMPACT_VERTICAL
        return LayoutMode.MINIMAL_SAFE


@dataclass
class SwarmNode:
    """Hierarchical status node representing a Lead, Worker, or QA sub-agent in ChatPane."""

    agent_id: str
    role: str  # "lead", "worker", "qa"
    status: str  # "IDLE", "RUNNING", "COMPLETED", "FAILED"
    worktree_branch: Optional[str] = None
    active_tool: Optional[str] = None
    summary: str = ""
    expanded: bool = True
    children: List[SwarmNode] = field(default_factory=list)


@dataclass(frozen=True)
class ChatMessage:
    """Single conversational turn or tool invocation entry rendered in ChatPane."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: float = 0.0
    is_streaming: bool = False


@dataclass(frozen=True)
class LogEntry:
    """Structured log line stored in ProcessLogPane's bounded ring buffer."""

    source: LogSource
    line: str
    is_error: bool = False
    timestamp: float = 0.0


@dataclass(frozen=True)
class VirtualScreenBuffer:
    """
    Deterministic 2D character snapshot rendered by TUILayoutManager for
    headless verification and alternate-screen differential flushing.
    """

    dimensions: TerminalDimensions
    lines: Tuple[str, ...]
    ansi_payload: str
    focused_pane: FocusPane
    layout_mode: LayoutMode

    def contains_text(self, substring: str) -> bool:
        """Returns True if `substring` appears anywhere in the stripped text lines."""
        return any(substring in line for line in self.lines)


class TUIEventType(str, Enum):
    """Asynchronous event categories dispatched through TUIApp's event bus."""

    CHAT_TOKEN = "CHAT_TOKEN"
    CHAT_MESSAGE = "CHAT_MESSAGE"
    SWARM_UPDATE = "SWARM_UPDATE"
    DIFF_UPDATE = "DIFF_UPDATE"
    PROCESS_LOG = "PROCESS_LOG"
    HUD_UPDATE = "HUD_UPDATE"
    KEY_PRESS = "KEY_PRESS"
    RESIZE = "RESIZE"
    STOP = "STOP"


@dataclass(frozen=True)
class TUIEvent:
    """Immutable envelope pushed onto `TUIApp`'s non-blocking `asyncio.Queue`."""

    event_type: TUIEventType
    payload: Any = None


class ChatPane:
    """
    Left Pane: Renders streaming conversational Markdown, the hierarchical
    Multi-Agent Swarm execution tree (`Lead` -> `Worker` -> `QA`), and the prompt composer.
    """

    def __init__(self, max_messages: int = 500) -> None:
        self.max_messages = max_messages
        self.messages: Deque[ChatMessage] = deque(maxlen=max_messages)
        self.swarm_root: Optional[SwarmNode] = None
        self.input_buffer: str = ""
        self.scroll_offset: int = 0
        self.focused: bool = True
        self._dirty: bool = True

    def append_message(self, role: str, content: str, is_streaming: bool = False) -> None:
        """Appends a new message turn and resets scroll_offset if auto-scroll is active."""
        ...

    def stream_token(self, token_chunk: str) -> None:
        """Appends `token_chunk` to the active streaming assistant message in O(1) time."""
        ...

    def update_swarm_tree(self, root_node: SwarmNode) -> None:
        """Updates the live hierarchical sub-agent execution tree and marks the pane dirty."""
        ...

    def scroll(self, delta_lines: int) -> None:
        """Adjusts `scroll_offset` clamped to `[0, max_scroll]`."""
        ...

    def render(self, width: int, height: int) -> RenderableType:
        """Builds a bordered Rich `Panel` containing the Swarm `Tree`, chat history, and prompt bar."""
        ...


class DiffViewerPane:
    """
    Top-Right Pane: Displays real-time syntax-highlighted unified diffs and
    Wave 1 Tree-Sitter AST `SymbolDelta` outlines for modified worktree files.
    """

    def __init__(self) -> None:
        self.view_mode: DiffViewMode = DiffViewMode.UNIFIED_DIFF
        self.file_diffs: Dict[str, str] = {}
        self.symbol_deltas: Dict[str, SymbolDelta] = {}
        self.active_file: Optional[str] = None
        self.scroll_offset: int = 0
        self.focused: bool = False
        self._dirty: bool = True

    def update_diff(
        self,
        file_path: str,
        unified_diff: str,
        symbol_delta: Optional[SymbolDelta] = None,
    ) -> None:
        """Registers or updates the unified diff and AST symbol delta for `file_path`."""
        ...

    def toggle_view_mode(self) -> DiffViewMode:
        """Switches between `DiffViewMode.UNIFIED_DIFF` and `DiffViewMode.AST_OUTLINE`."""
        ...

    def cycle_file(self, step: int = 1) -> Optional[str]:
        """Advances `active_file` forward (`+1`) or backward (`-1`) across modified files."""
        ...

    def scroll(self, delta_lines: int) -> None:
        """Scrolls the active diff or AST outline vertically."""
        ...

    def render(self, width: int, height: int) -> RenderableType:
        """Renders syntax-highlighted unified diff hunks or a Tree-Sitter AST delta table."""
        ...


class ProcessLogPane:
    """
    Bottom-Right Pane: Streams non-blocking ring-buffered output from Wave 2
    Pytest TDD runners, Wave 4.2 CIWatcher jobs, and MCP servers.
    """

    def __init__(self, max_lines: int = 2000) -> None:
        self.max_lines = max_lines
        self.entries: Deque[LogEntry] = deque(maxlen=max_lines)
        self.active_filter: LogSource = LogSource.ALL
        self.scroll_offset: int = 0
        self.focused: bool = False
        self._dirty: bool = True

    def append_log(self, source: LogSource, line: str, is_error: bool = False) -> None:
        """Appends a sanitized log line to the ring buffer and marks the pane dirty."""
        ...

    def set_filter(self, source: LogSource) -> None:
        """Filters visible log entries by `LogSource` (`ALL`, `TDD`, `MCP`, `CI`, `SYSTEM`)."""
        ...

    def clear(self) -> None:
        """Clears all buffered log entries and resets `scroll_offset`."""
        ...

    def scroll(self, delta_lines: int) -> None:
        """Scrolls the filtered log viewport vertically."""
        ...

    def render(self, width: int, height: int) -> RenderableType:
        """Renders a bordered Rich `Panel` with source badges and pytest failure highlighting."""
        ...


class StatusHUD:
    """
    Bottom Status HUD: 2-line persistent telemetry bar displaying active ModelEndpoint,
    W4.1 Circuit Breaker health snapshot, session token/cost counters, and W3.1 Voice VAD status.
    """

    def __init__(self) -> None:
        self.active_endpoint: Optional[ModelEndpoint] = None
        self.health_snapshot: Optional[EndpointHealthSnapshot] = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.estimated_cost_usd: float = 0.0
        self.active_swarm_workers: int = 0
        self.vad_listening: bool = False
        self._dirty: bool = True

    def update_endpoint_health(self, snapshot: EndpointHealthSnapshot) -> None:
        """Updates the active model endpoint and circuit breaker state badge (`CLOSED`/`OPEN`/`HALF_OPEN`)."""
        ...

    def update_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """Updates cumulative session token counters and estimated USD expenditure."""
        ...

    def set_vad_status(self, listening: bool) -> None:
        """Updates the Voice Activity Detection (VAD) microphone indicator."""
        ...

    def render(self, width: int) -> RenderableType:
        """Renders the compact 2-row telemetry grid formatted for `width` columns."""
        ...


class TUILayoutManager:
    """
    Coordinates spatial geometry, responsive breakpoint adaptation, focus cycling,
    and Rich `Layout` composition across all four viewports.
    """

    def __init__(
        self,
        chat_pane: ChatPane,
        diff_pane: DiffViewerPane,
        log_pane: ProcessLogPane,
        status_hud: StatusHUD,
        dimensions: Optional[TerminalDimensions] = None,
    ) -> None:
        self.chat_pane = chat_pane
        self.diff_pane = diff_pane
        self.log_pane = log_pane
        self.status_hud = status_hud
        self.dimensions = dimensions or TerminalDimensions()
        self.focused_pane: FocusPane = FocusPane.CHAT

    def resize(self, columns: int, rows: int) -> LayoutMode:
        """Updates terminal dimensions and returns the active responsive `LayoutMode`."""
        ...

    def cycle_focus(self, reverse: bool = False) -> FocusPane:
        """Cycles keyboard focus across `CHAT` -> `DIFF` -> `LOGS` (or reverse on `Shift+Tab`)."""
        ...

    def build_layout(self) -> Layout:
        """Constructs and populates the Rich `Layout` tree according to `self.dimensions.layout_mode`."""
        ...

    def render_frame(self) -> VirtualScreenBuffer:
        """
        Renders the composed layout into a deterministic `VirtualScreenBuffer` matching
        `self.dimensions` for terminal output or headless unit test assertions.
        """
        ...


class TUIApp:
    """
    Top-level asynchronous Fullscreen TUI application controller managing the
    alternate screen buffer lifecycle, 60 FPS event coalescing loop, keyboard
    event dispatch, and seamless `/tui` switching with the standard REPL.
    """

    def __init__(
        self,
        layout_manager: Optional[TUILayoutManager] = None,
        on_submit_prompt: Optional[Callable[[str], Any]] = None,
        refresh_rate_hz: float = 60.0,
        output_writer: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.chat_pane = ChatPane()
        self.diff_pane = DiffViewerPane()
        self.log_pane = ProcessLogPane()
        self.status_hud = StatusHUD()
        self.layout_manager = layout_manager or TUILayoutManager(
            chat_pane=self.chat_pane,
            diff_pane=self.diff_pane,
            log_pane=self.log_pane,
            status_hud=self.status_hud,
        )
        self.on_submit_prompt = on_submit_prompt
        self.frame_interval: float = 1.0 / max(refresh_rate_hz, 1.0)
        self._output_writer = output_writer
        self._event_queue: asyncio.Queue[TUIEvent] = asyncio.Queue()
        self.is_running: bool = False
        self.in_alternate_screen: bool = False

    def post_event(self, event: TUIEvent) -> None:
        """Thread-safe / async-safe non-blocking enqueue of a `TUIEvent`."""
        ...

    def handle_key(self, key: str) -> Optional[str]:
        """
        Dispatches a single normalized keystroke (`Tab`, `Shift+Tab`, `Up`, `Down`,
        `Ctrl+D`, `Ctrl+L`, `Enter`, `/tui`, or printable char) to the focused pane.
        Returns a command string (e.g. `"/tui"`) if a mode transition was triggered.
        """
        ...

    async def stream_subprocess_logs(
        self,
        command: Sequence[str],
        source: LogSource = LogSource.TDD,
        cwd: Optional[str] = None,
    ) -> int:
        """
        Spawns an async subprocess (e.g., `pytest`) and streams stdout/stderr lines
        into `ProcessLogPane` via `_event_queue` without blocking LLM token rendering.
        """
        ...

    async def run(self) -> None:
        """Enters alternate screen (`\\x1b[?1049h`) and runs the 60 FPS event loop until stopped."""
        ...

    async def stop(self) -> None:
        """Flushes pending frames, exits alternate screen (`\\x1b[?1049l`), and restores REPL state."""
        ...
```

---

### 6. Exhaustive Unit & Headless Testing Strategy (`tests/unit/test_tui.py`)

All tests for W4.3 reside in `tests/unit/test_tui.py`. Because CI runners and automated test environments execute without an interactive PTY (`isatty() == False`), the test suite validates 100% of layout geometry, rendering, async event coalescing, and keyboard dispatch **headlessly** via `VirtualScreenBuffer` and injected `output_writer` / `TerminalDimensions` fixtures.

#### 6.1 Headless Fixture Architecture

- **`HeadlessScreenCapture`**: Captures every `VirtualScreenBuffer` frame and raw ANSI escape sequence (`\x1b[?1049h` enter / `\x1b[?1049l` exit) emitted by `TUIApp` into an inspectable list without writing to `sys.stdout`.
- **`SyntheticEventProducer`**: Pushes concurrent batches of `TUIEventType.CHAT_TOKEN`, `DIFF_UPDATE`, `PROCESS_LOG`, and `HUD_UPDATE` events onto `TUIApp._event_queue` to verify non-blocking coalescing under simulated multi-agent load.

#### 6.2 Core Test Scenarios in `tests/unit/test_tui.py`

1. **Responsive Layout Geometry & Terminal Resize (`test_layout_manager_responsive_breakpoints`)**
   - Instantiate `TUILayoutManager` at `TerminalDimensions(columns=140, rows=45)` and assert `render_frame()` produces `LayoutMode.STANDARD_SPLIT` with exact `140x45` character dimensions and all four viewports (`ChatPane`, `DiffViewerPane`, `ProcessLogPane`, `StatusHUD`) visible in `VirtualScreenBuffer.lines`.
   - Trigger `resize(90, 30)` and assert transition to `LayoutMode.COMPACT_VERTICAL`.
   - Trigger `resize(72, 16)` and assert graceful degradation to `LayoutMode.MINIMAL_SAFE` without raising layout division or truncation errors.

2. **Streaming Chat & Multi-Agent Swarm Tree Rendering (`test_chat_pane_streaming_and_swarm_tree`)**
   - Populate `ChatPane` with a 3-node `SwarmNode` hierarchy (`Lead` $\rightarrow$ `Worker-1 (feat/cb)` $\rightarrow$ `QA-Verifier`) and stream 10 incremental `CHAT_TOKEN` chunks (`["Implemented ", "circuit ", "breaker..."]`).
   - Render `VirtualScreenBuffer` and assert the sub-agent tree statuses, worktree branch labels, and concatenated assistant message appear intact with zero duplicated intermediate lines.

3. **Real-Time Unified Diff & AST Symbol Outline Toggling (`test_diff_viewer_pane_modes_and_file_cycling`)**
   - Feed two modified files (`src/cli_agent/tui.py` and `src/cli_agent/pr.py`) with unified diff hunks and `SymbolDelta(added_symbols=("TUIApp", "StatusHUD"), modified_symbols=("render_frame",))` into `DiffViewerPane`.
   - Verify default `DiffViewMode.UNIFIED_DIFF` renders `+` / `-` diff lines in the top-right viewport.
   - Dispatch `Ctrl+D` (`toggle_view_mode()`) and assert the viewport switches to `DiffViewMode.AST_OUTLINE`, displaying `+ TUIApp`, `+ StatusHUD`, and `~ render_frame`.
   - Dispatch `]` (`cycle_file(1)`) and assert `active_file` advances from `src/cli_agent/tui.py` to `src/cli_agent/pr.py`.

4. **Non-Blocking Async Subprocess Log Streaming & Ring Buffer (`test_process_log_pane_ring_buffer_and_filtering`)**
   - Set `ProcessLogPane(max_lines=50)` and stream 120 interleaved `LogSource.TDD` (`pytest`) and `LogSource.MCP` lines.
   - Assert ring buffer length is strictly capped at `50` (`len(log_pane.entries) == 50`) and oldest lines are evicted FIFO.
   - Apply `set_filter(LogSource.TDD)` and verify `VirtualScreenBuffer` excludes `LogSource.MCP` lines while highlighting `FAILED tests/unit/test_tui.py` in error styling.

5. **Status HUD Circuit Breaker & Telemetry Snapshot (`test_status_hud_circuit_breaker_states`)**
   - Pass an `EndpointHealthSnapshot` with `state=CircuitState.CLOSED` for `ModelEndpoint("anthropic/claude-sonnet-4-6", "us-east5", tier=1)` and assert `StatusHUD` renders `[CLOSED ●]` alongside token counters (`14.2k in / 3.8k out`).
   - Transition snapshot to `CircuitState.OPEN` with `cooldown_remaining_seconds=42.5` and `vad_listening=True`; assert `StatusHUD` immediately reflects `[OPEN ● 42.5s]` and `[MIC ACTIVE ●]`.

6. **Keyboard Event Dispatch, Focus Cycling & `/tui` Mode Toggle (`test_keyboard_dispatch_and_repl_toggle`)**
   - Verify pressing `Tab` cycles `layout_manager.focused_pane` deterministically through ` FocusPane.CHAT` $\rightarrow$ `FocusPane.DIFF` $\rightarrow$ `FocusPane.LOGS` $\rightarrow$ `FocusPane.CHAT`, and `Shift+Tab` reverses the cycle.
   - Verify `Up`/`Down` arrow keys scroll only the currently focused pane (`chat_pane.scroll_offset` vs. `diff_pane.scroll_offset` vs. `log_pane.scroll_offset`).
   - Type `"/tui"` followed by `"Enter"` in `ChatPane`; assert `TUIApp.handle_key("Enter")` triggers `TUIApp.stop()`, emits the alternate-screen exit sequence `\x1b[?1049l` to `HeadlessScreenCapture`, and preserves all buffered messages and diffs for the standard REPL.
