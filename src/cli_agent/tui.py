"""
Fullscreen Split-Pane Terminal User Interface (Wave 4 Milestone 3).

Provides a concurrent spatial terminal workspace with dedicated viewports for:
1. Left Pane: Streaming Chat & Hierarchical Multi-Agent Swarm Tree
2. Top-Right Pane: Real-Time AST & Unified Diff Viewer
3. Bottom-Right Pane: Ring-Buffered Subprocess / TDD / MCP Logs
4. Bottom Status HUD: Model, Circuit Breaker State, Tokens, and Voice VAD
"""

from __future__ import annotations

import asyncio
import io
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .circuit_breaker import CircuitState, EndpointHealthSnapshot, ModelEndpoint
from .pr import SymbolDelta


class FocusPane(StrEnum):
    """Viewports eligible for keyboard focus and scrollback."""

    CHAT = "CHAT"
    DIFF = "DIFF"
    LOGS = "LOGS"


class DiffViewMode(StrEnum):
    """Display modes for the Top-Right Diff Viewer."""

    UNIFIED_DIFF = "UNIFIED_DIFF"
    AST_OUTLINE = "AST_OUTLINE"


class LogSource(StrEnum):
    """Categorized origin of subprocess telemetry lines."""

    ALL = "ALL"
    TDD = "TDD"
    MCP = "MCP"
    CI = "CI"


class LayoutMode(StrEnum):
    """Responsive terminal layout geometries based on viewport breakpoints."""

    STANDARD_SPLIT = "STANDARD_SPLIT"
    COMPACT_VERTICAL = "COMPACT_VERTICAL"
    MINIMAL_SAFE = "MINIMAL_SAFE"


class TUIEventType(StrEnum):
    """Asynchronous events dispatched to the TUI event queue."""

    CHAT_TOKEN = "CHAT_TOKEN"
    SWARM_UPDATE = "SWARM_UPDATE"
    DIFF_UPDATE = "DIFF_UPDATE"
    PROCESS_LOG = "PROCESS_LOG"
    HUD_UPDATE = "HUD_UPDATE"


@dataclass
class TerminalDimensions:
    """Terminal width (columns) and height (rows)."""

    columns: int = 120
    rows: int = 35


@dataclass
class VirtualScreenBuffer:
    """Headless snapshot of rendered character lines and text."""

    columns: int
    rows: int
    raw_text: str
    lines: list[str] = field(default_factory=list)


@dataclass
class SwarmNode:
    """Hierarchical node representing an agent inside the multi-agent swarm tree."""

    agent_id: str
    role: str  # "Lead", "Worker", "QA"
    status: str  # "IDLE", "RUNNING", "COMPLETED", "FAILED"
    active_tool: str | None = None
    worktree_branch: str | None = None
    children: list[SwarmNode] = field(default_factory=list)


@dataclass
class TUIEvent:
    """Immutable message emitted by concurrent worker tasks into TUIApp queue."""

    event_type: TUIEventType
    payload: dict[str, Any] = field(default_factory=dict)


class ChatPane:
    """Left viewport displaying Markdown conversational stream and Swarm execution tree."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.streaming_tokens: list[str] = []
        self.swarm_root: SwarmNode | None = None
        self.scroll_offset: int = 0
        self.focused: bool = True

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def append_streaming_token(self, token: str) -> None:
        self.streaming_tokens.append(token)

    def commit_streaming_turn(self, role: str = "assistant") -> None:
        if self.streaming_tokens:
            full_text = "".join(self.streaming_tokens)
            self.messages.append({"role": role, "content": full_text})
            self.streaming_tokens.clear()

    def update_swarm(self, root: SwarmNode) -> None:
        self.swarm_root = root

    def render(self) -> Panel:
        body = Text()

        # Swarm tree rendering if active
        if self.swarm_root:
            tree = Tree(f"[bold cyan]{self.swarm_root.role}[/bold cyan] ({self.swarm_root.status})")
            for child in self.swarm_root.children:
                branch_label = f" [{child.worktree_branch}]" if child.worktree_branch else ""
                tool_label = f" -> {child.active_tool}" if child.active_tool else ""
                tree.add(
                    f"[{child.role}] {child.agent_id} ({child.status}){branch_label}{tool_label}"
                )
            body.append("=== Multi-Agent Swarm Hierarchy ===\n", style="bold yellow")
            body.append(Text.from_markup(f"Active Root: {self.swarm_root.agent_id}\n"))

        # Render recent messages
        for msg in self.messages[-6:]:
            role_style = "bold green" if msg["role"] == "user" else "bold blue"
            body.append(f"{msg['role'].upper()}: ", style=role_style)
            body.append(f"{msg['content']}\n\n")

        # Render currently streaming chunk
        if self.streaming_tokens:
            body.append("ASSISTANT (streaming): ", style="bold magenta")
            body.append("".join(self.streaming_tokens))

        title = " [bold]Chat & Swarm Execution[/bold] "
        border_style = "bright_cyan" if self.focused else "dim"
        return Panel(body, title=title, border_style=border_style)


class DiffViewerPane:
    """Top-Right viewport for live unified diffs and AST symbol outlines."""

    def __init__(self) -> None:
        self.files: list[str] = []
        self.diffs: dict[str, str] = {}
        self.symbol_deltas: dict[str, SymbolDelta] = {}
        self.active_file_idx: int = 0
        self.mode: DiffViewMode = DiffViewMode.UNIFIED_DIFF
        self.scroll_offset: int = 0
        self.focused: bool = False

    def update_file_diff(
        self,
        file_path: str,
        diff_text: str,
        symbol_delta: SymbolDelta | None = None,
    ) -> None:
        if file_path not in self.files:
            self.files.append(file_path)
        self.diffs[file_path] = diff_text
        if symbol_delta:
            self.symbol_deltas[file_path] = symbol_delta

    def cycle_file(self, delta: int = 1) -> None:
        if not self.files:
            return
        self.active_file_idx = (self.active_file_idx + delta) % len(self.files)

    def toggle_view_mode(self) -> DiffViewMode:
        if self.mode == DiffViewMode.UNIFIED_DIFF:
            self.mode = DiffViewMode.AST_OUTLINE
        else:
            self.mode = DiffViewMode.UNIFIED_DIFF
        return self.mode

    @property
    def active_file(self) -> str | None:
        if not self.files:
            return None
        return self.files[self.active_file_idx]

    def render(self) -> Panel:
        active_f = self.active_file
        if not active_f:
            content = Text("No modified files in active worktree.", style="dim italic")
            return Panel(
                content,
                title=" Live Diff & AST Inspector ",
                border_style="bright_cyan" if self.focused else "dim",
            )

        body_renderable: Text | Syntax
        if self.mode == DiffViewMode.AST_OUTLINE:
            delta = self.symbol_deltas.get(active_f)
            outline = Text()
            outline.append(f"File: {active_f} (AST Symbol Outline)\n\n", style="bold yellow")
            if delta:
                for sym in delta.added_symbols:
                    outline.append(f"+ {sym} (Added)\n", style="bold green")
                for sym in delta.modified_symbols:
                    outline.append(f"~ {sym} (Modified)\n", style="bold yellow")
                for sym in delta.removed_symbols:
                    outline.append(f"- {sym} (Removed)\n", style="bold red")
            else:
                outline.append("No AST symbol delta computed for this file.\n", style="dim")
            body_renderable = outline
        else:
            raw_diff = self.diffs.get(active_f, "")
            body_renderable = Syntax(raw_diff, "diff", theme="monokai", line_numbers=True)

        mode_label = "[DIFF]" if self.mode == DiffViewMode.UNIFIED_DIFF else "[AST]"
        title = f" {mode_label} {active_f} ({self.active_file_idx + 1}/{len(self.files)}) "
        border_style = "bright_cyan" if self.focused else "dim"
        return Panel(body_renderable, title=title, border_style=border_style)


class ProcessLogPane:
    """Bottom-Right viewport buffering background subprocess, pytest, and MCP streams."""

    def __init__(self, max_lines: int = 2000) -> None:
        self.max_lines = max_lines
        self.entries: deque[tuple[LogSource, str]] = deque(maxlen=max_lines)
        self.active_filter: LogSource = LogSource.ALL
        self.scroll_offset: int = 0
        self.focused: bool = False

    def append_log(self, source: LogSource, line: str) -> None:
        clean = line.rstrip("\r\n")
        self.entries.append((source, clean))

    def set_filter(self, source: LogSource) -> None:
        self.active_filter = source

    def clear(self) -> None:
        self.entries.clear()

    def render(self) -> Panel:
        body = Text()
        filtered = [
            (src, txt)
            for src, txt in self.entries
            if self.active_filter == LogSource.ALL or src == self.active_filter
        ]

        # Display last 15 lines in viewport
        for src, txt in filtered[-15:]:
            tag_style = "dim cyan"
            if src == LogSource.TDD:
                tag_style = "bold magenta"
            elif src == LogSource.MCP:
                tag_style = "bold green"
            elif src == LogSource.CI:
                tag_style = "bold yellow"

            line_style = "normal"
            if "FAIL" in txt or "ERROR" in txt:
                line_style = "bold red"
            elif "PASSED" in txt or "SUCCESS" in txt:
                line_style = "bold green"

            body.append(f"[{src.value}] ", style=tag_style)
            body.append(f"{txt}\n", style=line_style)

        if not filtered:
            body.append("No log output captured yet.", style="dim italic")

        title = f" Subprocess & TDD Logs [{self.active_filter.value}] "
        border_style = "bright_cyan" if self.focused else "dim"
        return Panel(body, title=title, border_style=border_style)


class StatusHUD:
    """Docked bottom telemetry bar displaying Circuit Breaker, Model, and VAD state."""

    def __init__(self) -> None:
        self.endpoint: ModelEndpoint = ModelEndpoint("gemini-3.8-flash", "global", 3)
        self.circuit_snapshot: EndpointHealthSnapshot = EndpointHealthSnapshot(
            endpoint=self.endpoint,
            state=CircuitState.CLOSED,
            consecutive_failures=0,
            cooldown_remaining_seconds=0.0,
        )
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.vad_listening: bool = False
        self.active_swarms: int = 0

    def update_metrics(
        self,
        snapshot: EndpointHealthSnapshot,
        tokens_in: int = 0,
        tokens_out: int = 0,
        vad_listening: bool = False,
        active_swarms: int = 0,
    ) -> None:
        self.circuit_snapshot = snapshot
        self.endpoint = snapshot.endpoint
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.vad_listening = vad_listening
        self.active_swarms = active_swarms

    def render(self) -> Table:
        table = Table.grid(expand=True)
        table.add_column(justify="left", ratio=3)
        table.add_column(justify="center", ratio=2)
        table.add_column(justify="right", ratio=2)

        # Circuit status styling
        cb_state = self.circuit_snapshot.state
        if cb_state == CircuitState.CLOSED:
            cb_text = Text("[CLOSED ●]", style="bold green")
        elif cb_state == CircuitState.HALF_OPEN:
            cb_text = Text("[HALF_OPEN ◌]", style="bold yellow")
        else:
            sec = self.circuit_snapshot.cooldown_remaining_seconds
            cb_text = Text(f"[OPEN ● {sec:.1f}s]", style="bold red")

        mic_text = (
            Text("[MIC ACTIVE ●]", style="bold green")
            if self.vad_listening
            else Text("[MUTED ○]", style="dim")
        )

        model_info = Text.from_markup(
            f"Model: [bold]{self.endpoint.model_id}[/bold] ({self.endpoint.region}) Tier {self.endpoint.tier} | CB: "
        )
        model_info.append_text(cb_text)

        tokens_info = Text(
            f"Tokens: {self.tokens_in / 1000:.1f}k in / {self.tokens_out / 1000:.1f}k out | Swarm: {self.active_swarms}",
            style="cyan",
        )

        table.add_row(model_info, tokens_info, mic_text)
        return table


class TUILayoutManager:
    """Orchestrates responsive layout breakpoints and manages keyboard focus."""

    def __init__(
        self,
        chat_pane: ChatPane | None = None,
        diff_pane: DiffViewerPane | None = None,
        log_pane: ProcessLogPane | None = None,
        status_hud: StatusHUD | None = None,
        dimensions: TerminalDimensions | None = None,
    ) -> None:
        self.chat_pane = chat_pane or ChatPane()
        self.diff_pane = diff_pane or DiffViewerPane()
        self.log_pane = log_pane or ProcessLogPane()
        self.status_hud = status_hud or StatusHUD()
        self.dimensions = dimensions or TerminalDimensions(120, 35)
        self.focused_pane: FocusPane = FocusPane.CHAT
        self._sync_focus()

    def resize(self, columns: int, rows: int) -> None:
        self.dimensions = TerminalDimensions(columns=columns, rows=rows)

    def determine_layout_mode(self) -> LayoutMode:
        cols, rows = self.dimensions.columns, self.dimensions.rows
        if cols >= 100 and rows >= 24:
            return LayoutMode.STANDARD_SPLIT
        if cols >= 80 and rows >= 18:
            return LayoutMode.COMPACT_VERTICAL
        return LayoutMode.MINIMAL_SAFE

    def cycle_focus(self, reverse: bool = False) -> FocusPane:
        cycle = [FocusPane.CHAT, FocusPane.DIFF, FocusPane.LOGS]
        if reverse:
            cycle.reverse()
        idx = cycle.index(self.focused_pane)
        self.focused_pane = cycle[(idx + 1) % len(cycle)]
        self._sync_focus()
        return self.focused_pane

    def _sync_focus(self) -> None:
        self.chat_pane.focused = self.focused_pane == FocusPane.CHAT
        self.diff_pane.focused = self.focused_pane == FocusPane.DIFF
        self.log_pane.focused = self.focused_pane == FocusPane.LOGS

    def render_frame(self) -> VirtualScreenBuffer:
        mode = self.determine_layout_mode()
        root = Layout(name="root")

        # Split into main view area and bottom HUD
        root.split(
            Layout(name="main", ratio=1),
            Layout(name="hud", size=3),
        )
        root["hud"].update(Panel(self.status_hud.render(), border_style="dim"))

        if mode == LayoutMode.STANDARD_SPLIT:
            root["main"].split_row(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1),
            )
            root["right"].split(
                Layout(name="diff", ratio=1),
                Layout(name="logs", ratio=1),
            )
            root["left"].update(self.chat_pane.render())
            root["diff"].update(self.diff_pane.render())
            root["logs"].update(self.log_pane.render())

        elif mode == LayoutMode.COMPACT_VERTICAL:
            root["main"].split(
                Layout(name="top", ratio=2),
                Layout(name="bottom", ratio=1),
            )
            root["top"].update(self.chat_pane.render())
            root["bottom"].update(self.log_pane.render())

        else:  # MINIMAL_SAFE
            root["main"].update(self.chat_pane.render())

        # Render to in-memory console buffer
        buf = io.StringIO()
        console = Console(
            file=buf,
            width=self.dimensions.columns,
            height=self.dimensions.rows,
            force_terminal=True,
            color_system="standard",
        )
        console.print(root)
        output_str = buf.getvalue()
        lines = output_str.splitlines()

        return VirtualScreenBuffer(
            columns=self.dimensions.columns,
            rows=self.dimensions.rows,
            raw_text=output_str,
            lines=lines,
        )


class TUIApp:
    """Non-blocking terminal application driver managing the event loop and key dispatch."""

    def __init__(
        self,
        layout_manager: TUILayoutManager | None = None,
        on_submit_prompt: Callable[[str], Any] | None = None,
        output_writer: Callable[[str], None] | None = None,
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
        self._output_writer = output_writer or (lambda _: None)
        self._event_queue: asyncio.Queue[TUIEvent] = asyncio.Queue()
        self.is_running: bool = False
        self.in_alternate_screen: bool = False

    def post_event(self, event: TUIEvent) -> None:
        self._event_queue.put_nowait(event)

    def handle_key(self, key: str) -> str | None:
        """Dispatches key to focused pane; returns command (e.g. '/tui') if triggered."""
        if key == "Tab":
            self.layout_manager.cycle_focus(reverse=False)
            return None
        if key in ("Shift+Tab", "s-tab"):
            self.layout_manager.cycle_focus(reverse=True)
            return None
        if key == "Ctrl+D" or key == "c-d":
            self.diff_pane.toggle_view_mode()
            return None
        if key == "Ctrl+L" or key == "c-l":
            self.log_pane.clear()
            return None
        if key == "]":
            self.diff_pane.cycle_file(1)
            return None
        if key == "[":
            self.diff_pane.cycle_file(-1)
            return None
        if key in ("Up", "up"):
            if self.layout_manager.focused_pane == FocusPane.CHAT:
                self.chat_pane.scroll_offset += 1
            elif self.layout_manager.focused_pane == FocusPane.DIFF:
                self.diff_pane.scroll_offset += 1
            else:
                self.log_pane.scroll_offset += 1
            return None
        if key in ("Down", "down"):
            if self.layout_manager.focused_pane == FocusPane.CHAT:
                self.chat_pane.scroll_offset = max(0, self.chat_pane.scroll_offset - 1)
            elif self.layout_manager.focused_pane == FocusPane.DIFF:
                self.diff_pane.scroll_offset = max(0, self.diff_pane.scroll_offset - 1)
            else:
                self.log_pane.scroll_offset = max(0, self.log_pane.scroll_offset - 1)
            return None
        if key == "/tui":
            self.stop()
            return "/tui"
        return None

    def start(self) -> None:
        self.is_running = True
        self.in_alternate_screen = True
        self._output_writer("\x1b[?1049h")  # Enter alternate screen buffer

    def stop(self) -> None:
        self.is_running = False
        self.in_alternate_screen = False
        self._output_writer("\x1b[?1049l")  # Exit alternate screen buffer

    def run(self) -> VirtualScreenBuffer:
        """Boots the 4-zone fullscreen split-pane terminal workspace and renders a frame."""
        self.start()
        try:
            frame = self.layout_manager.render_frame()
            self._output_writer(frame.raw_text)
            return frame
        finally:
            self.stop()

    async def stream_subprocess_logs(
        self,
        command: Sequence[str],
        source: LogSource = LogSource.TDD,
        cwd: Path | None = None,
    ) -> int:
        """Runs async subprocess and feeds stdout/stderr into ProcessLogPane."""
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(stream: asyncio.StreamReader | None) -> None:
            if not stream:
                return
            while not stream.at_eof():
                line = await stream.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace")
                    self.log_pane.append_log(source, decoded)

        await asyncio.gather(
            read_stream(proc.stdout),
            read_stream(proc.stderr),
        )
        return await proc.wait()
