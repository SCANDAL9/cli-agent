"""
Main CLI entrypoint for cli-agent.
Provides an interactive terminal REPL and one-shot command execution with Rich formatting,
prompt_toolkit history/autocompletion, and real-time SSE streaming.
"""

import asyncio
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .circuit_breaker import QuotaCircuitBreaker
from .clipboard import (
    ImageAttachment,
    delete_kitty_image_sequence,
    is_kitty_terminal,
    load_image_from_path,
    read_clipboard_image,
    read_clipboard_text,
    render_kitty_image_preview,
)
from .code_graph import CodeGraphIndexer
from .config import DEFAULT_HISTORY_FILE, load_config, save_model_to_config
from .context import compact_history, extract_session_learnings
from .doctor import run_diagnostics
from .mcp import MCPRegistry, set_active_mcp_registry
from .memory import MemoryStore, resolve_project_key
from .models import ModelCatalogEntry, fetch_model_garden_catalog
from .pr import CIWatcher, PRSynthesizer, detect_git_host_provider
from .preflight import PreflightEngine, ProjectSelector, ServiceUsageManager
from .search import get_hybrid_search_engine
from .security import RiskLevel, SecurityPolicy
from .session import SessionStore, generate_session_id
from .tdd import AutonomousTDDEngine
from .team import MultiAgentTeam, set_active_team
from .theme import (
    MatugenPalette,
    build_prompt_toolkit_style,
    hex_to_ansi_fg,
    load_matugen_palette,
)
from .tools import set_active_sandbox
from .tui import TUIApp, TUILayoutManager
from .vertex import VertexAgentClient
from .voice import VoiceAgentEngine
from .worktree import DiffViewer, WorktreeManager, WorktreeSession

console = Console()

SLASH_COMMANDS = [
    "/auto",
    "/sessions",
    "/session",
    "/resume",
    "/memories",
    "/compact",
    "/doctor",
    "/setup",
    "/model",
    "/models",
    "/index",
    "/search",
    "/sandbox",
    "/team",
    "/mcp",
    "/tdd",
    "/fix",
    "/pr",
    "/tui",
    "/voice",
    "/clear",
    "/help",
    "/exit",
    "/quit",
]

COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/auto": "Toggle autonomous mode (auto-approve safe actions)",
    "/sessions": "List persistent conversation sessions",
    "/session": "Show active session details and turn count",
    "/resume": "Resume a previous conversation session",
    "/memories": "List persistent SQLite knowledge memories",
    "/compact": "Compact conversation history context",
    "/doctor": "Verify authentication, DB, AST, and health",
    "/setup": "Launch GCP project onboarding wizard",
    "/model": "Switch or inspect active Vertex AI model",
    "/models": "Explore Vertex AI Model Garden catalog",
    "/index": "Re-index workspace code AST symbols",
    "/search": "Hybrid BM25 + dense vector code search",
    "/sandbox": "Manage isolated Git worktree sandbox",
    "/team": "Run autonomous multi-agent swarm (ODD)",
    "/mcp": "Inspect and reload federated MCP servers",
    "/tdd": "Autonomous Red-Green-Refactor healing loop",
    "/fix": "Alias for /tdd autonomous repair",
    "/pr": "Synthesize or create GitHub/GitLab PR/MR",
    "/tui": "Launch fullscreen 4-zone Split-Pane UI",
    "/voice": "Bidirectional Gemini Live voice streaming",
    "/clear": "Clear active conversation history",
    "/help": "Show complete command overview and guide",
    "/exit": "Quit agent and save session learnings",
    "/quit": "Alias for /exit",
}


class AgentInputCompleter(Completer):
    """
    Intelligent dual-mode autocompleter:
    - '/' triggers slash command suggestions with metadata descriptions.
    - '@' triggers workspace file path autocompletion.
    - Falls back to general command autocompletion when appropriate.
    """

    def __init__(self, slash_completer: WordCompleter, path_completer: PathCompleter) -> None:
        self.slash_completer = slash_completer
        self.path_completer = path_completer

    def get_completions(self, document: Document, complete_event: Any) -> Any:
        text = document.text_before_cursor
        if text.startswith("/"):
            yield from self.slash_completer.get_completions(document, complete_event)
        elif "@" in text:
            at_idx = text.rfind("@")
            sub_text = text[at_idx + 1 :]
            sub_doc = Document(sub_text, cursor_position=len(sub_text))
            for comp in self.path_completer.get_completions(sub_doc, complete_event):
                yield Completion(
                    comp.text,
                    start_position=comp.start_position,
                    display=comp.display,
                    display_meta="file",
                )
        else:
            yield from self.slash_completer.get_completions(document, complete_event)


_slash_word_completer = WordCompleter(
    SLASH_COMMANDS, ignore_case=True, sentence=True, meta_dict=COMMAND_DESCRIPTIONS
)
_path_completer = PathCompleter(expanduser=True)
command_completer: Completer = AgentInputCompleter(
    slash_completer=_slash_word_completer, path_completer=_path_completer
)

matugen_palette: MatugenPalette = load_matugen_palette()
pt_style = build_prompt_toolkit_style(matugen_palette)


def format_bottom_toolbar(
    terminal_width: int,
    auto_approve: bool,
    model_name: str,
    image_count: int = 0,
) -> FormattedText:
    """Formats the REPL bottom toolbar with terminal-adaptive styling (no inverse contrast)."""
    term_w = max(20, terminal_width)
    bot_border = f"{'─' * term_w}\n"
    mode_str = "auto" if auto_approve else "default"
    mode_class = (
        "class:bottom-toolbar.badge"
        if auto_approve
        else "class:bottom-toolbar.badge.default"
    )
    model_disp = _format_model_badge(model_name)
    effort_disp = _get_model_effort(model_name)
    img_prefix = f"🖼 {image_count} img • " if image_count > 0 else ""
    status_text = f"{img_prefix}{mode_str} • {model_disp} • {effort_disp}"
    pad = " " * max(0, term_w - len(status_text) - 1)
    tokens: list[tuple[str, str]] = [
        ("class:prompt.box", bot_border),
        ("class:bottom-toolbar.text", pad),
    ]
    if img_prefix:
        tokens.append(("class:bottom-toolbar.model", img_prefix))
    tokens.extend(
        [
            (mode_class, mode_str),
            ("class:bottom-toolbar.text", " • "),
            ("class:bottom-toolbar.model", model_disp),
            ("class:bottom-toolbar.text", " • "),
            ("class:bottom-toolbar.text", effort_disp),
        ]
    )
    return FormattedText(tokens)


_IMAGE_TOKEN_RE = re.compile(r"\[Image #(\d+)\] ?")


class AlternateScreenWorkspace:
    """
    Isolates the interactive agent session inside the terminal's Alternate Screen Buffer
    (DEC private mode 1049: \\x1b[?1049h on enter, \\x1b[r\\x1b[?1049l on exit).
    Prevents scrolling back into prior shell commands and restores the original terminal cleanly.
    """

    def __init__(self, stream: Any | None = None, enabled: bool | None = None) -> None:
        self._stream = stream or sys.stdout
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._active = False

    def __enter__(self) -> "AlternateScreenWorkspace":
        if self._enabled and not self._active:
            # Enter alternate screen buffer, clear it, and move cursor to home (1,1)
            self._stream.write("\x1b[?1049h\x1b[2J\x1b[H")
            self._stream.flush()
            self._active = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._active:
            # Reset scroll margins and leave alternate screen buffer cleanly
            self._stream.write("\x1b[r\x1b[?1049l")
            self._stream.flush()
            self._active = False


class PinnedFooterManager:
    """
    Keeps the bottom 2 rows (horizontal separator + status/model bar) permanently pinned
    at the bottom of Kitty/ANSI terminals using DECSTBM scroll regions while the agent
    streams responses or executes tools. Always resets scroll margins cleanly on unpin().
    """

    def __init__(self, stream: Any | None = None, enabled: bool | None = None) -> None:
        self._stream = stream or sys.stdout
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._pinned = False
        self._rows = 24
        self._cols = 80

    def _refresh_size(self) -> tuple[int, int]:
        size = shutil.get_terminal_size((80, 24))
        self._cols = max(20, size.columns)
        self._rows = max(6, size.lines)
        return self._cols, self._rows

    def _paint_footer(
        self,
        activity: str,
        auto_approve: bool,
        model_name: str,
        image_count: int = 0,
    ) -> None:
        cols, rows = self._cols, self._rows
        border = "─" * cols
        mode_str = "auto" if auto_approve else "default"
        model_disp = _format_model_badge(model_name)
        effort_disp = _get_model_effort(model_name)
        img_prefix = f"🖼 {image_count} img • " if image_count > 0 else ""
        right_text = f"{img_prefix}{mode_str} • {model_disp} • {effort_disp}"
        left_text = activity.strip()
        max_left = max(0, cols - len(right_text) - 3)
        if len(left_text) > max_left:
            left_text = left_text[: max(0, max_left - 1)] + "…" if max_left > 1 else ""
        pad = " " * max(1, cols - len(left_text) - len(right_text) - 1)

        p = load_matugen_palette()
        if p.is_matugen:
            c_border = hex_to_ansi_fg(p.border)
            c_act = hex_to_ansi_fg(p.primary)
            c_muted = hex_to_ansi_fg(p.muted)
            c_mode = hex_to_ansi_fg(p.accent) if auto_approve else c_muted
            c_model = f"\x1b[1m{hex_to_ansi_fg(p.primary)}"
        else:
            c_border = "\x1b[2m"
            c_act = "\x1b[36m"
            c_muted = "\x1b[2m"
            c_mode = "\x1b[32m" if auto_approve else "\x1b[2m"
            c_model = "\x1b[1m"

        footer_line = (
            f"{c_act}{left_text}\x1b[0m"
            f"{pad}"
            f"{c_muted}{img_prefix}\x1b[0m"
            f"{c_mode}{mode_str}\x1b[0m"
            f"{c_muted} • \x1b[0m"
            f"{c_model}{model_disp}\x1b[0m"
            f"{c_muted} • {effort_disp}\x1b[0m"
        )

        # Save cursor, draw border on row (rows-1) and status on row (rows), restore cursor
        self._stream.write(
            f"\x1b7"
            f"\x1b[{rows - 1};1H\x1b[2K{c_border}{border}\x1b[0m"
            f"\x1b[{rows};1H\x1b[2K{footer_line}"
            f"\x1b8"
        )
        self._stream.flush()

    def pin(
        self,
        activity: str,
        auto_approve: bool,
        model_name: str,
        image_count: int = 0,
    ) -> None:
        """Locks bottom 2 rows with DECSTBM scroll region and paints the pinned footer."""
        if not self._enabled:
            return
        _, rows = self._refresh_size()
        scroll_bottom = max(2, rows - 2)
        # Set top..scroll_bottom scroll region; keep cursor inside scroll region
        self._stream.write(f"\x1b7\x1b[1;{scroll_bottom}r\x1b8")
        self._pinned = True
        self._paint_footer(
            activity=activity,
            auto_approve=auto_approve,
            model_name=model_name,
            image_count=image_count,
        )

    def update(
        self,
        activity: str,
        auto_approve: bool,
        model_name: str,
        image_count: int = 0,
    ) -> None:
        """Updates the pinned footer text without moving the main scroll cursor."""
        if not self._enabled or not self._pinned:
            return
        self._refresh_size()
        self._paint_footer(
            activity=activity,
            auto_approve=auto_approve,
            model_name=model_name,
            image_count=image_count,
        )

    def unpin(self) -> None:
        """Restores full-terminal scroll margins cleanly."""
        if not self._enabled or not self._pinned:
            return
        rows = self._rows
        # Reset DECSTBM to full screen and clear pinned footer rows before prompt_toolkit takes over
        self._stream.write(
            f"\x1b7"
            f"\x1b[{rows - 1};1H\x1b[2K"
            f"\x1b[{rows};1H\x1b[2K"
            f"\x1b[r"
            f"\x1b8"
        )
        self._stream.flush()
        self._pinned = False


def build_input_key_bindings(
    pending_images: list[ImageAttachment],
    enable_kitty_preview: bool = True,
) -> KeyBindings:
    """
    Builds prompt_toolkit key bindings supporting:
    - Ctrl+V: clipboard image paste (with native Kitty Graphics preview) and text fallback
    - Backspace / Delete: atomic deletion of [Image #N] tokens and their pending attachments
    """
    kb = KeyBindings()

    def _trigger_preview(attachment: ImageAttachment) -> None:
        if not enable_kitty_preview or not is_kitty_terminal():
            return
        try:
            run_in_terminal(lambda: render_kitty_image_preview(attachment))
        except Exception:
            pass

    def _trigger_delete_preview(attachment: ImageAttachment) -> None:
        if not enable_kitty_preview or not is_kitty_terminal():
            return
        seq = delete_kitty_image_sequence(9000 + attachment.index)
        if seq:
            try:
                sys.stdout.write(seq)
                sys.stdout.flush()
            except Exception:
                pass

    def _remove_attachment_by_index(img_index: int) -> ImageAttachment | None:
        for i, att in enumerate(pending_images):
            if att.index == img_index:
                return pending_images.pop(i)
        return None

    @kb.add("c-v")
    def _handle_ctrl_v(event: Any) -> None:
        """Ctrl+V pastes images only; plain text paste remains native Linux Ctrl+Shift+V."""
        img_result = read_clipboard_image()
        if img_result is not None:
            mime_type, raw_bytes = img_result
            next_idx = max((img.index for img in pending_images), default=0) + 1
            attachment = ImageAttachment(
                index=next_idx,
                mime_type=mime_type,
                data=raw_bytes,
                source="clipboard",
            )
            pending_images.append(attachment)
            event.current_buffer.insert_text(f"{attachment.token} ")
            _trigger_preview(attachment)
            return

        # Also support pasting an image file path copied from a file manager,
        # but ignore plain text so Ctrl+Shift+V handles text paste natively.
        clip_text = read_clipboard_text()
        if clip_text:
            path_img = load_image_from_path(clip_text)
            if path_img is not None:
                mime_type, raw_bytes = path_img
                next_idx = max((img.index for img in pending_images), default=0) + 1
                attachment = ImageAttachment(
                    index=next_idx,
                    mime_type=mime_type,
                    data=raw_bytes,
                    source=clip_text.strip(),
                )
                pending_images.append(attachment)
                event.current_buffer.insert_text(f"{attachment.token} ")
                _trigger_preview(attachment)

    @kb.add("backspace")
    def _handle_backspace(event: Any) -> None:
        buf = event.current_buffer
        text = buf.text
        pos = buf.cursor_position
        if pos > 0:
            for m in _IMAGE_TOKEN_RE.finditer(text):
                if m.start() < pos <= m.end():
                    img_idx = int(m.group(1))
                    buf.text = text[: m.start()] + text[m.end() :]
                    buf.cursor_position = m.start()
                    removed = _remove_attachment_by_index(img_idx)
                    if removed is not None:
                        _trigger_delete_preview(removed)
                    return
        buf.delete_before_cursor(count=1)

    @kb.add("delete")
    def _handle_delete(event: Any) -> None:
        buf = event.current_buffer
        text = buf.text
        pos = buf.cursor_position
        if pos < len(text):
            for m in _IMAGE_TOKEN_RE.finditer(text):
                if m.start() <= pos < m.end():
                    img_idx = int(m.group(1))
                    buf.text = text[: m.start()] + text[m.end() :]
                    buf.cursor_position = m.start()
                    removed = _remove_attachment_by_index(img_idx)
                    if removed is not None:
                        _trigger_delete_preview(removed)
                    return
        buf.delete(count=1)

    return kb


def _format_model_badge(name: str) -> str:
    mapping = {
        "gemini-3.8-flash": "Gemini 3.8 Flash",
        "gemini-3.1-pro": "Gemini 3.1 Pro",
        "gemini-3.7-flash": "Gemini 3.7 Flash",
        "gemini-4-hydrogen": "Gemini 4 Hydrogen",
        "gemini-4-neon": "Gemini 4 Neon",
        "gemini-4-carbon": "Gemini 4 Carbon",
        "gemini-4-argon": "Gemini 4 Argon",
    }
    return mapping.get(name, name)


def _get_model_effort(name: str) -> str:
    lower = name.lower()
    if "hydrogen" in lower:
        return "low"
    if "neon" in lower:
        return "medium"
    if "carbon" in lower:
        return "high"
    if "argon" in lower:
        return "high"
    if "flash" in lower:
        return "medium"
    if "pro" in lower:
        return "high"
    return "medium"


def print_banner(
    memory_count: int,
    project_key: str = "global",
    model_name: str = "gemini-3.8-flash",
    session_id: str | None = None,
    turn_count: int = 0,
) -> None:
    ws_name = Path(project_key).name if "/" in project_key else project_key
    session_badge = (
        f"Session: [bold yellow]{session_id}[/bold yellow] ({turn_count} turns)"
        if session_id
        else "[dim]Fresh Session[/dim]"
    )
    p = load_matugen_palette()
    c_primary = p.primary if p.is_matugen else "cyan"
    c_accent = p.accent if p.is_matugen else "green"
    c_secondary = p.secondary if p.is_matugen else "magenta"
    c_border = p.border if p.is_matugen else "dim cyan"
    theme_badge = f" [dim]•[/dim] [{c_secondary}]Matugen[/{c_secondary}]" if p.is_matugen else ""
    banner_text = (
        f"[bold {c_primary}]cli-agent[/bold {c_primary}] [dim]•[/dim] [bold {c_accent}]{model_name}[/bold {c_accent}] [dim]•[/dim] "
        f"[{c_secondary}]SQLite Memory ({memory_count} memories)[/{c_secondary}] [dim]•[/dim] [{c_primary}]AST Code Graph[/{c_primary}]{theme_badge}\n"
        f"[dim]Workspace:[/dim] [bold white]~/{ws_name}[/bold white] [dim]•[/dim] [dim]{session_badge}[/dim]\n"
        f"[dim]Shortcuts:[/dim] [bold {c_primary}]![/bold {c_primary}] [dim]shell[/dim]   "
        f"[bold {c_primary}]/[/bold {c_primary}] [dim]commands[/dim]   "
        f"[bold {c_primary}]@[/bold {c_primary}] [dim]files[/dim]   "
        f"[bold {c_primary}]Ctrl+V[/bold {c_primary}] [dim]paste image[/dim]   [dim]• type [bold]/help[/bold] for full reference[/dim]"
    )
    console.print(
        Panel(
            banner_text,
            title=f"[bold {c_primary}]❖ AI Developer Assistant[/bold {c_primary}]",
            border_style=c_border,
            box=box.ROUNDED,
        )
    )


def show_sessions(session_store: SessionStore, project_key: str | None = None) -> None:
    sessions = session_store.list_sessions(project_key=project_key, limit=20)
    if not sessions:
        label = f" for workspace '{project_key}'" if project_key else ""
        console.print(f"[dim yellow]No recorded sessions found{label}.[/dim yellow]")
        return

    table = Table(
        title=f"Saved Sessions ({'Workspace: ' + project_key if project_key else 'All Workspaces'})",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
    )
    table.add_column("Session ID", style="bold cyan", width=28)
    table.add_column("Project", style="blue", width=14)
    table.add_column("Turns", style="yellow", justify="right", width=6)
    table.add_column("Model", style="green", width=18)
    table.add_column("Last Active", style="dim", width=19)
    table.add_column("Title / First Prompt", style="white")

    for s in sessions:
        table.add_row(
            s.id,
            s.project_key,
            str(s.turn_count),
            s.model_name,
            s.updated_at[:19],
            s.title[:60],
        )

    console.print(table)


def show_memories(memory_store: MemoryStore, project_key: str = "global") -> None:
    recent = memory_store.get_recent_memories(limit=25, project_key=project_key)
    if not recent:
        console.print(
            f"[dim yellow]No memories recorded yet for workspace '{project_key}'. Ask the agent to save important decisions![/dim yellow]"
        )
        return

    table = Table(
        title=f"Persistent Memories (Workspace: {project_key})",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Project", style="blue", width=12)
    table.add_column("Scope", style="dim", width=8)
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Title", style="bold white", width=22)
    table.add_column("Content", style="green")

    for m in recent:
        table.add_row(
            str(m["id"]),
            m.get("project_key", "global"),
            m.get("scope", "project"),
            m["type"],
            m["title"],
            m["content"][:90] + ("..." if len(m["content"]) > 90 else ""),
        )

    console.print(table)


async def show_models(
    billing_project: str,
    active_model: str,
    force_refresh: bool = False,
) -> list[ModelCatalogEntry]:
    catalog = await fetch_model_garden_catalog(
        billing_project=billing_project, force_refresh=force_refresh
    )
    # Filter to only the viable Gemini models on Vertex AI requested by user
    gemini_focus = [
        m
        for m in catalog
        if any(
            k in m.short_name.lower()
            for k in (
                "gemini-3.8-flash",
                "gemini-3.1-pro",
                "gemini-4-hydrogen",
                "gemini-4-neon",
                "gemini-4-carbon",
                "gemini-4-argon",
            )
        )
    ]
    def model_sort_key(m: ModelCatalogEntry) -> int:
        name = m.short_name.lower()
        if "3.8-flash" in name:
            return 1
        if "3.1-pro" in name:
            return 2
        if "4-hydrogen" in name:
            return 3
        if "4-neon" in name:
            return 4
        if "4-carbon" in name:
            return 5
        if "4-argon" in name:
            return 6
        return 99

    display_catalog = gemini_focus if gemini_focus else catalog
    display_catalog = sorted(display_catalog, key=model_sort_key)

    table = Table(
        title="Vertex AI Model Garden (Gemini Endpoints)",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Publisher", style="cyan", width=12)
    table.add_column("Model Identifier", style="bold white", width=22)
    table.add_column("Stage", style="yellow", width=12)
    table.add_column("Effort Tier", style="cyan", width=16)
    table.add_column("Status", style="bold green", width=12)

    tier_map = {
        "gemini-3.8-flash": "low / medium",
        "gemini-3.1-pro": "medium / high",
        "gemini-4-hydrogen": "low (ultra-fast)",
        "gemini-4-neon": "medium (balanced)",
        "gemini-4-carbon": "medium / high (structural)",
        "gemini-4-argon": "high (deep synthesis)",
    }

    for idx, m in enumerate(display_catalog, 1):
        is_active = m.short_name == active_model or m.model_id == active_model
        active_mark = "[bold green]✔ ACTIVE[/bold green]" if is_active else ""
        tier = tier_map.get(m.short_name, "standard")
        table.add_row(
            str(idx),
            m.publisher.capitalize(),
            m.short_name,
            m.launch_stage,
            tier,
            active_mark,
        )

    console.print(table)
    console.print(
        "[dim]Switch with [bold]/model <#>[/bold] or [bold]/model <name>[/bold]. Refresh with [bold]/model refresh[/bold].[/dim]\n"
    )
    return display_catalog


def _build_model_picker_app(
    gemini_models: list[ModelCatalogEntry],
    descriptions: dict[str, str],
    active_model: str,
    palette: MatugenPalette | None = None,
) -> tuple[Application[str | None], dict[str, int]]:
    """
    Builds an interactive prompt_toolkit Application allowing arrow-key (↑/↓, j/k),
    direct number-key (1-9), Enter, and Esc navigation over the Gemini model catalog.
    """
    p = palette or load_matugen_palette()
    initial_idx = 0
    for idx, m in enumerate(gemini_models):
        if m.short_name == active_model or m.model_id == active_model:
            initial_idx = idx
            break

    state: dict[str, int] = {"selected": initial_idx}

    def _render_menu() -> FormattedText:
        sel = state["selected"]
        term_w = max(50, shutil.get_terminal_size((80, 24)).columns)
        sep = "─" * min(term_w - 4, 72)
        fragments: list[tuple[str, str]] = [
            ("class:picker.title", "  ❖ Switch Model\n"),
            ("class:picker.border", f"  {sep}\n"),
        ]
        for idx, m in enumerate(gemini_models):
            is_sel = idx == sel
            is_curr = m.short_name == active_model or m.model_id == active_model
            pointer = " ❯ " if is_sel else "   "
            curr_tag = "  ✔ active" if is_curr else ""
            row_style = "class:picker.selected" if is_sel else "class:picker.item"
            num_style = "class:picker.selected" if is_sel else "class:picker.num"
            desc = descriptions.get(m.short_name, "Google Vertex AI foundation model")

            fragments.append((row_style, f"{pointer}"))
            fragments.append((num_style, f"[{idx + 1}] "))
            fragments.append((row_style, f"{m.short_name}"))
            if curr_tag:
                fragments.append(("class:picker.active", curr_tag))
            fragments.append(("", "\n"))
            fragments.append(("class:picker.desc", f"       {desc}\n"))

        fragments.append(("class:picker.border", f"  {sep}\n"))
        fragments.append(
            (
                "class:picker.hint",
                f"  ↑/↓ navigate  •  1-{len(gemini_models)} quick select  •  Enter confirm  •  Esc cancel\n",
            )
        )
        return FormattedText(fragments)

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _move_up(event: Any) -> None:
        state["selected"] = (state["selected"] - 1) % len(gemini_models)

    @kb.add("down")
    @kb.add("j")
    def _move_down(event: Any) -> None:
        state["selected"] = (state["selected"] + 1) % len(gemini_models)

    @kb.add("enter")
    def _confirm(event: Any) -> None:
        event.app.exit(result=gemini_models[state["selected"]].short_name)

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    for num_idx in range(1, min(len(gemini_models), 9) + 1):
        def _make_num_handler(target_i: int) -> Any:
            def _pick_num(event: Any) -> None:
                state["selected"] = target_i
                event.app.exit(result=gemini_models[target_i].short_name)

            return _pick_num

        kb.add(str(num_idx))(_make_num_handler(num_idx - 1))

    picker_style = Style.from_dict(
        {
            "picker.title": f"{p.primary} bold" if p.is_matugen else "ansicyan bold",
            "picker.border": p.border if p.is_matugen else "dim",
            "picker.selected": (
                f"bg:{p.selection_bg} {p.selection_fg} bold"
                if p.is_matugen
                else "ansicyan bold reverse"
            ),
            "picker.item": p.foreground if p.is_matugen else "",
            "picker.num": f"{p.secondary} bold" if p.is_matugen else "bold",
            "picker.active": f"{p.accent} bold" if p.is_matugen else "ansigreen bold",
            "picker.desc": p.muted if p.is_matugen else "dim",
            "picker.hint": f"{p.muted} italic" if p.is_matugen else "dim italic",
        }
    )

    app: Application[str | None] = Application(
        layout=Layout(Window(content=FormattedTextControl(_render_menu), always_hide_cursor=True)),
        key_bindings=kb,
        style=picker_style,
        full_screen=False,
    )
    return app, state


async def prompt_model_switch_dialog(
    client: VertexAgentClient,
    billing_project: str,
) -> str | None:
    """
    Renders an interactive 'Switch Model' selector supporting arrow keys (↑/↓),
    number shortcuts (1-6), Enter, and Esc in TTY terminals, with text fallback for non-TTY.
    """
    catalog = await fetch_model_garden_catalog(billing_project=billing_project)
    canonical_keys = [
        "gemini-3.8-flash",
        "gemini-3.1-pro",
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "gemini-4-carbon",
        "gemini-4-argon",
    ]
    entries_map: dict[str, ModelCatalogEntry] = {}
    for m in catalog:
        for ck in canonical_keys:
            if m.short_name.lower() == ck:
                entries_map[ck] = m
    gemini_models = [
        entries_map.get(
            ck,
            ModelCatalogEntry(
                model_id=f"publishers/google/models/{ck}",
                publisher="google",
                short_name=ck,
                launch_stage="GA" if not ck.startswith("gemini-4") else "PREVIEW",
            ),
        )
        for ck in canonical_keys
    ]

    descriptions = {
        "gemini-3.8-flash": "Balanced speed and reasoning quality for most tasks (Effort: low/medium)",
        "gemini-3.1-pro": "Maximum reasoning & deep architecture refactoring (Effort: medium/high)",
        "gemini-3.1-pro-preview": "Maximum reasoning & deep architecture refactoring (Effort: medium/high)",
        "gemini-4-hydrogen": "Ultra-lightweight & ultra-fast for instantaneous completion & triage (Effort: low)",
        "gemini-4-neon": "High-efficiency coding & balanced multi-turn reasoning (Effort: medium)",
        "gemini-4-carbon": "Versatile foundation for complex refactoring & AST graph analysis (Effort: medium/high)",
        "gemini-4-argon": "Next-generation reasoning, multimodal & complex synthesis (Effort: high)",
    }

    # Interactive arrow-key & number-key selector when running in a real interactive TTY
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            app, _ = _build_model_picker_app(
                gemini_models=gemini_models,
                descriptions=descriptions,
                active_model=client.model_name,
                palette=load_matugen_palette(),
            )
            return await app.run_async()
        except Exception:
            pass

    p = load_matugen_palette()
    c_primary = p.primary if p.is_matugen else "cyan"
    c_accent = p.accent if p.is_matugen else "green"
    c_border = p.border if p.is_matugen else "cyan"

    content = Text()
    content.append("Search / Select Model:\n", style="bold")
    content.append("─" * 60 + "\n\n", style="dim")

    for idx, m in enumerate(gemini_models, 1):
        is_curr = m.short_name == client.model_name or m.model_id == client.model_name
        curr_badge = f" [bold {c_accent}](current)[/bold {c_accent}]" if is_curr else ""
        prefix = " > " if is_curr else "   "
        style_num = f"bold {c_primary}" if is_curr else "bold white"
        content.append(f"{prefix}[{idx}] ", style=style_num)
        content.append(f"{m.short_name}", style=f"bold {c_primary}" if is_curr else "bold")
        content.append(f"{curr_badge}\n")
        desc = descriptions.get(m.short_name, "Google Vertex AI foundation model")
        content.append(f"       {desc}\n\n", style="dim")

    content.append("Effort  ◀ ───●───◎───○───▶\n", style=f"dim {c_primary}")
    content.append("            low  medium  high\n", style="dim")
    content.append(f"Select a number [1-{len(gemini_models)}] or type name (Enter to cancel)", style="dim italic")

    console.print(
        Panel(
            content,
            title=f"[bold {c_primary}]Switch Model[/bold {c_primary}]",
            border_style=c_border,
            box=box.ROUNDED,
        )
    )

    try:
        sel = console.input(f"[bold white]Select model [1-{len(gemini_models)}] or name: [/bold white]").strip()
        if not sel:
            return None
        if sel.isdigit():
            num = int(sel)
            if 1 <= num <= len(gemini_models):
                return gemini_models[num - 1].short_name
        for m in gemini_models:
            if sel.lower() in m.short_name.lower():
                return m.short_name
        return sel
    except (KeyboardInterrupt, EOFError):
        return None


def make_security_policy(auto_approve: bool = False) -> SecurityPolicy:
    policy = SecurityPolicy(auto_approve=auto_approve)

    def interactive_confirm(tool_name: str, args: dict[str, Any], risk: RiskLevel) -> bool:
        border_col = "red" if risk == RiskLevel.HIGH else "yellow"
        badge = (
            "[bold red]🚨 CRITICAL / DESTRUCTIVE COMMAND DETECTED[/bold red]"
            if risk == RiskLevel.HIGH
            else f"[bold yellow]⚠️  Permission Request ({risk.value.upper()})[/bold yellow]"
        )
        content = Text()
        content.append("Action: ", style="bold")
        content.append(f"{tool_name}\n", style="bold cyan")
        if args:
            for k, v in args.items():
                content.append(f"  • {k}: ", style="dim")
                content.append(f"{repr(v)}\n", style="white")
        content.append("\nOptions: ", style="bold")
        content.append("[y] Yes   ", style="bold green")
        content.append("[n] No   ", style="bold red")
        content.append("[a] Always (this session)", style="bold yellow")

        console.print(
            Panel(
                content,
                title=badge,
                border_style=border_col,
                box=box.ROUNDED,
            )
        )
        try:
            choice = (
                console.input("[bold white]Select an option [y/n/a] (default 'n'): [/bold white]")
                .strip()
                .lower()
            )
            if choice in ("always", "a"):
                policy.grant_session_permission(tool_name, args)
                return True
            return choice in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False

    policy.confirm_callback = interactive_confirm
    return policy


_LATEX_SYMBOL_MAP: dict[str, str] = {
    # Arrows
    "rightarrow": "→",
    "to": "→",
    "leftarrow": "←",
    "gets": "←",
    "leftrightarrow": "↔",
    "Rightarrow": "⇒",
    "implies": "⇒",
    "Leftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "iff": "⇔",
    "uparrow": "↑",
    "downarrow": "↓",
    "updownarrow": "↕",
    "mapsto": "↦",
    "longrightarrow": "⟶",
    "longleftarrow": "⟵",
    "Longrightarrow": "⟹",
    "Longleftarrow": "⟸",
    # Math operators & relations
    "times": "×",
    "div": "÷",
    "pm": "±",
    "mp": "∓",
    "cdot": "·",
    "ast": "∗",
    "neq": "≠",
    "ne": "≠",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "approx": "≈",
    "equiv": "≡",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "propto": "∝",
    "infty": "∞",
    "dots": "…",
    "ldots": "…",
    "cdots": "⋯",
    "vdots": "⋮",
    "ddots": "⋱",
    "partial": "∂",
    "nabla": "∇",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "degree": "°",
    # Logic & sets
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "cup": "∪",
    "cap": "∩",
    "emptyset": "∅",
    "varnothing": "∅",
    "forall": "∀",
    "exists": "∃",
    "neg": "¬",
    "lnot": "¬",
    "land": "∧",
    "lor": "∨",
    "therefore": "∴",
    "because": "∵",
    "checkmark": "✓",
    # Greek letters
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}

_SUPERSCRIPT_MAP = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUBSCRIPT_MAP = str.maketrans("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

_LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_INLINE_CODE_SPLIT_RE = re.compile(r"(`[^`]+`)")
_MATH_BLOCK_RE = re.compile(
    r"\$\$(.+?)\$\$|\\\((.+?)\\\)|\\\[(.+?)\\\]|\$(?!\s)([^$\n]+?)(?<!\s)\$",
    re.DOTALL,
)


def _convert_latex_expr(expr: str) -> str:
    """Converts a LaTeX math expression into clean terminal-friendly Unicode."""
    out = expr.strip()
    # Unwrap common text/style wrappers like \text{...}, \mathrm{...}, \mathbf{...}
    out = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname|bf|rm)\{([^{}]*)\}", r"\1", out)
    # \frac{a}{b} -> a/b or (a)/(b)
    def _replace_frac(m: re.Match[str]) -> str:
        num, den = m.group(1).strip(), m.group(2).strip()
        n_str = num if len(num) <= 2 or num.isalnum() else f"({num})"
        d_str = den if len(den) <= 2 or den.isalnum() else f"({den})"
        return f"{n_str}/{d_str}"

    out = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", _replace_frac, out)
    out = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", out)
    # Replace known LaTeX symbol commands
    out = _LATEX_CMD_RE.sub(lambda m: _LATEX_SYMBOL_MAP.get(m.group(1), m.group(0)), out)
    # Simple superscripts and subscripts
    out = re.sub(r"\^\{([0-9+\-=()ni]+)\}", lambda m: m.group(1).translate(_SUPERSCRIPT_MAP), out)
    out = re.sub(r"\^([0-9ni])", lambda m: m.group(1).translate(_SUPERSCRIPT_MAP), out)
    out = re.sub(
        r"_\{([0-9+\-=()aehijklmnoprstuvx]+)\}",
        lambda m: m.group(1).translate(_SUBSCRIPT_MAP),
        out,
    )
    out = re.sub(r"_([0-9ijkmn])", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), out)
    # Remove LaTeX spacing commands (\, \; \: \!)
    out = re.sub(r"\\[,;:!]", " ", out)
    return out


def render_latex_unicode(text: str) -> str:
    """
    Replaces inline/display LaTeX math notations (such as $\\rightarrow$, $\\leq$, $O(n^2)$)
    with clean Unicode symbols while preserving inline code spans and currency values.
    """
    if "$" not in text and "\\" not in text:
        return text

    parts = _INLINE_CODE_SPLIT_RE.split(text)
    rendered_parts: list[str] = []

    for idx, segment in enumerate(parts):
        # Odd indices are inline `code` spans — leave untouched
        if idx % 2 == 1:
            rendered_parts.append(segment)
            continue

        def _replace_math(m: re.Match[str]) -> str:
            raw_match = m.group(0)
            inner = m.group(1) or m.group(2) or m.group(3) or m.group(4) or ""
            # For single-$ delimiters, avoid eating currency ranges like "$10 and $20"
            if m.group(4) is not None:
                has_latex_syntax = (
                    "\\" in inner
                    or "^" in inner
                    or "_" in inner
                    or any(op in inner for op in ("=", "<", ">", "≈", "≠", "≤", "≥"))
                )
                if not has_latex_syntax:
                    return raw_match
            return _convert_latex_expr(inner)

        rendered_parts.append(_MATH_BLOCK_RE.sub(_replace_math, segment))

    return "".join(rendered_parts)


class RichMarkdownStreamer:
    """
    Streams assistant responses progressively to the terminal with Rich Markdown interpretation,
    rendering completed block-level structures (headers, paragraphs, lists, code blocks) in real-time.
    Avoids ANSI full-buffer repainting repetition issues by streaming forward-only without Live cursor loops.
    Handles intermediate tool calls gracefully by flushing pending blocks before printing tool badges.
    """

    def __init__(
        self, console: Console, prompt_label: str = "\n[bold magenta]agent ❯[/bold magenta]\n"
    ) -> None:
        self.console = console
        self.prompt_label = prompt_label
        self.full_text = ""
        self._buffer = ""
        self._header_shown = False
        self._in_code_block = False
        self._live = None

    def on_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        if not self._header_shown and self.prompt_label:
            self.console.print(self.prompt_label, end="")
            self._header_shown = True

        self.full_text += chunk
        self._buffer += chunk
        self._flush_blocks(finalize=False)

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._flush_blocks(finalize=True)
        action_map = {
            "write_file": "Write",
            "edit_file": "Edit",
            "read_file": "Read",
            "run_command": "Bash",
            "search_code": "Search",
            "tdd_repair": "TDD",
            "mcp_call": "MCP",
        }
        action = action_map.get(name, name.replace("_", " ").title())
        target = (
            args.get("path")
            or args.get("file_path")
            or args.get("command")
            or args.get("query")
            or args.get("target")
            or ""
        )
        target_display = f" [bold white]{target}[/bold white]" if target else ""
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."

        self.console.print(
            f"\n[dim yellow]⚡ [bold]Calling Tool:[/bold] [bold cyan]▼ {action}[/bold cyan]{target_display} [bold green]✓[/bold green][/dim yellow]"
        )
        if name in ("write_file", "edit_file") and "content" in args:
            lines = args["content"].splitlines()
            self.console.print(f"  [dim cyan]({len(lines)} lines changed in {target})[/dim cyan]")

    def finalize(self) -> str:
        self._flush_blocks(finalize=True)
        self.console.print()
        return self.full_text

    def _flush_blocks(self, finalize: bool = False) -> None:
        while self._buffer:
            if not self._in_code_block:
                # Discard leading blank lines
                if self._buffer.startswith("\n") or self._buffer.startswith("\r"):
                    self._buffer = self._buffer.lstrip("\r\n")
                    continue

                # Check if a code fence starts
                code_start = self._buffer.find("```")
                if code_start != -1:
                    before = self._buffer[:code_start].strip()
                    if before:
                        self.console.print(Markdown(render_latex_unicode(before)))
                    self._buffer = self._buffer[code_start:]
                    self._in_code_block = True
                    continue

                # Paragraph or list block break (\n\n)
                dnl = self._buffer.find("\n\n")
                if dnl != -1:
                    block = self._buffer[:dnl].strip()
                    self._buffer = self._buffer[dnl + 2 :]
                    if block:
                        self.console.print(Markdown(render_latex_unicode(block)))
                    continue

                # Header line or divider ending with newline
                nl = self._buffer.find("\n")
                if nl != -1:
                    line = self._buffer[:nl].strip()
                    if line.startswith("#") or line in ("---", "***", "___"):
                        self.console.print(Markdown(render_latex_unicode(line)))
                        self._buffer = self._buffer[nl + 1 :]
                        continue

                # If finalizing, render any remaining trailing buffer
                if finalize:
                    rem = self._buffer.strip()
                    if rem:
                        self.console.print(Markdown(render_latex_unicode(rem)))
                    self._buffer = ""
                break
            else:
                # Inside code block: look for closing code fence ```
                closing = self._buffer[3:].find("```")
                if closing != -1:
                    closing_idx = 3 + closing + 3
                    if len(self._buffer) > closing_idx and self._buffer[closing_idx] == "\n":
                        closing_idx += 1
                    block = self._buffer[:closing_idx].strip()
                    self._buffer = self._buffer[closing_idx:]
                    self._in_code_block = False
                    if block:
                        self.console.print(Markdown(block))
                    continue
                if finalize:
                    block = self._buffer.strip()
                    if block:
                        self.console.print(Markdown(block))
                    self._buffer = ""
                break


async def run_interactive(
    auto_approve: bool = False,
    model_override: str | None = None,
    max_turns: int | None = None,
    resume_session_id: str | None = None,
    continue_last: bool = False,
) -> None:
    with AlternateScreenWorkspace():
        await _run_interactive_session(
            auto_approve=auto_approve,
            model_override=model_override,
            max_turns=max_turns,
            resume_session_id=resume_session_id,
            continue_last=continue_last,
        )


async def _run_interactive_session(
    auto_approve: bool = False,
    model_override: str | None = None,
    max_turns: int | None = None,
    resume_session_id: str | None = None,
    continue_last: bool = False,
) -> None:
    cfg = load_config()
    model = model_override or cfg.model
    memory_store = MemoryStore()
    session_store = SessionStore()
    project_key = resolve_project_key()
    security_policy = make_security_policy(auto_approve=auto_approve or cfg.auto_approve)
    circuit_breaker = QuotaCircuitBreaker()
    client = VertexAgentClient(
        project_id=cfg.project,
        location=cfg.location,
        model_name=model,
        memory_store=memory_store,
        security_policy=security_policy,
        fallback_router=None,
        circuit_breaker=circuit_breaker,
        timeout=cfg.timeout,
    )

    active_session_id: str = generate_session_id()
    conversation_history: list[dict[str, Any]] = []

    if resume_session_id:
        target = session_store.get_session(resume_session_id)
        if target:
            active_session_id = target.id
            conversation_history = [dict(m) for m in target.messages]
            if not model_override:
                model = target.model_name
                client.switch_model(model)
            console.print(
                f"[bold green]✔ Resumed session [bold cyan]{target.id}[/bold cyan] ({len(conversation_history)} turns, model: {target.model_name})[/bold green]\n"
                f"[dim]Title: {target.title} | Updated: {target.updated_at}[/dim]"
            )
            if target.preview:
                console.print(f"[dim]Context snippet: {target.preview}[/dim]")
        else:
            console.print(
                f"[bold red]Session '{resume_session_id}' not found. Starting a new session.[/bold red]"
            )
    elif continue_last:
        latest = session_store.get_latest_session(project_key=project_key)
        if latest:
            active_session_id = latest.id
            conversation_history = [dict(m) for m in latest.messages]
            if not model_override:
                model = latest.model_name
                client.switch_model(model)
            console.print(
                f"[bold green]✔ Resumed session [bold cyan]{latest.id}[/bold cyan] ({len(conversation_history)} turns, model: {latest.model_name})[/bold green]\n"
                f"[dim]Title: {latest.title} | Updated: {latest.updated_at}[/dim]"
            )
            if latest.preview:
                console.print(f"[dim]Context snippet: {latest.preview}[/dim]")
        else:
            console.print(
                f"[dim yellow]No prior session found for workspace '{project_key}'. Starting a new session.[/dim yellow]"
            )

    worktree_manager = WorktreeManager()
    active_sandbox: WorktreeSession | None = None

    team_coordinator = MultiAgentTeam(client=client)
    set_active_team(team_coordinator)

    mcp_registry = MCPRegistry()
    set_active_mcp_registry(mcp_registry)
    try:
        await mcp_registry.connect_all()
    except Exception:
        pass

    code_graph_indexer = CodeGraphIndexer(client.code_graph_store)
    hybrid_search_engine = get_hybrid_search_engine()
    try:
        code_graph_indexer.index_directory(".", project_key=project_key)
        await hybrid_search_engine.index_directory(".", project_key=project_key)
    except Exception:
        pass

    recent_mems = memory_store.get_recent_memories(limit=50, project_key=project_key)
    print_banner(
        len(recent_mems),
        project_key=project_key,
        model_name=client.model_name,
        session_id=active_session_id,
        turn_count=len(conversation_history),
    )

    DEFAULT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    pending_images: list[ImageAttachment] = []
    input_kb = build_input_key_bindings(pending_images)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(DEFAULT_HISTORY_FILE)),
        completer=command_completer,
        style=pt_style,
        key_bindings=input_kb,
    )

    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        action_map = {
            "write_file": "Write",
            "edit_file": "Edit",
            "read_file": "Read",
            "run_command": "Bash",
            "search_code": "Search",
            "tdd_repair": "TDD",
            "mcp_call": "MCP",
        }
        action = action_map.get(name, name.replace("_", " ").title())
        target = (
            args.get("path")
            or args.get("file_path")
            or args.get("command")
            or args.get("query")
            or args.get("target")
            or ""
        )
        target_display = f" [bold white]{target}[/bold white]" if target else ""
        console.print(
            f"\n[dim yellow]⚙ [bold]Calling Tool:[/bold] [bold cyan]▼ {action}[/bold cyan]{target_display} [bold green]✓[/bold green][/dim yellow]"
        )

    turns_completed = 0
    while True:
        if max_turns is not None and turns_completed >= max_turns:
            break
        try:
            w = max(20, shutil.get_terminal_size((80, 24)).columns)

            def get_bottom_toolbar() -> AnyFormattedText:
                term_w = shutil.get_terminal_size((80, 24)).columns
                return format_bottom_toolbar(
                    terminal_width=term_w,
                    auto_approve=security_policy.auto_approve,
                    model_name=client.model_name,
                    image_count=len(pending_images),
                )

            top_border = f"\n{'─' * w}\n"
            user_input = (
                await session.prompt_async(
                    [("class:prompt.box", top_border), ("class:prompt", "❯ ")],
                    placeholder="What would you like to do?",
                    bottom_toolbar=get_bottom_toolbar,
                )
            ).strip()
            if user_input:
                term_w = max(20, shutil.get_terminal_size((80, 24)).columns)
                console.print(f"[dim]{'─' * term_w}[/dim]")
        except (KeyboardInterrupt, EOFError):
            if conversation_history:
                session_store.save_session(
                    session_id=active_session_id,
                    messages=conversation_history,
                    model_name=client.model_name,
                    project_key=project_key,
                )
            console.print("\n[dim]Session terminated.[/dim]")
            break
        turns_completed += 1

        if not user_input:
            continue

        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if not shell_cmd:
                console.print(
                    "[dim]Usage: [bold]!<command>[/bold] (e.g. [cyan]!pytest[/cyan], [cyan]!git status[/cyan])[/dim]"
                )
                continue
            console.print(
                f"[bold cyan]▼ Executing Shell:[/bold cyan] [bold white]{shell_cmd}[/bold white]"
            )
            from .tools import run_command as exec_run_command

            res = exec_run_command(shell_cmd)
            status_style = "bold green" if res["exit_code"] == 0 else "bold red"
            mark = "✓" if res["exit_code"] == 0 else "✗"
            console.print(f"[{status_style}]Exit code: {res['exit_code']} {mark}[/{status_style}]")
            out = res.get("output", "")
            if out.strip():
                console.print(
                    Panel(
                        out.strip(),
                        title=f"[dim]Output: {shell_cmd}[/dim]",
                        border_style="dim",
                        box=box.ROUNDED,
                    )
                )
            continue

        if user_input in ("/exit", "/quit", "exit", "quit"):
            console.print("[bold green]Hasta luego![/bold green]")
            if active_sandbox:
                active_sandbox.cleanup()
                set_active_sandbox(None)
            set_active_team(None)
            await mcp_registry.shutdown_all()
            set_active_mcp_registry(None)
            if conversation_history:
                session_store.save_session(
                    session_id=active_session_id,
                    messages=conversation_history,
                    model_name=client.model_name,
                    project_key=project_key,
                )
                console.print(
                    f"[dim]Session saved: [bold yellow]{active_session_id}[/bold yellow]. "
                    f"Resume with: [bold cyan]cli-agent --continue[/bold cyan] or [bold cyan]cli-agent --session {active_session_id}[/bold cyan][/dim]"
                )
            if len(conversation_history) >= 2:
                learnings = extract_session_learnings(conversation_history)
                memory_store.save_memory(
                    title=f"Session Summary ({project_key})",
                    content=learnings,
                    memory_type="convention",
                    project_key=project_key,
                )
            break

        if user_input == "/clear":
            conversation_history.clear()
            pending_images.clear()
            active_session_id = generate_session_id()
            console.print(
                "[dim cyan]Conversation history cleared. Started fresh session: "
                f"[bold yellow]{active_session_id}[/bold yellow] (SQLite memories preserved).[/dim cyan]"
            )
            continue

        if user_input.startswith("/auto"):
            parts = user_input.split(maxsplit=1)
            arg = parts[1].strip().lower() if len(parts) > 1 else ""
            if arg in ("on", "1", "true", "yes", "enable"):
                security_policy.auto_approve = True
                console.print(
                    "[bold green]✔ Autonomous mode ENABLED[/bold green]: Safe and Medium risk tools (editing files, reading, running tests) are now pre-approved. (Destructive commands will still ask for confirmation)."
                )
            elif arg in ("off", "0", "false", "no", "disable"):
                security_policy.auto_approve = False
                console.print(
                    "[bold yellow]✔ Autonomous mode DISABLED[/bold yellow]: All mutating actions will require interactive approval."
                )
            elif not arg:
                # Toggle
                security_policy.auto_approve = not security_policy.auto_approve
                state = "ENABLED" if security_policy.auto_approve else "DISABLED"
                style = "green" if security_policy.auto_approve else "yellow"
                console.print(
                    f"[bold {style}]✔ Autonomous mode {state}[/bold {style}]. Use [bold]/auto on[/bold] or [bold]/auto off[/bold] to set explicitly."
                )
            else:
                console.print("[dim]Usage: [bold]/auto[/bold] (toggle) or [bold]/auto on|off[/bold][/dim]")
            continue

        if user_input in ("/sessions", "/session list"):
            show_sessions(session_store, project_key=project_key)
            continue

        if user_input == "/session":
            console.print(
                f"[bold cyan]Active Session:[/bold cyan] [bold yellow]{active_session_id}[/bold yellow] "
                f"({len(conversation_history)} turns) | Workspace: [bold blue]{project_key}[/bold blue] | Model: [bold green]{client.model_name}[/bold green]"
            )
            continue

        if user_input.startswith("/resume") or (
            user_input.startswith("/session ") and not user_input.startswith("/sessions")
        ):
            parts = user_input.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            if not arg:
                show_sessions(session_store, project_key=project_key)
                try:
                    chosen_session = console.input(
                        "[bold white]Enter Session ID to resume (or press Enter for latest): [/bold white]"
                    ).strip()
                    arg = chosen_session
                except (KeyboardInterrupt, EOFError):
                    continue

            target = (
                session_store.get_session(arg)
                if arg
                else session_store.get_latest_session(project_key=project_key)
            )
            if target:
                if conversation_history:
                    session_store.save_session(
                        session_id=active_session_id,
                        messages=conversation_history,
                        model_name=client.model_name,
                        project_key=project_key,
                    )
                active_session_id = target.id
                conversation_history = [dict(m) for m in target.messages]
                if not model_override:
                    client.switch_model(target.model_name)
                console.print(
                    f"[bold green]✔ Switched to session [bold cyan]{target.id}[/bold cyan] "
                    f"({len(conversation_history)} turns, model: {target.model_name})[/bold green]\n"
                    f"[dim]Title: {target.title} | Updated: {target.updated_at}[/dim]"
                )
                if target.preview:
                    console.print(f"[dim]Context snippet: {target.preview}[/dim]")
            else:
                target_label = arg if arg else "latest"
                console.print(f"[bold red]Session '{target_label}' not found.[/bold red]")
            continue

        if user_input in ("/memories", "/memory"):
            show_memories(memory_store, project_key=project_key)
            continue

        if user_input == "/compact":
            conversation_history = compact_history(conversation_history, max_turns=2, keep_recent=2)
            console.print("[dim cyan]Context history manually compacted into summary.[/dim cyan]")
            continue

        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            if not arg:
                chosen_model = await prompt_model_switch_dialog(
                    client=client, billing_project=client.project_id
                )
                if chosen_model:
                    client.switch_model(chosen_model)
                    save_model_to_config(chosen_model)
                    console.print(
                        f"[bold green]✔ Switched active model to:[/bold green] [bold cyan]{client.model_name}[/bold cyan] [dim](publisher: {client.publisher})[/dim]"
                    )
            elif arg == "refresh":
                with console.status(
                    "[bold cyan]Refreshing Model Garden catalog from Vertex AI...[/bold cyan]"
                ):
                    await show_models(
                        billing_project=client.project_id,
                        active_model=client.model_name,
                        force_refresh=True,
                    )
            else:
                catalog = await fetch_model_garden_catalog(billing_project=client.project_id)
                chosen_entry = None
                if arg.isdigit():
                    num = int(arg)
                    if 1 <= num <= len(catalog):
                        chosen_entry = catalog[num - 1]
                else:
                    for entry in catalog:
                        if (
                            entry.short_name.lower() == arg.lower()
                            or entry.model_id.lower() == arg.lower()
                        ):
                            chosen_entry = entry
                            break

                target_model = chosen_entry.short_name if chosen_entry else arg
                client.switch_model(target_model)
                save_model_to_config(target_model)
                console.print(
                    f"[bold green]Switched active model to:[/bold green] [bold cyan]{client.model_name}[/bold cyan] [dim](publisher: {client.publisher})[/dim]"
                )
            continue

        if user_input == "/index":
            with console.status(
                "[bold cyan]Indexing workspace AST symbols & hybrid search chunks...[/bold cyan]"
            ):
                indexed_count = code_graph_indexer.index_directory(".", project_key=project_key)
                search_chunks = await hybrid_search_engine.index_directory(
                    ".", project_key=project_key
                )
                stats = client.code_graph_store.get_stats(project_key=project_key)
            console.print(
                f"[bold green]Indexed {indexed_count} files.[/bold green] Total: [bold cyan]{stats['symbols']}[/bold cyan] symbols across [bold cyan]{stats['files']}[/bold cyan] files ({stats['edges']} edges) | [bold yellow]{search_chunks} code chunks[/bold yellow] vectorized."
            )
            continue

        if user_input.startswith("/search") or user_input.startswith("/find"):
            parts = user_input.split(maxsplit=1)
            query = parts[1].strip() if len(parts) > 1 else ""
            if not query:
                try:
                    query = console.input("[bold white]Enter code search query: [/bold white]").strip()
                except (KeyboardInterrupt, EOFError):
                    continue
                if not query:
                    continue

            with console.status(
                "[bold cyan]Running Hybrid Code Search (BM25 + Dense Vectors + RRF)...[/bold cyan]"
            ):
                search_results = await hybrid_search_engine.search(
                    query=query, project_key=project_key, limit=5
                )

            if not search_results:
                console.print(
                    "[dim yellow]No matching code chunks found for this query.[/dim yellow]"
                )
            else:
                table = Table(
                    title=f"Hybrid Code Search Results: '{query}'",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Score (RRF)", style="bold cyan", width=12)
                table.add_column("Symbol", style="bold white", width=26)
                table.add_column("Location", style="yellow", width=36)
                table.add_column("Snippet Preview", style="dim")

                for res in search_results:
                    loc = f"{res['file_path']}:{res['start_line']}-{res['end_line']}"
                    snippet = res["content"].strip().splitlines()[0][:60]
                    table.add_row(
                        str(res["rrf_score"]),
                        f"{res['symbol_name']} ({res['symbol_type']})",
                        loc,
                        snippet,
                    )
                console.print(table)
            continue

        if user_input.startswith("/sandbox"):
            parts = user_input.split(maxsplit=1)
            action = parts[1].strip().lower() if len(parts) > 1 else "status"

            if action in ("on", "start", "create"):
                if active_sandbox and active_sandbox.is_active:
                    console.print(
                        f"[dim yellow]Sandbox is already active on branch: {active_sandbox.branch_name}[/dim yellow]"
                    )
                else:
                    try:
                        active_sandbox = worktree_manager.create_session()
                        set_active_sandbox(active_sandbox.worktree_dir)
                        console.print(
                            f"[bold green]✔ Ephemeral Git Worktree Activated![/bold green]\n"
                            f"  [dim]Branch:[/dim] [bold cyan]{active_sandbox.branch_name}[/bold cyan]\n"
                            f"  [dim]Sandbox Dir:[/dim] [yellow]{active_sandbox.worktree_dir}[/yellow]\n"
                            f"  [dim]All file mutations and commands are now completely isolated from your workspace.[/dim]"
                        )
                    except Exception as e:
                        console.print(f"[bold red]Failed creating sandbox worktree:[/bold red] {e}")
            elif action == "diff":
                if not active_sandbox or not active_sandbox.is_active:
                    console.print(
                        "[dim yellow]No active sandbox. Type '/sandbox on' to start one.[/dim yellow]"
                    )
                else:
                    diff_text = active_sandbox.get_diff()
                    console.print(DiffViewer.render(diff_text))
            elif action == "apply":
                if not active_sandbox or not active_sandbox.is_active:
                    console.print("[dim yellow]No active sandbox to apply.[/dim yellow]")
                else:
                    summary = active_sandbox.get_summary()
                    if summary.files_changed == 0:
                        console.print(
                            "[dim yellow]No modifications made in sandbox to apply.[/dim yellow]"
                        )
                    else:
                        console.print(DiffViewer.render_summary(summary))
                        confirm = (
                            console.input(
                                "[bold white]Merge these changes to your active workspace? [y/N]: [/bold white]"
                            )
                            .strip()
                            .lower()
                        )
                        if confirm in ("y", "yes"):
                            res = active_sandbox.apply_to_main()
                            if res["status"] == "applied":
                                console.print(
                                    f"[bold green]✔ Patch successfully merged![/bold green] Applied to {res['files_changed']} files (+{res['insertions']} -{res['deletions']})."
                                )
                                active_sandbox.cleanup()
                                set_active_sandbox(None)
                                active_sandbox = None
                            else:
                                console.print(
                                    f"[bold red]Merge failed:[/bold red] {res.get('message')}"
                                )
                        else:
                            console.print("[dim]Merge aborted. Sandbox remains active.[/dim]")
            elif action in ("discard", "off", "cancel"):
                if not active_sandbox or not active_sandbox.is_active:
                    console.print("[dim yellow]No active sandbox to discard.[/dim yellow]")
                else:
                    active_sandbox.cleanup()
                    set_active_sandbox(None)
                    active_sandbox = None
                    console.print(
                        "[bold green]✔ Sandbox discarded cleanly.[/bold green] Your workspace remains untouched."
                    )
            else:
                if active_sandbox and active_sandbox.is_active:
                    summary = active_sandbox.get_summary()
                    console.print(
                        f"[bold cyan]Active Sandbox:[/bold cyan] {active_sandbox.branch_name}\n"
                        f"  [dim]Directory:[/dim] {active_sandbox.worktree_dir}\n"
                        f"  [dim]Changes:[/dim] {summary.files_changed} files modified (+{summary.insertions} -{summary.deletions})\n"
                        f"  [dim]Commands:[/dim] /sandbox diff, /sandbox apply, /sandbox discard"
                    )
                else:
                    console.print(
                        "[dim]Sandbox is currently [bold]inactive[/bold]. Type [bold]/sandbox on[/bold] to isolate changes in an ephemeral worktree.[/dim]"
                    )
            continue

        if user_input.startswith("/team") or user_input.startswith("/swarm"):
            parts = user_input.split(maxsplit=1)
            goal = parts[1].strip() if len(parts) > 1 else ""
            if not goal:
                try:
                    goal = console.input(
                        "[bold white]Enter goal for multi-agent swarm: [/bold white]"
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    continue
                if not goal:
                    continue

            console.print(
                f"\n[bold magenta]🚀 Initiating Multi-Agent Swarm (ODD Pipeline)...[/bold magenta]\n[dim]Goal: {goal}[/dim]\n"
            )

            def on_team_event(event_type: str, message: str) -> None:
                if event_type == "stage":
                    console.print(f"\n[bold cyan]▶ {message}[/bold cyan]")
                elif event_type == "spawn":
                    console.print(f"  [dim yellow]⚡ {message}[/dim yellow]")
                elif event_type == "tool_call":
                    console.print(f"    [dim]{message}[/dim]")

            with console.status("[bold cyan]Multi-Agent Swarm working...[/bold cyan]"):
                workflow_res = await team_coordinator.run_workflow(goal, on_event=on_team_event)

            console.print(Markdown(workflow_res.summary))
            continue

        if user_input.startswith("/mcp"):
            parts = user_input.split(maxsplit=1)
            action = parts[1].strip().lower() if len(parts) > 1 else "list"

            if action == "reload":
                with console.status("[bold cyan]Reloading MCP servers...[/bold cyan]"):
                    await mcp_registry.shutdown_all()
                    count = await mcp_registry.connect_all()
                console.print(
                    f"[bold green]✔ MCP Reloaded![/bold green] Connected to [bold cyan]{count}[/bold cyan] servers ({len(mcp_registry.tools)} tools federated)."
                )
            else:
                if not mcp_registry.tools:
                    console.print(
                        f"[dim yellow]No federated MCP tools active.\nRegister servers in [bold]{mcp_registry.config_path}[/bold] (JSON format) and run '/mcp reload'.[/dim yellow]"
                    )
                else:
                    table = Table(
                        title=f"Federated MCP Tools ({len(mcp_registry.servers)} servers, {len(mcp_registry.tools)} tools)",
                        show_header=True,
                        header_style="bold magenta",
                    )
                    table.add_column("Server", style="cyan", width=14)
                    table.add_column("Tool Name", style="bold white", width=28)
                    table.add_column("Description", style="dim")

                    for tool in mcp_registry.tools.values():
                        table.add_row(tool.server_name, tool.full_name, tool.description[:80])
                    console.print(table)
            continue

        if user_input.startswith("/tdd") or user_input.startswith("/fix"):
            parts = user_input.split(maxsplit=1)
            tdd_target = parts[1].strip() if len(parts) > 1 else ""
            if not tdd_target:
                try:
                    tdd_target = console.input(
                        "[bold white]Enter failing test target (e.g. tests/unit/test_calc.py): [/bold white]"
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    continue
                if not tdd_target:
                    continue

            test_cmd = tdd_target if ("pytest" in tdd_target or "python" in tdd_target) else f"pytest {tdd_target}"
            console.print(
                f"\n[bold magenta]🛠 Initiating Autonomous TDD Repair Loop...[/bold magenta]\n"
                f"[dim]Reproducing Target: {test_cmd}[/dim]\n"
            )

            tdd_engine = AutonomousTDDEngine()

            def on_tdd_event(evt_type: str, msg: str) -> None:
                if evt_type == "phase_red":
                    console.print(f"[bold red]▶ {msg}[/bold red]")
                elif evt_type == "phase_patch":
                    console.print(f"[bold yellow]▶ {msg}[/bold yellow]")
                elif evt_type == "phase_green":
                    console.print(f"[bold green]▶ {msg}[/bold green]")
                elif evt_type == "phase_regression":
                    console.print(f"[bold cyan]▶ {msg}[/bold cyan]")
                else:
                    console.print(f"  [dim]{msg}[/dim]")

            with console.status("[bold cyan]TDD Self-Healing Engine executing...[/bold cyan]"):
                tdd_result = await tdd_engine.run_cycle(
                    reproducing_command=test_cmd,
                    on_event=on_tdd_event,
                )

            if tdd_result.success:
                console.print(f"\n[bold green]✔ {tdd_result.final_status}[/bold green]")
                if tdd_result.diff:
                    console.print("\n[bold cyan]Verified Patch Diff:[/bold cyan]")
                    console.print(
                        Syntax(tdd_result.diff, "diff", theme="monokai", line_numbers=True)
                    )
            else:
                console.print(f"\n[bold red]✖ {tdd_result.final_status}[/bold red]")
            continue

        if user_input.startswith("/setup"):
            console.print(
                "\n[bold magenta]🛠 Google Cloud & Vertex AI Guided Setup Wizard[/bold magenta]\n"
            )
            pf_engine = PreflightEngine()
            with console.status(
                "[bold cyan]Inspecting local Google Cloud environment...[/bold cyan]"
            ):
                report = pf_engine.run_checks()

            console.print(
                f"  [bold]Active Account:[/bold] [cyan]{report.account or 'None (Not Logged In)'}[/cyan]"
            )
            console.print(
                f"  [bold]Target Project:[/bold] [cyan]{report.project or 'None (Unset)'}[/cyan]"
            )
            console.print(f"  [bold]Active Model:[/bold]   [cyan]{client.model_name}[/cyan]\n")

            projects = ProjectSelector.list_accessible_projects()
            if projects:
                p_table = Table(title="Accessible Google Cloud Projects", show_header=True)
                p_table.add_column("#", width=4)
                p_table.add_column("Project ID", style="bold cyan")
                p_table.add_column("Project Name", style="white")
                p_table.add_column("Number", style="dim")

                for idx, proj in enumerate(projects, 1):
                    p_table.add_row(
                        str(idx), proj["projectId"], proj["name"], proj["projectNumber"]
                    )
                console.print(p_table)

                sel = console.input(
                    "[bold white]Select project number to switch (or press Enter to keep current): [/bold white]"
                ).strip()
                if sel.isdigit() and 1 <= int(sel) <= len(projects):
                    chosen_proj = projects[int(sel) - 1]["projectId"]
                    ProjectSelector.save_project_to_config(chosen_proj)
                    client.project_id = chosen_proj
                    console.print(
                        f"[bold green]✔ Project updated to:[/bold green] [bold cyan]{chosen_proj}[/bold cyan] (saved to config.toml)"
                    )

            target_proj = client.project_id
            if not ServiceUsageManager.is_api_enabled(target_proj):
                console.print(
                    f"[bold yellow]⚠ Vertex AI API is disabled on '{target_proj}'.[/bold yellow]"
                )
                enable_confirm = (
                    console.input(
                        "[bold white]Enable aiplatform.googleapis.com now? [Y/n]: [/bold white]"
                    )
                    .strip()
                    .lower()
                )
                if enable_confirm in ("", "y", "yes"):
                    with console.status("[bold cyan]Enabling Vertex AI API...[/bold cyan]"):
                        ok = ServiceUsageManager.enable_api(target_proj)
                    if ok:
                        console.print(
                            "[bold green]✔ Vertex AI API enabled successfully![/bold green]"
                        )
                    else:
                        console.print(
                            f"[bold red]Failed enabling API. Run:[/bold red] gcloud services enable aiplatform.googleapis.com --project={target_proj}"
                        )
            else:
                console.print("[bold green]✔ Vertex AI API is active and ready.[/bold green]")
            continue

        if user_input.startswith("/doctor"):
            auto_fix = "--fix" in user_input
            with console.status("[bold cyan]Running health diagnostics...[/bold cyan]"):
                diag = await run_diagnostics(
                    token_provider=client.token_provider,
                    memory_store=memory_store,
                    code_graph_store=client.code_graph_store,
                )
                if auto_fix:
                    pf = PreflightEngine()
                    pf.run_checks(auto_fix=True)

            table = Table(
                title="System & Workspace Diagnostics",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Component", style="bold white", width=18)
            table.add_column("Status", width=12)
            table.add_column("Detail", style="dim")

            tok_str = (
                "[green]Valid[/green]"
                if diag["token_status"] == "valid"
                else f"[red]{diag['token_status']}[/red]"
            )
            db_str = (
                "[green]Healthy[/green]"
                if diag["database_status"] == "healthy"
                else f"[red]{diag['database_status']}[/red]"
            )
            cg_str = (
                "[green]Healthy[/green]"
                if diag.get("code_graph_status") == "healthy"
                else f"[red]{diag.get('code_graph_status')}[/red]"
            )
            cg_stats = diag.get("code_graph_stats", {})

            table.add_row(
                "Google Account",
                "[cyan]Active[/cyan]",
                diag.get("gcloud_account", "Unauthenticated"),
            )
            table.add_row("OAuth2 Token", tok_str, "")
            table.add_row("Target Project", "[cyan]Configured[/cyan]", client.project_id)
            table.add_row("SQLite Database", db_str, diag["database_path"])
            table.add_row(
                "AST Code Graph",
                cg_str,
                f"{cg_stats.get('symbols', 0)} symbols / {cg_stats.get('files', 0)} files ({cg_stats.get('edges', 0)} edges)",
            )
            table.add_row(
                "Git Sandbox Engine",
                "[green]Ready[/green]",
                "Ephemeral branches & atomic diff reviews available",
            )
            table.add_row(
                "Multi-Agent Swarm",
                "[green]Ready[/green]",
                "ODD Swarm (Architect, Feature Worker, QA Verifier)",
            )
            table.add_row(
                "MCP Federation",
                "[green]Ready[/green]",
                f"{diag.get('mcp_servers', 0)} servers / {diag.get('mcp_tools', 0)} tools active",
            )
            table.add_row(
                "TDD Repair Engine",
                "[green]Ready[/green]"
                if diag.get("tdd_status") == "ready"
                else "[yellow]Partial[/yellow]",
                "Autonomous Red-Green-Refactor loop & regression guards",
            )
            table.add_row(
                "Hybrid Code Search",
                "[green]Ready[/green]",
                "FTS5 BM25 + Dense Vectors + Reciprocal Rank Fusion",
            )
            table.add_row(
                "Git Host Platform",
                "[cyan]Detected[/cyan]",
                str(diag.get("git_host_platform", "unknown")),
            )
            table.add_row(
                "Circuit Breaker",
                "[green]Ready[/green]",
                str(diag.get("circuit_breaker_status", "CLOSED")),
            )
            table.add_row(
                "Split-Pane TUI",
                "[green]Ready[/green]",
                str(diag.get("tui_status", "ready")),
            )
            table.add_row(
                "Voice Engine",
                "[green]Ready[/green]",
                str(diag.get("voice_status", "headless_ready")),
            )
            table.add_row("Active Workspace", "[cyan]Detected[/cyan]", diag["workspace_key"])
            table.add_row("Model Endpoint", "[cyan]Configured[/cyan]", client.model_name)
            console.print(table)
            continue

        if user_input.startswith("/pr"):
            parts = user_input.split(maxsplit=1)
            action = parts[1].strip().lower() if len(parts) > 1 else "preview"
            if action == "create":
                provider = detect_git_host_provider(Path("."))
                synthesizer = PRSynthesizer(worktree_path=Path("."), client=client)
                pr_payload = synthesizer.generate_pr()
                pr_meta = provider.create_or_update_pr(
                    title=pr_payload["title"],
                    body=pr_payload["body"],
                    source_branch="HEAD",
                    target_branch="main",
                )
                console.print(
                    f"[bold green]✔ PR/MR Created or Updated![/bold green] #{pr_meta.number} -> {pr_meta.url}"
                )
            elif action in ("watch", "ci"):
                provider = detect_git_host_provider(Path("."))
                watcher = CIWatcher(provider=provider)
                watch_res = watcher.watch()
                console.print(
                    f"[bold cyan]CI Watch Status:[/bold cyan] {watch_res.final_status} "
                    f"(heal attempts: {watch_res.heal_attempts_used})"
                )
            else:
                synthesizer = PRSynthesizer(worktree_path=Path("."), client=client)
                pr_payload = synthesizer.generate_pr()
                console.print(
                    Panel(
                        pr_payload["commit_message"],
                        title="[bold cyan]Conventional Commit[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print(
                    Panel(
                        Markdown(pr_payload["body"]),
                        title=f"[bold magenta]ISO/IEC 25010 PR Preview: {pr_payload['title']}[/bold magenta]",
                        border_style="magenta",
                    )
                )
            continue

        if user_input.startswith("/tui"):
            app = TUIApp(layout_manager=TUILayoutManager())
            app.run()
            continue

        if user_input.startswith("/voice"):
            voice_engine = VoiceAgentEngine(client=client)
            voice_res = await voice_engine.run_session()
            console.print(
                f"[bold cyan]Voice Session ({voice_res.get('status', 'ready')}):[/bold cyan] "
                f"Endpoint={voice_res.get('websocket_url')}"
            )
            continue

        if user_input == "/help":
            console.print(
                "[bold]Available commands:[/bold]\n"
                " - [bold]/auto [on|off][/bold]: Toggle autonomous mode (auto-approve safe & medium actions in this session)\n"
                " - [bold]/sessions[/bold]: List persistent conversation sessions for current workspace\n"
                " - [bold]/resume [id][/bold]: Resume a previous session (or latest if omitted)\n"
                " - [bold]/session[/bold]: Show active session ID, turn count, and model\n"
                " - [bold]/setup[/bold]: Launch interactive Google Cloud project and onboarding setup wizard\n"
                " - [bold]/memories[/bold]: List persistent memories for current workspace\n"
                " - [bold]/compact[/bold]: Compact conversation context\n"
                " - [bold]/doctor[/bold]: Verify authentication, database, AST index, and workspace health\n"
                " - [bold]/model <name>[/bold]: Switch or inspect model\n"
                " - [bold]/index[/bold]: Re-index workspace code AST symbols and relations\n"
                " - [bold]/search <query>[/bold]: Hybrid BM25 + vector search with Reciprocal Rank Fusion\n"
                " - [bold]/sandbox [on|diff|apply|discard][/bold]: Manage isolated Git worktree sandboxing\n"
                " - [bold]/team <goal>[/bold]: Run autonomous multi-agent swarm (Architect -> Worker -> QA)\n"
                " - [bold]/mcp [list|reload][/bold]: Inspect and reload federated MCP servers and tools\n"
                " - [bold]/tdd <target>[/bold]: Run autonomous TDD self-healing loop on a failing test\n"
                " - [bold]/pr [preview|create|watch][/bold]: Synthesize, create, or watch GitHub/GitLab PR/MRs\n"
                " - [bold]/tui[/bold]: Launch fullscreen 4-zone Split-Pane Terminal UI\n"
                " - [bold]/voice[/bold]: Start bidirectional Gemini Live voice streaming session\n"
                " - [bold]/clear[/bold]: Clear current chat context\n"
                " - [bold]/exit[/bold]: Quit the agent and persist session summary"
            )
            continue

        attached_images = [img for img in pending_images if img.token in user_input]
        pending_images.clear()
        user_parts: list[dict[str, Any]] = [{"text": user_input}]
        if attached_images:
            summary_items = ", ".join(
                f"{img.token} ({img.mime_type}, {img.size_label})" for img in attached_images
            )
            console.print(f"[dim cyan]🖼 Attached {len(attached_images)} image(s): {summary_items}[/dim cyan]")
            for img in attached_images:
                user_parts.append(img.to_gemini_part())

        conversation_history.append({"role": "user", "parts": user_parts})
        conversation_history = compact_history(conversation_history, max_turns=12, keep_recent=6)

        streamer = RichMarkdownStreamer(console=console)
        pinned_footer = PinnedFooterManager()
        status = console.status("[dim cyan]Thinking...[/dim cyan]", spinner="dots")
        status_active = False

        def _handle_tool_call(
            name: str,
            args: dict[str, Any],
            _status: Any = status,
            _streamer: RichMarkdownStreamer = streamer,
            _footer: PinnedFooterManager = pinned_footer,
        ) -> None:
            nonlocal status_active
            if status_active:
                _status.stop()
                status_active = False
            _footer.update(
                activity=f"⚡ Tool: {name}",
                auto_approve=security_policy.auto_approve,
                model_name=client.model_name,
            )
            _streamer.on_tool_call(name, args)

        try:
            pinned_footer.pin(
                activity="● Thinking...",
                auto_approve=security_policy.auto_approve,
                model_name=client.model_name,
            )
            status.start()
            status_active = True
            stream_iter = client.stream_chat(
                conversation_history, on_tool_call=_handle_tool_call
            )
            try:
                first_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                first_chunk = None
            finally:
                if status_active:
                    status.stop()
                    status_active = False

            if first_chunk is not None:
                pinned_footer.update(
                    activity="● Streaming response...",
                    auto_approve=security_policy.auto_approve,
                    model_name=client.model_name,
                )
                streamer.on_chunk(first_chunk)
                async for chunk in stream_iter:
                    streamer.on_chunk(chunk)
            full_response = streamer.finalize()
        except Exception as e:
            if status_active:
                status.stop()
                status_active = False
            if streamer._live is not None:
                streamer._live.stop()
            # Remove unanswered user turn so it doesn't corrupt subsequent prompts
            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history.pop()
            console.print(
                f"\n[bold red]✖ No hubo respuesta del modelo '[bold]{client.model_name}[/bold]':[/bold red] {e}\n"
                "[bold yellow]→ Cambia de modelo con [bold]/model[/bold] para continuar.[/bold yellow]"
            )
            continue
        finally:
            pinned_footer.unpin()

        if full_response.strip():
            conversation_history.append({"role": "model", "parts": [{"text": full_response}]})
            session_store.save_session(
                session_id=active_session_id,
                messages=conversation_history,
                model_name=client.model_name,
                project_key=project_key,
            )
        else:
            # Remove unanswered user turn and notify the user instead of failing silently
            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history.pop()
            console.print(
                "[dim yellow]⚠ El modelo completó el turno sin texto de respuesta "
                f"(modelo activo: [bold]{client.model_name}[/bold]). Intenta reformular o cambiar de modelo con [bold]/model[/bold].[/dim yellow]"
            )


def print_help() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]cli-agent[/bold cyan] - Autonomous AI Developer Assistant\n\n"
            "[bold]Usage:[/bold]\n"
            "  cli-agent [options] [prompt]\n\n"
            "[bold]Session Options:[/bold]\n"
            "  [bold cyan]-c, --continue[/bold cyan]              Resume the most recent session in this workspace\n"
            "  [bold cyan]--resume [SESSION_ID][/bold cyan]       Resume a specific session (or latest if omitted)\n"
            "  [bold cyan]--session [SESSION_ID][/bold cyan]      Alias for --resume\n"
            "  [bold cyan]--sessions, --list-sessions[/bold cyan] List recorded sessions for current workspace and exit\n\n"
            "[bold]Execution Options:[/bold]\n"
            "  [bold cyan]--model MODEL_NAME[/bold cyan]          Override model (e.g. gemini-3.8-flash, claude-3-7-sonnet)\n"
            "  [bold cyan]--auto-approve[/bold cyan]              Auto-approve tool actions without prompting\n"
            "  [bold cyan]--max-turns N[/bold cyan]               Limit conversational loop turns\n"
            "  [bold cyan]--tui[/bold cyan]                       Launch Split-Pane Terminal UI\n"
            "  [bold cyan]--voice[/bold cyan]                     Launch bidirectional Gemini Live voice session\n"
            "  [bold cyan]-h, --help[/bold cyan]                  Show this help overview and exit\n\n"
            "[bold]REPL Slash Commands:[/bold]\n"
            "  /sessions, /resume [id], /session, /memories, /model, /pr, /doctor, /exit",
            title="[bold magenta]cli-agent CLI Help[/bold magenta]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


async def run_oneshot(
    prompt: str,
    auto_approve: bool = False,
    model_override: str | None = None,
    resume_session_id: str | None = None,
    continue_last: bool = False,
) -> None:
    cfg = load_config()
    model = model_override or cfg.model
    memory_store = MemoryStore()
    session_store = SessionStore()
    project_key = resolve_project_key()
    security_policy = make_security_policy(auto_approve=auto_approve or cfg.auto_approve)
    circuit_breaker = QuotaCircuitBreaker()
    client = VertexAgentClient(
        project_id=cfg.project,
        location=cfg.location,
        model_name=model,
        memory_store=memory_store,
        security_policy=security_policy,
        fallback_router=None,
        circuit_breaker=circuit_breaker,
        timeout=cfg.timeout,
    )

    active_session_id: str = generate_session_id()
    messages: list[dict[str, Any]] = []

    if resume_session_id:
        target = session_store.get_session(resume_session_id)
        if target:
            active_session_id = target.id
            messages = [dict(m) for m in target.messages]
            if not model_override:
                client.switch_model(target.model_name)
    elif continue_last:
        latest = session_store.get_latest_session(project_key=project_key)
        if latest:
            active_session_id = latest.id
            messages = [dict(m) for m in latest.messages]
            if not model_override:
                client.switch_model(latest.model_name)

    messages.append({"role": "user", "parts": [{"text": prompt}]})
    messages = compact_history(messages, max_turns=12, keep_recent=6)

    streamer = RichMarkdownStreamer(console=console, prompt_label="")
    try:
        with console.status("[dim cyan]Thinking...[/dim cyan]", spinner="dots"):
            stream_iter = client.stream_chat(messages, on_tool_call=streamer.on_tool_call)
            try:
                first_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                first_chunk = None
        if first_chunk is not None:
            streamer.on_chunk(first_chunk)
            async for chunk in stream_iter:
                streamer.on_chunk(chunk)
        full_response = streamer.finalize()
        if full_response.strip():
            messages.append({"role": "model", "parts": [{"text": full_response}]})
            session_store.save_session(
                session_id=active_session_id,
                messages=messages,
                model_name=client.model_name,
                project_key=project_key,
            )
    except Exception as e:
        if streamer._live is not None:
            streamer._live.stop()
        err_msg = str(e) or repr(e)
        console.print(
            f"[bold red]✖ No hubo respuesta del modelo '{client.model_name}' ({type(e).__name__}):[/bold red] {err_msg}\n"
            "[bold yellow]→ Cambia de modelo con [bold]--model <nombre>[/bold] o [bold]/model[/bold].[/bold yellow]"
        )
        sys.exit(1)


def main() -> None:
    raw_args = sys.argv[1:]

    if "-h" in raw_args or "--help" in raw_args:
        print_help()
        return

    if "--sessions" in raw_args or "--list-sessions" in raw_args:
        show_sessions(SessionStore(), project_key=resolve_project_key())
        return

    auto_approve = "--auto-approve" in raw_args
    tui_mode = "--tui" in raw_args
    voice_mode = "--voice" in raw_args

    continue_last = False
    resume_session_id: str | None = None
    model_override = None
    max_turns: int | None = None

    cli_args: list[str] = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg in ("--auto-approve", "--tui", "--voice"):
            i += 1
        elif arg in ("-c", "--continue"):
            continue_last = True
            i += 1
        elif arg in ("--resume", "--session"):
            if i + 1 < len(raw_args) and not raw_args[i + 1].startswith("-"):
                resume_session_id = raw_args[i + 1]
                i += 2
            else:
                continue_last = True
                i += 1
        elif arg == "--model":
            if i + 1 < len(raw_args):
                model_override = raw_args[i + 1]
                i += 2
            else:
                i += 1
        elif arg == "--max-turns":
            if i + 1 < len(raw_args):
                try:
                    max_turns = int(raw_args[i + 1])
                except ValueError:
                    max_turns = None
                i += 2
            else:
                i += 1
        else:
            cli_args.append(arg)
            i += 1

    if tui_mode:
        TUIApp(layout_manager=TUILayoutManager()).run()
        return

    if voice_mode:
        cfg = load_config()
        model = model_override or cfg.model
        client = VertexAgentClient(
            project_id=cfg.project,
            location=cfg.location,
            model_name=model,
        )
        asyncio.run(VoiceAgentEngine(client=client).run_session())
        return

    interactive_kwargs: dict[str, Any] = {
        "auto_approve": auto_approve,
        "model_override": model_override,
        "max_turns": max_turns,
    }
    if resume_session_id is not None:
        interactive_kwargs["resume_session_id"] = resume_session_id
    if continue_last:
        interactive_kwargs["continue_last"] = continue_last

    oneshot_kwargs: dict[str, Any] = {
        "auto_approve": auto_approve,
        "model_override": model_override,
    }
    if resume_session_id is not None:
        oneshot_kwargs["resume_session_id"] = resume_session_id
    if continue_last:
        oneshot_kwargs["continue_last"] = continue_last

    if cli_args:
        prompt = " ".join(cli_args)
        asyncio.run(run_oneshot(prompt, **oneshot_kwargs))
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if prompt:
            asyncio.run(run_oneshot(prompt, **oneshot_kwargs))
        else:
            asyncio.run(run_interactive(**interactive_kwargs))
    else:
        asyncio.run(run_interactive(**interactive_kwargs))


if __name__ == "__main__":
    main()
