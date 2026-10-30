"""Unit tests for Fullscreen Split-Pane TUI (Wave 4 Milestone 3)."""

from pathlib import Path

import pytest

from cli_agent.circuit_breaker import CircuitState, EndpointHealthSnapshot, ModelEndpoint
from cli_agent.pr import SymbolDelta
from cli_agent.tui import (
    ChatPane,
    DiffViewerPane,
    DiffViewMode,
    FocusPane,
    LayoutMode,
    LogSource,
    ProcessLogPane,
    StatusHUD,
    SwarmNode,
    TerminalDimensions,
    TUIApp,
    TUILayoutManager,
)


def test_layout_manager_responsive_breakpoints() -> None:
    chat = ChatPane()
    diff = DiffViewerPane()
    logs = ProcessLogPane()
    hud = StatusHUD()

    mgr = TUILayoutManager(chat, diff, logs, hud, TerminalDimensions(140, 45))
    assert mgr.determine_layout_mode() == LayoutMode.STANDARD_SPLIT
    frame = mgr.render_frame()
    assert frame.columns == 140
    assert frame.rows == 45
    assert len(frame.lines) > 0

    mgr.resize(90, 20)
    assert mgr.determine_layout_mode() == LayoutMode.COMPACT_VERTICAL

    mgr.resize(70, 15)
    assert mgr.determine_layout_mode() == LayoutMode.MINIMAL_SAFE


def test_chat_pane_streaming_and_swarm_tree() -> None:
    chat = ChatPane()
    chat.add_message("user", "Implement TUI layout")

    swarm = SwarmNode(
        agent_id="lead-arch",
        role="Lead",
        status="RUNNING",
        children=[
            SwarmNode(
                agent_id="worker-1",
                role="Worker",
                status="COMPLETED",
                worktree_branch="feat/tui",
                active_tool="write_file",
            ),
            SwarmNode(
                agent_id="qa-verifier",
                role="QA",
                status="RUNNING",
                active_tool="pytest",
            ),
        ],
    )
    chat.update_swarm(swarm)

    chat.append_streaming_token("Working on ")
    chat.append_streaming_token("split-pane...")
    panel = chat.render()
    assert panel is not None

    chat.commit_streaming_turn()
    assert len(chat.messages) == 2
    assert chat.messages[1]["content"] == "Working on split-pane..."


def test_diff_viewer_pane_modes_and_file_cycling() -> None:
    diff_pane = DiffViewerPane()
    diff_pane.update_file_diff(
        "src/cli_agent/tui.py",
        "@@ -1,3 +1,5 @@\n+class TUIApp:\n+    pass",
        SymbolDelta(
            file_path="src/cli_agent/tui.py",
            change_type="ADDED",
            added_symbols=("TUIApp", "StatusHUD"),
            modified_symbols=("render_frame",),
        ),
    )
    diff_pane.update_file_diff(
        "src/cli_agent/pr.py",
        "@@ -10,3 +10,4 @@\n+class GitHostProvider:\n",
    )

    assert diff_pane.active_file == "src/cli_agent/tui.py"
    diff_pane.cycle_file(1)
    assert diff_pane.active_file == "src/cli_agent/pr.py"
    diff_pane.cycle_file(-1)
    assert diff_pane.active_file == "src/cli_agent/tui.py"

    # Toggle to AST Outline
    assert diff_pane.mode == DiffViewMode.UNIFIED_DIFF
    diff_pane.toggle_view_mode()
    assert diff_pane.mode == DiffViewMode.AST_OUTLINE
    panel = diff_pane.render()
    assert panel is not None


def test_process_log_pane_ring_buffer_and_filtering() -> None:
    log_pane = ProcessLogPane(max_lines=50)
    for i in range(120):
        src = LogSource.TDD if i % 2 == 0 else LogSource.MCP
        log_pane.append_log(src, f"log entry {i}")

    # Ring buffer strictly respects max_lines
    assert len(log_pane.entries) == 50
    # FIFO: oldest entry should be 120 - 50 = 70
    assert log_pane.entries[0][1] == "log entry 70"

    log_pane.set_filter(LogSource.TDD)
    panel = log_pane.render()
    assert panel is not None

    log_pane.clear()
    assert len(log_pane.entries) == 0


def test_status_hud_circuit_breaker_states() -> None:
    hud = StatusHUD()
    ep = ModelEndpoint("anthropic/claude-sonnet-4-6", "us-east5", 1)

    # CLOSED state
    snap_closed = EndpointHealthSnapshot(
        endpoint=ep,
        state=CircuitState.CLOSED,
        consecutive_failures=0,
        cooldown_remaining_seconds=0.0,
    )
    hud.update_metrics(snap_closed, tokens_in=14200, tokens_out=3800, vad_listening=False)
    table_closed = hud.render()
    assert table_closed is not None

    # OPEN state with live countdown & active mic
    snap_open = EndpointHealthSnapshot(
        endpoint=ep,
        state=CircuitState.OPEN,
        consecutive_failures=1,
        cooldown_remaining_seconds=42.5,
    )
    hud.update_metrics(snap_open, tokens_in=25000, tokens_out=6000, vad_listening=True)
    table_open = hud.render()
    assert table_open is not None


def test_keyboard_dispatch_and_repl_toggle() -> None:
    emitted_codes: list[str] = []

    def mock_writer(code: str) -> None:
        emitted_codes.append(code)

    app = TUIApp(output_writer=mock_writer)
    app.start()
    assert app.is_running is True
    assert app.in_alternate_screen is True
    assert "\x1b[?1049h" in emitted_codes

    # Focus cycling
    assert app.layout_manager.focused_pane == FocusPane.CHAT
    app.handle_key("Tab")
    assert app.layout_manager.focused_pane == FocusPane.DIFF
    app.handle_key("Tab")
    assert app.layout_manager.focused_pane == FocusPane.LOGS
    app.handle_key("Shift+Tab")
    assert app.layout_manager.focused_pane == FocusPane.DIFF

    # Mode and file cycling keys
    app.handle_key("Ctrl+D")
    assert app.diff_pane.mode == DiffViewMode.AST_OUTLINE

    # Scroll keys
    app.handle_key("Up")
    assert app.diff_pane.scroll_offset == 1
    app.handle_key("Down")
    assert app.diff_pane.scroll_offset == 0

    # Command toggle /tui exits alternate screen
    cmd = app.handle_key("/tui")
    assert cmd == "/tui"
    assert app.is_running is False
    assert app.in_alternate_screen is False
    assert "\x1b[?1049l" in emitted_codes


@pytest.mark.asyncio
async def test_tui_app_stream_subprocess_logs(tmp_path: Path) -> None:
    app = TUIApp()
    returncode = await app.stream_subprocess_logs(
        [
            "python",
            "-c",
            "import sys; print('stdout_stream'); sys.stderr.write('stderr_stream\\n')",
        ],
        source=LogSource.TDD,
        cwd=tmp_path,
    )
    assert returncode == 0
    assert len(app.log_pane.entries) >= 2
    log_texts = [entry[1] for entry in app.log_pane.entries]
    assert any("stdout_stream" in t for t in log_texts)
    assert any("stderr_stream" in t for t in log_texts)


def test_tui_post_event_and_empty_diff() -> None:
    from cli_agent.tui import TUIEvent, TUIEventType

    app = TUIApp()
    app.post_event(TUIEvent(TUIEventType.CHAT_TOKEN, {"token": "hello"}))
    assert app._event_queue.qsize() == 1

    # Empty diff rendering
    empty_diff = DiffViewerPane()
    assert empty_diff.render() is not None
