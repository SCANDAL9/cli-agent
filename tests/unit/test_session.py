"""
Unit tests for Session persistence and resumption engine in cli-agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cli_agent.main import (
    main,
    print_help,
    render_session_transcript,
    run_oneshot,
    show_sessions,
)
from cli_agent.session import (
    SessionRecord,
    SessionStore,
    derive_session_title,
    generate_session_id,
)


@pytest.fixture
def temp_session_store(tmp_path: Path) -> SessionStore:
    db_file = tmp_path / "test_session.db"
    return SessionStore(db_path=db_file)


def test_generate_session_id() -> None:
    s1 = generate_session_id()
    s2 = generate_session_id()
    assert s1.startswith("sess_")
    assert s2.startswith("sess_")
    assert s1 != s2


def test_derive_session_title() -> None:
    messages = [
        {"role": "user", "parts": [{"text": "/setup"}]},
        {"role": "model", "parts": [{"text": "Welcome to setup"}]},
        {"role": "user", "parts": [{"text": "Fix the database migration script"}]},
    ]
    title = derive_session_title(messages)
    assert title == "Fix the database migration script"

    # Truncation > 50 chars
    long_msg = [
        {
            "role": "user",
            "parts": [
                {"text": "A" * 60}
            ],
        }
    ]
    truncated_title = derive_session_title(long_msg)
    assert len(truncated_title) == 50
    assert truncated_title.endswith("...")

    # Fallback
    assert derive_session_title([], fallback="Default Fallback") == "Default Fallback"


def test_session_record_properties() -> None:
    messages = [
        {"role": "user", "parts": [{"text": "First user message"}]},
        {"role": "model", "parts": [{"text": "First model reply"}]},
        {"role": "user", "parts": [{"text": "Second user message\nSecond line"}]},
    ]
    record = SessionRecord(
        id="sess_123456",
        project_key="proj_test",
        title="Test Session",
        model_name="gemini-3.8-flash",
        messages=messages,
        created_at="2025-01-01 10:00:00",
        updated_at="2025-01-01 10:05:00",
    )
    assert record.turn_count == 3
    assert record.preview == "Second user message"

    # Preview fallback to title
    record_empty = SessionRecord(
        id="sess_empty",
        project_key="proj_test",
        title="Empty Session",
        model_name="gemini-3.8-flash",
        messages=[],
        created_at="2025-01-01 10:00:00",
        updated_at="2025-01-01 10:05:00",
    )
    assert record_empty.preview == "Empty Session"


def test_save_and_get_session(temp_session_store: SessionStore) -> None:
    sess_id = "sess_20250101_abcdef"
    msgs = [{"role": "user", "parts": [{"text": "Hello world"}]}]

    record = temp_session_store.save_session(
        session_id=sess_id,
        messages=msgs,
        model_name="gemini-3.8-flash",
        project_key="workspace-a",
        title="Hello Test",
    )
    assert record.id == sess_id
    assert record.project_key == "workspace-a"
    assert record.title == "Hello Test"
    assert len(record.messages) == 1

    # Exact get
    fetched = temp_session_store.get_session(sess_id)
    assert fetched is not None
    assert fetched.id == sess_id
    assert fetched.messages == msgs

    # Prefix match (>= 6 chars)
    prefix_fetched = temp_session_store.get_session("sess_20250101")
    assert prefix_fetched is not None
    assert prefix_fetched.id == sess_id

    # Non-existent
    assert temp_session_store.get_session("sess_nonexistent_123") is None


def test_save_session_update_on_conflict(temp_session_store: SessionStore) -> None:
    sess_id = "sess_20250101_conflict"
    msgs_1 = [{"role": "user", "parts": [{"text": "Turn 1"}]}]
    temp_session_store.save_session(
        session_id=sess_id,
        messages=msgs_1,
        model_name="gemini-3.8-flash",
        project_key="workspace-a",
    )

    msgs_2 = msgs_1 + [{"role": "model", "parts": [{"text": "Turn 2"}]}]
    updated = temp_session_store.save_session(
        session_id=sess_id,
        messages=msgs_2,
        model_name="gemini-3.7-flash",
        project_key="workspace-a",
    )
    assert updated.turn_count == 2
    assert updated.model_name == "gemini-3.7-flash"


def test_get_latest_session(temp_session_store: SessionStore) -> None:
    assert temp_session_store.get_latest_session() is None

    import time
    temp_session_store.save_session(
        session_id="sess_workspace_a_old",
        messages=[{"role": "user", "parts": [{"text": "A1"}]}],
        model_name="gemini-3.8-flash",
        project_key="ws-a",
    )
    time.sleep(0.01)
    temp_session_store.save_session(
        session_id="sess_workspace_b_new",
        messages=[{"role": "user", "parts": [{"text": "B1"}]}],
        model_name="gemini-3.8-flash",
        project_key="ws-b",
    )

    latest_global = temp_session_store.get_latest_session()
    assert latest_global is not None
    assert latest_global.id == "sess_workspace_b_new"

    latest_ws_a = temp_session_store.get_latest_session(project_key="ws-a")
    assert latest_ws_a is not None
    assert latest_ws_a.id == "sess_workspace_a_old"


def test_list_and_delete_sessions(temp_session_store: SessionStore) -> None:
    for i in range(5):
        temp_session_store.save_session(
            session_id=f"sess_{i:02d}",
            messages=[{"role": "user", "parts": [{"text": f"Msg {i}"}]}],
            model_name="gemini-3.8-flash",
            project_key="ws-x" if i % 2 == 0 else "ws-y",
        )

    all_sessions = temp_session_store.list_sessions()
    assert len(all_sessions) == 5

    ws_x_sessions = temp_session_store.list_sessions(project_key="ws-x")
    assert len(ws_x_sessions) == 3

    # Delete
    assert temp_session_store.delete_session("sess_00") is True
    assert temp_session_store.delete_session("sess_00") is False
    assert len(temp_session_store.list_sessions()) == 4

    # Clear workspace
    cleared_count = temp_session_store.clear_sessions(project_key="ws-x")
    assert cleared_count == 2
    assert len(temp_session_store.list_sessions()) == 2

    # Clear all
    temp_session_store.clear_sessions()
    assert len(temp_session_store.list_sessions()) == 0


def test_get_session_corrupt_json(temp_session_store: SessionStore) -> None:
    sess_id = "sess_corrupt_json"
    temp_session_store.save_session(
        session_id=sess_id,
        messages=[],
        model_name="gemini-3.8-flash",
        project_key="ws",
    )
    with temp_session_store._get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET messages_json = 'INVALID_JSON' WHERE id = ?",
            (sess_id,),
        )
        conn.commit()

    record = temp_session_store.get_session(sess_id)
    assert record is not None
    assert record.messages == []


def test_show_sessions_visual(temp_session_store: SessionStore) -> None:
    # Empty case
    show_sessions(temp_session_store, project_key="empty_ws")

    # Populated case
    temp_session_store.save_session(
        session_id="sess_show_test_01",
        messages=[{"role": "user", "parts": [{"text": "Test prompt"}]}],
        model_name="gemini-3.8-flash",
        project_key="test_ws",
    )
    show_sessions(temp_session_store, project_key="test_ws")
    show_sessions(temp_session_store, project_key=None)


def test_print_help() -> None:
    print_help()


def test_main_cli_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Test --help
    with patch("cli_agent.main.print_help") as mock_help:
        monkeypatch.setattr("sys.argv", ["cli-agent", "--help"])
        main()
        mock_help.assert_called_once()

    # Test --sessions
    with patch("cli_agent.main.show_sessions") as mock_show:
        monkeypatch.setattr("sys.argv", ["cli-agent", "--sessions"])
        main()
        mock_show.assert_called_once()

    # Test --continue / -c flag passes continue_last=True to run_interactive
    with patch("cli_agent.main.run_interactive", new_callable=MagicMock) as mock_interactive, patch(
        "cli_agent.main.asyncio.run", side_effect=lambda coro: None
    ):
        monkeypatch.setattr("sys.argv", ["cli-agent", "--continue"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        main()
        mock_interactive.assert_called_once_with(
            auto_approve=False,
            model_override=None,
            max_turns=None,
            continue_last=True,
        )

    # Test -c shorthand
    with patch("cli_agent.main.run_interactive", new_callable=MagicMock) as mock_interactive, patch(
        "cli_agent.main.asyncio.run", side_effect=lambda coro: None
    ):
        monkeypatch.setattr("sys.argv", ["cli-agent", "-c"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        main()
        mock_interactive.assert_called_once_with(
            auto_approve=False,
            model_override=None,
            max_turns=None,
            continue_last=True,
        )

    # Test --resume <id>
    with patch("cli_agent.main.run_interactive", new_callable=MagicMock) as mock_interactive, patch(
        "cli_agent.main.asyncio.run", side_effect=lambda coro: None
    ):
        monkeypatch.setattr("sys.argv", ["cli-agent", "--resume", "sess_20250101_abcdef"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        main()
        mock_interactive.assert_called_once_with(
            auto_approve=False,
            model_override=None,
            max_turns=None,
            resume_session_id="sess_20250101_abcdef",
        )

    # Test --session <id>
    with patch("cli_agent.main.run_interactive", new_callable=MagicMock) as mock_interactive, patch(
        "cli_agent.main.asyncio.run", side_effect=lambda coro: None
    ):
        monkeypatch.setattr("sys.argv", ["cli-agent", "--session", "sess_custom_id"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        main()
        mock_interactive.assert_called_once_with(
            auto_approve=False,
            model_override=None,
            max_turns=None,
            resume_session_id="sess_custom_id",
        )

    # Test --continue with explicit prompt triggers run_oneshot
    with patch("cli_agent.main.run_oneshot", new_callable=MagicMock) as mock_oneshot, patch(
        "cli_agent.main.asyncio.run", side_effect=lambda coro: None
    ):
        monkeypatch.setattr(
            "sys.argv", ["cli-agent", "--continue", "resume", "and", "do", "task"]
        )
        main()
        mock_oneshot.assert_called_once_with(
            "resume and do task",
            auto_approve=False,
            model_override=None,
            continue_last=True,
        )


@pytest.mark.asyncio
async def test_run_oneshot_with_resumed_session(tmp_path: Path) -> None:
    db_file = tmp_path / "oneshot_session.db"
    store = SessionStore(db_path=db_file)
    sess_id = "sess_oneshot_target"
    store.save_session(
        session_id=sess_id,
        messages=[{"role": "user", "parts": [{"text": "Prior message"}]}],
        model_name="gemini-3.8-flash",
        project_key="default",
    )

    async def mock_stream(*args: Any, **kwargs: Any) -> Any:
        yield "Response chunk"

    with patch("cli_agent.main.SessionStore", return_value=store), patch(
        "cli_agent.main.resolve_project_key", return_value="default"
    ), patch("cli_agent.main.VertexAgentClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.model_name = "gemini-3.8-flash"
        mock_client.stream_chat = mock_stream
        mock_client_cls.return_value = mock_client

        await run_oneshot(
            prompt="Follow up question",
            resume_session_id=sess_id,
        )

    # Verify session was updated with both prior message, new user prompt, and model response
    updated = store.get_session(sess_id)
    assert updated is not None
    assert updated.turn_count == 3
    assert updated.messages[-1]["role"] == "model"


def test_render_session_transcript_replays_full_history() -> None:
    from rich.console import Console

    rec_console = Console(record=True, width=100)
    messages = [
        {
            "role": "user",
            "parts": [
                {"text": "[Image #1] Explica esta arquitectura"},
                {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
            ],
        },
        {
            "role": "model",
            "parts": [
                {"text": "### Resumen\n\nFlujo: Cliente $\\rightarrow$ Servidor (`O(n^2)`)."},
            ],
        },
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "read_file", "response": {"ok": True}}}],
        },
    ]

    render_session_transcript(messages, console_obj=rec_console)
    output = rec_console.export_text()
    assert "Explica esta arquitectura" in output
    assert "1 image(s) attached (image/png)" in output
    assert "kanon ❯" in output
    assert "Cliente → Servidor" in output
    assert "read_file" not in output

