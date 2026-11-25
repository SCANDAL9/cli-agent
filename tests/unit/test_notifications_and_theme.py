"""
Unit tests for soft Matugen Rich Markdown theme and Kitty OSC 99 / notify-send background notifications.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.markdown import Markdown

from cli_agent.notifications import (
    build_kitty_osc99_notification,
    send_background_notification,
)
from cli_agent.theme import MatugenPalette, build_rich_theme


@pytest.mark.unit
def test_build_rich_theme_uses_subtle_matugen_code_highlight() -> None:
    palette = MatugenPalette(
        foreground="#e4e3d7",
        background="#13140d",
        surface="#201f21",
        primary="#bccf81",
        secondary="#a1d0c5",
        accent="#dde1f9",
        muted="#929092",
        border="#474648",
        is_matugen=True,
    )
    theme = build_rich_theme(palette)
    assert "markdown.code" in theme.styles
    code_style = str(theme.styles["markdown.code"])
    assert "#bccf81" in code_style
    assert "#201f21" in code_style
    assert "black" not in code_style

    # Renders Markdown cleanly without errors
    console = Console(theme=theme, record=True, width=80)
    console.print(Markdown("Commit `bdd6554` updates `scripts/kanon.sh`."))
    assert "bdd6554" in console.export_text()


@pytest.mark.unit
def test_kitty_osc99_notification_format_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = build_kitty_osc99_notification(
        title="kanon — Permiso requerido (HIGH)",
        body="run_command: rm -rf /tmp/foo",
        urgency="critical",
        only_when_unfocused=True,
    )
    assert "\x1b]99;" in seq
    assert "o=unfocused" in seq
    assert "u=2" in seq
    assert "a=focus" in seq
    assert "p=title;kanon — Permiso requerido (HIGH)" in seq
    assert "p=body;run_command: rm -rf /tmp/foo" in seq

    # Inside Kitty -> writes OSC 99 directly to stream
    monkeypatch.setenv("KITTY_WINDOW_ID", "10")
    buf = io.StringIO()
    ok = send_background_notification(
        title="kanon — Tarea completada",
        body="3 commits creados.",
        stream=buf,
    )
    assert ok is True
    assert "o=unfocused" in buf.getvalue()
    assert "3 commits creados." in buf.getvalue()


@pytest.mark.unit
def test_kitty_osc99_notification_sanitizes_and_bounds_payloads() -> None:
    title = " \tTask\x1b[31m " + "T" * 110 + "\n\x07"
    body = " \tDetails\n" + "B" * 200 + "\x1b\x07"

    sequence = build_kitty_osc99_notification(title=title, body=body)

    expected_title = ("Task[31m " + "T" * 110)[:99] + "…"
    expected_body = ("Details " + "B" * 200)[:179] + "…"
    assert f"p=title;{expected_title}\x1b\\" in sequence
    assert f"p=body;{expected_body}\x1b\\" in sequence
    assert "\x1b[31m" not in sequence
    assert "\x07" not in sequence


@pytest.mark.unit
def test_background_notification_fallback_to_notify_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    buf = io.StringIO()
    with (
        patch("cli_agent.notifications.shutil.which", return_value="/usr/bin/notify-send"),
        patch("cli_agent.notifications.subprocess.Popen") as mock_popen,
    ):
        ok = send_background_notification(
            title="kanon — Tarea completada",
            body="Listo",
            urgency="normal",
            stream=buf,
        )
        assert ok is True
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[:5] == ["/usr/bin/notify-send", "-a", "kanon", "-u", "normal"]
