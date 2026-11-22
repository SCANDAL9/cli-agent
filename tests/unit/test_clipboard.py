"""
Unit tests for clipboard image/text paste (Ctrl+V) and multimodal Gemini inlineData parts.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cli_agent.clipboard import (
    ImageAttachment,
    build_kitty_graphics_sequence,
    delete_kitty_image_sequence,
    is_kitty_terminal,
    load_image_from_path,
    read_clipboard_image,
    read_clipboard_text,
    render_kitty_image_preview,
)
from cli_agent.context import compact_history, estimate_context_tokens
from cli_agent.main import (
    AlternateScreenWorkspace,
    PinnedFooterManager,
    build_input_key_bindings,
    format_bottom_toolbar,
)

SAMPLE_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"


@pytest.mark.unit
def test_image_attachment_to_gemini_part_and_labels() -> None:
    att = ImageAttachment(index=1, mime_type="image/png", data=SAMPLE_PNG)
    assert att.token == "[Image #1]"
    assert att.size_label == f"{len(SAMPLE_PNG)} B"

    part = att.to_gemini_part()
    assert "inlineData" in part
    assert part["inlineData"]["mimeType"] == "image/png"
    assert base64.b64decode(part["inlineData"]["data"]) == SAMPLE_PNG

    kb_att = ImageAttachment(index=2, mime_type="image/jpeg", data=b"x" * 2048)
    assert kb_att.size_label == "2.0 KB"


@pytest.mark.unit
def test_load_image_from_path_and_uri(tmp_path: Path) -> None:
    img_file = tmp_path / "shot.png"
    img_file.write_bytes(SAMPLE_PNG)

    loaded = load_image_from_path(img_file)
    assert loaded == ("image/png", SAMPLE_PNG)

    loaded_uri = load_image_from_path(f"file://{img_file}")
    assert loaded_uri == ("image/png", SAMPLE_PNG)

    # Unsupported extension returns None
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello", encoding="utf-8")
    assert load_image_from_path(txt_file) is None


@pytest.mark.unit
def test_read_clipboard_image_wayland(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        if "--list-types" in cmd:
            return SimpleNamespace(returncode=0, stdout="text/plain\nimage/png\n")
        if "--type" in cmd and "image/png" in cmd:
            return SimpleNamespace(returncode=0, stdout=SAMPLE_PNG)
        return SimpleNamespace(returncode=1, stdout=b"")

    with (
        patch("cli_agent.clipboard.shutil.which", side_effect=lambda c: "/usr/bin/wl-paste" if c == "wl-paste" else None),
        patch("cli_agent.clipboard.subprocess.run", side_effect=fake_run),
    ):
        res = read_clipboard_image()
        assert res == ("image/png", SAMPLE_PNG)


@pytest.mark.unit
def test_read_clipboard_text_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")

    with (
        patch("cli_agent.clipboard.shutil.which", side_effect=lambda c: "/usr/bin/wl-paste" if c == "wl-paste" else None),
        patch(
            "cli_agent.clipboard.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="copied text"),
        ),
    ):
        assert read_clipboard_text() == "copied text"


@pytest.mark.unit
def test_ctrl_v_keybinding_attaches_image_and_falls_back_to_text(tmp_path: Path) -> None:
    pending: list[ImageAttachment] = []
    kb = build_input_key_bindings(pending, enable_kitty_preview=False)
    ctrl_v_handler = next(b.handler for b in kb.bindings if b.keys == ("c-v",))

    # 1. Image in clipboard -> attaches [Image #1]
    buf = MagicMock()
    event = SimpleNamespace(current_buffer=buf)
    with patch("cli_agent.main.read_clipboard_image", return_value=("image/png", SAMPLE_PNG)):
        ctrl_v_handler(event)

    assert len(pending) == 1
    assert pending[0].token == "[Image #1]"
    buf.insert_text.assert_called_once_with("[Image #1] ")

    # 2. Image file path in clipboard text -> attaches [Image #2]
    img_path = tmp_path / "diagram.webp"
    img_path.write_bytes(b"RIFF....WEBP")
    buf2 = MagicMock()
    event2 = SimpleNamespace(current_buffer=buf2)
    with (
        patch("cli_agent.main.read_clipboard_image", return_value=None),
        patch("cli_agent.main.read_clipboard_text", return_value=str(img_path)),
    ):
        ctrl_v_handler(event2)

    assert len(pending) == 2
    assert pending[1].token == "[Image #2]"
    assert pending[1].mime_type == "image/webp"
    buf2.insert_text.assert_called_once_with("[Image #2] ")

    # 3. Plain text in clipboard -> ignored by Ctrl+V (Ctrl+Shift+V handles text natively)
    buf3 = MagicMock()
    event3 = SimpleNamespace(current_buffer=buf3)
    with (
        patch("cli_agent.main.read_clipboard_image", return_value=None),
        patch("cli_agent.main.read_clipboard_text", return_value="hello world"),
    ):
        ctrl_v_handler(event3)

    assert len(pending) == 2
    buf3.insert_text.assert_not_called()


@pytest.mark.unit
def test_bottom_toolbar_and_context_with_images() -> None:
    tb = format_bottom_toolbar(
        terminal_width=100, auto_approve=False, model_name="gemini-4-hydrogen", image_count=2
    )
    rendered = "".join(part[1] for part in tb)
    assert "🖼 2 img" in rendered

    att = ImageAttachment(index=1, mime_type="image/png", data=SAMPLE_PNG)
    history = [
        {"role": "user", "parts": [{"text": "[Image #1] What is this?"}, att.to_gemini_part()]},
        {"role": "model", "parts": [{"text": "It is a 1x1 PNG."}]},
    ]
    tokens = estimate_context_tokens(history)
    assert 200 <= tokens <= 400

    # Compaction summarizes image parts cleanly
    long_history = history * 8
    compacted = compact_history(long_history, max_turns=4, keep_recent=2)
    assert "[image attachment]: image/png" in compacted[0]["parts"][0]["text"]


@pytest.mark.unit
def test_atomic_backspace_and_delete_remove_image_token_and_attachment() -> None:
    att1 = ImageAttachment(index=1, mime_type="image/png", data=SAMPLE_PNG)
    att2 = ImageAttachment(index=2, mime_type="image/png", data=SAMPLE_PNG)
    pending = [att1, att2]
    kb = build_input_key_bindings(pending, enable_kitty_preview=False)
    bs_handler = next(b.handler for b in kb.bindings if b.keys == ("c-h",))
    del_handler = next(b.handler for b in kb.bindings if b.keys == ("delete",))

    # Backspace right after "[Image #1] " deletes the whole token & removes att1
    buf = SimpleNamespace(
        text="[Image #1] hola [Image #2] ",
        cursor_position=len("[Image #1] "),
        delete_before_cursor=MagicMock(),
        delete=MagicMock(),
    )
    bs_handler(SimpleNamespace(current_buffer=buf))
    assert buf.text == "hola [Image #2] "
    assert buf.cursor_position == 0
    assert [a.index for a in pending] == [2]

    # Forward delete at start of "[Image #2] " deletes the whole token & removes att2
    buf.cursor_position = len("hola ")
    del_handler(SimpleNamespace(current_buffer=buf))
    assert buf.text == "hola "
    assert pending == []

    # Normal backspace outside any image token delegates to delete_before_cursor
    buf.cursor_position = 4
    bs_handler(SimpleNamespace(current_buffer=buf))
    buf.delete_before_cursor.assert_called_once_with(count=1)


@pytest.mark.unit
def test_kitty_graphics_protocol_and_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITTY_WINDOW_ID", "10")
    assert is_kitty_terminal() is True

    seq = build_kitty_graphics_sequence(SAMPLE_PNG, rows=6, image_id=9001)
    assert seq.startswith("\x1b_Ga=T,f=100,q=2,r=6,i=9001,m=0;")
    assert seq.endswith("\x1b\\")

    del_seq = delete_kitty_image_sequence(9001)
    assert del_seq == "\x1b_Ga=d,d=I,i=9001,q=2\x1b\\"

    # Multi-chunk encoding for payloads > 4096 base64 chars
    big_seq = build_kitty_graphics_sequence(b"A" * 5000, rows=8, image_id=9002)
    assert "m=1;" in big_seq
    assert "\x1b_Gm=0;" in big_seq

    buf = io.StringIO()
    att = ImageAttachment(index=1, mime_type="image/png", data=SAMPLE_PNG)
    assert render_kitty_image_preview(att, stream=buf, rows=6) is True
    out = buf.getvalue()
    assert "[Image #1]" in out
    assert "\x1b_Ga=T,f=100,q=2,r=6,i=9001,m=0;" in out


@pytest.mark.unit
def test_pinned_footer_manager_decstbm_lifecycle() -> None:
    stream = io.StringIO()
    mgr = PinnedFooterManager(stream=stream, enabled=True)
    mgr.pin(activity="● Thinking...", auto_approve=False, model_name="gemini-4-hydrogen")
    mgr.update(activity="● Streaming response...", auto_approve=True, model_name="gemini-4-hydrogen", image_count=1)
    mgr.unpin()

    output = stream.getvalue()
    # DECSTBM scroll region set, footer painted, and scroll region reset (\x1b[r)
    assert "\x1b[1;" in output
    assert "Thinking..." in output
    assert "Streaming response..." in output
    assert "Gemini 4 Hydrogen" in output
    assert "\x1b[r" in output


@pytest.mark.unit
def test_alternate_screen_workspace_enter_and_exit() -> None:
    stream = io.StringIO()
    with AlternateScreenWorkspace(stream=stream, enabled=True):
        assert stream.getvalue() == "\x1b[3J\x1b[2J\x1b[H"
    assert stream.getvalue().endswith("\x1b[r")

    # Resets scroll margins even if an exception is raised inside
    stream_err = io.StringIO()
    with pytest.raises(RuntimeError), AlternateScreenWorkspace(stream=stream_err, enabled=True):
        raise RuntimeError("boom")
    assert stream_err.getvalue().endswith("\x1b[r")


