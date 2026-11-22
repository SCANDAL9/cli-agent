"""
Background desktop notifications for kanon (cli-agent).
Uses Kitty's native OSC 99 notification protocol (with focus-aware `o=unfocused`
and click-to-focus `a=focus`) when running inside Kitty, falling back to `notify-send`
on other Linux desktop terminals.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TextIO

from .clipboard import is_kitty_terminal

_URGENCY_TO_OSC99: dict[str, int] = {
    "low": 0,
    "normal": 1,
    "critical": 2,
}


def _sanitize_osc_text(text: str, max_len: int = 180) -> str:
    """Strips ANSI/control characters and collapses whitespace for OSC 99 payloads."""
    cleaned = " ".join(str(text or "").replace("\x1b", "").replace("\x07", "").split())
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1] + "…"
    return cleaned


def build_kitty_osc99_notification(
    title: str,
    body: str = "",
    *,
    urgency: str = "normal",
    only_when_unfocused: bool = True,
    notif_id: str = "kanon",
) -> str:
    """
    Encodes a Kitty OSC 99 desktop notification escape sequence.
    When `only_when_unfocused=True`, Kitty only displays the notification if the
    terminal window is in the background (`o=unfocused`), and clicking it focuses Kitty (`a=focus`).
    """
    safe_title = _sanitize_osc_text(title, max_len=100) or "kanon"
    safe_body = _sanitize_osc_text(body, max_len=180)
    u_val = _URGENCY_TO_OSC99.get(urgency.lower(), 1)
    occ = "unfocused" if only_when_unfocused else "always"

    if not safe_body:
        return f"\x1b]99;i={notif_id}:d=1:o={occ}:u={u_val}:a=focus:p=title;{safe_title}\x1b\\"

    part_title = f"\x1b]99;i={notif_id}:d=0:o={occ}:u={u_val}:a=focus:p=title;{safe_title}\x1b\\"
    part_body = f"\x1b]99;i={notif_id}:d=1:p=body;{safe_body}\x1b\\"
    return part_title + part_body


def send_background_notification(
    title: str,
    body: str = "",
    *,
    urgency: str = "normal",
    only_when_unfocused: bool = True,
    stream: TextIO | None = None,
) -> bool:
    """
    Dispatches a non-blocking background notification when kanon requires permission
    or finishes a task. Does nothing in non-TTY headless/test environments unless `stream` is given.
    """
    out = stream or sys.stdout
    is_tty = bool(getattr(out, "isatty", lambda: False)())
    if stream is None and not is_tty:
        return False

    if is_kitty_terminal():
        seq = build_kitty_osc99_notification(
            title=title,
            body=body,
            urgency=urgency,
            only_when_unfocused=only_when_unfocused,
        )
        try:
            out.write(seq)
            out.flush()
            return True
        except Exception:
            return False

    notify_bin = shutil.which("notify-send")
    if notify_bin:
        safe_title = _sanitize_osc_text(title, max_len=100) or "kanon"
        safe_body = _sanitize_osc_text(body, max_len=180)
        urg = urgency.lower() if urgency.lower() in _URGENCY_TO_OSC99 else "normal"
        cmd = [notify_bin, "-a", "kanon", "-u", urg, safe_title]
        if safe_body:
            cmd.append(safe_body)
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            return False

    return False
