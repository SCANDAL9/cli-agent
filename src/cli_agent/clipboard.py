"""
Clipboard image & text ingestion for cli-agent.
Supports pasting images via Ctrl+V across Wayland (wl-paste), X11 (xclip/xsel),
and macOS (pngpaste/pbpaste), converting them into Vertex AI multimodal inlineData parts.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import unquote, urlparse

SUPPORTED_IMAGE_MIME_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
)

EXTENSION_TO_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB safety cap for Vertex AI inlineData


@dataclass
class ImageAttachment:
    """Represents a pasted or attached image ready to send as a Gemini multimodal part."""

    index: int
    mime_type: str
    data: bytes
    source: str = "clipboard"

    @property
    def token(self) -> str:
        """Inline placeholder token inserted into the prompt buffer."""
        return f"[Image #{self.index}]"

    @property
    def size_label(self) -> str:
        """Human-readable byte size string."""
        size = len(self.data)
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    def to_gemini_part(self) -> dict[str, Any]:
        """Serializes this image into a Vertex AI Gemini REST API inlineData part."""
        return {
            "inlineData": {
                "mimeType": self.mime_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            }
        }


def load_image_from_path(raw_path: str | Path) -> tuple[str, bytes] | None:
    """Loads an image from a local filesystem path or file:// URI if valid and supported."""
    text = str(raw_path).strip().strip("'\"")
    if not text:
        return None

    if text.startswith("file://"):
        parsed = urlparse(text)
        text = unquote(parsed.path)

    try:
        path = Path(text).expanduser()
        if not path.is_file():
            return None
        mime = EXTENSION_TO_MIME.get(path.suffix.lower())
        if not mime:
            return None
        if path.stat().st_size > MAX_IMAGE_BYTES:
            return None
        data = path.read_bytes()
        if not data:
            return None
        return mime, data
    except Exception:
        return None


def _read_wayland_clipboard_image() -> tuple[str, bytes] | None:
    """Attempts to read an image from the Wayland clipboard via wl-paste."""
    if not shutil.which("wl-paste"):
        return None
    try:
        types_proc = subprocess.run(
            ["wl-paste", "--list-types"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if types_proc.returncode != 0:
            return None
        available = [line.strip() for line in types_proc.stdout.splitlines() if line.strip()]
        for mime in SUPPORTED_IMAGE_MIME_TYPES:
            if mime in available:
                img_proc = subprocess.run(
                    ["wl-paste", "--no-newline", "--type", mime],
                    capture_output=True,
                    timeout=5,
                )
                if img_proc.returncode == 0 and 0 < len(img_proc.stdout) <= MAX_IMAGE_BYTES:
                    return mime, img_proc.stdout

        # Support copying image files from desktop file managers (text/uri-list)
        if "text/uri-list" in available:
            uri_proc = subprocess.run(
                ["wl-paste", "--no-newline", "--type", "text/uri-list"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if uri_proc.returncode == 0:
                for line in uri_proc.stdout.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        loaded = load_image_from_path(line)
                        if loaded is not None:
                            return loaded
    except Exception:
        return None
    return None


def _read_x11_clipboard_image() -> tuple[str, bytes] | None:
    """Attempts to read an image from the X11 clipboard via xclip."""
    if not shutil.which("xclip"):
        return None
    try:
        targets_proc = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if targets_proc.returncode != 0:
            return None
        available = [line.strip() for line in targets_proc.stdout.splitlines() if line.strip()]
        for mime in SUPPORTED_IMAGE_MIME_TYPES:
            if mime in available:
                img_proc = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-t", mime, "-o"],
                    capture_output=True,
                    timeout=5,
                )
                if img_proc.returncode == 0 and 0 < len(img_proc.stdout) <= MAX_IMAGE_BYTES:
                    return mime, img_proc.stdout
    except Exception:
        return None
    return None


def _read_macos_clipboard_image() -> tuple[str, bytes] | None:
    """Attempts to read an image from the macOS clipboard via pngpaste."""
    if not shutil.which("pngpaste"):
        return None
    try:
        img_proc = subprocess.run(
            ["pngpaste", "-"],
            capture_output=True,
            timeout=5,
        )
        if img_proc.returncode == 0 and 0 < len(img_proc.stdout) <= MAX_IMAGE_BYTES:
            return "image/png", img_proc.stdout
    except Exception:
        return None
    return None


def read_clipboard_image() -> tuple[str, bytes] | None:
    """
    Reads an image from the system clipboard if available.
    Prefers Wayland (wl-paste) when WAYLAND_DISPLAY is active, falling back to X11 (xclip)
    and macOS (pngpaste). Returns (mime_type, raw_bytes) or None.
    """
    if os.getenv("WAYLAND_DISPLAY"):
        res = _read_wayland_clipboard_image()
        if res is not None:
            return res

    res = _read_x11_clipboard_image()
    if res is not None:
        return res

    if not os.getenv("WAYLAND_DISPLAY"):
        res = _read_wayland_clipboard_image()
        if res is not None:
            return res

    return _read_macos_clipboard_image()


def read_clipboard_text() -> str | None:
    """Reads plain text from the system clipboard as a fallback for Ctrl+V."""
    candidates: list[list[str]] = []
    if os.getenv("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        candidates.append(["wl-paste", "--no-newline"])
    if shutil.which("xclip"):
        candidates.append(["xclip", "-selection", "clipboard", "-o"])
    if shutil.which("xsel"):
        candidates.append(["xsel", "--clipboard", "--output"])
    if not os.getenv("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        candidates.append(["wl-paste", "--no-newline"])
    if shutil.which("pbpaste"):
        candidates.append(["pbpaste"])

    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except Exception:
            continue
    return None


def is_kitty_terminal() -> bool:
    """Returns True when running inside a Kitty terminal supporting Kitty Graphics Protocol."""
    if os.getenv("KITTY_WINDOW_ID"):
        return True
    term = os.getenv("TERM", "").lower()
    return "kitty" in term


def build_kitty_graphics_sequence(
    png_bytes: bytes,
    rows: int = 8,
    cols: int | None = None,
    image_id: int | None = None,
) -> str:
    """
    Encodes raw PNG bytes into a chunked Kitty Graphics Protocol escape sequence.
    Uses q=2 to suppress terminal response codes so prompt_toolkit stdin stays clean.
    """
    if not png_bytes:
        return ""

    b64_data = base64.b64encode(png_bytes).decode("ascii")
    chunk_size = 4096
    chunks = [b64_data[i : i + chunk_size] for i in range(0, len(b64_data), chunk_size)]
    seq_parts: list[str] = []

    for idx, chunk in enumerate(chunks):
        more = 1 if idx < len(chunks) - 1 else 0
        if idx == 0:
            header_items = ["a=T", "f=100", "q=2", f"r={max(1, rows)}"]
            if cols is not None and cols > 0:
                header_items.append(f"c={cols}")
            if image_id is not None and image_id > 0:
                header_items.append(f"i={image_id}")
            header_items.append(f"m={more}")
            header = ",".join(header_items)
            seq_parts.append(f"\x1b_G{header};{chunk}\x1b\\")
        else:
            seq_parts.append(f"\x1b_Gm={more};{chunk}\x1b\\")

    return "".join(seq_parts)


def delete_kitty_image_sequence(image_id: int) -> str:
    """Builds a Kitty Graphics Protocol sequence to delete a previously displayed image by ID."""
    if image_id <= 0:
        return ""
    return f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"


def render_kitty_image_preview(
    attachment: ImageAttachment,
    stream: TextIO | None = None,
    rows: int = 8,
) -> bool:
    """
    Renders a compact inline preview of an ImageAttachment using Kitty's Graphics Protocol.
    For PNG images, emits pure-Python Kitty Graphics sequences directly (zero latency).
    For JPEG/WebP/GIF images, falls back to `kitten icat` when available.
    Returns True if a graphical preview was emitted.
    """
    if not is_kitty_terminal():
        return False

    out = stream or sys.stdout
    # Use a stable Kitty image ID offset to avoid colliding with other terminal graphics
    kitty_img_id = 9000 + attachment.index

    if attachment.mime_type == "image/png":
        seq = build_kitty_graphics_sequence(
            attachment.data, rows=rows, image_id=kitty_img_id
        )
        if seq:
            out.write(f"\n\x1b[2m🖼 {attachment.token} ({attachment.mime_type}, {attachment.size_label})\x1b[0m\n")
            out.write(seq)
            out.write("\n")
            out.flush()
            return True
        return False

    kitten_bin = shutil.which("kitten")
    if kitten_bin:
        try:
            out.write(f"\n\x1b[2m🖼 {attachment.token} ({attachment.mime_type}, {attachment.size_label})\x1b[0m\n")
            out.flush()
            proc = subprocess.run(
                [
                    kitten_bin,
                    "icat",
                    "--transfer-mode=stream",
                    "--stdin=yes",
                    "--align=left",
                    f"--place=32x{max(1, rows)}@0x0",
                ],
                input=attachment.data,
                timeout=3,
            )
            if proc.returncode == 0:
                out.write("\n")
                out.flush()
                return True
        except Exception:
            return False

    return False

