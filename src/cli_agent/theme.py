"""
Dynamic Matugen theme integration for Kitty & prompt_toolkit / Rich.
Reads Kitty's live Matugen color palette (e.g. /tmp/kitty-matugen-colors.conf)
and exposes Material You colors for Rich panels, prompt_toolkit styles, and ANSI footers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.styles import Style

DEFAULT_MATUGEN_PATHS: tuple[Path, ...] = (
    Path("/tmp/kitty-matugen-colors.conf"),
    Path.home() / ".config/kitty/matugen-colors.conf",
    Path.home() / ".cache/matugen/kitty-colors.conf",
)

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class MatugenPalette:
    """Normalized Material You color palette derived from Kitty Matugen colors."""

    foreground: str = "#e5e2e3"
    background: str = "#131315"
    surface: str = "#201f21"
    primary: str = "#38bdf8"
    secondary: str = "#818cf8"
    accent: str = "#34d399"
    muted: str = "#929092"
    border: str = "#474648"
    error: str = "#ffb4ab"
    selection_bg: str = "#45464f"
    selection_fg: str = "#e2e1ec"
    is_matugen: bool = False


def parse_kitty_colors_file(path: Path) -> dict[str, str]:
    """Parses `key #rrggbb` entries from a Kitty color configuration file."""
    colors: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and _HEX_RE.match(parts[1]):
                colors[parts[0]] = parts[1].lower()
    except Exception:
        return {}
    return colors


def load_matugen_palette(config_path: Path | None = None) -> MatugenPalette:
    """
    Loads the active Matugen color palette from Kitty's generated color file.
    Falls back to sensible defaults if no Matugen file is present.
    """
    candidates = (config_path,) if config_path is not None else DEFAULT_MATUGEN_PATHS
    raw: dict[str, str] = {}
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            raw = parse_kitty_colors_file(candidate)
            if raw:
                break

    if not raw:
        return MatugenPalette(is_matugen=False)

    return MatugenPalette(
        foreground=raw.get("foreground", "#e5e2e3"),
        background=raw.get("background", "#131315"),
        surface=raw.get("inactive_tab_background", raw.get("background", "#201f21")),
        primary=raw.get(
            "active_tab_background",
            raw.get("active_border_color", raw.get("color4", "#38bdf8")),
        ),
        secondary=raw.get("url_color", raw.get("color6", "#818cf8")),
        accent=raw.get("color10", raw.get("color14", raw.get("color2", "#34d399"))),
        muted=raw.get("color0", raw.get("inactive_border_color", "#929092")),
        border=raw.get("inactive_border_color", raw.get("color0", "#474648")),
        error=raw.get("bell_border_color", raw.get("color1", "#ffb4ab")),
        selection_bg=raw.get("selection_background", "#45464f"),
        selection_fg=raw.get("selection_foreground", raw.get("foreground", "#e2e1ec")),
        is_matugen=True,
    )


def hex_to_ansi_fg(hex_color: str) -> str:
    """Converts a `#rrggbb` hex color to a 24-bit ANSI foreground escape code."""
    if not _HEX_RE.match(hex_color):
        return ""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"\x1b[38;2;{r};{g};{b}m"


def build_prompt_toolkit_style(palette: MatugenPalette | None = None) -> Style:
    """Builds a prompt_toolkit Style harmonized with Matugen (while preserving terminal adaptation)."""
    p = palette or load_matugen_palette()
    if p.is_matugen:
        return Style.from_dict(
            {
                "prompt": f"{p.primary} bold",
                "prompt.box": p.border,
                "placeholder": f"{p.muted} italic",
                "bottom-toolbar": "noreverse",
                "bottom-toolbar.text": p.muted,
                "bottom-toolbar.model": f"{p.primary} bold",
                "bottom-toolbar.badge": f"{p.accent} bold",
                "bottom-toolbar.badge.default": p.muted,
                "completion-menu": f"bg:{p.surface} {p.foreground}",
                "completion-menu.completion": f"bg:{p.surface} {p.foreground}",
                "completion-menu.completion.current": f"bg:{p.selection_bg} {p.selection_fg} bold",
                "completion-menu.meta.completion": f"bg:{p.surface} {p.muted}",
                "completion-menu.meta.completion.current": f"bg:{p.selection_bg} {p.selection_fg}",
                "scrollbar.background": f"bg:{p.surface}",
                "scrollbar.button": f"bg:{p.primary}",
            }
        )
    return Style.from_dict(
        {
            "prompt": "ansicyan bold",
            "prompt.box": "dim",
            "placeholder": "dim italic",
            "bottom-toolbar": "noreverse",
            "bottom-toolbar.text": "dim",
            "bottom-toolbar.model": "bold",
            "bottom-toolbar.badge": "ansigreen bold",
            "bottom-toolbar.badge.default": "dim",
            "completion-menu": "bg:default",
            "completion-menu.completion": "bg:default",
            "completion-menu.completion.current": "reverse bold",
            "completion-menu.meta.completion": "dim",
            "completion-menu.meta.completion.current": "reverse",
            "scrollbar.background": "dim",
            "scrollbar.button": "ansicyan",
        }
    )
