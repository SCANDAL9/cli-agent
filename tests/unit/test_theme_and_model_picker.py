"""
Unit tests for Matugen Kitty color integration and interactive arrow/number model picker.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cli_agent.config import load_config, normalize_effort, save_effort_to_config
from cli_agent.main import _build_model_picker_app
from cli_agent.models import ModelCatalogEntry
from cli_agent.theme import (
    MatugenPalette,
    build_prompt_toolkit_style,
    hex_to_ansi_fg,
    load_matugen_palette,
    parse_kitty_colors_file,
)
from cli_agent.vertex import VertexAgentClient


@pytest.mark.unit
def test_parse_and_load_matugen_palette(tmp_path: Path) -> None:
    conf = tmp_path / "kitty-matugen-colors.conf"
    conf.write_text(
        "# Matugen generated\n"
        "foreground #e5e2e3\n"
        "background #131315\n"
        "active_tab_background #c3c6d6\n"
        "inactive_tab_background #201f21\n"
        "url_color #c1c5dd\n"
        "inactive_border_color #474648\n"
        "bell_border_color #ffb4ab\n"
        "selection_background #45464f\n"
        "selection_foreground #e2e1ec\n"
        "color0 #929092\n"
        "color10 #dde1f9\n",
        encoding="utf-8",
    )

    raw = parse_kitty_colors_file(conf)
    assert raw["foreground"] == "#e5e2e3"
    assert raw["active_tab_background"] == "#c3c6d6"

    palette = load_matugen_palette(config_path=conf)
    assert palette.is_matugen is True
    assert palette.primary == "#c3c6d6"
    assert palette.secondary == "#c1c5dd"
    assert palette.accent == "#dde1f9"
    assert palette.border == "#474648"
    assert palette.selection_bg == "#45464f"

    # Style uses Matugen hex tokens and preserves noreverse bottom-toolbar
    style = build_prompt_toolkit_style(palette)
    style_dict = dict(style.style_rules)
    assert style_dict["bottom-toolbar"] == "noreverse"
    assert "#c3c6d6" in style_dict["prompt"]

    # Truecolor ANSI conversion
    assert hex_to_ansi_fg("#c3c6d6") == "\x1b[38;2;195;198;214m"
    assert hex_to_ansi_fg("invalid") == ""


@pytest.mark.unit
def test_load_matugen_palette_fallback_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.conf"
    palette = load_matugen_palette(config_path=missing)
    assert palette.is_matugen is False


@pytest.mark.unit
def test_model_picker_arrow_keys_number_shortcuts_and_cancel() -> None:
    models = [
        ModelCatalogEntry("publishers/google/models/gemini-3.8-flash", "google", "gemini-3.8-flash", "GA"),
        ModelCatalogEntry("publishers/google/models/gemini-3.1-pro", "google", "gemini-3.1-pro", "GA"),
        ModelCatalogEntry("publishers/google/models/gemini-4-hydrogen", "google", "gemini-4-hydrogen", "PREVIEW"),
    ]
    app, state = _build_model_picker_app(
        gemini_models=models,
        descriptions={"gemini-3.8-flash": "Flash", "gemini-3.1-pro": "Pro", "gemini-4-hydrogen": "Hydrogen"},
        active_model="gemini-3.8-flash",
        active_effort="medium",
        palette=MatugenPalette(is_matugen=False),
    )
    assert state["selected"] == 0
    assert state["effort_idx"] == 1

    bindings = app.key_bindings.bindings
    down_h = next(b.handler for b in bindings if b.keys == ("down",))
    up_h = next(b.handler for b in bindings if b.keys == ("up",))
    left_h = next(b.handler for b in bindings if b.keys == ("left",))
    right_h = next(b.handler for b in bindings if b.keys == ("right",))
    enter_h = next(b.handler for b in bindings if b.keys == ("c-m",))
    esc_h = next(b.handler for b in bindings if b.keys == ("escape",))
    key_3_h = next(b.handler for b in bindings if b.keys == ("3",))

    mock_app = MagicMock()
    ev = SimpleNamespace(app=mock_app)

    # Arrow down -> index 1
    down_h(ev)
    assert state["selected"] == 1

    # Arrow up -> back to index 0
    up_h(ev)
    assert state["selected"] == 0

    # Left/right arrows slide effort low <-> medium <-> high (clamped)
    right_h(ev)
    assert state["effort_idx"] == 2  # high
    right_h(ev)
    assert state["effort_idx"] == 2  # clamped at high
    left_h(ev)
    left_h(ev)
    assert state["effort_idx"] == 0  # low
    left_h(ev)
    assert state["effort_idx"] == 0  # clamped at low
    right_h(ev)
    right_h(ev)
    assert state["effort_idx"] == 2  # high

    # Arrow down + Enter -> selects (gemini-3.1-pro, high)
    down_h(ev)
    enter_h(ev)
    mock_app.exit.assert_called_with(result=("gemini-3.1-pro", "high"))

    # Number shortcut '3' -> immediately selects (gemini-4-hydrogen, high)
    key_3_h(ev)
    assert state["selected"] == 2
    mock_app.exit.assert_called_with(result=("gemini-4-hydrogen", "high"))

    # Esc -> cancels with None
    esc_h(ev)
    mock_app.exit.assert_called_with(result=None)


@pytest.mark.unit
def test_real_effort_config_persistence_and_vertex_generation_config(tmp_path: Path) -> None:
    assert normalize_effort("HIGH") == "high"
    assert normalize_effort("invalid", default="low") == "low"

    cfg_file = tmp_path / "config.toml"
    assert save_effort_to_config("high", config_path=cfg_file) is True
    loaded = load_config(config_path=cfg_file)
    assert loaded.effort == "high"

    client = VertexAgentClient(project_id="test-proj", model_name="gemini-3.8-flash", effort="low")
    assert client._build_generation_config() == {
        "temperature": 0.2,
        "thinkingConfig": {"thinkingLevel": "LOW"},
    }
    client.set_effort("high")
    assert client._build_generation_config()["thinkingConfig"] == {"thinkingLevel": "HIGH"}

    # Non-Google publisher omits Gemini thinkingConfig
    client.switch_model("publishers/anthropic/models/claude-sonnet-5")
    assert "thinkingConfig" not in client._build_generation_config()

