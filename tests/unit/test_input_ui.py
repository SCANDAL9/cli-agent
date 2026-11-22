"""
Unit tests for Rich user input interface and prompt_toolkit enhancements.
"""

from unittest.mock import patch

import pytest
from prompt_toolkit.completion import PathCompleter, WordCompleter
from prompt_toolkit.document import Document
from rich.console import Console

from cli_agent.main import (
    COMMAND_DESCRIPTIONS,
    SLASH_COMMANDS,
    AgentInputCompleter,
    RichMarkdownStreamer,
    format_bottom_toolbar,
    make_security_policy,
    print_banner,
    prompt_model_switch_dialog,
    pt_style,
    render_latex_unicode,
    show_models,
)
from cli_agent.security import RiskLevel
from cli_agent.vertex import VertexAgentClient


def test_command_descriptions_cover_slash_commands() -> None:
    for cmd in SLASH_COMMANDS:
        assert cmd in COMMAND_DESCRIPTIONS
        assert len(COMMAND_DESCRIPTIONS[cmd]) > 5


def test_agent_input_completer_slash_and_at() -> None:
    slash_c = WordCompleter(SLASH_COMMANDS, sentence=True, meta_dict=COMMAND_DESCRIPTIONS)
    path_c = PathCompleter()
    completer = AgentInputCompleter(slash_completer=slash_c, path_completer=path_c)

    # Autocomplete slash commands
    doc_slash = Document("/do", cursor_position=3)
    completions = list(completer.get_completions(doc_slash, None))
    match_names = [c.text for c in completions]
    assert "/doctor" in match_names

    # Autocomplete file references after @
    doc_at = Document("inspect file @pyproj", cursor_position=20)
    at_completions = list(completer.get_completions(doc_at, None))
    assert len(at_completions) > 0

    # Default fallback completions
    doc_empty = Document("", cursor_position=0)
    empty_comps = list(completer.get_completions(doc_empty, None))
    assert len(empty_comps) > 0


def test_pt_style_contains_modern_tokens() -> None:
    style_rules = pt_style.style_rules
    rule_keys = [rule[0] for rule in style_rules]
    assert "prompt" in rule_keys
    assert "prompt.box" in rule_keys
    assert "placeholder" in rule_keys
    assert "bottom-toolbar" in rule_keys
    assert "bottom-toolbar.text" in rule_keys
    assert "bottom-toolbar.model" in rule_keys
    assert "bottom-toolbar.badge" in rule_keys
    assert "completion-menu" in rule_keys
    # Ensure bottom toolbar adapts to both dark and light terminal themes without inverting
    style_dict = dict(style_rules)
    assert style_dict["bottom-toolbar"] == "noreverse"


def test_format_bottom_toolbar_auto_and_default() -> None:
    # Test default mode (user confirmation required)
    tb_default = format_bottom_toolbar(terminal_width=80, auto_approve=False, model_name="gemini-3.8-flash")
    raw_text = "".join(part[1] for part in tb_default)
    assert "default" in raw_text
    assert "Gemini 3.8 Flash" in raw_text
    assert "medium" in raw_text
    styles = [part[0] for part in tb_default]
    assert "class:bottom-toolbar.badge.default" in styles

    # Test auto mode (auto-approve enabled)
    tb_auto = format_bottom_toolbar(terminal_width=80, auto_approve=True, model_name="gemini-4-carbon")
    raw_text_auto = "".join(part[1] for part in tb_auto)
    assert "auto" in raw_text_auto
    assert "Gemini 4 Carbon" in raw_text_auto
    assert "high" in raw_text_auto
    styles_auto = [part[0] for part in tb_auto]
    assert "class:bottom-toolbar.badge" in styles_auto


def test_print_banner_renders_rich_panel() -> None:
    console = Console(record=True, width=100)
    # Patch console inside main or capture output
    from unittest.mock import patch
    with patch("cli_agent.main.console", console):
        print_banner(
            memory_count=5,
            project_key="my-workspace",
            model_name="gemini-3.8-flash",
            session_id="sess-test-123",
            turn_count=3,
        )
    output = console.export_text()
    assert "cli-agent" in output
    assert "gemini-3.8-flash" in output
    assert "my-workspace" in output
    assert "sess-test-123" in output
    assert "Shortcuts" in output


def test_rich_tool_call_formatting() -> None:
    console = Console(record=True, width=100)
    streamer = RichMarkdownStreamer(console=console)

    streamer.on_tool_call("write_file", {"path": "src/fib.py", "content": "def fib(): pass\n"})
    output = console.export_text()
    assert "Calling Tool" in output
    assert "Write" in output
    assert "src/fib.py" in output

    streamer.on_tool_call("run_command", {"command": "pytest -v"})
    output2 = console.export_text()
    assert "Calling Tool" in output2
    assert "Bash" in output2
    assert "pytest -v" in output2


def test_security_policy_rich_panel_confirm() -> None:
    with (
        patch("cli_agent.main.console.print") as mock_print,
        patch("cli_agent.main.console.input", return_value="y") as mock_input,
    ):
        policy = make_security_policy(auto_approve=False)
        assert policy.confirm_callback is not None
        res = policy.confirm_callback("run_command", {"command": "rm -rf /tmp/test"}, RiskLevel.HIGH)
        assert res is True
        assert mock_print.called
        assert mock_input.called


@pytest.mark.asyncio
async def test_prompt_model_switch_dialog_options() -> None:
    client = VertexAgentClient(project_id="test-proj", model_name="gemini-3.8-flash")
    # Test picking by number
    with patch("cli_agent.main.console.input", return_value="2"):
        picked = await prompt_model_switch_dialog(client=client, billing_project="test-proj")
        assert picked is not None
        assert "3.1-pro" in picked

    # Test picking by search name (argon)
    with patch("cli_agent.main.console.input", return_value="argon"):
        picked_argon = await prompt_model_switch_dialog(client=client, billing_project="test-proj")
        assert picked_argon is not None
        assert "4-argon" in picked_argon

    # Test picking by search name (hydrogen)
    with patch("cli_agent.main.console.input", return_value="hydrogen"):
        picked_h = await prompt_model_switch_dialog(client=client, billing_project="test-proj")
        assert picked_h is not None
        assert "4-hydrogen" in picked_h

    # Test picking by search name (neon)
    with patch("cli_agent.main.console.input", return_value="neon"):
        picked_ne = await prompt_model_switch_dialog(client=client, billing_project="test-proj")
        assert picked_ne is not None
        assert "4-neon" in picked_ne

    # Test picking by search name (carbon)
    with patch("cli_agent.main.console.input", return_value="carbon"):
        picked_c = await prompt_model_switch_dialog(client=client, billing_project="test-proj")
        assert picked_c is not None
        assert "4-carbon" in picked_c

    # Test empty cancel
    with patch("cli_agent.main.console.input", return_value=""):
        cancelled = await prompt_model_switch_dialog(client=client, billing_project="test-proj")
        assert cancelled is None


@pytest.mark.asyncio
async def test_show_models_filters_to_gemini_focus() -> None:
    models = await show_models("test-proj", active_model="gemini-3.8-flash")
    shorts = [m.short_name for m in models]
    assert any("3.8-flash" in s for s in shorts)
    assert any("3.1-pro" in s for s in shorts)
    assert any("4-hydrogen" in s for s in shorts)
    assert any("4-neon" in s for s in shorts)
    assert any("4-carbon" in s for s in shorts)
    assert any("4-argon" in s for s in shorts)
    assert len(models) == 6


def test_format_bottom_toolbar_spans_full_wide_terminal() -> None:
    tb = format_bottom_toolbar(terminal_width=160, auto_approve=False, model_name="gemini-4-hydrogen")
    border_text = tb[0][1]
    assert border_text == f"{'─' * 160}\n"


def test_render_latex_unicode_and_streamer() -> None:
    assert render_latex_unicode(r"Oscuridad $\rightarrow$ Chainsaw Man") == "Oscuridad → Chainsaw Man"
    assert render_latex_unicode(r"Complejidad $O(n^2)$ y $\alpha + \beta \leq \infty$") == "Complejidad O(n²) y α + β ≤ ∞"
    assert render_latex_unicode(r"Fracción $\frac{a}{b}$ y $\sqrt{x}$") == "Fracción a/b y √(x)"
    # Preserve currency values and inline code spans
    assert render_latex_unicode("Cuesta $10 y $20 en total") == "Cuesta $10 y $20 en total"
    assert render_latex_unicode(r"Literal `$\rightarrow$` intacto") == r"Literal `$\rightarrow$` intacto"

    console = Console(record=True, width=100)
    streamer = RichMarkdownStreamer(console=console, prompt_label="")
    streamer.on_chunk(r"- Acción $\rightarrow$ **Chainsaw Man**" + "\n\n")
    streamer.finalize()
    out = console.export_text()
    assert "→" in out
    assert r"\rightarrow" not in out


