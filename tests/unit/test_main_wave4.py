"""Unit tests for Wave 4 REPL commands and CLI flags in main.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cli_agent.main import (
    SLASH_COMMANDS,
    main,
    make_security_policy,
    print_banner,
    run_interactive,
    run_oneshot,
    show_memories,
    show_models,
)
from cli_agent.memory import MemoryStore
from cli_agent.models import ModelCatalogEntry
from cli_agent.pr import CIStatus, CIWatcher, CIWatchResult, GitPlatform, PRMetadata, PRSynthesizer
from cli_agent.security import RiskLevel
from cli_agent.tui import TUIApp, TUILayoutManager
from cli_agent.voice import VoiceAgentEngine


def test_slash_commands_include_wave4() -> None:
    assert "/pr" in SLASH_COMMANDS
    assert "/tui" in SLASH_COMMANDS
    assert "/voice" in SLASH_COMMANDS


def test_print_banner_and_show_memories(tmp_path: Path) -> None:
    print_banner(5, project_key="test-proj", model_name="gemini-2.5-pro")
    store = MemoryStore(db_path=tmp_path / "mem.db")
    show_memories(store, project_key="test-proj")
    store.save_memory("Title 1", "Content 1", project_key="test-proj")
    show_memories(store, project_key="test-proj")


@pytest.mark.asyncio
async def test_show_models_table() -> None:
    entries = [
        ModelCatalogEntry(
            publisher="google",
            model_id="publishers/google/models/gemini-2.5-pro",
            short_name="gemini-2.5-pro",
            launch_stage="GA",
        )
    ]
    with patch("cli_agent.main.fetch_model_garden_catalog", AsyncMock(return_value=entries)):
        res = await show_models("proj-1", "gemini-2.5-pro")
        assert len(res) == 1


def test_make_security_policy_callbacks() -> None:
    policy = make_security_policy(auto_approve=False)
    assert policy.confirm_callback is not None
    with patch("cli_agent.main.console.input", return_value="y"):
        assert policy.confirm_callback("run_command", {"command": "ls"}, RiskLevel.MEDIUM) is True
    with patch("cli_agent.main.console.input", return_value="always"):
        assert policy.confirm_callback("run_command", {"command": "rm"}, RiskLevel.HIGH) is True
    with patch("cli_agent.main.console.input", side_effect=KeyboardInterrupt):
        assert policy.confirm_callback("run_command", {"command": "ls"}, RiskLevel.MEDIUM) is False


@pytest.mark.asyncio
async def test_wave4_repl_commands_pr_tui_voice_doctor_help(tmp_path: Path) -> None:
    inputs = [
        "",
        "/help",
        "/doctor",
        "/pr",
        "/pr create",
        "/pr watch",
        "/tui",
        "/voice",
        "/clear",
        "/compact",
        "/memories",
        "/exit",
    ]
    mock_session = MagicMock()
    mock_session.prompt_async = AsyncMock(side_effect=inputs)

    mock_pr_meta = PRMetadata(
        platform=GitPlatform.GITHUB,
        number=101,
        title="feat(core): wave 4",
        body="## Summary",
        url="https://github.com/acme/repo/pull/101",
        source_branch="HEAD",
        target_branch="main",
        head_sha="abc1234",
    )
    mock_provider = MagicMock()
    mock_provider.platform = GitPlatform.GITHUB
    mock_provider.create_or_update_pr.return_value = mock_pr_meta
    mock_provider.find_existing_pr.return_value = mock_pr_meta
    mock_provider.get_ci_status.return_value = CIStatus.SUCCESS

    mock_watch_res = CIWatchResult(
        pr_metadata=mock_pr_meta,
        final_status=CIStatus.SUCCESS,
        heal_attempts_used=0,
    )

    diag_mock = {
        "token_status": "valid",
        "database_status": "healthy",
        "code_graph_status": "healthy",
        "code_graph_stats": {"symbols": 10, "files": 2, "edges": 5},
        "workspace_key": "cli-agent",
        "database_path": str(tmp_path / "mem.db"),
        "gcloud_account": "dev@example.com",
        "mcp_servers": 1,
        "mcp_tools": 3,
        "tdd_status": "ready",
        "git_host_platform": "github",
        "circuit_breaker_status": "CLOSED",
        "tui_status": "ready",
        "voice_status": "headless_ready",
        "issues": [],
        "healthy": True,
    }

    with (
        patch("cli_agent.main.PromptSession", return_value=mock_session),
        patch("cli_agent.main.MemoryStore", return_value=MemoryStore(db_path=tmp_path / "m.db")),
        patch("cli_agent.main.CodeGraphIndexer") as mock_indexer_cls,
        patch("cli_agent.main.get_hybrid_search_engine") as mock_search_fn,
        patch("cli_agent.main.detect_git_host_provider", return_value=mock_provider),
        patch("cli_agent.main.CIWatcher") as mock_watcher_cls,
        patch("cli_agent.main.run_diagnostics", AsyncMock(return_value=diag_mock)),
        patch("cli_agent.main.TUIApp") as mock_tui_cls,
        patch("cli_agent.main.VoiceAgentEngine") as mock_voice_cls,
    ):
        mock_indexer_cls.return_value.index_directory.return_value = 1
        mock_search_fn.return_value.index_directory = AsyncMock(return_value=1)
        mock_watcher_cls.return_value.watch.return_value = mock_watch_res
        mock_voice_cls.return_value.run_session = AsyncMock(
            return_value={"status": "completed", "websocket_url": "wss://test"}
        )

        await run_interactive(auto_approve=True, model_override="gemini-2.5-pro")

        assert mock_provider.create_or_update_pr.called
        assert mock_watcher_cls.return_value.watch.called
        assert mock_tui_cls.return_value.run.called
        assert mock_voice_cls.return_value.run_session.called


@pytest.mark.asyncio
async def test_run_interactive_max_turns_and_chat(tmp_path: Path) -> None:
    mock_session = MagicMock()
    mock_session.prompt_async = AsyncMock(side_effect=["Hello agent", "Second turn"])

    async def fake_stream(*args: Any, **kwargs: Any):
        yield "Hello "
        yield "human!"

    with (
        patch("cli_agent.main.PromptSession", return_value=mock_session),
        patch("cli_agent.main.MemoryStore", return_value=MemoryStore(db_path=tmp_path / "m2.db")),
        patch("cli_agent.main.CodeGraphIndexer"),
        patch("cli_agent.main.get_hybrid_search_engine") as mock_search_fn,
        patch("cli_agent.main.VertexAgentClient") as mock_client_cls,
    ):
        mock_search_fn.return_value.index_directory = AsyncMock(return_value=1)
        mock_client = mock_client_cls.return_value
        mock_client.model_name = "gemini-2.5-pro"
        mock_client.stream_chat = fake_stream

        await run_interactive(auto_approve=True, max_turns=1)
        assert mock_session.prompt_async.call_count == 1


@pytest.mark.asyncio
async def test_run_oneshot_success(tmp_path: Path) -> None:
    async def fake_stream(*args: Any, **kwargs: Any):
        yield "Oneshot response"

    with (
        patch("cli_agent.main.MemoryStore", return_value=MemoryStore(db_path=tmp_path / "o.db")),
        patch("cli_agent.main.VertexAgentClient") as mock_client_cls,
    ):
        mock_client_cls.return_value.stream_chat = fake_stream
        await run_oneshot("test prompt", auto_approve=True, model_override="gemini-2.5-pro")


def test_main_cli_flags_tui_voice_and_max_turns() -> None:
    # 1. --tui flag
    with (
        patch("sys.argv", ["cli-agent", "--tui"]),
        patch("cli_agent.main.TUIApp") as mock_tui_cls,
    ):
        main()
        assert mock_tui_cls.return_value.run.called

    # 2. --voice flag
    with (
        patch("sys.argv", ["cli-agent", "--voice", "--model", "gemini-2.0-flash-exp"]),
        patch("cli_agent.main.VertexAgentClient"),
        patch("cli_agent.main.VoiceAgentEngine") as mock_voice_cls,
    ):
        mock_voice_cls.return_value.run_session = AsyncMock(return_value={"status": "completed"})
        main()
        assert mock_voice_cls.return_value.run_session.called

    # 3. --max-turns flag in interactive mode
    with (
        patch("sys.argv", ["cli-agent", "--auto-approve", "--max-turns", "2"]),
        patch("sys.stdin.isatty", return_value=True),
        patch("cli_agent.main.run_interactive", new_callable=AsyncMock) as mock_run_int,
    ):
        main()
        mock_run_int.assert_called_once_with(
            auto_approve=True,
            model_override=None,
            max_turns=2,
        )

    # 4. One-shot prompt via CLI args
    with (
        patch("sys.argv", ["cli-agent", "Explain AST"]),
        patch("cli_agent.main.run_oneshot", new_callable=AsyncMock) as mock_oneshot,
    ):
        main()
        mock_oneshot.assert_called_once_with(
            "Explain AST",
            auto_approve=False,
            model_override=None,
        )


@pytest.mark.asyncio
async def test_direct_wave4_helpers_tui_voice_pr(tmp_path: Path) -> None:
    # TUIApp.run() and default TUILayoutManager()
    app = TUIApp(layout_manager=TUILayoutManager())
    frame = app.run()
    assert frame.columns == 120
    assert app.is_running is False

    # VoiceAgentEngine default session & run_session()
    voice = VoiceAgentEngine(
        client=MagicMock(project_id="p", location="us-central1", model_name="m")
    )
    res = await voice.run_session()
    assert res["status"] == "completed"
    assert "wss://" in res["websocket_url"]

    # PRSynthesizer.generate_pr() and CIWatcher.watch()
    synth = PRSynthesizer(
        worktree_path=tmp_path, git_runner=lambda _a, _c: (0, "M\tsrc/app.py\n", "")
    )
    pr_data = synth.generate_pr(task_objective="Implement feature X")
    assert "feat(" in pr_data["title"]
    assert "ISO/IEC 25010" in pr_data["body"]

    provider = MagicMock()
    provider.platform = GitPlatform.GITHUB
    provider.find_existing_pr.return_value = None
    provider.get_ci_status.return_value = CIStatus.SUCCESS
    watcher = CIWatcher(provider=provider, poll_interval_seconds=0.01, sleep_fn=lambda _: None)
    watch_res = watcher.watch()
    assert watch_res.final_status == CIStatus.SUCCESS
