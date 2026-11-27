from __future__ import annotations

import hashlib
import shlex
import subprocess
from pathlib import Path
from typing import Any

import cli_agent.tools as tools


def _stub_os_sandbox(monkeypatch: Any, tmp_path: Path) -> None:
    executable_paths = {name: tmp_path / f"fake-{name}" for name in ("bwrap", "bash")}
    for path in executable_paths.values():
        path.touch()
    monkeypatch.setattr(tools.shutil, "which", lambda name: str(executable_paths[name]))
    monkeypatch.setattr(tools, "_bwrap_available", lambda: True, raising=False)


def test_run_command_fails_closed_without_os_isolation(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(tools, "_bwrap_available", lambda: False, raising=False)

    result = tools.run_command("echo should-not-run", cwd=str(tmp_path))

    assert "OS isolation is unavailable" in result.get("error", "")
    assert result.get("isolation") == "unavailable"


def test_rejected_command_leaves_workspace_data_unchanged(tmp_path: Path, monkeypatch: Any) -> None:
    marker = tmp_path / "workspace-state.txt"
    marker.write_text("original", encoding="utf-8")
    original_hash = hashlib.sha256(marker.read_bytes()).hexdigest()
    monkeypatch.setattr(tools, "_bwrap_available", lambda: False, raising=False)

    result = tools.run_command(
        f"printf changed > {shlex.quote(str(marker))}",
        cwd=str(tmp_path),
    )

    assert hashlib.sha256(marker.read_bytes()).hexdigest() == original_hash
    assert "OS isolation is unavailable" in result.get("error", "")


def test_reduced_isolation_requires_explicit_opt_in_and_is_disclosed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(tools, "_bwrap_available", lambda: False, raising=False)

    result = tools.run_command(
        "printf reduced",
        cwd=str(tmp_path),
        allow_reduced_isolation=True,
        network_access=True,
    )

    assert result.get("exit_code") == 0
    assert result.get("stdout") == "reduced"
    assert result.get("isolation") == "reduced"
    assert "does not provide OS process isolation" in result.get("warning", "")


def test_reduced_fallback_cannot_bypass_default_network_denial(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(tools, "_bwrap_available", lambda: False, raising=False)

    result = tools.run_command(
        "echo should-not-run", cwd=str(tmp_path), allow_reduced_isolation=True
    )

    assert "cannot enforce network isolation" in result.get("error", "")
    assert result.get("isolation") == "unavailable"


def test_os_sandbox_hides_host_credentials_and_disables_network_by_default(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="sandboxed", stderr="")

    monkeypatch.setattr(tools, "_bwrap_available", lambda: True, raising=False)
    _stub_os_sandbox(monkeypatch, tmp_path)
    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    monkeypatch.setenv("HOST_SECRET_FOR_TEST", "must-not-be-forwarded")

    result = tools.run_command("printf sandboxed", cwd=str(tmp_path))

    assert result.get("exit_code") == 0
    assert result.get("isolation") == "os"
    args, kwargs = calls[0]
    assert "--unshare-user" in args
    assert "--unshare-net" in args
    assert "--clearenv" in args
    assert "--tmpfs" in args
    assert "--ro-bind-try" in args
    assert "HOST_SECRET_FOR_TEST" not in args
    assert kwargs.get("shell") is not True
    assert "HOST_SECRET_FOR_TEST" not in kwargs["env"]


def test_worker_cwd_prefix_cannot_escape_active_sandbox(tmp_path: Path, monkeypatch: Any) -> None:
    allowed = tmp_path / "worker"
    sibling = tmp_path / "worker-evil"
    allowed.mkdir()
    sibling.mkdir()
    marker = sibling / "escaped.txt"
    monkeypatch.setattr(tools, "_bwrap_available", lambda: False, raising=False)
    monkeypatch.setattr(tools, "_resolve_tool_path", lambda _raw_path: sibling)

    with tools.worker_execution_scope(allowed):
        result = tools.run_command(
            f"touch {marker}",
            cwd=str(sibling),
            network_access=True,
            allow_reduced_isolation=True,
        )

    assert "outside active sandbox" in result.get("error", "")
    assert not marker.exists()


def test_network_access_is_only_enabled_when_requested(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(tools, "_bwrap_available", lambda: True, raising=False)
    _stub_os_sandbox(monkeypatch, tmp_path)
    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    result = tools.run_command("printf ok", cwd=str(tmp_path), network_access=True)

    assert result.get("exit_code") == 0
    assert "--unshare-net" not in calls[0]


def test_background_process_fails_closed_without_os_isolation(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(tools, "_bwrap_available", lambda: False, raising=False)
    spawned = False

    def fake_popen(*_args: Any, **_kwargs: Any) -> None:
        nonlocal spawned
        spawned = True

    monkeypatch.setattr(tools.subprocess, "Popen", fake_popen)

    result = tools.start_background_process("echo should-not-run", cwd=str(tmp_path))

    assert "OS isolation is unavailable" in result.get("error", "")
    assert result.get("isolation") == "unavailable"
    assert not spawned


def test_background_worker_cwd_prefix_cannot_escape_active_sandbox(
    tmp_path: Path, monkeypatch: Any
) -> None:
    allowed = tmp_path / "worker"
    sibling = tmp_path / "worker-evil"
    allowed.mkdir()
    sibling.mkdir()
    monkeypatch.setattr(tools, "_bwrap_available", lambda: False, raising=False)
    monkeypatch.setattr(tools, "_resolve_tool_path", lambda _raw_path: sibling)
    spawned = False

    def fake_popen(*_args: Any, **_kwargs: Any) -> None:
        nonlocal spawned
        spawned = True

    monkeypatch.setattr(tools.subprocess, "Popen", fake_popen)
    with tools.worker_execution_scope(allowed):
        result = tools.start_background_process(
            "echo should-not-run",
            cwd=str(sibling),
            network_access=True,
            allow_reduced_isolation=True,
        )

    assert "outside active sandbox" in result.get("error", "")
    assert not spawned


def test_command_schemas_expose_high_risk_capability_controls() -> None:
    declarations = {item["name"]: item for item in tools.GEMINI_TOOLS_DECLARATIONS}
    for name in ("run_command", "start_background_process"):
        properties = declarations[name]["parameters"]["properties"]
        assert properties["network_access"]["type"] == "BOOLEAN"
        assert properties["allow_reduced_isolation"]["type"] == "BOOLEAN"


def test_bwrap_masks_repository_git_config(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    git_dir = workspace / ".git"
    git_dir.mkdir(parents=True)
    config = git_dir / "config"
    config.write_text('[http "https://example.invalid"]\n\textraheader = Bearer host-secret\n')
    bwrap = tmp_path / "bwrap"
    shell = tmp_path / "bash"
    sanitized_config = tmp_path / "empty-git-config"
    bwrap.touch()
    shell.touch()
    sanitized_config.write_text("")

    argv = tools.build_bwrap_argv(
        "git status",
        bwrap_path=str(bwrap),
        shell_path=str(shell),
        workspace=workspace,
        cwd=workspace,
        network_access=False,
        sanitized_git_config=sanitized_config,
    )

    mask = ["--ro-bind", str(sanitized_config), str(config)]
    assert any(argv[index : index + len(mask)] == mask for index in range(len(argv)))
    assert "host-secret" not in argv
