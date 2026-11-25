"""Tests for the declarative workflow loader, runner, and isolated rollback."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cli_agent.security import SecurityPolicy
from cli_agent.storage import SQLiteEngine
from cli_agent.workflows import (
    WorkflowEngine,
    WorkflowHistoryStore,
    WorkflowLoader,
    WorkflowPermissionBroker,
    WorkflowValidationError,
    handle_workflow_command,
    parse_workflow_command,
)
from cli_agent.workflows.checkpoint import WorkflowCheckpointManager
from cli_agent.workflows.commands import format_verify_gate, format_workflow_inputs
from cli_agent.workflows.runner import (
    WorkflowAborted,
    WorkflowExecutionError,
    execute_shell_command,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Workflow Test"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "workflow@example.invalid"], cwd=path, check=True
    )
    (path / "tracked.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def _workflow_file(root: Path, *, failing_gate: bool = False) -> Path:
    directory = root / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    gate = (
        '[steps.verify_gate]\ntype = "command_exit_code"\n'
        'command = "printf verify"\nexpected_exit_code = 0\n'
    )
    if failing_gate:
        gate = gate.replace('command = "printf verify"', 'command = "printf failed"')
        gate = gate.replace(
            "expected_exit_code = 0", 'expected_exit_code = 1\nexpected_output = "failed"'
        )
    path = directory / "demo.toml"
    path.write_text(
        'id = "demo"\nname = "Demo workflow"\n'
        'description = "Pass state between sequential steps."\nversion = "1.0.0"\n\n'
        '[inputs.feature]\ntype = "string"\ndescription = "Feature name"\n\n'
        '[[steps]]\nid = "write"\nname = "Write a file"\n'
        'action = "agent_prompt"\nprompt = "Implement {{ feature }}"\n\n'
        '[[steps]]\nid = "verify"\nname = "Verify"\naction = "verify_gate"\n' + gate,
        encoding="utf-8",
    )
    return path


def _make_engine(
    root: Path,
    *,
    command_executor: Any = None,
    agent_executor: Any = None,
    confirm: bool = True,
) -> WorkflowEngine:
    if not (root / ".git").exists():
        _init_repo(root)
    loader = WorkflowLoader(
        workspace_root=root,
        user_config_dir=root / "user-config",
        include_builtins=False,
    )
    history = WorkflowHistoryStore(SQLiteEngine(root / ".git" / "workflow-history.db"))
    security = SecurityPolicy(
        auto_approve=False,
        confirm_callback=lambda _name, _arguments, _risk: confirm,
    )
    return WorkflowEngine(
        loader=loader,
        history=history,
        security_policy=security,
        command_executor=command_executor,
        agent_executor=agent_executor,
    )


@pytest.mark.unit
def test_workflow_loader_validates_toml_schema_and_dag(tmp_path: Path):
    _workflow_file(tmp_path)

    workflows = WorkflowLoader(
        workspace_root=tmp_path, user_config_dir=tmp_path / "home", include_builtins=False
    ).list()

    assert [workflow.id for workflow in workflows] == ["demo"]
    assert [step.id for step in workflows[0].steps] == ["write", "verify"]
    assert workflows[0].steps[1].depends_on == ("write",)


@pytest.mark.unit
def test_workflow_presentation_includes_inputs_and_verify_gate_details(tmp_path: Path):
    _workflow_file(tmp_path)
    definition = WorkflowLoader(
        workspace_root=tmp_path, user_config_dir=tmp_path / "home", include_builtins=False
    ).get("demo")

    inputs = format_workflow_inputs(definition)
    gate = format_verify_gate(definition.steps[1])

    assert "feature (string, required): Feature name" in inputs
    assert "printf verify" in gate
    assert "expected exit 0" in gate


@pytest.mark.unit
def test_workflow_presentation_handles_steps_without_inputs_or_verify_gates(tmp_path: Path):
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "simple.toml").write_text(
        'id = "simple"\nname = "Simple workflow"\ndescription = "No inputs or gate"\n'
        'version = "1.0.0"\n\n[[steps]]\nid = "analyze"\nname = "Analyze"\n'
        'action = "agent_prompt"\nprompt = "Review the repository"\n',
        encoding="utf-8",
    )
    definition = WorkflowLoader(
        workspace_root=tmp_path,
        user_config_dir=tmp_path / "home",
        include_builtins=False,
    ).get("simple")

    assert format_workflow_inputs(definition) == "No inputs"
    assert format_verify_gate(definition.steps[0]) == "—"


@pytest.mark.unit
def test_workflow_history_adds_tag_column_to_existing_runs_table(tmp_path: Path):
    database_path = tmp_path / "workflow-history.db"
    engine = SQLiteEngine(database_path)
    engine.execute(
        """CREATE TABLE workflow_runs (
            run_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            status TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            branch_name TEXT,
            worktree_path TEXT,
            error TEXT,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT
        )"""
    )
    engine.execute(
        "INSERT INTO workflow_runs (run_id, workflow_id, status, inputs_json) VALUES (?, ?, ?, ?)",
        ("existing", "demo", "completed", "{}"),
    )

    history = WorkflowHistoryStore(engine)

    columns = engine.execute("PRAGMA table_info(workflow_runs)")
    assert "tag_name" in {column["name"] for column in columns}
    assert history.get_run("existing")["status"] == "completed"
    assert history.get_run("existing")["tag_name"] is None


@pytest.mark.unit
def test_workflow_loader_rejects_unknown_dependencies_and_cycles(tmp_path: Path):
    workflow_file = _workflow_file(tmp_path)
    workflow_file.write_text(
        workflow_file.read_text(encoding="utf-8").replace(
            'action = "verify_gate"', 'action = "verify_gate"\ndepends_on = ["missing"]'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown dependency"):
        WorkflowLoader(workspace_root=tmp_path, user_config_dir=tmp_path / "home").list()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_runner_executes_steps_in_order_and_interpolates_state(tmp_path: Path):
    _init_repo(tmp_path)
    _workflow_file(tmp_path)
    (tmp_path / "tracked.txt").write_text("dirty tracked input\n", encoding="utf-8")
    (tmp_path / "user-input.txt").write_text("untracked input\n", encoding="utf-8")
    progress: list[dict[str, Any]] = []
    commands: list[tuple[str, Path]] = []

    async def agent_executor(prompt: str, cwd: Path) -> str:
        progress.append({"prompt": prompt, "cwd": cwd})
        assert (cwd / "tracked.txt").read_text(encoding="utf-8") == "dirty tracked input\n"
        assert (cwd / "user-input.txt").read_text(encoding="utf-8") == "untracked input\n"
        return "written"

    async def command_executor(
        command: str, cwd: Path, timeout: float, abort_event: asyncio.Event
    ) -> dict[str, Any]:
        del timeout, abort_event
        commands.append((command, cwd))
        return {"exit_code": 0, "stdout": "verify", "stderr": ""}

    engine = _make_engine(
        tmp_path,
        command_executor=command_executor,
        agent_executor=agent_executor,
    )
    engine.add_event_listener(progress.append)

    run_id = engine.start("demo", {"feature": "storage"})
    result = await engine.wait(run_id)

    assert result.status == "completed", result.error
    assert next(event for event in progress if "prompt" in event)["prompt"].startswith(
        "Implement storage\n\nAvailable workflow state"
    )
    assert commands[0][0] == "printf verify"
    assert commands[0][1] != tmp_path
    assert any(event.get("status") == "completed" for event in progress if isinstance(event, dict))
    assert engine.history.list_runs(limit=1)[0]["status"] == "completed"
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "dirty tracked input\n"
    assert (tmp_path / "user-input.txt").read_text(encoding="utf-8") == "untracked input\n"
    _remove_retained_worktree(result.worktree_path, result.branch_name, tmp_path)
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_runner_self_heal_retries_and_persists_the_successful_attempt(
    tmp_path: Path,
):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "retry.toml").write_text(
        'id = "retry"\nname = "Retry workflow"\ndescription = "Self-heal test"\n'
        'version = "1.0.0"\n\n[[steps]]\nid = "check"\nname = "Check"\n'
        'action = "shell"\ncommand = "printf check"\n\n[steps.on_failure]\n'
        'action = "self_heal"\nprompt = "Repair after: {{ error }}"\nmax_retries = 1\n',
        encoding="utf-8",
    )
    attempts: list[str] = []
    repairs: list[str] = []
    events: list[dict[str, Any]] = []

    async def command_executor(command: str, *_args: Any) -> dict[str, Any]:
        attempts.append(command)
        if len(attempts) == 1:
            return {"exit_code": 1, "stdout": "", "stderr": "transient check failure"}
        return {"exit_code": 0, "stdout": "check passed", "stderr": ""}

    async def agent_executor(prompt: str, _cwd: Path) -> str:
        repairs.append(prompt)
        return "repair applied"

    engine = _make_engine(
        tmp_path,
        command_executor=command_executor,
        agent_executor=agent_executor,
    )
    engine.add_event_listener(events.append)

    run_id = engine.start("retry", {})
    result = await engine.wait(run_id)
    history = engine.history.get_run(run_id)
    step = history["steps"][0]

    assert result.status == "completed", result.error
    assert attempts == ["printf check", "printf check"]
    assert repairs == [
        "Repair after: Shell step 'check' exited with code 1: transient check failure"
    ]
    assert step["status"] == "completed"
    assert step["attempt"] == 2
    assert step["exit_code"] == 0
    assert step["output"]["stdout"] == "check passed"
    assert [(event["status"], event["attempt"]) for event in events if event.get("step_id")] == [
        ("running", 1),
        ("retrying", 1),
        ("running", 2),
        ("completed", 2),
    ]
    _remove_retained_worktree(result.worktree_path, result.branch_name, tmp_path)
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_verify_gate_persists_expected_stdout_and_stderr(tmp_path: Path):
    _init_repo(tmp_path)
    workflow_path = _workflow_file(tmp_path)
    workflow_path.write_text(
        workflow_path.read_text(encoding="utf-8").replace(
            "expected_exit_code = 0\n",
            'expected_exit_code = 0\nexpected_output = "verified"\nexpected_stderr = "warning"\n',
        ),
        encoding="utf-8",
    )

    async def command_executor(*_args: Any) -> dict[str, Any]:
        return {"exit_code": 0, "stdout": "verified output", "stderr": "warning: slow"}

    engine = _make_engine(
        tmp_path,
        command_executor=command_executor,
        agent_executor=lambda *_args: asyncio.sleep(0, result="implemented"),
    )
    run_id = engine.start("demo", {"feature": "gate"})

    result = await engine.wait(run_id)
    run = engine.history.get_run(run_id)
    verify_step = next(step for step in run["steps"] if step["step_id"] == "verify")
    gate_result = verify_step["output"]["verify_gate"]

    assert result.status == "completed", result.error
    assert gate_result["stdout"] == "verified output"
    assert gate_result["stderr"] == "warning: slow"
    _remove_retained_worktree(result.worktree_path, result.branch_name, tmp_path)
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_retry_exhaustion_persists_failure_and_events(tmp_path: Path):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "retry.toml").write_text(
        'id = "retry"\nname = "Retry workflow"\ndescription = "Retry exhaustion test"\n'
        'version = "1.0.0"\n\n[[steps]]\nid = "check"\nname = "Check"\n'
        'action = "shell"\ncommand = "printf check"\n\n[steps.on_failure]\n'
        'action = "self_heal"\nprompt = "Repair after: {{ error }}"\nmax_retries = 1\n',
        encoding="utf-8",
    )
    attempts: list[str] = []
    repairs: list[str] = []
    events: list[dict[str, Any]] = []

    async def command_executor(command: str, *_args: Any) -> dict[str, Any]:
        attempts.append(command)
        return {"exit_code": 1, "stdout": "", "stderr": "persistent failure"}

    async def agent_executor(prompt: str, _cwd: Path) -> str:
        repairs.append(prompt)
        return "repair attempted"

    engine = _make_engine(
        tmp_path,
        command_executor=command_executor,
        agent_executor=agent_executor,
    )
    engine.add_event_listener(events.append)
    run_id = engine.start("retry", {})

    result = await engine.wait(run_id)
    run = engine.history.get_run(run_id)
    step = run["steps"][0]

    assert result.status == "failed"
    assert "persistent failure" in result.error
    assert attempts == ["printf check", "printf check"]
    assert len(repairs) == 1
    assert step["status"] == "failed"
    assert step["attempt"] == 2
    assert "persistent failure" in step["error"]
    assert [(event["status"], event["attempt"]) for event in events if event.get("step_id")] == [
        ("running", 1),
        ("retrying", 1),
        ("running", 2),
        ("failed", 2),
    ]
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_engine_validates_and_normalizes_typed_inputs(tmp_path: Path):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "typed.toml").write_text(
        'id = "typed"\nname = "Typed workflow"\ndescription = "Input validation"\n'
        'version = "1.0.0"\n\n[inputs.title]\ntype = "string"\nrequired = true\n'
        '\n[inputs.count]\ntype = "integer"\nrequired = false\ndefault = 2\n'
        '\n[inputs.enabled]\ntype = "boolean"\nrequired = false\ndefault = false\n'
        '\n[inputs.source_path]\ntype = "path"\nrequired = false\ndefault = "src/main.py"\n'
        '\n[[steps]]\nid = "analyze"\nname = "Analyze"\naction = "agent_prompt"\n'
        'prompt = "Analyze {{ title }}"\n',
        encoding="utf-8",
    )
    engine = _make_engine(
        tmp_path,
        agent_executor=lambda _prompt, _cwd: "analysis complete",
    )
    invalid_inputs = [
        ({}, "Missing required workflow input 'title'"),
        ({"title": "demo", "unexpected": "value"}, "Unknown workflow inputs"),
        ({"title": "demo", "count": "not-an-integer"}, "must be an integer"),
        ({"title": "demo", "count": True}, "must be an integer"),
        ({"title": "demo", "enabled": "yes"}, "must be 'true' or 'false'"),
        ({"title": "demo", "enabled": 1}, "must be a boolean"),
        ({"title": "demo", "source_path": "../outside.py"}, "must remain inside"),
        ({"title": "demo", "source_path": "/etc/passwd"}, "must remain inside"),
    ]

    for supplied, message in invalid_inputs:
        with pytest.raises(WorkflowValidationError, match=message):
            engine.start("typed", supplied)

    run_id = engine.start("typed", {"title": "demo", "count": "7", "enabled": "TRUE"})
    result = await engine.wait(run_id)

    assert result.status == "completed", result.error
    assert engine.history.get_run(run_id)["inputs"] == {
        "title": "demo",
        "count": 7,
        "enabled": True,
        "source_path": "src/main.py",
    }
    _remove_retained_worktree(result.worktree_path, result.branch_name, tmp_path)
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_shell_command_captures_output_and_returns_when_process_exits(
    tmp_path: Path,
):
    result = await asyncio.wait_for(
        execute_shell_command("printf hello; printf warning >&2", tmp_path, 5, asyncio.Event()),
        timeout=2,
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "hello"
    assert result["stderr"] == "warning"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_shell_command_timeout_terminates_the_process(tmp_path: Path):
    with pytest.raises(WorkflowExecutionError, match="Command timed out after 0.1 seconds"):
        await asyncio.wait_for(
            execute_shell_command("sleep 30", tmp_path, 0.1, asyncio.Event()),
            timeout=2,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_shell_command_bounds_output_and_redacts_secrets(tmp_path: Path):
    command = (
        'python3 -c \'import sys; sys.stdout.write("x" * 1_100_000); '
        'sys.stderr.write("password=example-secret-value")\''
    )

    result = await execute_shell_command(command, tmp_path, 5, asyncio.Event())

    assert result["exit_code"] == 0
    assert result["stdout_truncated"] is True
    assert len(result["stdout"]) < 1_100_000
    assert result["stderr"] == "password=[REDACTED_SECRET]"
    assert "example-secret-value" not in result["stderr"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_shell_command_stops_when_abort_event_is_set(tmp_path: Path):
    abort_event = asyncio.Event()

    async def request_abort() -> None:
        await asyncio.sleep(0.05)
        abort_event.set()

    abort_task = asyncio.create_task(request_abort())
    command = subprocess.list2cmdline([sys.executable, "-c", "import time; time.sleep(30)"])

    with pytest.raises(WorkflowAborted, match="Workflow aborted by the user"):
        await asyncio.wait_for(execute_shell_command(command, tmp_path, 5, abort_event), timeout=3)

    await abort_task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_abort_event_between_steps_discards_checkpoint(tmp_path: Path):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "abort-between-steps.toml").write_text(
        'id = "abort-between-steps"\nname = "Abort between steps"\n'
        'description = "Abort after one successful step."\nversion = "1.0.0"\n\n'
        '[[steps]]\nid = "first"\nname = "First"\naction = "shell"\n'
        'command = "printf first"\n\n'
        '[[steps]]\nid = "second"\nname = "Second"\naction = "shell"\n'
        'command = "printf second"\n',
        encoding="utf-8",
    )
    commands: list[str] = []

    async def command_executor(
        command: str, _cwd: Path, _timeout: float, abort_event: asyncio.Event
    ) -> dict[str, Any]:
        commands.append(command)
        abort_event.set()
        return {"exit_code": 0, "stdout": "first", "stderr": ""}

    engine = _make_engine(tmp_path, command_executor=command_executor)
    run_id = engine.start("abort-between-steps", {})

    result = await engine.wait(run_id)

    assert result.status == "aborted"
    assert result.completed_steps == ("first",)
    assert commands == ["printf first"]
    assert engine.history.get_run(run_id)["status"] == "aborted"
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "committed\n"
    branches = subprocess.run(
        ["git", "branch", "--list", "cli-agent-wf-*"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tags = subprocess.run(
        ["git", "tag", "--list", "cli-agent-wf-*"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert not branches.strip()
    assert not tags.strip()
    await engine.aclose()


@pytest.mark.unit
def test_workflow_checkpoint_rejects_escaping_symlink_and_cleans_partial_git_state(
    tmp_path: Path,
):
    _init_repo(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside data\n", encoding="utf-8")
    link = tmp_path / "escape.txt"
    link.symlink_to(outside)

    with pytest.raises(WorkflowValidationError, match="refusing to copy it"):
        WorkflowCheckpointManager(tmp_path).create(run_id="escape")

    branches = subprocess.run(
        ["git", "branch", "--list", "cli-agent-wf-escape-branch"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tags = subprocess.run(
        ["git", "tag", "--list", "cli-agent-wf-escape"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert link.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside data\n"
    assert not branches.strip()
    assert not tags.strip()
    assert sum(line.startswith("worktree ") for line in worktrees) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_gate_failure_discards_only_isolated_checkpoint(tmp_path: Path):
    _init_repo(tmp_path)
    _workflow_file(tmp_path, failing_gate=True)
    (tmp_path / "tracked.txt").write_text("user's staged and unstaged baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("user's final content\n", encoding="utf-8")
    (tmp_path / "user-note.txt").write_text("keep me\n", encoding="utf-8")
    before_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    before_file = (tmp_path / "tracked.txt").read_bytes()
    engine = _make_engine(
        tmp_path,
        agent_executor=lambda _prompt, _cwd: asyncio.sleep(0, result="implemented"),
        command_executor=lambda *_args: asyncio.sleep(
            0, result={"exit_code": 0, "stdout": "failed", "stderr": ""}
        ),
    )

    run_id = engine.start("demo", {"feature": "rollback"})
    result = await engine.wait(run_id)

    after_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-uall"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert result.status == "failed"
    assert (tmp_path / "tracked.txt").read_bytes() == before_file
    assert (tmp_path / "user-note.txt").read_text(encoding="utf-8") == "keep me\n"
    assert after_status == before_status
    assert engine.history.get_run(run_id)["status"] == "failed"
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_failure_retains_checkpoint_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "failure.toml").write_text(
        'id = "failure"\nname = "Failure workflow"\ndescription = "Cleanup failure test"\n'
        'version = "1.0.0"\n\n[[steps]]\nid = "check"\nname = "Check"\n'
        'action = "shell"\ncommand = "printf check"\n',
        encoding="utf-8",
    )
    checkpoint_manager = WorkflowCheckpointManager(tmp_path)
    created_checkpoints = []
    create_checkpoint = checkpoint_manager.create

    def capture_checkpoint(**kwargs: Any):
        checkpoint = create_checkpoint(**kwargs)
        created_checkpoints.append(checkpoint)
        return checkpoint

    monkeypatch.setattr(checkpoint_manager, "create", capture_checkpoint)
    engine = _make_engine(
        tmp_path,
        command_executor=lambda *_args: asyncio.sleep(
            0, result={"exit_code": 1, "stdout": "", "stderr": "verification failed"}
        ),
    )
    engine.checkpoint_manager = checkpoint_manager
    monkeypatch.setattr(engine, "_discard_checkpoint", lambda _checkpoint: "cleanup denied")
    run_id = engine.start("failure", {})

    try:
        result = await engine.wait(run_id)
        run = engine.history.get_run(run_id)

        assert result.status == "failed"
        assert "Checkpoint cleanup failed: cleanup denied" in result.error
        assert result.worktree_path is not None
        assert result.branch_name is not None
        assert result.tag_name is not None
        assert Path(result.worktree_path).is_dir()
        assert run["worktree_path"] == result.worktree_path
        assert run["branch_name"] == result.branch_name
        assert run["tag_name"] == result.tag_name
    finally:
        for checkpoint in created_checkpoints:
            checkpoint.discard()
        await engine.aclose()


@pytest.mark.asyncio
async def test_workflow_abort_cancels_running_command_and_removes_its_branch(tmp_path: Path):
    _init_repo(tmp_path)
    workflow = tmp_path / ".cli-agent" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "slow.toml").write_text(
        'id = "slow"\nname = "Slow workflow"\n'
        'description = "Cancelable command test."\nversion = "1.0.0"\n\n'
        '[[steps]]\nid = "sleep"\nname = "Sleep"\naction = "shell"\n'
        'command = "sleep 30"\n',
        encoding="utf-8",
    )
    engine = _make_engine(tmp_path)
    run_id = engine.start("slow", {})
    await asyncio.sleep(0.1)

    command_result = await handle_workflow_command(engine, "/workflow abort")
    result = command_result.payload

    assert result is not None and result.status == "aborted"
    assert engine.history.get_run(run_id)["status"] == "aborted"
    branches = subprocess.run(
        ["git", "branch", "--list", "cli-agent-wf-*"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert not branches.strip()
    tags = subprocess.run(
        ["git", "tag", "--list", "cli-agent-wf-*"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert not tags.strip()
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_command_handler_lists_describes_and_starts(tmp_path: Path):
    _workflow_file(tmp_path)
    engine = _make_engine(
        tmp_path,
        agent_executor=lambda _prompt, _cwd: asyncio.sleep(0, result="done"),
        command_executor=lambda *_args: asyncio.sleep(
            0, result={"exit_code": 0, "stdout": "verify", "stderr": ""}
        ),
    )

    listed = await handle_workflow_command(engine, "/workflow list")
    described = await handle_workflow_command(engine, "/workflow describe demo")
    started = await handle_workflow_command(engine, "/workflow run demo feature=hello")
    result = await engine.wait(started.payload)

    assert any(workflow.id == "demo" for workflow in listed.payload)
    assert described.payload.id == "demo"
    assert started.action == "run"
    assert result.status == "completed"
    _remove_retained_worktree(result.worktree_path, result.branch_name, tmp_path)
    await engine.aclose()


@pytest.mark.unit
def test_bundled_workflow_recipes_validate(tmp_path: Path):
    workflows = WorkflowLoader(workspace_root=tmp_path).list()

    assert {workflow.id for workflow in workflows} >= {
        "strict-tdd",
        "code-health",
        "security-audit",
        "dependency-audit",
    }
    by_id = {workflow.id: workflow for workflow in workflows}
    assert any(step.id == "verify_ast" for step in by_id["strict-tdd"].steps)
    assert {step.tool_name for step in by_id["code-health"].steps if step.action == "tool"} >= {
        "search_code",
        "lsp_workspace_diagnostics",
    }
    dependency_commands = [step.command or "" for step in by_id["dependency-audit"].steps]
    assert any("deptry" in command for command in dependency_commands)
    assert any("pip-audit" in command for command in dependency_commands)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_tool_step_fails_on_explicit_error_status(tmp_path: Path):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "diagnostics.toml").write_text(
        'id = "diagnostics"\nname = "Diagnostics"\ndescription = "Tool gate test"\n'
        'version = "1.0.0"\n\n[[steps]]\nid = "diagnose"\nname = "Diagnose"\n'
        'action = "tool"\ntool_name = "lsp_workspace_diagnostics"\n',
        encoding="utf-8",
    )
    engine = _make_engine(tmp_path)
    engine.tool_executor = lambda *_args: {"status": "failed", "error_count": 2}

    run_id = engine.start("diagnostics", {})
    result = await engine.wait(run_id)

    assert result.status == "failed"
    assert "lsp_workspace_diagnostics" in (result.error or "")
    await engine.aclose()


@pytest.mark.unit
def test_workflow_command_parser_preserves_quoted_inputs():
    parsed = parse_workflow_command(
        '/workflow run strict-tdd feature_description="Fix retry handling" test_name=test_retry'
    )

    assert parsed.action == "run"
    assert parsed.workflow_id == "strict-tdd"
    assert parsed.inputs == {
        "feature_description": "Fix retry handling",
        "test_name": "test_retry",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_shell_command_requires_security_approval(tmp_path: Path):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "denied.toml").write_text(
        'id = "denied"\nname = "Denied"\ndescription = "Permission test"\nversion = "1.0.0"\n\n'
        '[[steps]]\nid = "touch"\nname = "Touch a file"\naction = "shell"\n'
        'command = "touch denied.txt"\n',
        encoding="utf-8",
    )
    called: list[str] = []

    async def command_executor(command: str, *_args: Any) -> dict[str, Any]:
        called.append(command)
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    engine = _make_engine(tmp_path, command_executor=command_executor, confirm=False)
    run_id = engine.start("denied", {})
    result = await engine.wait(run_id)

    assert result.status == "failed"
    assert "security policy" in (result.error or "").lower()
    assert called == []
    assert not (tmp_path / "denied.txt").exists()
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_permission_broker_waits_for_repl_decision(tmp_path: Path):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "approved.toml").write_text(
        'id = "approved"\nname = "Approved"\ndescription = "Approval broker test"\n'
        'version = "1.0.0"\n\n[[steps]]\nid = "touch"\nname = "Touch a file"\n'
        'action = "shell"\ncommand = "touch approved.txt"\n',
        encoding="utf-8",
    )
    broker = WorkflowPermissionBroker()
    commands: list[str] = []

    async def command_executor(command: str, *_args: Any) -> dict[str, Any]:
        commands.append(command)
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    engine = _make_engine(tmp_path, command_executor=command_executor)
    engine.permission_callback = broker.request
    run_id = engine.start("approved", {})
    for _ in range(100):
        if broker.pending is not None:
            break
        await asyncio.sleep(0)

    assert broker.pending is not None
    assert broker.pending.tool_name == "run_command"
    assert commands == []
    assert broker.resolve(True)
    result = await engine.wait(run_id)

    assert result.status == "completed", result.error
    assert commands == ["touch approved.txt"]
    assert broker.pending is None
    _remove_retained_worktree(result.worktree_path, result.branch_name, tmp_path)
    await engine.aclose()


@pytest.mark.unit
def test_workflow_loader_requires_evidence_for_expected_failures(tmp_path: Path):
    workflow_file = _workflow_file(tmp_path)
    text = workflow_file.read_text(encoding="utf-8").replace(
        "expected_exit_code = 0", "expected_exit_code = 1"
    )
    workflow_file.write_text(text, encoding="utf-8")
    with pytest.raises(WorkflowValidationError, match="add expected_output"):
        WorkflowLoader(
            workspace_root=tmp_path,
            user_config_dir=tmp_path / "home",
            include_builtins=False,
        ).list()


@pytest.mark.unit
def test_workflow_loader_supports_yaml_when_parser_is_installed(tmp_path: Path):
    pytest.importorskip("yaml")
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "yaml.yml").write_text(
        "id: yaml-demo\nname: YAML demo\ndescription: YAML parser test\nversion: 1.0.0\n"
        "steps:\n  - id: analyze\n    name: Analyze\n    action: agent_prompt\n"
        "    prompt: 'Review {{ target }}'\ninputs:\n  target:\n    type: string\n",
        encoding="utf-8",
    )

    workflows = WorkflowLoader(workspace_root=tmp_path, user_config_dir=tmp_path / "home").list()

    yaml_workflow = next(workflow for workflow in workflows if workflow.id == "yaml-demo")
    assert yaml_workflow.steps[0].prompt == "Review {{ target }}"


@pytest.mark.unit
def test_workflow_loader_reports_missing_optional_yaml_parser(monkeypatch, tmp_path: Path):
    workflow_file = tmp_path / "yaml.yml"
    workflow_file.write_text("id: yaml-demo\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(WorkflowValidationError, match="YAML workflows require PyYAML"):
        WorkflowLoader(workspace_root=tmp_path, include_builtins=False).load_file(workflow_file)


@pytest.mark.unit
def test_workflow_loader_rejects_oversized_files_before_reading(tmp_path: Path):
    workflow_file = tmp_path / "too-large.toml"
    workflow_file.write_bytes(b"x" * (1_000_001))

    with pytest.raises(WorkflowValidationError, match="cannot exceed 1 MB"):
        WorkflowLoader(workspace_root=tmp_path, include_builtins=False).load_file(workflow_file)


@pytest.mark.unit
def test_workflow_loader_normalizes_toml_parse_errors(tmp_path: Path):
    workflow_file = tmp_path / "malformed.toml"
    workflow_file.write_text("[broken", encoding="utf-8")

    with pytest.raises(WorkflowValidationError, match="Unable to read workflow 'malformed.toml'"):
        WorkflowLoader(workspace_root=tmp_path, include_builtins=False).load_file(workflow_file)


@pytest.mark.unit
def test_workflow_loader_prefers_workspace_over_user_and_rejects_same_source_duplicates(
    tmp_path: Path,
):
    workspace_dir = tmp_path / ".cli-agent" / "workflows"
    user_dir = tmp_path / "user-workflows"
    workspace_dir.mkdir(parents=True)
    user_dir.mkdir()
    (user_dir / "shared.toml").write_text(
        'id = "shared"\nname = "User definition"\ndescription = "User version"\n'
        '[[steps]]\nid = "analyze"\nname = "Analyze"\naction = "agent_prompt"\n'
        'prompt = "Review the user definition"\n',
        encoding="utf-8",
    )
    workspace_file = workspace_dir / "shared.toml"
    workspace_file.write_text(
        'id = "shared"\nname = "Workspace definition"\ndescription = "Workspace version"\n'
        '[[steps]]\nid = "analyze"\nname = "Analyze"\naction = "agent_prompt"\n'
        'prompt = "Review the workspace definition"\n',
        encoding="utf-8",
    )
    loader = WorkflowLoader(
        workspace_root=tmp_path,
        user_config_dir=user_dir,
        include_builtins=False,
    )

    loaded = loader.list()

    assert len(loaded) == 1
    assert loaded[0].name == "Workspace definition"
    assert loaded[0].description == "Workspace version"
    assert loaded[0].source == str(workspace_file.resolve())

    (workspace_dir / "duplicate.toml").write_text(
        'id = "shared"\nname = "Duplicate"\ndescription = "Duplicate workspace id"\n'
        '[[steps]]\nid = "analyze"\nname = "Analyze"\naction = "agent_prompt"\n'
        'prompt = "Review the duplicate"\n',
        encoding="utf-8",
    )
    with pytest.raises(WorkflowValidationError, match="Duplicate workflow id 'shared'"):
        loader.list()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("max_retries", "message"),
    [
        ("true", "must be an integer"),
        ('"one"', "must be an integer"),
        ("-1", "between 0 and 3"),
        ("4", "between 0 and 3"),
    ],
)
def test_workflow_loader_rejects_invalid_self_heal_retry_limits(
    tmp_path: Path, max_retries: str, message: str
):
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "retry.toml").write_text(
        'id = "retry"\nname = "Retry workflow"\ndescription = "Retry validation"\n'
        'version = "1.0.0"\n\n[[steps]]\nid = "check"\nname = "Check"\n'
        'action = "shell"\ncommand = "printf check"\n\n[steps.on_failure]\n'
        f'action = "self_heal"\nmax_retries = {max_retries}\n',
        encoding="utf-8",
    )

    with pytest.raises(WorkflowValidationError, match=message):
        WorkflowLoader(workspace_root=tmp_path, include_builtins=False).list()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stdout", "expected_status"),
    [
        ("FAILED tests/test_example.py::test_example - AssertionError", "completed"),
        ("ERROR collecting tests/test_example.py", "failed"),
        ("no tests ran in 0.01s", "failed"),
        ("unrelated command output", "failed"),
    ],
)
@pytest.mark.asyncio
async def test_workflow_pytest_failure_gate_requires_actual_failed_test_output(
    tmp_path: Path, stdout: str, expected_status: str
):
    _init_repo(tmp_path)
    directory = tmp_path / ".cli-agent" / "workflows"
    directory.mkdir(parents=True)
    (directory / "expected-failure.toml").write_text(
        'id = "expected-failure"\nname = "Expected failure"\n'
        'description = "TDD gate proof"\nversion = "1.0.0"\n\n'
        '[[steps]]\nid = "verify"\nname = "Verify failing test"\n'
        'action = "verify_gate"\n\n[steps.verify_gate]\n'
        'type = "command_exit_code"\ncommand = "pytest -q tests/test_example.py"\n'
        "expected_exit_code = 1\n",
        encoding="utf-8",
    )
    engine = _make_engine(
        tmp_path,
        command_executor=lambda *_args: asyncio.sleep(
            0, result={"exit_code": 1, "stdout": stdout, "stderr": ""}
        ),
    )

    run_id = engine.start("expected-failure", {})
    result = await engine.wait(run_id)

    assert result.status == expected_status
    if expected_status == "completed":
        _remove_retained_worktree(result.worktree_path, result.branch_name, tmp_path)
    else:
        assert "did not prove the expected pytest failure" in (result.error or "")
    await engine.aclose()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("/not-workflow list", "must start with"),
        ("/workflow list extra", "does not accept additional arguments"),
        ("/workflow describe", "requires a workflow id"),
        ("/workflow run demo missing_equals", "key=value"),
        ("/workflow run demo feature=one feature=two", "provided more than once"),
    ],
)
def test_workflow_command_parser_rejects_ambiguous_or_malformed_inputs(command: str, message: str):
    with pytest.raises(ValueError, match=message):
        parse_workflow_command(command)


def _remove_retained_worktree(path: str | None, branch: str | None, repo_root: Path) -> None:
    tag = f"{branch.removesuffix('-branch')}" if branch is not None else None
    if path is None or branch is None:
        return
    subprocess.run(["git", "worktree", "remove", "--force", path], cwd=repo_root, check=True)
    subprocess.run(["git", "branch", "-D", branch], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "tag", "-d", tag], cwd=repo_root, check=True, capture_output=True)
    Path(path).parent.rmdir()
