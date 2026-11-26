"""Sequential DAG workflow execution with permissions, durable history, and abort."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import signal
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..events import EventBus, EventTopic
from ..security import PermissionCallback, SecurityPolicy
from ..tools import (
    _build_safe_env,
    _redact_secrets,
    _window_command_output,
    worker_execution_scope,
)
from .checkpoint import WorkflowCheckpoint, WorkflowCheckpointManager
from .history import WorkflowHistoryStore
from .loader import WorkflowLoader
from .models import (
    _TEMPLATE_PATTERN,
    WorkflowDefinition,
    WorkflowRunResult,
    WorkflowStep,
    WorkflowStepResult,
    WorkflowValidationError,
    _is_pytest_command,
)

CommandExecutor = Callable[[str, Path, float, asyncio.Event], Any]
AgentExecutor = Callable[[str, Path], Any]
ToolExecutor = Callable[[str, dict[str, Any]], Any]


class WorkflowAborted(RuntimeError):
    """Raised inside the runner after a user requests an abort."""


class WorkflowExecutionError(RuntimeError):
    """Raised when a workflow action or verification gate fails."""


async def _read_bounded(
    stream: asyncio.StreamReader | None, limit: int = 1_000_000
) -> tuple[str, bool]:
    if stream is None:
        return "", False
    kept = bytearray()
    truncated = False
    while chunk := await stream.read(65536):
        remaining = limit - len(kept)
        if remaining > 0:
            kept.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    decoded = kept.decode("utf-8", errors="replace")
    redacted = _redact_secrets(decoded)
    return _window_command_output(redacted)[0], truncated


async def execute_shell_command(
    command: str,
    cwd: Path,
    timeout_seconds: float,
    abort_event: asyncio.Event,
) -> dict[str, Any]:
    """Execute one permission-approved command with timeout and cancellation cleanup."""
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        env=_build_safe_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    stdout_task = asyncio.create_task(_read_bounded(process.stdout))
    stderr_task = asyncio.create_task(_read_bounded(process.stderr))
    wait_task = asyncio.create_task(process.wait())
    abort_task = asyncio.create_task(abort_event.wait())

    async def stop_process() -> None:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.returncode is None:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            if process.returncode is None:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            else:
                await asyncio.sleep(0.05)
        except TimeoutError:
            pass
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        if process.returncode is None:
            await process.wait()

    try:
        done, _pending = await asyncio.wait(
            {wait_task, abort_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done and abort_event.is_set():
            await stop_process()
            raise WorkflowAborted("Workflow aborted by the user.")
        if wait_task not in done:
            await stop_process()
            raise WorkflowExecutionError(f"Command timed out after {timeout_seconds:g} seconds.")
        await stop_process()
        stdout, stdout_truncated = await stdout_task
        stderr, stderr_truncated = await stderr_task
        return {
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
    except BaseException:
        await stop_process()
        raise
    finally:
        if not abort_task.done():
            abort_task.cancel()
        pending_tasks: list[asyncio.Task[Any]] = [abort_task]
        if not wait_task.done():
            pending_tasks.append(wait_task)
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
                pending_tasks.append(task)
        await asyncio.gather(*pending_tasks, return_exceptions=True)


def _interpolate(value: Any, variables: dict[str, Any], *, shell: bool = False) -> Any:
    if isinstance(value, dict):
        return {key: _interpolate(item, variables, shell=shell) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate(item, variables, shell=shell) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: Any) -> str:
        name = match.group(1)
        if name not in variables:
            raise WorkflowValidationError(f"Workflow variable '{name}' is not defined yet.")
        current = variables[name]
        if isinstance(current, (dict, list)):
            current = json.dumps(current, ensure_ascii=False, sort_keys=True)
        text = str(current)
        return shlex.quote(text) if shell else text

    return _TEMPLATE_PATTERN.sub(replace, value)


class WorkflowEngine:
    """Runs one workflow at a time inside a disposable or retained Git worktree."""

    def __init__(
        self,
        *,
        loader: WorkflowLoader,
        history: WorkflowHistoryStore,
        security_policy: SecurityPolicy,
        checkpoint_manager: WorkflowCheckpointManager | None = None,
        event_bus: EventBus | None = None,
        command_executor: CommandExecutor | None = None,
        agent_executor: AgentExecutor | None = None,
        tool_executor: ToolExecutor | None = None,
        permission_callback: PermissionCallback | None = None,
    ) -> None:
        self.loader = loader
        self.history = history
        self.security_policy = security_policy
        self.checkpoint_manager = checkpoint_manager
        self.event_bus = event_bus
        self.command_executor = command_executor or execute_shell_command
        self.agent_executor = agent_executor
        self.permission_callback = permission_callback
        if tool_executor is None:
            from ..tools import execute_tool

            tool_executor = execute_tool
        self.tool_executor = tool_executor
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._active_task: asyncio.Task[WorkflowRunResult] | None = None
        self._active_run_id: str | None = None
        self._abort_event: asyncio.Event | None = None

    @property
    def is_running(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def add_event_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def list_workflows(self) -> list[WorkflowDefinition]:
        return self.loader.list()

    def describe(self, workflow_id: str) -> WorkflowDefinition:
        return self.loader.get(workflow_id)

    def start(self, workflow_id: str, inputs: dict[str, Any]) -> str:
        if self.is_running:
            raise WorkflowExecutionError(
                "A workflow is already running; abort it before starting another."
            )
        workflow = self.loader.get(workflow_id)
        normalized_inputs = self._validate_inputs(workflow, inputs)
        run_id = uuid.uuid4().hex
        self._abort_event = asyncio.Event()
        self._active_run_id = run_id
        self._active_task = asyncio.create_task(
            self._execute(run_id, workflow, normalized_inputs),
            name=f"workflow-{workflow.id}-{run_id[:8]}",
        )
        return run_id

    async def wait(self, run_id: str | None = None) -> WorkflowRunResult:
        if self._active_task is None or self._active_run_id is None:
            raise WorkflowExecutionError("There is no workflow run to wait for.")
        if run_id is not None and run_id != self._active_run_id:
            raise WorkflowExecutionError(f"Workflow run '{run_id}' is not the active run.")
        return await self._active_task

    async def abort(self) -> WorkflowRunResult | None:
        task = self._active_task
        if task is None:
            return None
        if task.done():
            return await task
        if self._abort_event is not None:
            self._abort_event.set()
        task.cancel()
        return await task

    async def aclose(self) -> None:
        if self.is_running:
            await self.abort()

    def _validate_inputs(
        self, workflow: WorkflowDefinition, supplied: dict[str, Any]
    ) -> dict[str, Any]:
        unknown = set(supplied) - set(workflow.inputs)
        if unknown:
            raise WorkflowValidationError(f"Unknown workflow inputs: {', '.join(sorted(unknown))}.")
        values: dict[str, Any] = {}
        for name, definition in workflow.inputs.items():
            value = supplied.get(name, definition.default)
            if value is None:
                if definition.required:
                    raise WorkflowValidationError(f"Missing required workflow input '{name}'.")
                continue
            if definition.value_type in {"string", "path"} and not isinstance(value, str):
                raise WorkflowValidationError(f"Workflow input '{name}' must be a string/path.")
            if definition.value_type == "integer" and isinstance(value, str):
                try:
                    value = int(value)
                except ValueError as exc:
                    raise WorkflowValidationError(
                        f"Workflow input '{name}' must be an integer."
                    ) from exc
            if definition.value_type == "integer" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise WorkflowValidationError(f"Workflow input '{name}' must be an integer.")
            if definition.value_type == "boolean" and isinstance(value, str):
                normalized = value.strip().lower()
                if normalized not in {"true", "false"}:
                    raise WorkflowValidationError(
                        f"Workflow input '{name}' must be 'true' or 'false'."
                    )
                value = normalized == "true"
            if definition.value_type == "boolean" and not isinstance(value, bool):
                raise WorkflowValidationError(f"Workflow input '{name}' must be a boolean.")
            if definition.value_type == "path":
                candidate = Path(value)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise WorkflowValidationError(
                        f"Workflow path input '{name}' must remain inside the workflow worktree."
                    )
            values[name] = value
        return values

    def _emit(self, payload: dict[str, Any]) -> None:
        safe_payload = {
            key: _redact_secrets(value) if isinstance(value, str) else value
            for key, value in payload.items()
        }
        if self.event_bus is not None:
            self.event_bus.publish_nowait(EventTopic.WORKFLOW_STEP_STATE, safe_payload)
        for listener in tuple(self._listeners):
            try:
                listener(safe_payload)
            except Exception:
                continue

    @contextmanager
    def _confirmation_scope(self) -> Iterator[None]:
        if self.permission_callback is None:
            yield
            return
        with self.security_policy.confirmation_scope(self.permission_callback):
            yield

    async def _check_permission(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[bool, str]:
        with self._confirmation_scope():
            return await self.security_policy.check_permission_async(tool_name, arguments)

    @staticmethod
    def _discard_checkpoint(checkpoint: WorkflowCheckpoint | None) -> str | None:
        if checkpoint is None:
            return None
        try:
            checkpoint.discard()
        except Exception as exc:
            return str(exc)
        return None

    async def _execute(
        self,
        run_id: str,
        workflow: WorkflowDefinition,
        inputs: dict[str, Any],
    ) -> WorkflowRunResult:
        self.history.start_run(run_id, workflow.id, inputs)
        self._emit(
            {"run_id": run_id, "workflow_id": workflow.id, "status": "starting", "step_id": None}
        )
        checkpoint: WorkflowCheckpoint | None = None
        completed: list[str] = []
        current_step_id: str | None = None
        try:
            checkpoint_manager = self.checkpoint_manager or WorkflowCheckpointManager(
                self.loader.workspace_root
            )
            checkpoint = checkpoint_manager.create(
                run_id=run_id, branch_prefix=workflow.git_tag_prefix
            )
            self.history.set_checkpoint(
                run_id,
                checkpoint.branch_name,
                checkpoint.tag_name,
                str(checkpoint.worktree_path),
            )
            variables = dict(inputs)
            for step in workflow.steps:
                current_step_id = step.id
                if self._abort_event is not None and self._abort_event.is_set():
                    raise WorkflowAborted("Workflow aborted by the user.")
                step_result = await self._execute_step_with_retries(
                    run_id, workflow, step, variables, checkpoint.worktree_path
                )
                if step_result.status != "completed":
                    raise WorkflowExecutionError(step_result.error or f"Step '{step.id}' failed.")
                variables[step.id] = step_result.output
                variables[f"{step.id}_output"] = step_result.output
                completed.append(step.id)
                current_step_id = None
            checkpoint.keep()
            self.history.finish_run(run_id, status="completed", keep_checkpoint=True)
            run_result = WorkflowRunResult(
                run_id=run_id,
                workflow_id=workflow.id,
                status="completed",
                branch_name=checkpoint.branch_name,
                tag_name=checkpoint.tag_name,
                worktree_path=str(checkpoint.worktree_path),
                completed_steps=tuple(completed),
            )
            self._emit(
                {
                    "run_id": run_id,
                    "workflow_id": workflow.id,
                    "status": "completed",
                    "step_id": None,
                    "branch_name": checkpoint.branch_name,
                    "tag_name": checkpoint.tag_name,
                    "worktree_path": str(checkpoint.worktree_path),
                }
            )
            return run_result
        except asyncio.CancelledError:
            cleanup_error = self._discard_checkpoint(checkpoint)
            error = "Workflow aborted by the user."
            if cleanup_error:
                error = f"{error} Checkpoint cleanup failed: {cleanup_error}"
            status = "failed" if cleanup_error else "aborted"
            if current_step_id is not None:
                try:
                    self.history.update_step(
                        run_id,
                        current_step_id,
                        status="aborted",
                        error=error,
                        finished=True,
                    )
                except Exception:
                    pass
            self.history.finish_run(
                run_id,
                status=status,
                error=error,
                keep_checkpoint=bool(cleanup_error),
            )
            self._emit(
                {
                    "run_id": run_id,
                    "workflow_id": workflow.id,
                    "status": status,
                    "step_id": current_step_id,
                    "error": error,
                }
            )
            return WorkflowRunResult(
                run_id=run_id,
                workflow_id=workflow.id,
                status=status,
                branch_name=checkpoint.branch_name if checkpoint and cleanup_error else None,
                tag_name=checkpoint.tag_name if checkpoint and cleanup_error else None,
                worktree_path=str(checkpoint.worktree_path)
                if checkpoint and cleanup_error
                else None,
                error=error,
                completed_steps=tuple(completed),
            )
        except WorkflowAborted as exc:
            cleanup_error = self._discard_checkpoint(checkpoint)
            error = str(exc)
            if cleanup_error:
                error = f"{error} Checkpoint cleanup failed: {cleanup_error}"
            status = "failed" if cleanup_error else "aborted"
            self.history.finish_run(
                run_id,
                status=status,
                error=error,
                keep_checkpoint=bool(cleanup_error),
            )
            self._emit(
                {
                    "run_id": run_id,
                    "workflow_id": workflow.id,
                    "status": status,
                    "step_id": None,
                    "error": error,
                }
            )
            return WorkflowRunResult(
                run_id=run_id,
                workflow_id=workflow.id,
                status=status,
                branch_name=checkpoint.branch_name if checkpoint and cleanup_error else None,
                tag_name=checkpoint.tag_name if checkpoint and cleanup_error else None,
                worktree_path=str(checkpoint.worktree_path)
                if checkpoint and cleanup_error
                else None,
                error=error,
                completed_steps=tuple(completed),
            )
        except Exception as exc:
            cleanup_error = self._discard_checkpoint(checkpoint)
            error = str(exc)
            if cleanup_error:
                error = f"{error} Checkpoint cleanup failed: {cleanup_error}"
            self.history.finish_run(
                run_id,
                status="failed",
                error=error,
                keep_checkpoint=bool(cleanup_error),
            )
            self._emit(
                {
                    "run_id": run_id,
                    "workflow_id": workflow.id,
                    "status": "failed",
                    "step_id": None,
                    "error": error,
                }
            )
            return WorkflowRunResult(
                run_id=run_id,
                workflow_id=workflow.id,
                status="failed",
                branch_name=checkpoint.branch_name if checkpoint and cleanup_error else None,
                tag_name=checkpoint.tag_name if checkpoint and cleanup_error else None,
                worktree_path=str(checkpoint.worktree_path)
                if checkpoint and cleanup_error
                else None,
                error=error,
                completed_steps=tuple(completed),
            )

    async def _execute_step_with_retries(
        self,
        run_id: str,
        workflow: WorkflowDefinition,
        step: WorkflowStep,
        variables: dict[str, Any],
        cwd: Path,
    ) -> WorkflowStepResult:
        retries = step.max_retries if step.retry_prompt and self.agent_executor else 0
        for attempt in range(1, retries + 2):
            if self._abort_event is not None and self._abort_event.is_set():
                raise WorkflowAborted("Workflow aborted by the user.")
            self.history.update_step(
                run_id, step.id, status="running", attempt=attempt, finished=False
            )
            self._emit(
                {
                    "run_id": run_id,
                    "workflow_id": workflow.id,
                    "step_id": step.id,
                    "status": "running",
                    "attempt": attempt,
                    "name": step.name,
                }
            )
            try:
                result = await self._execute_step(step, variables, cwd)
                self.history.update_step(
                    run_id,
                    step.id,
                    status="completed",
                    attempt=attempt,
                    output=result.output,
                    exit_code=result.exit_code,
                    finished=True,
                )
                self._emit(
                    {
                        "run_id": run_id,
                        "workflow_id": workflow.id,
                        "step_id": step.id,
                        "status": "completed",
                        "attempt": attempt,
                    }
                )
                return result
            except WorkflowAborted:
                raise
            except Exception as exc:
                message = str(exc)
                self.history.update_step(
                    run_id,
                    step.id,
                    status="retrying" if attempt <= retries else "failed",
                    attempt=attempt,
                    error=message,
                    finished=attempt > retries,
                )
                self._emit(
                    {
                        "run_id": run_id,
                        "workflow_id": workflow.id,
                        "step_id": step.id,
                        "status": "retrying" if attempt <= retries else "failed",
                        "attempt": attempt,
                        "error": message,
                    }
                )
                if attempt > retries:
                    return WorkflowStepResult(step.id, "failed", error=message)
                heal_context = dict(variables)
                heal_context["error"] = message
                prompt = str(_interpolate(step.retry_prompt, heal_context))
                with worker_execution_scope(cwd), self._confirmation_scope():
                    await self._call(self.agent_executor, prompt, cwd)
        return WorkflowStepResult(step.id, "failed", error="Retry limit exhausted.")

    async def _execute_step(
        self, step: WorkflowStep, variables: dict[str, Any], cwd: Path
    ) -> WorkflowStepResult:
        if self._abort_event is not None and self._abort_event.is_set():
            raise WorkflowAborted("Workflow aborted by the user.")
        output: Any = None
        exit_code: int | None = None
        if step.action == "agent_prompt":
            if self.agent_executor is None:
                raise WorkflowExecutionError("No agent prompt executor is configured.")
            prompt = str(_interpolate(step.prompt, variables))
            prior_state = dict(variables)
            if prior_state:
                context_json = _redact_secrets(
                    json.dumps(prior_state, ensure_ascii=False, default=str)[:16_000]
                )
                prompt = f"{prompt}\n\nAvailable workflow state from prior steps:\n{context_json}"
            with worker_execution_scope(cwd), self._confirmation_scope():
                output = await self._call(self.agent_executor, prompt, cwd)
        elif step.action == "tool":
            params = _interpolate(step.params, variables)
            allowed, reason = await self._check_permission(step.tool_name or "", params)
            if not allowed:
                raise WorkflowExecutionError(f"Tool '{step.tool_name}' was denied: {reason}")
            with worker_execution_scope(cwd):
                output = await self._call(self.tool_executor, step.tool_name or "", params)
            if isinstance(output, dict) and output.get("status") in {"error", "failed"}:
                detail = output.get("error") or output.get("message") or "reported a failure"
                raise WorkflowExecutionError(f"Tool '{step.tool_name}' {detail}.")
        elif step.action == "shell":
            command = str(_interpolate(step.command, variables, shell=True))
            output = await self._run_command(command, cwd, 60)
            exit_code = int(output.get("exit_code", 1)) if isinstance(output, dict) else 1
            if exit_code != 0:
                raise WorkflowExecutionError(
                    f"Shell step '{step.id}' exited with code {exit_code}: {output.get('stderr', '')}"
                )
        elif step.action != "verify_gate":
            raise WorkflowExecutionError(f"Unsupported step action '{step.action}'.")

        if step.verify_gate is not None:
            gate_output = await self._run_gate(step, variables, cwd)
            output = {"action_output": output, "verify_gate": gate_output}
            exit_code = int(gate_output["exit_code"])
        return WorkflowStepResult(step.id, "completed", output=output, exit_code=exit_code)

    async def _run_gate(
        self, step: WorkflowStep, variables: dict[str, Any], cwd: Path
    ) -> dict[str, Any]:
        gate = step.verify_gate
        if gate is None:
            raise WorkflowExecutionError(f"Step '{step.id}' has no verify gate.")
        command = str(_interpolate(gate.command, variables, shell=True))
        result = await self._run_command(command, cwd, gate.timeout_seconds)
        exit_code = int(result.get("exit_code", 1))
        if exit_code != gate.expected_exit_code:
            raise WorkflowExecutionError(
                f"Verify gate for '{step.id}' expected exit {gate.expected_exit_code}, got {exit_code}."
            )
        expected_output = (
            str(_interpolate(gate.expected_output, variables)) if gate.expected_output else None
        )
        expected_stderr = (
            str(_interpolate(gate.expected_stderr, variables)) if gate.expected_stderr else None
        )
        if expected_output and expected_output not in str(result.get("stdout", "")):
            raise WorkflowExecutionError(
                f"Verify gate for '{step.id}' did not find expected stdout text."
            )
        if expected_stderr and expected_stderr not in str(result.get("stderr", "")):
            raise WorkflowExecutionError(
                f"Verify gate for '{step.id}' did not find expected stderr text."
            )
        if gate.expected_exit_code != 0 and not (gate.expected_output or gate.expected_stderr):
            stdout = str(result.get("stdout", ""))
            if (
                gate.expected_exit_code != 1
                or not _is_pytest_command(gate.command)
                or "ERROR " in stdout
                or "no tests ran" in stdout.lower()
                or "FAILED" not in stdout
            ):
                raise WorkflowExecutionError(
                    f"Verify gate for '{step.id}' did not prove the expected pytest failure."
                )
        return result

    async def _run_command(self, command: str, cwd: Path, timeout_seconds: float) -> dict[str, Any]:
        allowed, reason = await self._check_permission("run_command", {"command": command})
        if not allowed:
            raise WorkflowExecutionError(f"Command rejected by security policy: {reason}")
        result = await self._call(
            self.command_executor,
            command,
            cwd,
            timeout_seconds,
            self._abort_event or asyncio.Event(),
        )
        if not isinstance(result, dict):
            raise WorkflowExecutionError("Command executor returned an invalid result.")
        return result

    @staticmethod
    async def _call(callback: Callable[..., Any] | None, *arguments: Any) -> Any:
        if callback is None:
            raise WorkflowExecutionError("Required workflow executor is not configured.")
        if inspect.iscoroutinefunction(callback):
            return await callback(*arguments)
        result = await asyncio.to_thread(callback, *arguments)
        if inspect.isawaitable(result):
            return await result
        return result
