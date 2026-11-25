"""Validated value objects for declarative engineering workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class WorkflowValidationError(ValueError):
    """Raised when a workflow definition is malformed or unsafe to execute."""


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TEMPLATE_PATTERN = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
_ALLOWED_INPUT_TYPES = {"string", "integer", "boolean", "path"}
_ALLOWED_ACTIONS = {"tool", "agent_prompt", "verify_gate", "shell"}


def _is_pytest_command(command: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\s)(?:uv\s+run\s+)?(?:python(?:\d+(?:\.\d+)?)?\s+-m\s+)?pytest(?:\s|$)",
            command,
        )
    )


@dataclass(frozen=True)
class WorkflowInput:
    name: str
    value_type: str
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass(frozen=True)
class VerifyGate:
    gate_type: str
    command: str
    expected_exit_code: int = 0
    timeout_seconds: int = 60
    expected_output: str | None = None
    expected_stderr: str | None = None


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    name: str
    action: str
    depends_on: tuple[str, ...] = ()
    prompt: str | None = None
    command: str | None = None
    tool_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    verify_gate: VerifyGate | None = None
    retry_prompt: str | None = None
    max_retries: int = 0


@dataclass(frozen=True)
class WorkflowDefinition:
    id: str
    name: str
    description: str
    version: str
    inputs: dict[str, WorkflowInput]
    steps: tuple[WorkflowStep, ...]
    git_tag_prefix: str = "cli-agent-wf-"
    source: str = ""


@dataclass(frozen=True)
class WorkflowRunResult:
    run_id: str
    workflow_id: str
    status: str
    branch_name: str | None = None
    tag_name: str | None = None
    worktree_path: str | None = None
    error: str | None = None
    completed_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowStepResult:
    step_id: str
    status: str
    output: Any = None
    error: str | None = None
    exit_code: int | None = None


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowValidationError(f"'{field_name}' must be a non-empty string.")
    return value.strip()


def _validate_id(value: Any, field_name: str) -> str:
    result = _require_string(value, field_name)
    if not _ID_PATTERN.fullmatch(result):
        raise WorkflowValidationError(
            f"'{field_name}' must start with a lowercase letter and contain only lowercase letters, digits, '_' or '-'."
        )
    return result


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowValidationError(f"'{field_name}' must be a table/object.")
    return value


def _reject_unknown(data: dict[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise WorkflowValidationError(
            f"Unknown field(s) in '{field_name}': {', '.join(sorted(map(str, unknown)))}."
        )


def _parse_gate(value: Any, field_name: str) -> VerifyGate:
    data = _as_mapping(value, field_name)
    _reject_unknown(
        data,
        {
            "type",
            "command",
            "expected_exit_code",
            "timeout_seconds",
            "expected_output",
            "expected_stderr",
        },
        field_name,
    )
    gate_type = _require_string(data.get("type"), f"{field_name}.type")
    if gate_type != "command_exit_code":
        raise WorkflowValidationError(f"Unsupported verify gate type '{gate_type}'.")
    command = _require_string(data.get("command"), f"{field_name}.command")
    expected_exit = data.get("expected_exit_code", 0)
    timeout = data.get("timeout_seconds", 60)
    if isinstance(expected_exit, bool) or not isinstance(expected_exit, int):
        raise WorkflowValidationError(f"'{field_name}.expected_exit_code' must be an integer.")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 300:
        raise WorkflowValidationError(f"'{field_name}.timeout_seconds' must be between 1 and 300.")
    expected_output = data.get("expected_output")
    expected_stderr = data.get("expected_stderr")
    for name, item in (("expected_output", expected_output), ("expected_stderr", expected_stderr)):
        if item is not None and not isinstance(item, str):
            raise WorkflowValidationError(f"'{field_name}.{name}' must be a string.")
    if (
        expected_exit != 0
        and not (expected_output or expected_stderr)
        and not (expected_exit == 1 and _is_pytest_command(command))
    ):
        raise WorkflowValidationError(
            f"'{field_name}' expects a failing exit code; add expected_output or expected_stderr so unrelated errors cannot pass the gate."
        )
    return VerifyGate(
        gate_type=gate_type,
        command=command,
        expected_exit_code=expected_exit,
        timeout_seconds=timeout,
        expected_output=expected_output,
        expected_stderr=expected_stderr,
    )


def parse_workflow(data: Any, *, source: str = "") -> WorkflowDefinition:
    """Validate untrusted TOML/YAML objects before making them executable."""
    root = _as_mapping(data, "workflow")
    _reject_unknown(
        root, {"id", "name", "description", "version", "inputs", "checkpoints", "steps"}, "workflow"
    )
    workflow_id = _validate_id(root.get("id"), "id")
    name = _require_string(root.get("name"), "name")
    description = _require_string(root.get("description"), "description")
    version = _require_string(root.get("version", "1.0.0"), "version")

    raw_inputs = root.get("inputs", {})
    raw_inputs = _as_mapping(raw_inputs, "inputs")
    inputs: dict[str, WorkflowInput] = {}
    for raw_name, raw_definition in raw_inputs.items():
        input_name = _validate_id(raw_name, "input name")
        definition = _as_mapping(raw_definition, f"inputs.{input_name}")
        _reject_unknown(
            definition,
            {"type", "description", "required", "default"},
            f"inputs.{input_name}",
        )
        value_type = _require_string(definition.get("type"), f"inputs.{input_name}.type")
        if value_type not in _ALLOWED_INPUT_TYPES:
            raise WorkflowValidationError(
                f"Unsupported input type '{value_type}' for '{input_name}'."
            )
        description_text = definition.get("description", "")
        if not isinstance(description_text, str):
            raise WorkflowValidationError(f"'inputs.{input_name}.description' must be a string.")
        required = definition.get("required", "default" not in definition)
        if not isinstance(required, bool):
            raise WorkflowValidationError(f"'inputs.{input_name}.required' must be a boolean.")
        inputs[input_name] = WorkflowInput(
            name=input_name,
            value_type=value_type,
            description=description_text,
            required=required,
            default=definition.get("default"),
        )

    raw_checkpoints = root.get("checkpoints", {})
    checkpoint_data = _as_mapping(raw_checkpoints, "checkpoints")
    _reject_unknown(checkpoint_data, {"enabled", "git_tag_prefix"}, "checkpoints")
    if checkpoint_data.get("enabled", True) is not True:
        raise WorkflowValidationError(
            "Workflow checkpoints cannot be disabled; execution always uses an isolated Git worktree."
        )
    git_tag_prefix = checkpoint_data.get("git_tag_prefix", "cli-agent-wf-")
    if not isinstance(git_tag_prefix, str) or not git_tag_prefix.strip():
        raise WorkflowValidationError("'checkpoints.git_tag_prefix' must be a non-empty string.")

    raw_steps = root.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise WorkflowValidationError("'steps' must be a non-empty array.")
    if len(raw_steps) > 100:
        raise WorkflowValidationError("A workflow cannot define more than 100 steps.")

    steps: list[WorkflowStep] = []
    known_ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps):
        field_name = f"steps[{index}]"
        item = _as_mapping(raw_step, field_name)
        _reject_unknown(
            item,
            {
                "id",
                "name",
                "action",
                "depends_on",
                "prompt",
                "command",
                "tool_name",
                "params",
                "verify_gate",
                "on_failure",
            },
            field_name,
        )
        step_id = _validate_id(item.get("id"), f"{field_name}.id")
        if step_id in known_ids:
            raise WorkflowValidationError(f"Duplicate step id '{step_id}'.")
        known_ids.add(step_id)
        step_name = _require_string(item.get("name", step_id), f"{field_name}.name")
        action = _require_string(item.get("action"), f"{field_name}.action")
        if action not in _ALLOWED_ACTIONS:
            raise WorkflowValidationError(f"Unsupported action '{action}' in step '{step_id}'.")

        raw_dependencies = item.get("depends_on")
        if raw_dependencies is None:
            dependencies = (steps[-1].id,) if steps else ()
        elif isinstance(raw_dependencies, list) and all(
            isinstance(dep, str) for dep in raw_dependencies
        ):
            dependencies = tuple(raw_dependencies)
        else:
            raise WorkflowValidationError(
                f"'{field_name}.depends_on' must be an array of step ids."
            )

        prompt = item.get("prompt")
        command = item.get("command")
        tool_name = item.get("tool_name")
        params = item.get("params", {})
        if action == "agent_prompt":
            prompt = _require_string(prompt, f"{field_name}.prompt")
        if action == "shell":
            command = _require_string(command, f"{field_name}.command")
        if action == "tool":
            tool_name = _require_string(tool_name, f"{field_name}.tool_name")
            params = _as_mapping(params, f"{field_name}.params")
        verify_gate = (
            _parse_gate(item["verify_gate"], f"{field_name}.verify_gate")
            if "verify_gate" in item
            else None
        )
        if action == "verify_gate" and verify_gate is None:
            raise WorkflowValidationError(f"Step '{step_id}' requires a verify_gate definition.")

        raw_failure = item.get("on_failure", {})
        failure = _as_mapping(raw_failure, f"{field_name}.on_failure")
        _reject_unknown(
            failure,
            {"action", "prompt", "max_retries"},
            f"{field_name}.on_failure",
        )
        retry_prompt: str | None = None
        max_retries = 0
        if failure:
            if failure.get("action") != "self_heal":
                raise WorkflowValidationError(f"Unsupported on_failure action in step '{step_id}'.")
            retry_prompt = _require_string(
                failure.get("prompt", "Use the error details to repair this workflow step."),
                f"{field_name}.on_failure.prompt",
            )
            raw_retries = failure.get("max_retries", 1)
            if isinstance(raw_retries, bool) or not isinstance(raw_retries, int):
                raise WorkflowValidationError(
                    f"'{field_name}.on_failure.max_retries' must be an integer."
                )
            if not 0 <= raw_retries <= 3:
                raise WorkflowValidationError(
                    f"'{field_name}.on_failure.max_retries' must be between 0 and 3."
                )
            max_retries = raw_retries

        steps.append(
            WorkflowStep(
                id=step_id,
                name=step_name,
                action=action,
                depends_on=dependencies,
                prompt=prompt,
                command=command,
                tool_name=tool_name,
                params=dict(params),
                verify_gate=verify_gate,
                retry_prompt=retry_prompt,
                max_retries=max_retries,
            )
        )

    step_ids = {step.id for step in steps}
    for step in steps:
        missing = set(step.depends_on) - step_ids
        if missing:
            raise WorkflowValidationError(
                f"Step '{step.id}' has unknown dependency: {', '.join(sorted(missing))}."
            )
        if step.id in step.depends_on:
            raise WorkflowValidationError(f"Step '{step.id}' cannot depend on itself.")

    ordered = _topological_order(steps)
    return WorkflowDefinition(
        id=workflow_id,
        name=name,
        description=description,
        version=version,
        inputs=inputs,
        steps=tuple(ordered),
        git_tag_prefix=git_tag_prefix.strip(),
        source=source,
    )


def _topological_order(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    by_id = {step.id: step for step in steps}
    pending = {step.id: set(step.depends_on) for step in steps}
    result: list[WorkflowStep] = []
    while pending:
        ready = [step.id for step in steps if step.id in pending and not pending[step.id]]
        if not ready:
            raise WorkflowValidationError("Workflow step dependencies contain a cycle.")
        for step_id in ready:
            result.append(by_id[step_id])
            pending.pop(step_id)
            for dependencies in pending.values():
                dependencies.discard(step_id)
    return result
