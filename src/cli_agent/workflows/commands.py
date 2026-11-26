"""Small parser and dispatcher for the interactive /workflow command family."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from .models import WorkflowDefinition, WorkflowRunResult, WorkflowStep
from .runner import WorkflowEngine


@dataclass(frozen=True)
class WorkflowCommand:
    action: str
    workflow_id: str | None = None
    inputs: dict[str, str] | None = None


@dataclass(frozen=True)
class WorkflowCommandResult:
    action: str
    message: str
    payload: Any = None


def format_workflow_inputs(workflow: WorkflowDefinition) -> str:
    """Format required/optional input descriptions for workflow listings."""
    if not workflow.inputs:
        return "No inputs"
    formatted: list[str] = []
    for item in workflow.inputs.values():
        requirement = "required" if item.required else "optional"
        description = f": {item.description}" if item.description else ""
        formatted.append(f"{item.name} ({item.value_type}, {requirement}){description}")
    return "\n".join(formatted)


def format_verify_gate(step: WorkflowStep) -> str:
    """Describe a step's automated gate without hiding its expected evidence."""
    gate = step.verify_gate
    if gate is None:
        return "—"
    details = [f"expected exit {gate.expected_exit_code}", f"timeout {gate.timeout_seconds}s"]
    if gate.expected_output:
        details.append(f"stdout contains {gate.expected_output!r}")
    if gate.expected_stderr:
        details.append(f"stderr contains {gate.expected_stderr!r}")
    return f"{gate.gate_type}: {gate.command} ({'; '.join(details)})"


def parse_workflow_command(command: str) -> WorkflowCommand:
    parts = shlex.split(command)
    if not parts or parts[0] != "/workflow":
        raise ValueError("Workflow commands must start with '/workflow'.")
    action = parts[1].lower() if len(parts) > 1 else "list"
    if action not in {"list", "describe", "run", "abort"}:
        raise ValueError("Use /workflow list|describe <id>|run <id> [key=value]|abort.")
    if action in {"list", "abort"}:
        if len(parts) > 2:
            raise ValueError(f"/workflow {action} does not accept additional arguments.")
        return WorkflowCommand(action)
    if len(parts) < 3:
        raise ValueError(f"/workflow {action} requires a workflow id.")
    workflow_id = parts[2]
    if action == "describe":
        if len(parts) > 3:
            raise ValueError("/workflow describe accepts one workflow id.")
        return WorkflowCommand(action, workflow_id)
    inputs: dict[str, str] = {}
    for assignment in parts[3:]:
        key, separator, value = assignment.partition("=")
        if not separator or not key:
            raise ValueError(f"Workflow input '{assignment}' must use key=value syntax.")
        if key in inputs:
            raise ValueError(f"Workflow input '{key}' was provided more than once.")
        inputs[key] = value
    return WorkflowCommand(action, workflow_id, inputs)


async def handle_workflow_command(engine: WorkflowEngine, command: str) -> WorkflowCommandResult:
    parsed = parse_workflow_command(command)
    if parsed.action == "list":
        workflows = engine.list_workflows()
        return WorkflowCommandResult("list", f"Found {len(workflows)} workflow(s).", workflows)
    if parsed.action == "describe":
        definition: WorkflowDefinition = engine.describe(parsed.workflow_id or "")
        return WorkflowCommandResult("describe", definition.description, definition)
    if parsed.action == "run":
        run_id = engine.start(parsed.workflow_id or "", parsed.inputs or {})
        return WorkflowCommandResult("run", f"Started workflow run {run_id}.", run_id)
    if not engine.is_running:
        return WorkflowCommandResult("abort", "No workflow run is active.")
    result: WorkflowRunResult | None = await engine.abort()
    if result is None:
        return WorkflowCommandResult("abort", "No workflow run is active.")
    return WorkflowCommandResult(
        "abort", f"Workflow run finished with status {result.status}.", result
    )
