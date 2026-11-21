"""
Hierarchical Multi-Agent Teams (Swarm) for cli-agent.
Implements Organic Driven Development (ODD) with role-specialized agent loops:
- Lead Architect: Requirements analysis, task graph decomposition, synthesis.
- Feature Worker: Bounded, isolated implementation with scoped edit surfaces.
- QA Verifier: Independent validation, test execution, regression detection.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .tools import (
    GEMINI_TOOLS_DECLARATIONS,
    reset_allowed_write_paths,
    set_allowed_write_paths,
)
from .vertex import VertexAgentClient

MAX_DELEGATION_DEPTH = 3


class AgentRole(StrEnum):
    ARCHITECT = "architect"
    WORKER = "worker"
    QA = "qa"


ROLE_ALLOWED_TOOLS: dict[AgentRole, set[str]] = {
    AgentRole.ARCHITECT: {
        "read_file",
        "list_directory",
        "search_code",
        "hybrid_search_code",
        "get_symbol_outline",
        "find_symbol_definition",
        "find_symbol_references",
        "search_memory",
        "save_memory",
        "delegate_task",
    },
    AgentRole.WORKER: {
        "read_file",
        "list_directory",
        "search_code",
        "hybrid_search_code",
        "get_symbol_outline",
        "find_symbol_definition",
        "write_file",
        "edit_file",
    },
    AgentRole.QA: {
        "read_file",
        "list_directory",
        "search_code",
        "hybrid_search_code",
        "run_command",
    },
}

ROLE_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.ARCHITECT: """You are the Lead Architect in an autonomous multi-agent engineering swarm.
Your job is to analyze the user goal, inspect the codebase architecture, formulate an execution plan,
and coordinate feature workers and QA verifiers.
Always be modular, disciplined, and verify before assuming.""",
    AgentRole.WORKER: """You are a dedicated Feature Worker in an autonomous multi-agent swarm.
You have a clean, bounded context. Your sole mission is to execute the assigned task unit.
You must strictly restrict edits to your approved files. Never modify files outside your approved scope.
When done, summarize the exact modifications made.""",
    AgentRole.QA: """You are an independent QA Verifier in an autonomous multi-agent swarm.
You have zero code mutation privileges. Your job is to verify implementation correctness by inspecting files
and running the test suites and linters via `run_command`.
Produce an objective PASS or FAIL determination with clear failure logs if tests fail.""",
}


@dataclass
class TaskAssignment:
    task_id: str
    role: AgentRole
    objective: str
    allowed_files: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    parent_task_id: str | None = None
    max_turns: int = 8


@dataclass
class WorkerResult:
    task_id: str
    role: AgentRole
    status: str  # "success" | "failed" | "rejected"
    summary: str
    files_modified: list[str] = field(default_factory=list)
    turn_count: int = 0
    test_output: str | None = None
    error: str | None = None


@dataclass
class TeamWorkflowResult:
    goal: str
    plan: str
    worker_results: list[WorkerResult]
    qa_result: WorkerResult | None
    status: str  # "success" | "failed"
    summary: str


class MultiAgentTeam:
    """Coordinates hierarchical multi-agent teams with strict role isolation and recursion guards."""

    def __init__(self, client: VertexAgentClient, max_depth: int = MAX_DELEGATION_DEPTH):
        self.client = client
        self.max_depth = max_depth
        self._active_workers: list[WorkerResult] = []

    def _get_role_tools(self, role: AgentRole) -> list[dict[str, Any]]:
        """Filters global tool declarations according to the role's privilege matrix."""
        allowed = ROLE_ALLOWED_TOOLS.get(role, set())
        return [decl for decl in GEMINI_TOOLS_DECLARATIONS if decl.get("name") in allowed]

    async def spawn_worker(
        self,
        assignment: TaskAssignment,
        depth: int = 0,
        on_event: Callable[[str, str], None] | None = None,
    ) -> WorkerResult:
        """Spawns an isolated subagent worker with bounded context and approved surfaces."""
        if depth >= self.max_depth:
            return WorkerResult(
                task_id=assignment.task_id,
                role=assignment.role,
                status="rejected",
                summary="Delegation depth limit exceeded.",
                error=f"Max delegation depth ({self.max_depth}) reached. Subagent spawn blocked.",
            )

        if on_event:
            on_event(
                "spawn",
                f"Spawning {assignment.role.value.upper()} worker for task [{assignment.task_id}]: {assignment.objective}",
            )

        role_tools = self._get_role_tools(assignment.role)
        base_prompt = ROLE_SYSTEM_PROMPTS.get(assignment.role, "")
        scoped_system_prompt = (
            f"{base_prompt}\n\n"
            f"### ASSIGNED TASK [{assignment.task_id}]:\n"
            f"Objective: {assignment.objective}\n"
            f"Approved Edit Files: {', '.join(assignment.allowed_files) if assignment.allowed_files else 'None (read-only)'}\n"
            f"Context Reference Files: {', '.join(assignment.context_files) if assignment.context_files else 'None'}\n"
        )

        files_modified: list[str] = []
        turn_counter = 0

        def track_tool(name: str, args: dict[str, Any]) -> None:
            nonlocal turn_counter
            turn_counter += 1
            if name in ("write_file", "edit_file"):
                target_path = str(args.get("path", ""))
                if target_path and target_path not in files_modified:
                    files_modified.append(target_path)
            if on_event:
                on_event("tool_call", f"[{assignment.role.value}] {name}")

        user_content = (
            f"Execute task: {assignment.objective}\n"
            f"Allowed files: {assignment.allowed_files}\n"
            f"Context files: {assignment.context_files}"
        )
        initial_messages = [{"role": "user", "parts": [{"text": user_content}]}]

        # Enforce hard write restriction scope at contextvar level if worker has allowed_files
        token = None
        if assignment.role == AgentRole.WORKER and assignment.allowed_files:
            token = set_allowed_write_paths(assignment.allowed_files)

        try:
            summary = await self.client.chat(
                messages=initial_messages,
                on_tool_call=track_tool,
                max_turns=assignment.max_turns,
                tools_declarations=role_tools,
                system_instruction=scoped_system_prompt,
            )
            result = WorkerResult(
                task_id=assignment.task_id,
                role=assignment.role,
                status="success",
                summary=summary,
                files_modified=files_modified,
                turn_count=turn_counter,
            )
        except Exception as e:
            result = WorkerResult(
                task_id=assignment.task_id,
                role=assignment.role,
                status="failed",
                summary="",
                files_modified=files_modified,
                turn_count=turn_counter,
                error=str(e),
            )
        finally:
            if token is not None:
                reset_allowed_write_paths(token)

        self._active_workers.append(result)
        return result

    async def run_workflow(
        self,
        goal: str,
        max_feedback_rounds: int = 3,
        on_event: Callable[[str, str], None] | None = None,
    ) -> TeamWorkflowResult:
        """
        Coordinates the complete Organic Driven Development (ODD) multi-agent workflow:
        1. Lead Architect formulates the architectural plan.
        2. Feature Worker implements the discrete task unit.
        3. Independent QA Verifier runs test suite and validates regressions.
        4. If QA fails and rounds remain, loops feedback back to Worker to self-heal.
        5. Architect synthesizes the final outcome and records episodic memory reflection.
        """
        # Step 1: Lead Architect Planning
        if on_event:
            on_event("stage", "Phase 1: Lead Architect Analysis & Planning")

        plan_prompt = (
            f"You are the Lead Architect. Break down the following request into a concrete, minimal implementation plan:\n"
            f"Request: {goal}\n"
            f"Identify target files to edit and tests to verify."
        )
        plan = await self.client.chat(
            messages=[{"role": "user", "parts": [{"text": plan_prompt}]}],
            system_instruction=ROLE_SYSTEM_PROMPTS[AgentRole.ARCHITECT],
            max_turns=3,
        )

        all_worker_results: list[WorkerResult] = []
        qa_result: WorkerResult | None = None
        current_objective = f"Implement feature according to plan:\n{plan}\nGoal: {goal}"
        accumulated_files: list[str] = []

        # Iterative Worker <-> QA self-healing loop
        for round_idx in range(1, max_feedback_rounds + 1):
            if on_event:
                stage_name = (
                    f"Phase 2: Feature Worker Implementation (Round {round_idx})"
                    if round_idx > 1
                    else "Phase 2: Feature Worker Implementation"
                )
                on_event("stage", stage_name)

            worker_assignment = TaskAssignment(
                task_id=f"feat-worker-r{round_idx}",
                role=AgentRole.WORKER,
                objective=current_objective,
                allowed_files=[],
                max_turns=8,
            )
            w_res = await self.spawn_worker(worker_assignment, depth=1, on_event=on_event)
            all_worker_results.append(w_res)
            for f in w_res.files_modified:
                if f not in accumulated_files:
                    accumulated_files.append(f)

            if on_event:
                stage_name = (
                    f"Phase 3: Independent QA Verification (Round {round_idx})"
                    if round_idx > 1
                    else "Phase 3: Independent QA Verification"
                )
                on_event("stage", stage_name)

            qa_assignment = TaskAssignment(
                task_id=f"qa-verifier-r{round_idx}",
                role=AgentRole.QA,
                objective=(
                    f"Verify the changes implemented by the Feature Worker.\n"
                    f"Modified files: {accumulated_files}\n"
                    f"Run relevant test suites via run_command (e.g. pytest). Output PASS or FAIL with diagnostics."
                ),
                max_turns=4,
            )
            qa_result = await self.spawn_worker(qa_assignment, depth=1, on_event=on_event)

            # Check if QA passed
            qa_summary_upper = (qa_result.summary or "").upper()
            is_pass = qa_result.status == "success" and "FAIL" not in qa_summary_upper

            if is_pass:
                if on_event:
                    on_event("qa_pass", f"Round {round_idx}: QA verified all tests passed successfully!")
                break

            if round_idx < max_feedback_rounds:
                if on_event:
                    on_event(
                        "qa_feedback",
                        f"Round {round_idx}: QA reported test failures. Routing diagnostics back to Feature Worker...",
                    )
                current_objective = (
                    f"Fix the test failures identified by QA Verifier in round {round_idx}.\n"
                    f"QA Diagnostics and Error Trace:\n{qa_result.summary}\n"
                    f"Goal: {goal}"
                )

        # Step 4: Outcome Synthesis & Reflection
        if on_event:
            on_event("stage", "Phase 4: Synthesis & Final Review")

        overall_status = "success" if (qa_result and qa_result.status == "success" and "FAIL" not in (qa_result.summary or "").upper()) else "failed"
        last_worker = all_worker_results[-1] if all_worker_results else None
        final_summary = (
            f"### Multi-Agent Swarm Outcome ({overall_status.upper()})\n\n"
            f"**Goal:** {goal}\n\n"
            f"**Architecture Plan:**\n{plan}\n\n"
            f"**Feature Worker Results:**\n- Status: {last_worker.status if last_worker else 'none'}\n- Modified Files: {accumulated_files}\n- Iterations: {len(all_worker_results)}\n- Summary: {last_worker.summary if last_worker else ''}\n\n"
            f"**QA Verification:**\n- Status: {qa_result.status if qa_result else 'unknown'}\n- Diagnostics: {qa_result.summary if qa_result else ''}\n"
        )

        # Record episodic reflection automatically in persistent memory
        try:
            from .tools import memory_store

            memory_store.record_episodic_reflection(
                goal=goal,
                summary=final_summary,
                files_modified=accumulated_files,
            )
        except Exception:
            pass

        return TeamWorkflowResult(
            goal=goal,
            plan=plan,
            worker_results=all_worker_results,
            qa_result=qa_result,
            status=overall_status,
            summary=final_summary,
        )


# Global active team instance for tool delegation
_active_team: MultiAgentTeam | None = None


def set_active_team(team: MultiAgentTeam | None) -> None:
    """Sets or clears the active MultiAgentTeam coordinator for tools."""
    global _active_team
    _active_team = team


def get_active_team() -> MultiAgentTeam | None:
    """Retrieves the active MultiAgentTeam coordinator."""
    return _active_team
