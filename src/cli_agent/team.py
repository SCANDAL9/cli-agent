"""
Hierarchical Multi-Agent Teams (Swarm) for cli-agent.
Implements Organic Driven Development (ODD) with role-specialized agent loops:
- Lead Architect: Requirements analysis, task graph decomposition, synthesis.
- Feature Worker: Bounded, isolated implementation with scoped edit surfaces.
- QA Verifier: Independent validation, test execution, regression detection.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .tools import GEMINI_TOOLS_DECLARATIONS
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

        self._active_workers.append(result)
        return result

    async def run_workflow(
        self,
        goal: str,
        on_event: Callable[[str, str], None] | None = None,
    ) -> TeamWorkflowResult:
        """
        Coordinates the complete Organic Driven Development (ODD) multi-agent workflow:
        1. Lead Architect formulates the architectural plan.
        2. Feature Worker implements the discrete task unit.
        3. Independent QA Verifier runs test suite and validates regressions.
        4. Architect synthesizes the final outcome.
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

        # Step 2: Feature Worker Implementation
        if on_event:
            on_event("stage", "Phase 2: Feature Worker Implementation")

        worker_assignment = TaskAssignment(
            task_id="feat-worker-1",
            role=AgentRole.WORKER,
            objective=f"Implement feature according to plan:\n{plan}\nGoal: {goal}",
            allowed_files=[],
            max_turns=8,
        )
        worker_result = await self.spawn_worker(worker_assignment, depth=1, on_event=on_event)

        # Step 3: Independent QA Verification
        if on_event:
            on_event("stage", "Phase 3: Independent QA Verification")

        qa_assignment = TaskAssignment(
            task_id="qa-verifier-1",
            role=AgentRole.QA,
            objective=(
                f"Verify the changes implemented by the Feature Worker.\n"
                f"Modified files: {worker_result.files_modified}\n"
                f"Run relevant test suites via run_command (e.g. pytest). Output PASS or FAIL with diagnostics."
            ),
            max_turns=4,
        )
        qa_result = await self.spawn_worker(qa_assignment, depth=1, on_event=on_event)

        # Step 4: Outcome Synthesis
        if on_event:
            on_event("stage", "Phase 4: Synthesis & Final Review")

        overall_status = "success" if qa_result.status == "success" else "failed"
        final_summary = (
            f"### Multi-Agent Swarm Outcome ({overall_status.upper()})\n\n"
            f"**Goal:** {goal}\n\n"
            f"**Architecture Plan:**\n{plan}\n\n"
            f"**Feature Worker Results:**\n- Status: {worker_result.status}\n- Modified Files: {worker_result.files_modified}\n- Summary: {worker_result.summary}\n\n"
            f"**QA Verification:**\n- Status: {qa_result.status}\n- Diagnostics: {qa_result.summary}\n"
        )

        return TeamWorkflowResult(
            goal=goal,
            plan=plan,
            worker_results=[worker_result],
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
