"""
Hierarchical Multi-Agent Teams (Swarm) for cli-agent.
Implements Organic Driven Development (ODD) with role-specialized agent loops:
- Lead Architect: Requirements analysis, task graph decomposition, synthesis.
- Feature Worker: Bounded, isolated implementation with scoped edit surfaces.
- QA Verifier: Independent validation, test execution, regression detection.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .events import EventBus, EventTopic, SwarmNodeStatePayload
from .tools import GEMINI_TOOLS_DECLARATIONS, worker_execution_scope
from .vertex import VertexAgentClient
from .worktree import (
    ASTThreeWayMerger,
    MergeResult,
    WorktreeManager,
    WorktreeSession,
    _is_git_repo,
)

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
    merge_result: MergeResult | None = None
    dag_plan: SwarmDAGPlan | None = None


class DAGNodeStatus(StrEnum):
    """Lifecycle states of a single node within the Swarm execution DAG."""

    PENDING = "PENDING"  # Waiting on upstream dependency nodes
    READY = "READY"  # All dependencies COMPLETED; eligible for scheduling
    RUNNING = "RUNNING"  # Active inside dedicated WorktreeSession
    COMPLETED = "COMPLETED"  # Worker finished and worktree diff captured
    FAILED = "FAILED"  # Execution error, timeout, or merge conflict
    SKIPPED = "SKIPPED"  # Upstream dependency failed; node bypassed
    REJECTED = "REJECTED"  # Blocked by recursion guard or security policy


@dataclass(slots=True)
class SwarmDAGNode:
    """Represents an atomic unit of work in the Lead Architect's execution DAG."""

    node_id: str
    role: AgentRole
    objective: str
    allowed_files: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    max_turns: int = 10
    timeout_seconds: float = 180.0
    status: DAGNodeStatus = DAGNodeStatus.PENDING
    worktree_branch: str | None = None
    worktree_dir: Path | None = None
    result: WorkerResult | None = None


@dataclass(slots=True)
class SwarmDAGPlan:
    """Validated Directed Acyclic Graph of agent tasks synthesized by the Lead Architect."""

    plan_id: str
    goal: str
    rationale: str = ""
    nodes: dict[str, SwarmDAGNode] = field(default_factory=dict)
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.nodes, list):
            self.nodes = {n.node_id: n for n in self.nodes}


class DAGValidationError(ValueError):
    """Raised when a SwarmDAGPlan violates structural invariants."""


class DAGCycleError(DAGValidationError):
    """Raised when Kahn's algorithm detects a dependency cycle in a SwarmDAGPlan."""

    def __init__(self, message: str, cycle_nodes: list[str] | None = None) -> None:
        super().__init__(message)
        self.cycle_nodes: list[str] = cycle_nodes or []


class DAGPlanner:
    """Validates and topologically stratifies a SwarmDAGPlan using Kahn's In-Degree Algorithm."""

    @staticmethod
    def validate_and_stratify(
        plan: SwarmDAGPlan,
        auto_serialize_shared_files: bool = False,
    ) -> list[list[SwarmDAGNode]]:
        """
        Enforces referential integrity and acyclicity via Kahn's algorithm,
        returning topological execution tiers [T_0, T_1, ..., T_k].
        """
        if not plan.nodes:
            return []

        # 1. Referential Integrity Check
        for node_id, node in plan.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in plan.nodes:
                    raise DAGValidationError(
                        f"Unknown dependency node_id '{dep_id}' referenced by node '{node_id}'."
                    )
                if dep_id == node_id:
                    raise DAGCycleError(
                        f"Self-dependency cycle detected on node '{node_id}'.",
                        cycle_nodes=[node_id],
                    )

        # 2. Kahn's In-Degree Topological Stratification
        in_degree: dict[str, int] = {}
        dependents: dict[str, list[str]] = {nid: [] for nid in plan.nodes}

        for node_id, node in plan.nodes.items():
            unique_deps = list(dict.fromkeys(node.dependencies))
            in_degree[node_id] = len(unique_deps)
            for dep_id in unique_deps:
                dependents[dep_id].append(node_id)

        current_tier_ids = [nid for nid, deg in in_degree.items() if deg == 0]
        tiers: list[list[SwarmDAGNode]] = []
        visited_count = 0

        while current_tier_ids:
            tier_nodes = [plan.nodes[nid] for nid in current_tier_ids]
            tiers.append(tier_nodes)
            visited_count += len(current_tier_ids)

            next_tier_ids: list[str] = []
            for nid in current_tier_ids:
                for child_id in dependents[nid]:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        next_tier_ids.append(child_id)
            current_tier_ids = next_tier_ids

        if visited_count < len(plan.nodes):
            cycle_nodes = [nid for nid, deg in in_degree.items() if deg > 0]
            raise DAGCycleError(
                f"Dependency cycle detected in SwarmDAGPlan '{plan.plan_id}': {cycle_nodes}",
                cycle_nodes=cycle_nodes,
            )

        # 3. Optional auto-serialization of sibling nodes with overlapping allowed_files
        if auto_serialize_shared_files:
            edge_added = False
            for tier in tiers:
                for i in range(len(tier)):
                    u = tier[i]
                    u_files = {Path(f).as_posix().lstrip("./") for f in u.allowed_files}
                    if not u_files:
                        continue
                    for j in range(i + 1, len(tier)):
                        v = tier[j]
                        v_files = {Path(f).as_posix().lstrip("./") for f in v.allowed_files}
                        if u_files & v_files and u.node_id not in v.dependencies:
                            v.dependencies.append(u.node_id)
                            edge_added = True
            if edge_added:
                return DAGPlanner.validate_and_stratify(plan, auto_serialize_shared_files=False)

        return tiers


class DAGTaskScheduler:
    """Executes a SwarmDAGPlan with structured concurrency and bounded parallelism."""

    def __init__(
        self,
        team: MultiAgentTeam,
        worktree_manager: WorktreeManager | None = None,
        event_bus: EventBus | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.team = team
        self.worktree_manager = worktree_manager or team.worktree_manager
        self.event_bus = event_bus if event_bus is not None else team.event_bus
        self.max_concurrency = max(1, max_concurrency)
        self.worker_sessions: dict[str, WorktreeSession] = {}

    def validate_dag(self, plan: SwarmDAGPlan) -> list[list[SwarmDAGNode]]:
        """Validates the DAG plan and returns its topological tiers."""
        return DAGPlanner.validate_and_stratify(plan)

    async def _emit_node_state(
        self,
        plan_id: str,
        node: SwarmDAGNode,
        active_tool: str | None = None,
        error: str | None = None,
    ) -> None:
        """Publishes a SwarmNodeStatePayload event to the EventBus if configured."""
        if self.event_bus is None:
            return
        payload = SwarmNodeStatePayload(
            plan_id=plan_id,
            node_id=node.node_id,
            role=node.role.value,
            status=node.status.value,
            active_tool=active_tool,
            worktree_branch=node.worktree_branch,
            dependencies=tuple(node.dependencies),
            error=error if error is not None else (node.result.error if node.result else None),
        )
        await self.event_bus.publish(EventTopic.SWARM_NODE_STATE, payload)

    async def _execute_isolated_worker(
        self,
        plan: SwarmDAGPlan,
        node: SwarmDAGNode,
        base_branch_ref: str,
    ) -> WorkerResult:
        """Provisions a per-worker worktree session (when enabled) and runs the node inside ContextVars."""
        session: WorktreeSession | None = None
        if (
            self.worktree_manager is not None
            and self.team.repo_root is not None
            and _is_git_repo(self.team.repo_root)
            and node.role == AgentRole.WORKER
        ):
            upstream = [
                self.worker_sessions[dep_id]
                for dep_id in node.dependencies
                if dep_id in self.worker_sessions
            ]
            session = self.worktree_manager.create_worker_session(
                plan_id=plan.plan_id,
                node_id=node.node_id,
                base_ref=base_branch_ref,
                repo_path=self.team.repo_root,
                upstream_sessions=upstream,
            )
            node.worktree_branch = session.branch_name
            node.worktree_dir = session.worktree_dir
            self.worker_sessions[node.node_id] = session
        elif node.worktree_branch is None:
            node.worktree_branch = f"swarm/{plan.plan_id}/{node.node_id}"

        effective_allowed: list[str] | None
        if node.role == AgentRole.QA:
            effective_allowed = []
        elif node.allowed_files:
            effective_allowed = list(node.allowed_files)
        else:
            effective_allowed = None

        assignment = TaskAssignment(
            task_id=node.node_id,
            role=node.role,
            objective=node.objective,
            allowed_files=list(node.allowed_files),
            context_files=list(node.context_files),
            max_turns=node.max_turns,
        )

        with worker_execution_scope(
            sandbox_dir=node.worktree_dir,
            allowed_files=effective_allowed,
            worker_id=node.node_id,
        ):
            worker_res = await self.team.spawn_worker(assignment, depth=1)

        if session is not None and session.worktree_dir.exists():
            wt_files = session.get_summary().files
            for f in wt_files:
                if f not in worker_res.files_modified:
                    worker_res.files_modified.append(f)

        return worker_res

    async def execute_dag(
        self,
        plan: SwarmDAGPlan,
        base_branch_ref: str = "HEAD",
    ) -> dict[str, WorkerResult]:
        """Schedules all DAG nodes concurrently respecting dependency edges and semaphore limits."""
        self.validate_dag(plan)
        if not plan.nodes:
            return {}

        effective_limit = max(1, min(self.max_concurrency, max(1, plan.max_concurrency)))
        semaphore = asyncio.Semaphore(effective_limit)
        completion_events: dict[str, asyncio.Event] = {
            node_id: asyncio.Event() for node_id in plan.nodes
        }
        results: dict[str, WorkerResult] = {}

        async def _run_node(node: SwarmDAGNode) -> None:
            try:
                # 1. Await all upstream dependencies
                for dep_id in node.dependencies:
                    await completion_events[dep_id].wait()
                    dep_node = plan.nodes[dep_id]
                    if dep_node.status != DAGNodeStatus.COMPLETED:
                        node.status = DAGNodeStatus.SKIPPED
                        skip_res = WorkerResult(
                            task_id=node.node_id,
                            role=node.role,
                            status="skipped",
                            summary=(
                                f"Skipped because upstream dependency '{dep_id}' "
                                f"{dep_node.status.value}."
                            ),
                        )
                        node.result = skip_res
                        results[node.node_id] = skip_res
                        await self._emit_node_state(plan.plan_id, node)
                        return

                node.status = DAGNodeStatus.READY
                await self._emit_node_state(plan.plan_id, node)

                # 2. Acquire bounded concurrency slot
                async with semaphore:
                    node.status = DAGNodeStatus.RUNNING
                    await self._emit_node_state(plan.plan_id, node)

                    # 3. Execute inside isolated per-worker WorktreeSession & ContextVars
                    worker_res = await asyncio.wait_for(
                        self._execute_isolated_worker(plan, node, base_branch_ref),
                        timeout=node.timeout_seconds,
                    )
                    node.result = worker_res
                    results[node.node_id] = worker_res
                    if worker_res.status == "rejected":
                        node.status = DAGNodeStatus.REJECTED
                    elif worker_res.status == "success":
                        node.status = DAGNodeStatus.COMPLETED
                    else:
                        node.status = DAGNodeStatus.FAILED
                    await self._emit_node_state(plan.plan_id, node, error=worker_res.error)

            except TimeoutError:
                node.status = DAGNodeStatus.FAILED
                timeout_res = WorkerResult(
                    task_id=node.node_id,
                    role=node.role,
                    status="failed",
                    summary="",
                    error=f"Node '{node.node_id}' exceeded timeout of {node.timeout_seconds}s.",
                )
                node.result = timeout_res
                results[node.node_id] = timeout_res
                await self._emit_node_state(plan.plan_id, node, error=timeout_res.error)
            except Exception as exc:
                node.status = DAGNodeStatus.FAILED
                err_res = WorkerResult(
                    task_id=node.node_id,
                    role=node.role,
                    status="failed",
                    summary="",
                    error=str(exc),
                )
                node.result = err_res
                results[node.node_id] = err_res
                await self._emit_node_state(plan.plan_id, node, error=err_res.error)
            finally:
                completion_events[node.node_id].set()

        async with asyncio.TaskGroup() as tg:
            for dag_node in plan.nodes.values():
                tg.create_task(_run_node(dag_node), name=f"swarm-node-{dag_node.node_id}")

        return results


class MultiAgentTeam:
    """Coordinates hierarchical multi-agent teams with strict role isolation and recursion guards."""

    def __init__(
        self,
        client: VertexAgentClient,
        max_depth: int = MAX_DELEGATION_DEPTH,
        worktree_manager: WorktreeManager | None = None,
        event_bus: EventBus | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.client = client
        self.max_depth = max_depth
        self.worktree_manager = worktree_manager
        self.event_bus = event_bus
        self.repo_root = repo_root.resolve() if repo_root is not None else None
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

    async def _synthesize_dag_plan(
        self,
        goal: str,
        max_concurrency: int = 4,
    ) -> SwarmDAGPlan:
        """Uses the Lead Architect to decompose a goal into a SwarmDAGPlan."""
        plan_id = uuid.uuid4().hex[:8]
        prompt = (
            "You are the Lead Architect. Decompose the user goal into a JSON DAG plan with keys "
            "'rationale' and 'nodes' (list of objects with node_id, role, objective, "
            "allowed_files, context_files, dependencies).\n"
            f"Goal: {goal}"
        )
        raw_response = await self.client.chat(
            messages=[{"role": "user", "parts": [{"text": prompt}]}],
            system_instruction=ROLE_SYSTEM_PROMPTS[AgentRole.ARCHITECT],
            max_turns=3,
        )
        try:
            text = raw_response.strip()
            if "```" in text:
                for block in text.split("```"):
                    cleaned = block.removeprefix("json").strip()
                    if cleaned.startswith("{"):
                        text = cleaned
                        break
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "nodes" in parsed:
                nodes: dict[str, SwarmDAGNode] = {}
                for item in parsed["nodes"]:
                    nid = str(item["node_id"])
                    role_str = str(item.get("role", "worker")).lower()
                    role = (
                        AgentRole(role_str)
                        if role_str in AgentRole._value2member_map_
                        else AgentRole.WORKER
                    )
                    nodes[nid] = SwarmDAGNode(
                        node_id=nid,
                        role=role,
                        objective=str(item.get("objective", goal)),
                        allowed_files=list(item.get("allowed_files", [])),
                        context_files=list(item.get("context_files", [])),
                        dependencies=list(item.get("dependencies", [])),
                    )
                if nodes:
                    return SwarmDAGPlan(
                        plan_id=plan_id,
                        goal=goal,
                        rationale=str(parsed.get("rationale", raw_response)),
                        nodes=nodes,
                        max_concurrency=max_concurrency,
                    )
        except Exception:
            pass

        default_worker = SwarmDAGNode(
            node_id="feat-worker-1",
            role=AgentRole.WORKER,
            objective=f"Implement feature according to plan:\n{raw_response}\nGoal: {goal}",
        )
        default_qa = SwarmDAGNode(
            node_id="qa-verify",
            role=AgentRole.QA,
            objective=f"Verify changes for goal: {goal}",
            dependencies=["feat-worker-1"],
        )
        return SwarmDAGPlan(
            plan_id=plan_id,
            goal=goal,
            rationale=raw_response,
            nodes={
                default_worker.node_id: default_worker,
                default_qa.node_id: default_qa,
            },
            max_concurrency=max_concurrency,
        )

    async def execute_dag_workflow(
        self,
        goal: str,
        dag_plan: SwarmDAGPlan | None = None,
        max_concurrency: int = 4,
        auto_apply: bool = True,
    ) -> TeamWorkflowResult:
        """
        Coordinates concurrent DAG-based multi-agent swarm execution:
        1. Validates and topologically stratifies the SwarmDAGPlan (or synthesizes one via Architect).
        2. Executes WORKER nodes concurrently in isolated per-worker Git worktrees.
        3. Reconciles worker worktrees into an integration worktree via ASTThreeWayMerger.
        4. Runs QA verification inside the integration worktree.
        5. Atomically promotes verified changes to main workspace (or rolls back untouched on failure).
        """
        plan = dag_plan or await self._synthesize_dag_plan(goal, max_concurrency=max_concurrency)
        DAGPlanner.validate_and_stratify(plan)

        # Partition into non-QA worker nodes and trailing QA nodes
        worker_nodes: dict[str, SwarmDAGNode] = {}
        qa_nodes: list[SwarmDAGNode] = []
        for nid, node in plan.nodes.items():
            if node.role == AgentRole.QA:
                qa_nodes.append(node)
            else:
                worker_nodes[nid] = node

        worker_subplan = SwarmDAGPlan(
            plan_id=plan.plan_id,
            goal=plan.goal,
            rationale=plan.rationale,
            nodes=worker_nodes,
            max_concurrency=min(max_concurrency, plan.max_concurrency),
        )

        scheduler = DAGTaskScheduler(
            team=self,
            worktree_manager=self.worktree_manager,
            event_bus=self.event_bus,
            max_concurrency=max_concurrency,
        )

        use_git_worktrees = (
            self.worktree_manager is not None
            and self.repo_root is not None
            and _is_git_repo(self.repo_root)
        )
        base_session: WorktreeSession | None = None
        integration_session: WorktreeSession | None = None
        merge_result: MergeResult | None = None
        qa_result: WorkerResult | None = None

        try:
            if use_git_worktrees and self.worktree_manager is not None:
                base_session = self.worktree_manager.create_worker_session(
                    plan_id=plan.plan_id,
                    node_id="base",
                    repo_path=self.repo_root,
                )
                integration_session = self.worktree_manager.create_worker_session(
                    plan_id=plan.plan_id,
                    node_id="integration",
                    repo_path=self.repo_root,
                )

            # Stage 1: Execute concurrent WORKER DAG
            worker_results_map = await scheduler.execute_dag(worker_subplan)
            worker_results = list(worker_results_map.values())

            all_workers_succeeded = all(
                node.status == DAGNodeStatus.COMPLETED for node in worker_nodes.values()
            )

            if not all_workers_succeeded:
                for qa_node in qa_nodes:
                    qa_node.status = DAGNodeStatus.SKIPPED
                    qa_node.result = WorkerResult(
                        task_id=qa_node.node_id,
                        role=AgentRole.QA,
                        status="skipped",
                        summary="Skipped because one or more upstream worker nodes failed.",
                    )
                    await scheduler._emit_node_state(plan.plan_id, qa_node)
                return TeamWorkflowResult(
                    goal=goal,
                    plan=plan.rationale,
                    worker_results=worker_results,
                    qa_result=qa_nodes[0].result if qa_nodes else None,
                    status="failed",
                    summary="Swarm DAG workflow failed during worker execution.",
                    merge_result=None,
                    dag_plan=plan,
                )

            # Stage 2: AST-Aware Three-Way Merge into Integration Worktree
            if (
                use_git_worktrees
                and base_session is not None
                and integration_session is not None
                and self.repo_root is not None
            ):
                merger = ASTThreeWayMerger(self.repo_root)
                merge_result = merger.merge_worker_sessions(
                    base_session=base_session,
                    worker_sessions=scheduler.worker_sessions,
                    integration_session=integration_session,
                )
                if merge_result.status in ("conflict", "error"):
                    for qa_node in qa_nodes:
                        qa_node.status = DAGNodeStatus.SKIPPED
                        qa_node.result = WorkerResult(
                            task_id=qa_node.node_id,
                            role=AgentRole.QA,
                            status="skipped",
                            summary=f"Skipped due to merge {merge_result.status}: {merge_result.message}",
                        )
                        await scheduler._emit_node_state(plan.plan_id, qa_node)
                    return TeamWorkflowResult(
                        goal=goal,
                        plan=plan.rationale,
                        worker_results=worker_results,
                        qa_result=qa_nodes[0].result if qa_nodes else None,
                        status="failed",
                        summary=f"Swarm DAG merge failed ({merge_result.status}): {merge_result.message}",
                        merge_result=merge_result,
                        dag_plan=plan,
                    )

            # Stage 3: Execute QA Verifier inside Integration Worktree
            active_qa_nodes = qa_nodes or [
                SwarmDAGNode(
                    node_id="qa-verify",
                    role=AgentRole.QA,
                    objective=f"Verify integrated changes for goal: {goal}",
                    dependencies=list(worker_nodes.keys()),
                    max_turns=4,
                )
            ]

            for qa_node in active_qa_nodes:
                if integration_session is not None:
                    qa_node.worktree_branch = integration_session.branch_name
                    qa_node.worktree_dir = integration_session.worktree_dir
                else:
                    qa_node.worktree_branch = f"swarm/{plan.plan_id}/integration"

                qa_node.status = DAGNodeStatus.READY
                await scheduler._emit_node_state(plan.plan_id, qa_node)
                qa_node.status = DAGNodeStatus.RUNNING
                await scheduler._emit_node_state(plan.plan_id, qa_node)

                qa_assignment = TaskAssignment(
                    task_id=qa_node.node_id,
                    role=AgentRole.QA,
                    objective=qa_node.objective,
                    allowed_files=[],
                    context_files=list(qa_node.context_files),
                    max_turns=qa_node.max_turns,
                )
                with worker_execution_scope(
                    sandbox_dir=qa_node.worktree_dir,
                    allowed_files=[],
                    worker_id=qa_node.node_id,
                ):
                    qa_res = await self.spawn_worker(qa_assignment, depth=1)

                summary_upper = qa_res.summary.strip().upper()
                if qa_res.status != "success" or summary_upper.startswith("FAIL"):
                    qa_res.status = "failed"
                    qa_node.status = DAGNodeStatus.FAILED
                else:
                    qa_node.status = DAGNodeStatus.COMPLETED

                qa_node.result = qa_res
                qa_result = qa_res
                await scheduler._emit_node_state(plan.plan_id, qa_node, error=qa_res.error)
                if qa_node.status == DAGNodeStatus.FAILED:
                    break

            if qa_result is not None and qa_result.status != "success":
                return TeamWorkflowResult(
                    goal=goal,
                    plan=plan.rationale,
                    worker_results=worker_results,
                    qa_result=qa_result,
                    status="failed",
                    summary=f"QA verification failed: {qa_result.summary}",
                    merge_result=merge_result,
                    dag_plan=plan,
                )

            # Stage 4: Atomic Promotion to Main Workspace
            if auto_apply and integration_session is not None:
                apply_res = integration_session.apply_to_main()
                if apply_res.get("status") in ("conflict", "error"):
                    return TeamWorkflowResult(
                        goal=goal,
                        plan=plan.rationale,
                        worker_results=worker_results,
                        qa_result=qa_result,
                        status="failed",
                        summary=f"Atomic promotion to main failed: {apply_res.get('message', '')}",
                        merge_result=merge_result,
                        dag_plan=plan,
                    )

            return TeamWorkflowResult(
                goal=goal,
                plan=plan.rationale,
                worker_results=worker_results,
                qa_result=qa_result,
                status="success",
                summary=f"Successfully executed {len(worker_results)} DAG worker(s) and verified integration.",
                merge_result=merge_result,
                dag_plan=plan,
            )

        finally:
            for w_sess in list(scheduler.worker_sessions.values()):
                w_sess.cleanup()
            if integration_session is not None:
                integration_session.cleanup()
            if base_session is not None:
                base_session.cleanup()


# Global active team instance for tool delegation
_active_team: MultiAgentTeam | None = None


def set_active_team(team: MultiAgentTeam | None) -> None:
    """Sets or clears the active MultiAgentTeam coordinator for tools."""
    global _active_team
    _active_team = team


def get_active_team() -> MultiAgentTeam | None:
    """Retrieves the active MultiAgentTeam coordinator."""
    return _active_team
