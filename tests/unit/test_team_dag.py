"""
Unit tests for Phase 2 of Wave 5: Concurrent DAG Task Planner & Scheduler (src/cli_agent/team.py).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cli_agent.events import EventBus, EventTopic, SwarmNodeStatePayload
from cli_agent.team import (
    AgentRole,
    DAGCycleError,
    DAGNodeStatus,
    DAGPlanner,
    DAGTaskScheduler,
    DAGValidationError,
    MultiAgentTeam,
    SwarmDAGNode,
    SwarmDAGPlan,
    TaskAssignment,
    WorkerResult,
)
from cli_agent.tools import write_file
from cli_agent.vertex import VertexAgentClient
from cli_agent.worktree import WorktreeManager


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock(spec=VertexAgentClient)
    client.chat = AsyncMock(return_value="Completed assigned task.")
    return client


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "dag_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "CI Bot"], cwd=repo_dir, check=True)

    (repo_dir / "service_a.py").write_text(
        "def compute_a() -> int:\n    return 1\n", encoding="utf-8"
    )
    (repo_dir / "service_b.py").write_text(
        "def compute_b() -> int:\n    return 2\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo_dir, check=True)
    return repo_dir


@pytest.mark.unit
class TestDAGPlanner:
    def test_validate_and_stratify_diamond_dag(self) -> None:
        node_a = SwarmDAGNode(
            node_id="feat-events",
            role=AgentRole.WORKER,
            objective="Implement EventBus",
            allowed_files=["src/cli_agent/events.py"],
        )
        node_b = SwarmDAGNode(
            node_id="feat-lsp",
            role=AgentRole.WORKER,
            objective="Implement LSP client",
            allowed_files=["src/cli_agent/lsp.py"],
        )
        node_c = SwarmDAGNode(
            node_id="feat-tui-wire",
            role=AgentRole.WORKER,
            objective="Wire EventBus and LSP to TUI",
            dependencies=["feat-events", "feat-lsp"],
            allowed_files=["src/cli_agent/tui.py"],
        )
        node_qa = SwarmDAGNode(
            node_id="qa-verify",
            role=AgentRole.QA,
            objective="Run verification suite",
            dependencies=["feat-tui-wire"],
        )

        # Also test passing list of nodes to SwarmDAGPlan
        plan = SwarmDAGPlan(
            plan_id="plan-diamond",
            goal="Build Wave 5 runtime",
            rationale="Diamond dependency topology",
            nodes=[node_a, node_b, node_c, node_qa],  # type: ignore[arg-type]
        )
        assert isinstance(plan.nodes, dict)

        tiers = DAGPlanner.validate_and_stratify(plan)
        assert len(tiers) == 3
        assert {n.node_id for n in tiers[0]} == {"feat-events", "feat-lsp"}
        assert [n.node_id for n in tiers[1]] == ["feat-tui-wire"]
        assert [n.node_id for n in tiers[2]] == ["qa-verify"]

    def test_unknown_dependency_raises_validation_error(self) -> None:
        node = SwarmDAGNode(
            node_id="worker-1",
            role=AgentRole.WORKER,
            objective="Task",
            dependencies=["non-existent-node"],
        )
        plan = SwarmDAGPlan(plan_id="p1", goal="g", nodes={"worker-1": node})

        with pytest.raises(DAGValidationError, match="Unknown dependency node_id"):
            DAGPlanner.validate_and_stratify(plan)

    def test_self_dependency_raises_cycle_error(self) -> None:
        node = SwarmDAGNode(
            node_id="self-loop",
            role=AgentRole.WORKER,
            objective="Task",
            dependencies=["self-loop"],
        )
        plan = SwarmDAGPlan(plan_id="p-self", goal="g", nodes={"self-loop": node})

        with pytest.raises(DAGCycleError) as exc_info:
            DAGPlanner.validate_and_stratify(plan)
        assert "self-loop" in exc_info.value.cycle_nodes

    def test_kahn_detects_multi_node_cycle(self) -> None:
        n1 = SwarmDAGNode(node_id="n1", role=AgentRole.WORKER, objective="1", dependencies=["n3"])
        n2 = SwarmDAGNode(node_id="n2", role=AgentRole.WORKER, objective="2", dependencies=["n1"])
        n3 = SwarmDAGNode(node_id="n3", role=AgentRole.WORKER, objective="3", dependencies=["n2"])
        root = SwarmDAGNode(node_id="root", role=AgentRole.WORKER, objective="0")
        plan = SwarmDAGPlan(
            plan_id="p-cycle",
            goal="cycle test",
            nodes={"root": root, "n1": n1, "n2": n2, "n3": n3},
        )

        with pytest.raises(DAGCycleError) as exc_info:
            DAGPlanner.validate_and_stratify(plan)
        assert set(exc_info.value.cycle_nodes) == {"n1", "n2", "n3"}

    def test_auto_serialize_shared_files_in_same_tier(self) -> None:
        u = SwarmDAGNode(
            node_id="u",
            role=AgentRole.WORKER,
            objective="Edit shared",
            allowed_files=["./src/shared.py"],
        )
        v = SwarmDAGNode(
            node_id="v",
            role=AgentRole.WORKER,
            objective="Also edit shared",
            allowed_files=["src/shared.py"],
        )
        plan = SwarmDAGPlan(plan_id="p-shared", goal="shared", nodes={"u": u, "v": v})

        tiers = DAGPlanner.validate_and_stratify(plan, auto_serialize_shared_files=True)
        assert len(tiers) == 2
        assert [n.node_id for n in tiers[0]] == ["u"]
        assert [n.node_id for n in tiers[1]] == ["v"]
        assert "u" in v.dependencies


@pytest.mark.unit
@pytest.mark.asyncio
class TestDAGTaskScheduler:
    async def test_bounded_semaphore_concurrency_and_event_telemetry(
        self, mock_client: AsyncMock
    ) -> None:
        team = MultiAgentTeam(client=mock_client)
        bus = EventBus()
        sub = bus.subscribe(EventTopic.SWARM_NODE_STATE)

        active_count = 0
        peak_concurrency = 0

        async def delayed_spawn(assignment: TaskAssignment, depth: int = 0) -> WorkerResult:
            nonlocal active_count, peak_concurrency
            active_count += 1
            peak_concurrency = max(peak_concurrency, active_count)
            await asyncio.sleep(0.03)
            active_count -= 1
            return WorkerResult(
                task_id=assignment.task_id,
                role=assignment.role,
                status="success",
                summary=f"Done {assignment.task_id}",
            )

        team.spawn_worker = delayed_spawn  # type: ignore[method-assign]

        nodes = {
            f"w{i}": SwarmDAGNode(
                node_id=f"w{i}",
                role=AgentRole.WORKER,
                objective=f"Parallel task {i}",
            )
            for i in range(4)
        }
        nodes["final"] = SwarmDAGNode(
            node_id="final",
            role=AgentRole.WORKER,
            objective="Join task",
            dependencies=["w0", "w1", "w2", "w3"],
        )
        plan = SwarmDAGPlan(
            plan_id="plan-sem",
            goal="Test concurrency limit",
            nodes=nodes,
            max_concurrency=2,
        )

        scheduler = DAGTaskScheduler(team=team, event_bus=bus, max_concurrency=2)
        results = await scheduler.execute_dag(plan)

        assert peak_concurrency == 2
        assert len(results) == 5
        assert all(r.status == "success" for r in results.values())
        assert all(n.status == DAGNodeStatus.COMPLETED for n in plan.nodes.values())

        envelopes = sub.drain()
        payloads = [
            env.payload for env in envelopes if isinstance(env.payload, SwarmNodeStatePayload)
        ]
        final_events = [p for p in payloads if p.node_id == "final"]
        assert [p.status for p in final_events] == ["READY", "RUNNING", "COMPLETED"]
        assert final_events[-1].dependencies == ("w0", "w1", "w2", "w3")

    async def test_upstream_failure_skips_downstream_dependents(
        self, mock_client: AsyncMock
    ) -> None:
        team = MultiAgentTeam(client=mock_client)
        bus = EventBus()
        sub = bus.subscribe(EventTopic.SWARM_NODE_STATE)

        async def selective_spawn(assignment: TaskAssignment, depth: int = 0) -> WorkerResult:
            if assignment.task_id == "failing-node":
                return WorkerResult(
                    task_id=assignment.task_id,
                    role=assignment.role,
                    status="failed",
                    summary="",
                    error="Unit test simulated error",
                )
            return WorkerResult(
                task_id=assignment.task_id,
                role=assignment.role,
                status="success",
                summary="OK",
            )

        team.spawn_worker = selective_spawn  # type: ignore[method-assign]

        n_fail = SwarmDAGNode(node_id="failing-node", role=AgentRole.WORKER, objective="Will fail")
        n_dep = SwarmDAGNode(
            node_id="dependent-node",
            role=AgentRole.WORKER,
            objective="Depends on failing-node",
            dependencies=["failing-node"],
        )
        n_ok = SwarmDAGNode(
            node_id="independent-node", role=AgentRole.WORKER, objective="Runs fine"
        )
        plan = SwarmDAGPlan(
            plan_id="plan-skip",
            goal="Skip downstream on failure",
            nodes={
                "failing-node": n_fail,
                "dependent-node": n_dep,
                "independent-node": n_ok,
            },
        )

        scheduler = DAGTaskScheduler(team=team, event_bus=bus)
        results = await scheduler.execute_dag(plan)

        assert n_fail.status == DAGNodeStatus.FAILED
        assert n_dep.status == DAGNodeStatus.SKIPPED
        assert results["dependent-node"].status == "skipped"
        assert n_ok.status == DAGNodeStatus.COMPLETED
        assert results["independent-node"].status == "success"

        events = [e.payload for e in sub.drain() if isinstance(e.payload, SwarmNodeStatePayload)]
        dep_statuses = [p.status for p in events if p.node_id == "dependent-node"]
        assert dep_statuses == ["SKIPPED"]

    async def test_node_timeout_marks_failed_and_skips_dependents(
        self, mock_client: AsyncMock
    ) -> None:
        team = MultiAgentTeam(client=mock_client)

        async def slow_spawn(assignment: TaskAssignment, depth: int = 0) -> WorkerResult:
            await asyncio.sleep(0.5)
            return WorkerResult(
                task_id=assignment.task_id,
                role=assignment.role,
                status="success",
                summary="Too late",
            )

        team.spawn_worker = slow_spawn  # type: ignore[method-assign]

        n_slow = SwarmDAGNode(
            node_id="slow-node",
            role=AgentRole.WORKER,
            objective="Times out",
            timeout_seconds=0.02,
        )
        n_after = SwarmDAGNode(
            node_id="after-slow",
            role=AgentRole.WORKER,
            objective="Skipped after timeout",
            dependencies=["slow-node"],
        )
        plan = SwarmDAGPlan(
            plan_id="plan-timeout",
            goal="Timeout test",
            nodes={"slow-node": n_slow, "after-slow": n_after},
        )

        scheduler = DAGTaskScheduler(team=team)
        results = await scheduler.execute_dag(plan)

        assert n_slow.status == DAGNodeStatus.FAILED
        assert "exceeded timeout" in (results["slow-node"].error or "")
        assert n_after.status == DAGNodeStatus.SKIPPED


@pytest.mark.unit
@pytest.mark.asyncio
class TestMultiAgentTeamExecuteDAGWorkflow:
    async def test_execute_dag_workflow_with_worktree_merge_and_atomic_apply(
        self, temp_git_repo: Path, tmp_path: Path, mock_client: AsyncMock
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_swarm")
        bus = EventBus()
        team = MultiAgentTeam(
            client=mock_client,
            worktree_manager=wm,
            event_bus=bus,
            repo_root=temp_git_repo,
        )

        async def custom_spawn(assignment: TaskAssignment, depth: int = 0) -> WorkerResult:
            if assignment.task_id == "worker-a":
                res = write_file(
                    "service_a.py",
                    "def compute_a() -> int:\n    return 100\n",
                    overwrite=True,
                )
                assert res.get("status") == "success"
                return WorkerResult(
                    task_id="worker-a",
                    role=AgentRole.WORKER,
                    status="success",
                    summary="Updated service_a.py",
                    files_modified=["service_a.py"],
                )
            if assignment.task_id == "worker-b":
                res = write_file(
                    "service_b.py",
                    "def compute_b() -> int:\n    return 200\n",
                    overwrite=True,
                )
                assert res.get("status") == "success"
                return WorkerResult(
                    task_id="worker-b",
                    role=AgentRole.WORKER,
                    status="success",
                    summary="Updated service_b.py",
                    files_modified=["service_b.py"],
                )
            return WorkerResult(
                task_id=assignment.task_id,
                role=AgentRole.QA,
                status="success",
                summary="PASS: All integration checks succeeded.",
            )

        team.spawn_worker = custom_spawn  # type: ignore[method-assign]

        plan = SwarmDAGPlan(
            plan_id="swarm01",
            goal="Update service A and service B concurrently",
            rationale="Disjoint service updates",
            nodes={
                "worker-a": SwarmDAGNode(
                    node_id="worker-a",
                    role=AgentRole.WORKER,
                    objective="Update service_a.py",
                    allowed_files=["service_a.py"],
                ),
                "worker-b": SwarmDAGNode(
                    node_id="worker-b",
                    role=AgentRole.WORKER,
                    objective="Update service_b.py",
                    allowed_files=["service_b.py"],
                ),
                "qa-verify": SwarmDAGNode(
                    node_id="qa-verify",
                    role=AgentRole.QA,
                    objective="Run tests on integrated changes",
                    dependencies=["worker-a", "worker-b"],
                ),
            },
        )

        outcome = await team.execute_dag_workflow(
            goal=plan.goal,
            dag_plan=plan,
            max_concurrency=2,
            auto_apply=True,
        )

        assert outcome.status == "success"
        assert outcome.merge_result is not None
        assert outcome.merge_result.status == "clean"
        assert sorted(outcome.merge_result.merged_files) == ["service_a.py", "service_b.py"]
        assert "return 100" in (temp_git_repo / "service_a.py").read_text(encoding="utf-8")
        assert "return 200" in (temp_git_repo / "service_b.py").read_text(encoding="utf-8")

    async def test_execute_dag_workflow_rolls_back_when_qa_fails(
        self, temp_git_repo: Path, tmp_path: Path, mock_client: AsyncMock
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_swarm_rollback")
        team = MultiAgentTeam(
            client=mock_client,
            worktree_manager=wm,
            repo_root=temp_git_repo,
        )

        async def custom_spawn(assignment: TaskAssignment, depth: int = 0) -> WorkerResult:
            if assignment.role == AgentRole.WORKER:
                write_file(
                    "service_a.py",
                    "def compute_a() -> int:\n    return 999\n",
                    overwrite=True,
                )
                return WorkerResult(
                    task_id=assignment.task_id,
                    role=AgentRole.WORKER,
                    status="success",
                    summary="Modified service_a.py",
                )
            return WorkerResult(
                task_id=assignment.task_id,
                role=AgentRole.QA,
                status="success",
                summary="FAIL: Regression detected in test_service_a",
            )

        team.spawn_worker = custom_spawn  # type: ignore[method-assign]

        plan = SwarmDAGPlan(
            plan_id="swarm-rb",
            goal="Attempt broken update",
            nodes={
                "w1": SwarmDAGNode(
                    node_id="w1",
                    role=AgentRole.WORKER,
                    objective="Mutate service_a.py",
                    allowed_files=["service_a.py"],
                ),
            },
        )

        outcome = await team.execute_dag_workflow(goal=plan.goal, dag_plan=plan, auto_apply=True)

        assert outcome.status == "failed"
        # Primary workspace must remain 100% untouched
        assert "return 1\n" in (temp_git_repo / "service_a.py").read_text(encoding="utf-8")

    async def test_architect_synthesizes_json_dag_plan(self, mock_client: AsyncMock) -> None:
        architect_json = json.dumps(
            {
                "rationale": "Split into 2 parallel workers and 1 QA node",
                "nodes": [
                    {
                        "node_id": "w-1",
                        "role": "worker",
                        "objective": "Build part 1",
                        "allowed_files": ["a.py"],
                        "dependencies": [],
                    },
                    {
                        "node_id": "qa-1",
                        "role": "qa",
                        "objective": "Verify part 1",
                        "dependencies": ["w-1"],
                    },
                ],
            }
        )
        mock_client.chat = AsyncMock(
            side_effect=[
                f"```json\n{architect_json}\n```",
                "Implemented part 1",
                "PASS: Verified part 1",
            ]
        )
        team = MultiAgentTeam(client=mock_client)
        outcome = await team.execute_dag_workflow(goal="Build feature X", auto_apply=False)

        assert outcome.status == "success"
        assert outcome.dag_plan is not None
        assert "w-1" in outcome.dag_plan.nodes
        assert "qa-1" in outcome.dag_plan.nodes
