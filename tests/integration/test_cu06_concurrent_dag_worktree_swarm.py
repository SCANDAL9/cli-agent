"""
Integration test for CU-06: Concurrent DAG Swarm with Isolated Per-Worker Git Worktrees
and Hard allowed_files ACL Enforcement (Wave 5 Milestone W5.1 & W5.2).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cli_agent.team import (
    AgentRole,
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
def repo_cu06(tmp_path: Path) -> Path:
    repo = tmp_path / "repo_cu06"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "swarm@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Swarm Bot"], cwd=repo, check=True)

    (repo / "module_a.py").write_text("def func_a() -> int:\n    return 1\n", encoding="utf-8")
    (repo / "module_b.py").write_text("def func_b() -> int:\n    return 2\n", encoding="utf-8")
    (repo / "protected_config.py").write_text("SECRET_KEY = 'super_secret'\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo, check=True)
    return repo


@pytest.mark.integration
@pytest.mark.asyncio
class TestCU06ConcurrentDAGWorktreeSwarm:
    async def test_concurrent_dag_execution_with_acl_and_worktree_merge(
        self, repo_cu06: Path, tmp_path: Path
    ) -> None:
        wm = WorktreeManager(base_dir=tmp_path / "wt_cu06")
        mock_client = AsyncMock(spec=VertexAgentClient)
        team = MultiAgentTeam(
            client=mock_client,
            worktree_manager=wm,
            repo_root=repo_cu06,
        )

        unauthorized_attempt_detected = False

        async def custom_spawn(assignment: TaskAssignment, depth: int = 0) -> WorkerResult:
            nonlocal unauthorized_attempt_detected
            if assignment.task_id == "node_a":
                res = write_file(
                    "module_a.py",
                    "def func_a() -> int:\n    return 100\n",
                    overwrite=True,
                )
                assert res.get("status") == "success"
                return WorkerResult(
                    task_id=assignment.task_id,
                    role=AgentRole.WORKER,
                    status="success",
                    summary="Updated module_a.py",
                    files_modified=["module_a.py"],
                )

            if assignment.task_id == "node_b":
                # 1. Attempt unauthorized write to protected_config.py
                unauth_res = write_file(
                    "protected_config.py",
                    "SECRET_KEY = 'hacked'\n",
                    overwrite=True,
                )
                assert unauth_res.get("violation_type") == "ALLOWED_FILES_ACL_VIOLATION"
                unauthorized_attempt_detected = True

                # 2. Valid write to module_b.py
                valid_res = write_file(
                    "module_b.py",
                    "def func_b() -> int:\n    return 200\n",
                    overwrite=True,
                )
                assert valid_res.get("status") == "success"
                return WorkerResult(
                    task_id=assignment.task_id,
                    role=AgentRole.WORKER,
                    status="success",
                    summary="Updated module_b.py",
                    files_modified=["module_b.py"],
                )

            # QA Node
            return WorkerResult(
                task_id=assignment.task_id,
                role=AgentRole.QA,
                status="success",
                summary="PASS: QA validated both modules.",
            )

        team.spawn_worker = custom_spawn  # type: ignore[method-assign]

        node_a = SwarmDAGNode(
            node_id="node_a",
            role=AgentRole.WORKER,
            objective="Update module_a",
            allowed_files=["module_a.py"],
            dependencies=[],
        )
        node_b = SwarmDAGNode(
            node_id="node_b",
            role=AgentRole.WORKER,
            objective="Update module_b",
            allowed_files=["module_b.py"],
            dependencies=[],
        )
        node_qa = SwarmDAGNode(
            node_id="node_qa",
            role=AgentRole.QA,
            objective="QA verification of both modules",
            allowed_files=[],
            dependencies=["node_a", "node_b"],
        )

        plan = SwarmDAGPlan(
            plan_id="cu06-plan",
            goal="Concurrently upgrade module_a and module_b with zero ACL leaks",
            rationale="Disjoint module modifications",
            nodes={"node_a": node_a, "node_b": node_b, "node_qa": node_qa},
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
        assert unauthorized_attempt_detected is True

        # Assert primary workspace on disk updated cleanly
        assert (repo_cu06 / "module_a.py").read_text(
            encoding="utf-8"
        ) == "def func_a() -> int:\n    return 100\n"
        assert (repo_cu06 / "module_b.py").read_text(
            encoding="utf-8"
        ) == "def func_b() -> int:\n    return 200\n"
        assert (repo_cu06 / "protected_config.py").read_text(
            encoding="utf-8"
        ) == "SECRET_KEY = 'super_secret'\n"
