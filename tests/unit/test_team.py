"""
Unit tests for Hierarchical Multi-Agent Teams (Swarm) module.
"""

from unittest.mock import AsyncMock

import pytest

from cli_agent.team import (
    MAX_DELEGATION_DEPTH,
    ROLE_ALLOWED_TOOLS,
    AgentRole,
    MultiAgentTeam,
    TaskAssignment,
    set_active_team,
)
from cli_agent.tools import delegate_task
from cli_agent.vertex import VertexAgentClient


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock(spec=VertexAgentClient)
    client.chat = AsyncMock(return_value="Architect/Worker/QA analysis completed.")
    return client


@pytest.mark.unit
class TestAgentRoleMatrix:
    def test_role_privilege_isolation(self) -> None:
        architect_tools = ROLE_ALLOWED_TOOLS[AgentRole.ARCHITECT]
        worker_tools = ROLE_ALLOWED_TOOLS[AgentRole.WORKER]
        qa_tools = ROLE_ALLOWED_TOOLS[AgentRole.QA]

        # Architect has delegation
        assert "delegate_task" in architect_tools
        assert "delegate_task" not in worker_tools
        assert "delegate_task" not in qa_tools

        # Worker has mutation privileges
        assert "write_file" in worker_tools
        assert "edit_file" in worker_tools
        assert "run_command" not in worker_tools

        # QA has verification only (zero mutations)
        assert "run_command" in qa_tools
        assert "write_file" not in qa_tools
        assert "edit_file" not in qa_tools


@pytest.mark.unit
@pytest.mark.asyncio
class TestMultiAgentTeamWorkerSpawning:
    async def test_spawn_worker_success(self, mock_client: AsyncMock) -> None:
        team = MultiAgentTeam(client=mock_client)
        assignment = TaskAssignment(
            task_id="task-001",
            role=AgentRole.WORKER,
            objective="Add validation helper",
            allowed_files=["src/val.py"],
        )

        events: list[tuple[str, str]] = []

        def on_event(etype: str, msg: str) -> None:
            events.append((etype, msg))

        res = await team.spawn_worker(assignment, depth=1, on_event=on_event)

        assert res.status == "success"
        assert res.task_id == "task-001"
        assert res.role == AgentRole.WORKER
        assert "Architect/Worker/QA" in res.summary
        assert any(e[0] == "spawn" for e in events)

        # Check client.chat called with role-filtered tools
        mock_client.chat.assert_called_once()
        _, kwargs = mock_client.chat.call_args
        tools_decl = kwargs.get("tools_declarations", [])
        tool_names = {t["name"] for t in tools_decl}
        assert "write_file" in tool_names
        assert "run_command" not in tool_names
        assert "delegate_task" not in tool_names

    async def test_spawn_worker_recursion_guard(self, mock_client: AsyncMock) -> None:
        team = MultiAgentTeam(client=mock_client, max_depth=MAX_DELEGATION_DEPTH)
        assignment = TaskAssignment(
            task_id="deep-task",
            role=AgentRole.WORKER,
            objective="Deep nested task",
        )

        res = await team.spawn_worker(assignment, depth=MAX_DELEGATION_DEPTH)
        assert res.status == "rejected"
        assert "depth limit" in res.summary
        assert res.error is not None
        mock_client.chat.assert_not_called()

    async def test_spawn_worker_client_error_handling(self, mock_client: AsyncMock) -> None:
        mock_client.chat.side_effect = RuntimeError("Vertex AI rate limit 429")
        team = MultiAgentTeam(client=mock_client)

        assignment = TaskAssignment(
            task_id="failing-task",
            role=AgentRole.QA,
            objective="Run test suite",
        )
        res = await team.spawn_worker(assignment, depth=0)
        assert res.status == "failed"
        assert "Vertex AI rate limit 429" in (res.error or "")


@pytest.mark.unit
@pytest.mark.asyncio
class TestMultiAgentTeamWorkflow:
    async def test_run_workflow_odd_pipeline(self, mock_client: AsyncMock) -> None:
        team = MultiAgentTeam(client=mock_client)
        events: list[str] = []

        def on_event(etype: str, msg: str) -> None:
            events.append(f"{etype}:{msg}")

        result = await team.run_workflow("Implement user auth caching", on_event=on_event)

        assert result.status == "success"
        assert result.goal == "Implement user auth caching"
        assert len(result.worker_results) == 1
        assert result.qa_result is not None
        assert "Phase 1: Lead Architect" in "\n".join(events)
        assert "Phase 2: Feature Worker" in "\n".join(events)
        assert "Phase 3: Independent QA" in "\n".join(events)


@pytest.mark.unit
@pytest.mark.asyncio
class TestDelegateTaskTool:
    async def test_delegate_task_without_active_team(self) -> None:
        set_active_team(None)
        res = await delegate_task(role="worker", objective="Build feature")
        assert "error" in res
        assert "not active" in res["error"]

    async def test_delegate_task_with_active_team(self, mock_client: AsyncMock) -> None:
        team = MultiAgentTeam(client=mock_client)
        set_active_team(team)

        try:
            res = await delegate_task(
                role="worker",
                objective="Create helper function",
                allowed_files=["src/helper.py"],
            )
            assert res.get("status") == "success"
            assert res.get("role") == "worker"
        finally:
            set_active_team(None)
