"""
Integration test for CU-03: Cross-Session Persistent Memory Sync & Scoping.
Verifies memory stored in Session 1 is recalled in Session 2 for the same workspace,
and isolated from unrelated workspaces.
"""

from pathlib import Path

import pytest

from cli_agent.memory import MemoryStore
from cli_agent.token_provider import TokenProvider
from cli_agent.vertex import VertexAgentClient


class MockFixedTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "mock-auth-token-cu03"


@pytest.mark.integration
@pytest.mark.asyncio
class TestCU03PersistentMemory:
    async def test_cross_session_memory_sync_and_isolation(self, httpx_mock, tmp_path: Path):
        db_file = tmp_path / "cu03_shared.db"
        shared_store = MemoryStore(db_path=db_file)

        # -------------------------------------------------------------
        # SESSION 1: In 'project-frontend', agent saves an architecture decision
        # -------------------------------------------------------------
        httpx_mock.add_response(
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "save_memory",
                                        "args": {
                                            "title": "Routing Architecture",
                                            "content": "Use TanStack Router for type-safe routes",
                                            "memory_type": "decision",
                                            "scope": "project",
                                        },
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )
        httpx_mock.add_response(
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Persisted TanStack Router decision to memory."}]
                        }
                    }
                ]
            }
        )

        client_session1 = VertexAgentClient(
            memory_store=shared_store,
            token_provider=MockFixedTokenProvider(),
        )

        res1 = await client_session1.chat(
            [{"role": "user", "parts": [{"text": "remember that we chose TanStack Router"}]}]
        )
        assert "Persisted TanStack Router" in res1

        # -------------------------------------------------------------
        # SESSION 2: New session in 'project-frontend' recalls that decision
        # -------------------------------------------------------------
        memories = shared_store.search_memory("TanStack Router")
        assert len(memories) >= 1
        assert "TanStack Router for type-safe routes" in memories[0]["content"]

        # -------------------------------------------------------------
        # SESSION 3: Different project 'backend-go' searches memory with its project_key
        # -------------------------------------------------------------
        backend_memories = shared_store.search_memory("TanStack Router", project_key="backend-go")
        # Should NOT find the frontend decision in backend project
        assert len(backend_memories) == 0
