"""
Integration test for CU-02: Agentic Tool Calling & Dispatch Loop.
Verifies end-to-end multi-turn interaction between VertexAgentClient and local tools.
"""

from pathlib import Path

import pytest

from cli_agent.memory import MemoryStore
from cli_agent.token_provider import TokenProvider
from cli_agent.vertex import VertexAgentClient


class MockFixedTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "mock-auth-token-cu02"


@pytest.mark.integration
@pytest.mark.asyncio
class TestCU02ToolOrchestration:
    async def test_agent_tool_calling_multi_turn_execution(self, httpx_mock, tmp_path: Path):
        test_file = tmp_path / "greeting.py"
        test_file.write_text("print('hello v1')\n", encoding="utf-8")

        # Mock Turn 1: Model requests edit_file
        tool_request = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "edit_file",
                                    "args": {
                                        "path": str(test_file),
                                        "target": "v1",
                                        "replacement": "v2",
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        }

        # Mock Turn 2: Model receives tool output and responds with final text
        final_response = {
            "candidates": [
                {"content": {"parts": [{"text": "Successfully updated greeting.py to v2."}]}}
            ]
        }

        httpx_mock.add_response(json=tool_request)
        httpx_mock.add_response(json=final_response)

        executed_tools = []

        def on_tool(name: str, args: dict):
            executed_tools.append((name, args))

        db_path = tmp_path / "cu02_mem.db"
        client = VertexAgentClient(
            memory_store=MemoryStore(db_path=db_path),
            token_provider=MockFixedTokenProvider(),
        )

        messages = [{"role": "user", "parts": [{"text": "update greeting to v2"}]}]
        result = await client.chat(messages, on_tool_call=on_tool)

        # Assertions
        assert result == "Successfully updated greeting.py to v2."
        assert len(executed_tools) == 1
        assert executed_tools[0][0] == "edit_file"
        assert test_file.read_text(encoding="utf-8") == "print('hello v2')\n"
