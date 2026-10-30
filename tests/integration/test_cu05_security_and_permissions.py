"""
Integration test for CU-05: Security Policy & Destructive Tool Gate.
Verifies interactive Human-in-the-Loop permission interception in the agent loop.
"""

from pathlib import Path

import pytest

from cli_agent.security import RiskLevel, SecurityPolicy
from cli_agent.token_provider import TokenProvider
from cli_agent.vertex import VertexAgentClient


class MockFixedTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "mock-auth-token-cu05"


@pytest.mark.integration
@pytest.mark.asyncio
class TestCU05SecurityAndPermissions:
    async def test_denied_tool_is_not_executed_and_model_receives_error(
        self, httpx_mock, tmp_path: Path
    ):
        danger_file = tmp_path / "critical.py"
        danger_file.write_text("ORIGINAL_CONTENT = True\n", encoding="utf-8")

        # Mock Turn 1: Model requests destructive write_file
        tool_call = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "write_file",
                                    "args": {
                                        "path": str(danger_file),
                                        "content": "OVERWRITTEN = True\n",
                                        "overwrite": True,
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        }

        # Mock Turn 2: Model receives rejection message and provides alternative
        polite_decline = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "I understand you declined the file modification. I will keep the original file untouched."
                            }
                        ]
                    }
                }
            ]
        }

        httpx_mock.add_response(json=tool_call)
        httpx_mock.add_response(json=polite_decline)

        # Policy callback explicitly denies permission
        callback_called = False

        def deny_all(tool_name: str, args: dict, risk: RiskLevel) -> bool:
            nonlocal callback_called
            callback_called = True
            return False

        policy = SecurityPolicy(auto_approve=False, confirm_callback=deny_all)
        client = VertexAgentClient(token_provider=MockFixedTokenProvider(), security_policy=policy)

        response = await client.chat(
            [{"role": "user", "parts": [{"text": "overwrite critical.py"}]}]
        )

        # Assertions
        assert callback_called is True
        assert "keep the original file untouched" in response
        # Verify file on disk was NEVER modified
        assert danger_file.read_text(encoding="utf-8") == "ORIGINAL_CONTENT = True\n"

    async def test_approved_tool_executes_successfully(self, httpx_mock, tmp_path: Path):
        dest_file = tmp_path / "allowed.txt"

        tool_call = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "write_file",
                                    "args": {
                                        "path": str(dest_file),
                                        "content": "hello approved",
                                    },
                                }
                            }
                        ]
                    }
                }
            ]
        }

        final_response = {"candidates": [{"content": {"parts": [{"text": "File created."}]}}]}

        httpx_mock.add_response(json=tool_call)
        httpx_mock.add_response(json=final_response)

        # Callback approves
        policy = SecurityPolicy(auto_approve=False, confirm_callback=lambda name, args, risk: True)
        client = VertexAgentClient(token_provider=MockFixedTokenProvider(), security_policy=policy)

        response = await client.chat([{"role": "user", "parts": [{"text": "create allowed.txt"}]}])

        assert response == "File created."
        assert dest_file.exists()
        assert dest_file.read_text(encoding="utf-8") == "hello approved"
