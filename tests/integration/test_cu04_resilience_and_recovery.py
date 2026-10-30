"""
Integration test for CU-04: Resiliency, Rate Limits & Error Handling.
Verifies graceful failure modes when encountering API errors and limits.
"""

import pytest

from cli_agent.token_provider import TokenProvider
from cli_agent.vertex import VertexAgentClient


class MockFixedTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "mock-auth-token-cu04"


@pytest.mark.integration
@pytest.mark.asyncio
class TestCU04ResilienceAndRecovery:
    async def test_vertex_429_rate_limit_raises_clean_runtime_error(self, httpx_mock):
        httpx_mock.add_response(
            status_code=429,
            text='{"error": {"message": "Resource has been exhausted (e.g. check quota)."}}',
        )

        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        messages = [{"role": "user", "parts": [{"text": "heavy query"}]}]

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            await client.chat(messages)

    async def test_vertex_500_api_error_surfaces_message(self, httpx_mock):
        httpx_mock.add_response(
            status_code=500,
            text='{"error": {"message": "Internal backend error in Vertex AI"}}',
        )

        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        messages = [{"role": "user", "parts": [{"text": "query"}]}]

        with pytest.raises(RuntimeError, match="Vertex AI API error \\(500\\)"):
            await client.chat(messages)

    async def test_empty_candidates_returns_fallback_message(self, httpx_mock):
        httpx_mock.add_response(json={"candidates": []})

        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        messages = [{"role": "user", "parts": [{"text": "query"}]}]

        res = await client.chat(messages)
        assert res == "No response candidates returned by model."

    async def test_max_turns_exceeded_returns_limit_message(self, httpx_mock):
        # Always request a tool call endlessly
        infinite_tool_call = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "list_directory",
                                    "args": {"path": "."},
                                }
                            }
                        ]
                    }
                }
            ]
        }

        # Return tool request 3 times
        httpx_mock.add_response(json=infinite_tool_call)
        httpx_mock.add_response(json=infinite_tool_call)
        httpx_mock.add_response(json=infinite_tool_call)

        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        messages = [{"role": "user", "parts": [{"text": "loop"}]}]

        res = await client.chat(messages, max_turns=3)
        assert res == "Agent reached maximum tool iterations limit."
