"""
Integration test for CU-01: REPL Journey & Conversation Turns.
Verifies pure text generation, multi-turn messages, and SSE streaming responses.
"""

import pytest

from cli_agent.token_provider import TokenProvider
from cli_agent.vertex import VertexAgentClient


class MockFixedTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "mock-auth-token-cu01"


@pytest.mark.integration
@pytest.mark.asyncio
class TestCU01ReplJourney:
    async def test_direct_text_response_without_tools(self, httpx_mock):
        model_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Hello! I am ready to help you architect and develop code."}
                        ]
                    }
                }
            ]
        }
        httpx_mock.add_response(json=model_reply)

        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        messages = [{"role": "user", "parts": [{"text": "Hello"}]}]

        response = await client.chat(messages)
        assert response == "Hello! I am ready to help you architect and develop code."

    async def test_streaming_sse_yields_text_chunks(self, httpx_mock):
        sse_data = (
            'data: {"candidates": [{"content": {"parts": [{"text": "Chunk 1 "}]}}]}\n\n'
            'data: {"candidates": [{"content": {"parts": [{"text": "Chunk 2"}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        httpx_mock.add_response(text=sse_data, headers={"content-type": "text/event-stream"})

        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        messages = [{"role": "user", "parts": [{"text": "Stream test"}]}]

        chunks = []
        async for chunk in client.stream_chat(messages):
            chunks.append(chunk)

        assert "".join(chunks) == "Chunk 1 Chunk 2"

    async def test_streaming_sse_with_tool_calling(self, httpx_mock, tmp_path):
        target = tmp_path / "stream_tool.txt"

        # SSE Stream 1: Model requests write_file
        sse_tool = (
            'data: {"candidates": [{"content": {"parts": [{"functionCall": {"name": "write_file", "args": {"path": "'
            + str(target)
            + '", "content": "streamed-file"}}}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        # SSE Stream 2: Model outputs final message after receiving tool response
        sse_final = (
            'data: {"candidates": [{"content": {"parts": [{"text": "File created via stream."}]}}]}\n\n'
            "data: [DONE]\n\n"
        )

        httpx_mock.add_response(text=sse_tool, headers={"content-type": "text/event-stream"})
        httpx_mock.add_response(text=sse_final, headers={"content-type": "text/event-stream"})

        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        messages = [{"role": "user", "parts": [{"text": "create file"}]}]

        collected = []
        async for token in client.stream_chat(messages):
            collected.append(token)

        assert "".join(collected) == "File created via stream."
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "streamed-file"

    async def test_streaming_sse_errors(self, httpx_mock):
        httpx_mock.add_response(status_code=429, text='{"error": "rate limit"}')
        client = VertexAgentClient(token_provider=MockFixedTokenProvider())
        with pytest.raises(RuntimeError, match="rate limit"):
            async for _ in client.stream_chat([{"role": "user", "parts": [{"text": "hi"}]}]):
                pass

        httpx_mock.add_response(status_code=500, text="Internal Server Error")
        with pytest.raises(RuntimeError, match="Vertex AI API error \\(500\\)"):
            async for _ in client.stream_chat([{"role": "user", "parts": [{"text": "hi"}]}]):
                pass
