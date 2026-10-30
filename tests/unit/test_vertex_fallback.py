"""Unit tests for VertexAgentClient stream_chat fallback router failover on 429."""

import pytest

from cli_agent.circuit_breaker import (
    CircuitState,
    FallbackRouter,
    ModelEndpoint,
    QuotaCircuitBreaker,
)
from cli_agent.token_provider import TokenProvider
from cli_agent.vertex import VertexAgentClient


class _MockFixedTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "mock-token-fallback"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_chat_fallback_on_429(httpx_mock) -> None:
    breaker = QuotaCircuitBreaker()
    ep_tier1 = ModelEndpoint("gemini-3.8-flash", "global", tier=1)
    ep_tier2 = ModelEndpoint("gemini-2.5-pro", "us-central1", tier=2)
    router = FallbackRouter(breaker, [ep_tier1, ep_tier2])

    # Tier 1 returns 429 rate limit
    httpx_mock.add_response(
        status_code=429,
        text='{"error": {"message": "Quota exceeded for gemini-3.8-flash"}}',
    )
    # Tier 2 returns successful 200 SSE stream
    sse_data = (
        'data: {"candidates": [{"content": {"parts": [{"text": "Fallback "}]}}]}\n\n'
        'data: {"candidates": [{"content": {"parts": [{"text": "succeeded!"}]}}]}\n\n'
        "data: [DONE]\n\n"
    )
    httpx_mock.add_response(
        status_code=200,
        text=sse_data,
        headers={"content-type": "text/event-stream"},
    )

    client = VertexAgentClient(
        project_id="test-project",
        location="global",
        model_name="gemini-3.8-flash",
        token_provider=_MockFixedTokenProvider(),
        fallback_router=router,
        circuit_breaker=breaker,
    )

    messages = [{"role": "user", "parts": [{"text": "Hello fallback"}]}]
    tokens: list[str] = []
    async for chunk in client.stream_chat(messages):
        tokens.append(chunk)

    assert "".join(tokens) == "Fallback succeeded!"
    assert client.model_name == "gemini-2.5-pro"
    assert client.location == "us-central1"
    assert breaker.get_state(ep_tier1) == CircuitState.OPEN
    assert breaker.get_state(ep_tier2) == CircuitState.CLOSED
