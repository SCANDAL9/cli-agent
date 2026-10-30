"""
Unit tests for TokenProvider (caching, TTL expiration, async fetch, error states).
"""

from unittest.mock import AsyncMock, patch

import pytest

from cli_agent.token_provider import TokenProvider


@pytest.mark.unit
@pytest.mark.asyncio
class TestTokenProvider:
    async def test_env_var_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("GCLOUD_ACCESS_TOKEN", "custom-env-token-xyz")
        provider = TokenProvider()
        token = await provider.get_token()
        assert token == "custom-env-token-xyz"

    async def test_token_caching_within_ttl(self):
        provider = TokenProvider(ttl_buffer_seconds=60, default_ttl_seconds=3600)

        # Mock the internal _fetch_token method
        fetch_mock = AsyncMock(return_value="fresh-fetched-token-123")
        provider._fetch_token = fetch_mock  # type: ignore

        token1 = await provider.get_token()
        token2 = await provider.get_token()

        assert token1 == "fresh-fetched-token-123"
        assert token2 == "fresh-fetched-token-123"
        assert fetch_mock.call_count == 1  # Only fetched once due to caching

    async def test_cache_clear_forces_refresh(self):
        provider = TokenProvider()
        fetch_mock = AsyncMock(side_effect=["token-turn-1", "token-turn-2"])
        provider._fetch_token = fetch_mock  # type: ignore

        t1 = await provider.get_token()
        assert t1 == "token-turn-1"

        provider.clear_cache()

        t2 = await provider.get_token()
        assert t2 == "token-turn-2"
        assert fetch_mock.call_count == 2

    async def test_gcloud_execution_failure_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("GCLOUD_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("VERTEX_TOKEN", raising=False)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (
            b"",
            b"Authentication failed. Please run gcloud auth login.",
        )
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            provider = TokenProvider()
            with pytest.raises(RuntimeError, match="Failed to obtain gcloud token"):
                await provider.get_token()

    async def test_gcloud_empty_token_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("GCLOUD_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("VERTEX_TOKEN", raising=False)

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"   \n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
            provider = TokenProvider()
            with pytest.raises(RuntimeError, match="empty access token"):
                await provider.get_token()
