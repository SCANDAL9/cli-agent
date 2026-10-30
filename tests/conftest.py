"""
Global pytest fixtures and configuration for cli-agent test suite.
"""

from collections.abc import Generator
from pathlib import Path

import pytest

from cli_agent.memory import MemoryStore
from cli_agent.token_provider import TokenProvider


class MockTokenProvider(TokenProvider):
    """TokenProvider for testing that returns a fixed token without invoking gcloud."""

    def __init__(self, token: str = "mock-vertex-access-token"):
        super().__init__()
        self._mock_token = token

    async def get_token(self) -> str:
        return self._mock_token


@pytest.fixture
def mock_token_provider() -> MockTokenProvider:
    return MockTokenProvider()


@pytest.fixture
def temp_memory_store(tmp_path: Path) -> Generator[MemoryStore, None, None]:
    db_file = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=db_file)
    yield store
