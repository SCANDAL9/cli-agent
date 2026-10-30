"""
Unit tests for the /doctor diagnostic module.
"""

from pathlib import Path

import pytest

from cli_agent.doctor import run_diagnostics
from cli_agent.memory import MemoryStore
from cli_agent.token_provider import TokenProvider


class MockHealthyTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        return "valid-gcloud-oauth2-token-example"


class MockFailingTokenProvider(TokenProvider):
    async def get_token(self) -> str:
        raise RuntimeError("gcloud command not found")


@pytest.mark.unit
@pytest.mark.asyncio
class TestDoctor:
    async def test_doctor_healthy_status(self, tmp_path: Path):
        db_file = tmp_path / "doctor_healthy.db"
        store = MemoryStore(db_path=db_file)
        store.save_memory(title="Diag Note", content="Initial check")

        report = await run_diagnostics(
            token_provider=MockHealthyTokenProvider(),
            memory_store=store,
        )

        assert report["healthy"] is True
        assert report["token_status"] == "valid"
        assert report["database_status"] == "healthy"
        assert report["code_graph_status"] == "healthy"
        assert report["tdd_status"] in ("ready", "partial")
        assert len(report["issues"]) == 0

    async def test_doctor_reports_failing_token(self, tmp_path: Path):
        store = MemoryStore(db_path=tmp_path / "doctor.db")
        report = await run_diagnostics(
            token_provider=MockFailingTokenProvider(),
            memory_store=store,
        )

        assert report["healthy"] is False
        assert report["token_status"] == "error"
        assert any("gcloud command not found" in issue for issue in report["issues"])
