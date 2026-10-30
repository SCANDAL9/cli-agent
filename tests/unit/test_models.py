"""
Unit tests for the dynamic Model Garden discovery and model management module.
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cli_agent.main import show_models
from cli_agent.models import (
    BASELINE_CATALOG,
    detect_publisher,
    fetch_model_garden_catalog,
    is_viable_coding_model,
    parse_model_entry,
)
from cli_agent.vertex import VertexAgentClient


@pytest.mark.unit
class TestModelCatalogFiltering:
    def test_detect_publisher(self) -> None:
        assert detect_publisher("publishers/anthropic/models/claude-sonnet-5") == "anthropic"
        assert detect_publisher("claude-opus-5") == "anthropic"
        assert detect_publisher("publishers/xai/models/grok-4.7") == "xai"
        assert detect_publisher("grok-beta") == "xai"
        assert detect_publisher("publishers/mistralai/models/codestral-2") == "mistralai"
        assert detect_publisher("mistral-large") == "mistralai"
        assert detect_publisher("publishers/openai/models/gpt-4o") == "openai"
        assert detect_publisher("gemini-3.8-flash") == "google"
        assert detect_publisher("publishers/google/models/gemini-4-argon") == "google"

    def test_is_viable_coding_model_valid(self) -> None:
        assert is_viable_coding_model("publishers/google/models/gemini-3.8-flash") is True
        assert is_viable_coding_model("publishers/anthropic/models/claude-sonnet-5") is True
        assert is_viable_coding_model("publishers/xai/models/grok-4.7") is True
        assert is_viable_coding_model("publishers/mistralai/models/codestral-2") is True
        assert is_viable_coding_model("publishers/deepseek-ai/models/deepseek-coder") is True

    def test_is_viable_coding_model_filtered_out(self) -> None:
        # Non-coding modalities
        assert is_viable_coding_model("publishers/google/models/text-embedding-004") is False
        assert is_viable_coding_model("publishers/google/models/image2video-fast") is False
        assert is_viable_coding_model("publishers/google/models/audio-transcribe-v2") is False
        assert is_viable_coding_model("publishers/google/models/tts-chirp-3") is False
        assert is_viable_coding_model("publishers/google/models/veo-generate-001") is False

        # Obsolete / deprecated checkpoints
        assert is_viable_coding_model("publishers/google/models/gemma@gemma-1-7b") is False
        assert is_viable_coding_model("publishers/meta/models/openllama-7b") is False

        # Non-whitelisted publishers or invalid format
        assert is_viable_coding_model("invalid-model-name") is False
        assert is_viable_coding_model("publishers/unknown-org/models/random-model") is False

    def test_parse_model_entry(self) -> None:
        valid_raw = {
            "name": "publishers/anthropic/models/claude-sonnet-5@default",
            "launchStage": "GA",
            "versionId": "default",
        }
        parsed = parse_model_entry(valid_raw)
        assert parsed is not None
        assert parsed.publisher == "anthropic"
        assert parsed.short_name == "claude-sonnet-5@default"
        assert parsed.launch_stage == "GA"
        assert parsed.can_predict is True

        invalid_raw = {
            "name": "publishers/google/models/text-embedding-preview",
            "launchStage": "PUBLIC_PREVIEW",
        }
        assert parse_model_entry(invalid_raw) is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchModelGardenCatalog:
    async def test_fetch_from_cache_hit(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "model_catalog.json"
        cached_data = {
            "timestamp": time.time(),
            "models": [
                {
                    "model_id": "publishers/google/models/gemini-test",
                    "publisher": "google",
                    "short_name": "gemini-test",
                    "launch_stage": "GA",
                    "version_id": "default",
                    "can_predict": True,
                }
            ],
        }
        cache_file.write_text(json.dumps(cached_data), encoding="utf-8")

        result = await fetch_model_garden_catalog(
            billing_project="mock-gcp-project",
            force_refresh=False,
            cache_path=cache_file,
        )

        assert len(result) == 1
        assert result[0].short_name == "gemini-test"

    async def test_fetch_with_subprocess_success(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "cache.json"
        mock_output = json.dumps(
            [
                {
                    "name": "publishers/google/models/gemini-3.8-flash",
                    "launchStage": "GA",
                    "versionId": "default",
                },
                {
                    "name": "publishers/anthropic/models/claude-sonnet-5",
                    "launchStage": "GA",
                    "versionId": "default",
                },
                {
                    "name": "publishers/google/models/text-embedding-005",
                    "launchStage": "GA",
                    "versionId": "default",
                },
            ]
        ).encode("utf-8")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (mock_output, b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            catalog = await fetch_model_garden_catalog(
                billing_project="mock-gcp-project",
                force_refresh=True,
                cache_path=cache_file,
            )

        assert len(catalog) == 2
        shorts = {m.short_name for m in catalog}
        assert "gemini-3.8-flash" in shorts
        assert "claude-sonnet-5" in shorts
        assert cache_file.is_file()

    async def test_fetch_subprocess_failure_falls_back_to_baseline(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "non_existent.json"

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("gcloud not found")):
            catalog = await fetch_model_garden_catalog(
                billing_project="mock-gcp-project",
                force_refresh=True,
                cache_path=cache_file,
            )

        assert len(catalog) == len(BASELINE_CATALOG)
        assert catalog[0].publisher == BASELINE_CATALOG[0].publisher


@pytest.mark.unit
class TestVertexClientModelSwitching:
    def test_switch_model_endpoints(self) -> None:
        client = VertexAgentClient(
            project_id="mock-gcp-project",
            location="us-central1",
            model_name="gemini-3.8-flash",
        )
        assert client.publisher == "google"
        assert "publishers/google/models/gemini-3.8-flash:generateContent" in client.endpoint

        # Switch to Anthropic Claude
        client.switch_model("publishers/anthropic/models/claude-sonnet-5")
        assert client.publisher == "anthropic"
        assert "publishers/anthropic/models/claude-sonnet-5:rawPredict" in client.endpoint

        # Switch to xAI Grok
        client.switch_model("grok-4.7")
        assert client.publisher == "xai"
        assert "publishers/xai/models/grok-4.7:rawPredict" in client.endpoint

        # Switch to Mistral Codestral
        client.switch_model("codestral-2")
        assert client.publisher == "mistralai"
        assert "publishers/mistralai/models/codestral-2:rawPredict" in client.endpoint


@pytest.mark.unit
@pytest.mark.asyncio
class TestShowModelsUI:
    async def test_show_models_renders_table(self) -> None:
        entries = await show_models(
            billing_project="mock-gcp-project",
            active_model="gemini-3.8-flash",
            force_refresh=False,
        )
        assert len(entries) > 0
        assert any(e.short_name == "gemini-3.8-flash" for e in entries)
